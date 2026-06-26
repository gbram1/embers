"""Tests for the ColdStartLoader fast/slow-path fork and snapshot invalidation —
the Phase 4↔5 seam. GPU ops are faked; this is pure orchestration + correctness."""

from embers import fingerprint as fp_mod
from embers.fingerprint import Fingerprint
from embers.loader import (
    ColdStartLoader,
    DiskSnapshotStore,
    Snapshot,
    SnapshotStore,
    default_fingerprint_fn,
)


def fp(**over) -> Fingerprint:
    base = dict(
        weights_hash="w", model_version="m", engine_version="vllm-0.8.5",
        gpu_type="NVIDIA A40", driver_cuda_version="580", dtype="bf16",
        max_seq_len=4096, tensor_parallel=1, captured_batch_shapes=(1, 2),
    )
    base.update(over)
    return Fingerprint(**base)


def make_loader(fingerprint, store=None):
    """Loader whose GPU ops just record calls and return tagged backends."""
    calls = {"cold_load": 0, "capture": 0, "restore": 0}

    def cold_load(model, gpu):
        calls["cold_load"] += 1
        return f"COLD:{model}@{gpu}"

    def capture(model, gpu, backend):
        calls["capture"] += 1
        return f"SNAP:{model}"

    def restore(model, gpu, snap):
        calls["restore"] += 1
        return f"RESTORED:{model}@{gpu}"

    fp_fn = fingerprint if callable(fingerprint) else (lambda m, g: fingerprint)
    loader = ColdStartLoader(fingerprint_fn=fp_fn, cold_load=cold_load,
                             capture=capture, restore=restore, store=store)
    return loader, calls


def test_first_launch_is_slow_path_and_captures():
    loader, calls = make_loader(fp())
    out = loader.launch("m", "g0")
    assert out == "COLD:m@g0"            # cold-loaded
    assert calls == {"cold_load": 1, "capture": 1, "restore": 0}
    assert loader.cold_loads == 1 and loader.restores == 0
    assert loader.store.get("m") is not None  # snapshot captured for next time


def test_second_launch_is_fast_path_restore():
    loader, calls = make_loader(fp())
    loader.launch("m", "g0")            # slow (captures)
    out = loader.launch("m", "g1")     # snapshot now exists + fingerprint matches
    assert out == "RESTORED:m@g1"
    assert calls == {"cold_load": 1, "capture": 1, "restore": 1}
    assert loader.restores == 1


def test_changed_weights_invalidate_snapshot():
    # fingerprint flips weights_hash on the 2nd launch (weights overwritten)
    seq = iter([fp(weights_hash="v1"), fp(weights_hash="v2")])
    loader, calls = make_loader(lambda m, g: next(seq))
    loader.launch("m", "g0")            # slow, captures snapshot tagged v1
    out = loader.launch("m", "g0")     # current is v2 → mismatch → slow again
    assert out == "COLD:m@g0"
    assert calls["restore"] == 0
    assert loader.cold_loads == 2
    assert loader.invalidations == 1


def test_cross_gpu_invalidates_snapshot():
    seq = iter([fp(gpu_type="NVIDIA A40"), fp(gpu_type="NVIDIA L4")])
    loader, calls = make_loader(lambda m, g: next(seq))
    loader.launch("m", "g0")
    loader.launch("m", "g0")
    assert calls["restore"] == 0       # never restore across GPU types
    assert loader.invalidations == 1


def test_distinct_models_have_distinct_snapshots():
    loader, calls = make_loader(fp())
    loader.launch("a", "g0")
    loader.launch("b", "g0")
    assert calls["cold_load"] == 2     # each cold-loads once
    loader.launch("a", "g0")           # 'a' restores
    assert calls["restore"] == 1


def test_recapture_after_invalidation_then_fast():
    seq = iter([fp(weights_hash="v1"), fp(weights_hash="v2"),
                fp(weights_hash="v2")])
    loader, calls = make_loader(lambda m, g: next(seq))
    loader.launch("m", "g0")           # slow, snap=v1
    loader.launch("m", "g0")           # v2 mismatch → slow, snap=v2
    loader.launch("m", "g0")           # v2 matches → fast
    assert loader.cold_loads == 2 and loader.restores == 1


def test_store_is_shareable():
    store = SnapshotStore()
    l1, _ = make_loader(fp(), store=store)
    l1.launch("m", "g0")
    l2, calls2 = make_loader(fp(), store=store)
    l2.launch("m", "g0")               # second loader reuses the shared snapshot
    assert calls2["restore"] == 1


# --- DiskSnapshotStore (persistence across restarts) ----------------------

def test_disk_store_persists_across_instances(tmp_path):
    s1 = DiskSnapshotStore(tmp_path)
    s1.put(Snapshot("my/model", fp(), handle={"pid": 123, "port": 19000}))
    # a fresh store over the same dir (simulating a control-plane restart)
    s2 = DiskSnapshotStore(tmp_path)
    got = s2.get("my/model")
    assert got is not None
    assert got.fingerprint == fp()
    assert got.handle == {"pid": 123, "port": 19000}


def test_disk_store_drop_removes_file(tmp_path):
    s = DiskSnapshotStore(tmp_path)
    s.put(Snapshot("m", fp(), handle="h"))
    s.drop("m")
    assert s.get("m") is None
    assert DiskSnapshotStore(tmp_path).get("m") is None   # gone from disk too


def test_disk_store_skips_corrupt_files(tmp_path):
    (tmp_path / "broken.json").write_text("{ not json")
    s = DiskSnapshotStore(tmp_path)        # must not raise on startup
    assert s.get("anything") is None


def test_disk_store_loader_invalidation_recaptures(tmp_path):
    # a persisted snapshot whose fingerprint no longer matches must recapture
    seq = iter([fp(weights_hash="v1"), fp(weights_hash="v2")])
    store = DiskSnapshotStore(tmp_path)
    loader, calls = make_loader(lambda m, g: next(seq), store=store)
    loader.launch("m", "g0")               # slow, persists v1
    loader.launch("m", "g0")               # v2 mismatch → recapture, not restore
    assert calls["restore"] == 0
    assert loader.invalidations == 1


# --- default_fingerprint_fn (real fingerprint wiring) ---------------------

def test_default_fingerprint_fn_builds_real_fingerprint(tmp_path, monkeypatch):
    (tmp_path / "w.safetensors").write_bytes(b"x" * 200)
    monkeypatch.setattr(fp_mod, "resolve_model_dir", lambda m, **k: tmp_path)
    monkeypatch.setattr(fp_mod, "_nvidia_smi",
                        lambda q: "NVIDIA A40" if q == "name" else "570")
    fn = default_fingerprint_fn(engine_version="vllm-0.8.5", dtype="bf16",
                                max_seq_len=4096, tensor_parallel=1,
                                captured_batch_shapes=(1, 2))
    fp_out = fn("my/model", "g0")
    assert fp_out.gpu_type == "NVIDIA A40"
    assert fp_out.model_version == "my/model"
    assert not fp_out.weights_hash.startswith("UNRESOLVED")
