"""GPU-free tests for the fingerprint gate — the restore correctness boundary
(never restore a stale or cross-GPU snapshot)."""

from embers import fingerprint as fp_mod
from embers.fingerprint import (
    Fingerprint,
    build_fingerprint,
    current_fingerprint,
    decide,
    hash_weights,
    resolve_model_dir,
)


def base() -> Fingerprint:
    return Fingerprint(
        weights_hash="deadbeef",
        model_version="qwen2.5-3b",
        engine_version="vllm-0.8.5",
        gpu_type="NVIDIA A40",
        driver_cuda_version="580.95.05",
        dtype="bfloat16",
        max_seq_len=4096,
        tensor_parallel=1,
        captured_batch_shapes=(1, 2, 4, 8),
    )


def replace(fp: Fingerprint, **kw) -> Fingerprint:
    d = {**fp.__dict__, **kw}
    return Fingerprint(**d)


def test_identical_restores():
    assert decide(base(), base()) is True


def test_changed_weights_invalidate():
    # weights overwritten in place — the hash must catch it
    assert decide(replace(base(), weights_hash="cafe"), base()) is False


def test_cross_gpu_hard_stop():
    assert decide(replace(base(), gpu_type="NVIDIA L4"), base()) is False


def test_driver_drift_hard_stop():
    assert decide(replace(base(), driver_cuda_version="575.00.00"), base()) is False


def test_batch_shape_change_invalidates():
    assert decide(replace(base(), captured_batch_shapes=(1, 2)), base()) is False


def test_digest_stable_and_distinct():
    assert base().digest() == base().digest()
    assert base().digest() != replace(base(), dtype="float16").digest()


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "fp.json"
    base().save(p)
    assert Fingerprint.load(p) == base()


def test_hash_weights_changes_with_content(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"weights-v1" * 1000)
    h1 = hash_weights(tmp_path)
    (tmp_path / "a.safetensors").write_bytes(b"weights-v2" * 1000)
    h2 = hash_weights(tmp_path)
    assert h1 != h2  # in-place weight edit must change the hash


def test_hash_weights_stable_and_order_independent(tmp_path):
    (tmp_path / "b.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "a.safetensors").write_bytes(b"y" * 100)
    assert hash_weights(tmp_path) == hash_weights(tmp_path)  # deterministic


def test_current_fingerprint_probes_gpu(monkeypatch):
    monkeypatch.setattr(fp_mod, "_nvidia_smi",
                        lambda q: "NVIDIA A40" if q == "name" else "580.95.05")
    fp = current_fingerprint(
        model_version="qwen2.5-3b", engine_version="vllm-0.8.5",
        weights_hash="abc", dtype="bfloat16", max_seq_len=4096,
        tensor_parallel=1, captured_batch_shapes=(1, 2, 4),
    )
    assert fp.gpu_type == "NVIDIA A40"
    assert fp.driver_cuda_version == "580.95.05"
    # a fresh probe with identical inputs should be restore-compatible
    monkeypatch.setattr(fp_mod, "_nvidia_smi",
                        lambda q: "NVIDIA A40" if q == "name" else "580.95.05")
    fp2 = current_fingerprint(
        model_version="qwen2.5-3b", engine_version="vllm-0.8.5",
        weights_hash="abc", dtype="bfloat16", max_seq_len=4096,
        tensor_parallel=1, captured_batch_shapes=(1, 2, 4),
    )
    assert decide(fp2, fp) is True


# --- resolve_model_dir & build_fingerprint --------------------------------

def test_resolve_local_dir(tmp_path):
    assert resolve_model_dir(str(tmp_path)) == tmp_path


def test_resolve_hf_cache_layout(tmp_path):
    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-3B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    got = resolve_model_dir("Qwen/Qwen2.5-3B", hf_home=str(tmp_path))
    assert got == snap


def test_resolve_returns_none_when_absent(tmp_path):
    assert resolve_model_dir("Missing/Model", hf_home=str(tmp_path)) is None


def test_build_fingerprint_hashes_real_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"real-weights" * 500)
    fp = build_fingerprint(
        "my/model", engine_version="vllm-0.8.5", dtype="bf16", max_seq_len=4096,
        tensor_parallel=1, captured_batch_shapes=(1, 2), model_dir=str(tmp_path),
        gpu_name="NVIDIA A40", driver="570")
    assert fp.gpu_type == "NVIDIA A40"
    assert fp.weights_hash == hash_weights(tmp_path)   # actual byte hash
    assert not fp.weights_hash.startswith("UNRESOLVED")


def test_build_fingerprint_marks_unresolved_weights(tmp_path):
    fp = build_fingerprint(
        "ghost/model", engine_version="v", dtype="bf16", max_seq_len=4096,
        tensor_parallel=1, captured_batch_shapes=(1,),
        gpu_name="g", driver="d")   # no model_dir, not in cache
    assert fp.weights_hash == "UNRESOLVED:ghost/model"
