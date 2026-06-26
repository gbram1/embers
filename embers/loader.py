"""Loader orchestrator — the fork between fast path and slow path. THIS is the
Phase 4↔5 seam: a `ColdStartLoader.launch(model, gpu_id) -> Backend` drops
straight into the autoscaler's `launch` callback.

    snapshot valid?  ──yes──►  restore GPU state directly (~9s, skip init)
                     ──no───►  cold-load + engine init (~57s), then capture a
                               fresh snapshot tagged with the fingerprint.

Validity is decided by the fingerprint (never guessed): a stale or cross-GPU
snapshot would serve silent wrong output, so a mismatch falls back to the slow
path and recaptures.

The GPU operations are injected so the orchestration is testable without a GPU;
the real implementations are the cuda-checkpoint capture/restore from
scripts/_rung2_* (run on a pod). Counters split scale-from-zero into fast
`restores` vs slow `cold_loads` — the platform's cold-start win, made visible.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embers.fingerprint import Fingerprint, build_fingerprint, decide


@dataclass
class Snapshot:
    """A captured post-init GPU state, tagged with the fingerprint it's valid
    for. `handle` is opaque to the orchestration (a disk path / restore token);
    for a persistent store it must be JSON-serialisable."""
    model: str
    fingerprint: Fingerprint
    handle: Any


class SnapshotStore:
    """Registry of model -> latest Snapshot. In-memory."""

    def __init__(self):
        self._by_model: dict[str, Snapshot] = {}

    def get(self, model: str) -> Snapshot | None:
        return self._by_model.get(model)

    def put(self, snapshot: Snapshot) -> None:
        self._by_model[snapshot.model] = snapshot

    def drop(self, model: str) -> None:
        self._by_model.pop(model, None)


class DiskSnapshotStore(SnapshotStore):
    """Snapshot registry persisted to disk (one JSON file per model), so the
    control plane recovers its snapshots across restarts. `handle` must be
    JSON-serialisable. Writes are atomic (tmp + rename)."""

    def __init__(self, directory: str | Path):
        super().__init__()
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def _path(self, model: str) -> Path:
        return self.dir / (model.replace("/", "--") + ".json")

    def _load_all(self) -> None:
        for f in self.dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                fp_d = dict(d["fingerprint"])
                fp_d["captured_batch_shapes"] = tuple(fp_d["captured_batch_shapes"])
                self._by_model[d["model"]] = Snapshot(
                    d["model"], Fingerprint(**fp_d), d["handle"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # skip corrupt entries rather than crash on startup

    def put(self, snapshot: Snapshot) -> None:
        super().put(snapshot)
        payload = {"model": snapshot.model,
                   "fingerprint": asdict(snapshot.fingerprint),
                   "handle": snapshot.handle}
        tmp = self._path(snapshot.model).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.rename(self._path(snapshot.model))

    def drop(self, model: str) -> None:
        super().drop(model)
        self._path(model).unlink(missing_ok=True)


def default_fingerprint_fn(*, engine_version: str, dtype: str, max_seq_len: int,
                           tensor_parallel: int,
                           captured_batch_shapes: tuple[int, ...]
                           ) -> Callable[[str, str], Fingerprint]:
    """A real `(model, gpu_id) -> Fingerprint` for production: hashes the actual
    weights and probes the live GPU/driver. Pass this to ColdStartLoader instead
    of a fake fingerprint_fn."""
    def fn(model: str, gpu_id: str) -> Fingerprint:
        return build_fingerprint(
            model, engine_version=engine_version, dtype=dtype,
            max_seq_len=max_seq_len, tensor_parallel=tensor_parallel,
            captured_batch_shapes=captured_batch_shapes)
    return fn


class ColdStartLoader:
    """Forks fast/slow path per launch, gated by the fingerprint.

    Injected GPU ops (signatures keep `gpu_id` so they target the placed device):
      fingerprint_fn(model, gpu_id) -> Fingerprint   # current environment
      cold_load(model, gpu_id)      -> Backend        # slow path (~57s)
      capture(model, gpu_id, backend) -> handle       # snapshot post-init state
      restore(model, gpu_id, snapshot) -> Backend     # fast path (~9s)
    """

    def __init__(self, *,
                 fingerprint_fn: Callable[[str, str], Fingerprint],
                 cold_load: Callable[[str, str], Any],
                 capture: Callable[[str, str, Any], Any],
                 restore: Callable[[str, str, Snapshot], Any],
                 store: SnapshotStore | None = None):
        self.fingerprint_fn = fingerprint_fn
        self.cold_load = cold_load
        self.capture = capture
        self.restore = restore
        self.store = store or SnapshotStore()
        self.cold_loads = 0    # slow path taken (snapshot miss or invalid)
        self.restores = 0      # fast path taken (valid snapshot)
        self.invalidations = 0  # snapshot existed but fingerprint mismatched

    def launch(self, model: str, gpu_id: str):
        """Autoscaler-compatible launch. Restore if a valid snapshot exists,
        else cold-load and capture a fresh one."""
        current = self.fingerprint_fn(model, gpu_id)
        snap = self.store.get(model)

        if snap is not None and decide(current, snap.fingerprint):
            self.restores += 1
            return self.restore(model, gpu_id, snap)   # FAST PATH

        if snap is not None:
            # snapshot exists but is stale/cross-GPU — must NOT restore it
            self.invalidations += 1
            self.store.drop(model)

        # SLOW PATH: cold-load, then capture a snapshot for next time
        self.cold_loads += 1
        backend = self.cold_load(model, gpu_id)
        handle = self.capture(model, gpu_id, backend)
        self.store.put(Snapshot(model, current, handle))
        return backend
