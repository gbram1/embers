"""Dependency fingerprint + restore gate — the snapshot-invalidation core.

A snapshot is frozen GPU/process state; restoring a stale or cross-GPU one
produces *silent wrong output* (worse than no snapshot). Every restore MUST pass
`decide()`:
exact fingerprint match → restore; anything else → run the slow path and capture
a fresh snapshot tagged with the new fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fingerprint:
    """Everything a snapshot depends on. A missing field here is a silent-
    corruption bug — adding a field without hashing it defeats the gate."""

    weights_hash: str          # hash of the ACTUAL weights, not a version label
    model_version: str
    engine_version: str        # vLLM version
    gpu_type: str              # e.g. "NVIDIA A40"
    driver_cuda_version: str   # driver + CUDA — state is NOT portable across these
    dtype: str
    max_seq_len: int
    tensor_parallel: int
    captured_batch_shapes: tuple[int, ...]

    def digest(self) -> str:
        """Stable hash over all fields — the snapshot's identity tag."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True))

    @staticmethod
    def load(path: str | Path) -> "Fingerprint":
        d = json.loads(Path(path).read_text())
        d["captured_batch_shapes"] = tuple(d["captured_batch_shapes"])
        return Fingerprint(**d)


def decide(current: Fingerprint, snapshot: Fingerprint) -> bool:
    """True → safe to restore. Mirrors validate.rs.

    GPU/driver drift is a HARD stop (never a best-effort match): restoring across
    an incompatible GPU/driver boundary risks silent corruption. Otherwise every
    field must match exactly.
    """
    if (current.gpu_type != snapshot.gpu_type
            or current.driver_cuda_version != snapshot.driver_cuda_version):
        return False
    return current == snapshot


def _nvidia_smi(query: str) -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.split("\n")[0].strip()


def hash_weights(model_dir: str | Path) -> str:
    """Hash the actual *.safetensors bytes (labels lie when weights are
    overwritten in place). Hashes file sizes + a sampled prefix of each shard for
    speed; swap to full-content hashing if paranoid."""
    h = hashlib.sha256()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        with open(f, "rb") as fh:
            h.update(fh.read(1 << 20))  # first 1 MiB of each shard
    return h.hexdigest()


def current_fingerprint(
    *, model_version: str, engine_version: str, weights_hash: str,
    dtype: str, max_seq_len: int, tensor_parallel: int,
    captured_batch_shapes: tuple[int, ...],
) -> Fingerprint:
    """Build the fingerprint for the live environment. GPU + driver are probed
    from nvidia-smi; the rest are supplied by the serving config."""
    return Fingerprint(
        weights_hash=weights_hash,
        model_version=model_version,
        engine_version=engine_version,
        gpu_type=_nvidia_smi("name"),
        driver_cuda_version=_nvidia_smi("driver_version"),
        dtype=dtype,
        max_seq_len=max_seq_len,
        tensor_parallel=tensor_parallel,
        captured_batch_shapes=tuple(captured_batch_shapes),
    )


def resolve_model_dir(model: str, hf_home: str | None = None) -> Path | None:
    """Find the on-disk weights for a model — a local path, or the HF cache
    snapshot dir (`models--org--name/snapshots/<rev>`). None if not found."""
    p = Path(model)
    if p.is_dir():
        return p
    base = Path(hf_home or os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    snaps = base / ("models--" + model.replace("/", "--")) / "snapshots"
    if snaps.is_dir():
        revs = sorted(d for d in snaps.iterdir() if d.is_dir())
        if revs:
            return revs[0]
    return None


def build_fingerprint(
    model: str, *, engine_version: str, dtype: str, max_seq_len: int,
    tensor_parallel: int, captured_batch_shapes: tuple[int, ...],
    model_dir: str | Path | None = None,
    gpu_name: str | None = None, driver: str | None = None,
) -> Fingerprint:
    """Real fingerprint for `model` in the live environment: hashes the actual
    weight bytes (resolving the model dir if not given) and probes GPU/driver.
    gpu_name/driver are injectable for tests; default to nvidia-smi."""
    md = Path(model_dir) if model_dir else resolve_model_dir(model)
    # If weights can't be located, mark it explicitly rather than silently
    # producing a label-only hash that could collide across edits.
    weights = hash_weights(md) if md else f"UNRESOLVED:{model}"
    return Fingerprint(
        weights_hash=weights,
        model_version=model,
        engine_version=engine_version,
        gpu_type=gpu_name or _nvidia_smi("name"),
        driver_cuda_version=driver or _nvidia_smi("driver_version"),
        dtype=dtype,
        max_seq_len=max_seq_len,
        tensor_parallel=tensor_parallel,
        captured_batch_shapes=tuple(captured_batch_shapes),
    )
