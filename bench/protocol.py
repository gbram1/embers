"""True-cold-start protocol enforcement — the #1 footgun.

Every run must be genuinely cold or the numbers are fake (run 2 reads weights
from RAM cache and you report a phantom speedup). See eng doc §7.2.
"""

import os
import shutil
import subprocess


def clear_vllm_compile_cache() -> None:
    """Delete vLLM's torch.compile cache so engine init is genuinely cold.

    A naive scale-to-zero deployment recompiles on every cold container, so the
    baseline must too — otherwise runs after the first reuse the cached compiled
    graph and under-report engine_init, the exact cost this project targets.
    Lives in ~/.cache (writable, no root) so it works even in containers."""
    shutil.rmtree(os.path.expanduser("~/.cache/vllm/torch_compile_cache"),
                  ignore_errors=True)


class CacheDropError(RuntimeError):
    """Raised when the OS page cache could not be dropped (no permission, or a
    container that doesn't expose the host's drop_caches). The caller decides
    whether to abort or continue with a warm-cache caveat."""


def drop_caches() -> None:
    """Drop the OS page cache. Skipping this is the most common way to produce
    garbage numbers (run 2 reads weights from RAM). Needs root: uses sudo unless
    we're already root (e.g. a root container)."""
    if os.geteuid() == 0:
        cmd = "sync && echo 3 > /proc/sys/vm/drop_caches"
    else:
        cmd = "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise CacheDropError(r.stderr.strip() or "could not drop page cache")


def assert_no_resident_model() -> None:
    """Confirm no model is resident on the GPU before a 'cold' run."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    used_mb = max(int(line) for line in out.stdout.split())
    if used_mb > 500:  # allow driver overhead, flag a resident model
        raise RuntimeError(f"GPU not cold: {used_mb} MiB resident before run")

# Also required (caller's responsibility): fresh process every run (never reuse
# a loaded model), and for the S3 tier ensure weights aren't already on disk.
