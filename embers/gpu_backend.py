"""Real-hardware cold-start: cuda-checkpoint park/unpark of a live vLLM serving
unit (Path-1, app-level scale-to-zero). Frees the GPU while keeping the process
parked in host RAM, then restores serving in ~9s instead of cold-loading ~57s.
Validated mechanism: scripts/_rung2_*. Needs driver ≥550 + the cuda-checkpoint
binary; runs on a GPU pod.

Why park/unpark, not snapshot-to-disk: full process-to-disk (CRIU) needs
CAP_SYS_ADMIN, which RunPod containers don't grant. cuda-checkpoint alone works
unprivileged.

The subprocess / HTTP / command boundaries are injected so the lifecycle logic is
unit-tested offline; defaults do the real thing on a pod. `GpuLauncher.launch`
is the autoscaler's launch callback; `GpuLauncher.deactivate` is its scale-down
(park) hook — together they make the platform's scale-to-zero genuinely fast.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager


def _server_ready(url: str, model: str, timeout: float = 2.0) -> bool:
    """True only when OUR serving unit answers /health with the right model —
    not just any 200 (RunPod proxies answer localhost ports generically)."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:  # noqa: S310
            if r.status != 200:
                return False
            body = json.loads(r.read())
            return body.get("status") == "ok" and body.get("model") == model
    except Exception:
        return False


CC = "/usr/local/bin/cuda-checkpoint"


def _cuda_device(gpu_id: str | None) -> str | None:
    """Physical CUDA index for a logical gpu_id. `detect_gpus` names GPUs
    'gpu0'/'gpu1'/… (the nvidia-smi index), so the trailing integer IS the
    device to pin. None (no digits / no id) → don't pin (single-GPU default)."""
    if not gpu_id:
        return None
    import re
    m = re.search(r"\d+$", gpu_id)
    return m.group() if m else None


def _devices(gpu_ids: list[str]) -> str | None:
    """CUDA_VISIBLE_DEVICES value for a GPU group (tensor-parallel unit spans
    several): 'gpu0','gpu1' → '0,1'. None if no id maps to an index."""
    idx = [d for d in (_cuda_device(g) for g in gpu_ids) if d is not None]
    return ",".join(idx) if idx else None


class GpuServingProcess:
    """A real `embers serve` subprocess (single-process vLLM, so one PID holds
    the CUDA context) plus its cuda-checkpoint park/unpark. Pinned to one
    physical GPU via CUDA_VISIBLE_DEVICES so a model's replicas land on the
    distinct GPUs the scheduler assigned (not all on GPU 0)."""

    def __init__(self, model: str, *, port: int, host: str = "127.0.0.1",
                 cc: str = CC, python: str | None = None,
                 cuda_device: str | None = None, tensor_parallel_size: int = 1,
                 adapters: dict[str, str] | None = None,
                 gpu_memory_utilization: float | None = None,
                 max_model_len: int | None = None,
                 spawn: Callable | None = None, run: Callable | None = None,
                 ready_check: Callable[[], bool] | None = None,
                 gpu_pid_finder: Callable[[int], int] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter):
        self.model, self.port, self.host = model, port, host
        self.cc = cc
        self.python = python or sys.executable
        self.cuda_device = cuda_device
        self.tensor_parallel_size = tensor_parallel_size
        self.adapters = dict(adapters or {})
        # fraction of the GPU this unit grabs (for packing many models on one GPU)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._spawn = spawn or (lambda args: subprocess.Popen(
            args, env=self._child_env()))
        self._run = run or (lambda args: subprocess.run(args, capture_output=True))
        self._ready = ready_check or (lambda: _server_ready(self.url, self.model))
        # cuda-checkpoint must target the pid(s) that actually OWN GPU memory.
        # AsyncLLMEngine (true streaming) runs the GPU in a worker child, and a
        # tensor-parallel unit has one worker per GPU; the default resolver finds
        # every GPU pid that is `pid` or a descendant. Injectable for tests.
        self._gpu_pid_finder = gpu_pid_finder or self._find_gpu_pids
        self._sleep, self._clock = sleep, clock
        self.proc = None
        self.pid: int | None = None
        self._gpu_pids: list[int] = []
        self.parked = False

    def _child_env(self) -> dict:
        """Env for the serving subprocess: pin it to its physical GPU and keep
        single-process vLLM (one PID owns the CUDA context, for park/unpark)."""
        env = dict(os.environ)
        env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if self.cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        return env

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ready_url(self) -> str:
        return f"{self.url}/ready"

    def start(self, timeout: float = 600.0) -> float:
        """Spawn the serving unit and block until /ready. Returns cold-load secs."""
        t0 = self._clock()
        args = [self.python, "-m", "embers.cli", "serve", self.model,
                "--host", self.host, "--port", str(self.port)]
        if self.tensor_parallel_size > 1:          # shard one model across N GPUs
            args += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.gpu_memory_utilization is not None:  # take only a fraction (packing)
            args += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.max_model_len is not None:
            args += ["--max-model-len", str(self.max_model_len)]
        for name, path in self.adapters.items():   # LoRA adapters off this base
            args += ["--lora", f"{name}={path}"]
        self.proc = self._spawn(args)
        self.pid = self.proc.pid
        for _ in range(int(timeout * 2)):
            if self._ready():
                return self._clock() - t0
            self._sleep(0.5)
        raise TimeoutError(f"{self.model} not ready within {timeout}s")

    def _find_gpu_pids(self, pid: int) -> list[int]:
        """ALL pids that hold GPU memory for this unit: `pid` itself, and/or its
        descendants. A single-process (DP) unit yields one; a tensor-parallel
        unit yields one worker rank per GPU (validated 2026-06-25: parking all
        ranks frees every GPU and restores cleanly). Falls back to `[pid]`."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            gpu_pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        except Exception:
            return [pid]
        if not gpu_pids:
            return [pid]
        parents = self._parent_map()
        found = []
        for gp in gpu_pids:                       # keep every GPU pid that is
            cur = gp                              # `pid` or a descendant of it
            for _ in range(30):
                if cur == pid:
                    found.append(gp)
                    break
                cur = parents.get(cur)
                if not cur or cur <= 1:
                    break
        return found or [pid]

    @staticmethod
    def _parent_map() -> dict[int, int]:
        out = subprocess.run(["ps", "-eo", "pid=,ppid="],
                             capture_output=True, text=True)
        m = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                m[int(parts[0])] = int(parts[1])
        return m

    def _cc(self, pid: int, action: str, *extra: str) -> None:
        r = self._run([self.cc, "--action", action, "--pid", str(pid), *extra])
        rc = getattr(r, "returncode", 0)        # injected fakes return None → 0
        if rc:
            err = getattr(r, "stderr", b"") or b""
            if isinstance(err, (bytes, bytearray)):
                err = err.decode(errors="replace")
            raise RuntimeError(
                f"cuda-checkpoint {action} pid={pid} failed (rc={rc}): "
                f"{str(err).strip()[:300]}")

    def park(self) -> float:
        """lock → checkpoint every GPU-holding rank pid: evict GPU memory to
        host, free the GPU(s). For tensor-parallel units, ALL ranks are locked
        (quiesced) BEFORE any is checkpointed — the order matters because the
        ranks share NCCL collectives. Process stays alive parked in host RAM."""
        t0 = self._clock()
        self._gpu_pids = self._gpu_pid_finder(self.pid)
        for p in self._gpu_pids:                  # lock ALL ranks first (quiesce)
            self._cc(p, "lock", "--timeout", "30000")
        for p in self._gpu_pids:                  # then checkpoint each
            self._cc(p, "checkpoint")
        self.parked = True
        return self._clock() - t0

    def unpark(self) -> float:
        """restore → unlock every rank pid: bring GPU state back (DP ~9s, TP
        ~16s). Restore ALL ranks before unlocking any (mirror of park)."""
        t0 = self._clock()
        pids = self._gpu_pids or [self.pid]
        for p in pids:
            self._cc(p, "restore")
        for p in pids:
            self._cc(p, "unlock")
        self.parked = False
        return self._clock() - t0

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()


class GpuLauncher:
    """Real cold-start orchestrator for the autoscaler. Tracks one serving
    process per model: launch = start (cold) or unpark (fast); deactivate =
    park. Counters split scale-from-zero into cold_loads vs restores."""

    def __init__(self, *, port_base: int = 8001, serve_host: str = "127.0.0.1",
                 adapters_by_model: dict[str, dict[str, str]] | None = None,
                 engine_by_model: dict[str, dict] | None = None,
                 make_process: Callable[..., GpuServingProcess] | None = None):
        self.port_base = port_base
        # host the serving units bind. 127.0.0.1 = same-box only (default);
        # 0.0.0.0 = reachable off-box (for a node serving a remote control plane).
        self.serve_host = serve_host
        self._adapters_by_model = adapters_by_model or {}
        # per-model engine knobs (gpu_memory_utilization, max_model_len) — for
        # packing several models on one GPU, each taking a fraction.
        self._engine_by_model = engine_by_model or {}
        self._make = make_process or (
            lambda model, port, gpu_ids: GpuServingProcess(
                model, port=port, host=self.serve_host,
                cuda_device=_devices(gpu_ids),
                tensor_parallel_size=len(gpu_ids),
                adapters=self._adapters_by_model.get(model) or None,
                **(self._engine_by_model.get(model) or {})))
        # keyed by (model, gpu_ids): a model's replicas are DISTINCT processes on
        # distinct GPU groups, so launch/park must address the specific replica —
        # not collapse them under one model key (which leaked GPUs and aliased
        # ports). A tensor-parallel unit's key is its whole GPU group.
        self.procs: dict[tuple[str, tuple[str, ...]], GpuServingProcess] = {}
        # one lock per PHYSICAL GPU, serialising cold-load / park / unpark on it.
        # A park frees GPU memory; if it overlaps another model's vLLM startup
        # (which memory-profiles the GPU), vLLM asserts ("memory not cleaned up").
        # So GPU-mutating ops on the same GPU must not run concurrently.
        self._gpu_locks: dict[str, threading.Lock] = {}
        self._gpu_locks_guard = threading.Lock()
        self._next = 0
        self.cold_loads = 0
        self.restores = 0
        self.unpark_failures = 0
        self.park_failures = 0
        self.last_seconds: float | None = None

    @contextmanager
    def _gpu_guard(self, gpu_ids: list[str]):
        """Hold the lock(s) for the physical GPU(s) so cold-load/park/unpark on
        the same GPU can't overlap (sorted acquire order → no deadlock)."""
        with self._gpu_locks_guard:
            locks = [self._gpu_locks.setdefault(g, threading.Lock())
                     for g in sorted(set(gpu_ids))]
        for lk in locks:
            lk.acquire()
        try:
            yield
        finally:
            for lk in reversed(locks):
                lk.release()

    def _cold_start(self, model: str, gpu_ids: list[str]):
        port = self.port_base + self._next
        self._next += 1
        proc = self._make(model, port, gpu_ids)
        self.procs[(model, tuple(gpu_ids))] = proc
        self.last_seconds = proc.start()
        self.cold_loads += 1
        return proc

    def launch(self, model: str, gpu_ids: list[str]):
        with self._gpu_guard(gpu_ids):           # serialise GPU ops on this GPU
            return self._launch(model, gpu_ids)

    def _launch(self, model: str, gpu_ids: list[str]):
        from embers.gateway import HttpBackend

        key = (model, tuple(gpu_ids))
        proc = self.procs.get(key)
        if proc is None:                         # never started → cold load
            proc = self._cold_start(model, gpu_ids)
        elif proc.parked:                        # parked → fast unpark (restore)
            try:
                self.last_seconds = proc.unpark()
                self.restores += 1
            except Exception:                    # noqa: BLE001
                # unpark failed (dead/corrupt parked state) — never serve broken:
                # discard it and cold-load a fresh process.
                self.unpark_failures += 1
                try:
                    proc.stop()
                except Exception:                # noqa: BLE001
                    pass
                proc = self._cold_start(model, gpu_ids)
        return HttpBackend(model, proc.url)

    def deactivate(self, model: str, gpu_ids: list[str]) -> None:
        """Scale-down hook: park the specific (model, gpu_ids) replica (free its
        GPU group, keep it warm). If the park fails, kill the process and drop
        it — a half-parked process strands the GPUs, so the next launch must
        cold-load fresh (process death frees the GPUs). Re-raises so the caller
        knows the GPUs weren't cleanly freed."""
        with self._gpu_guard(gpu_ids):           # serialise GPU ops on this GPU
            self._deactivate(model, gpu_ids)

    def _deactivate(self, model: str, gpu_ids: list[str]) -> None:
        key = (model, tuple(gpu_ids))
        proc = self.procs.get(key)
        if proc is None or proc.parked:
            return
        # park() locks+checkpoints every rank pid (DP=1, TP=N) — multi-rank TP
        # park/restore is hardware-validated.
        # If it fails (e.g. older driver), the fallback below kills + drops so the
        # next launch cold-loads fresh.
        try:
            proc.park()
        except Exception:
            self.park_failures += 1
            try:
                proc.stop()
            except Exception:        # noqa: BLE001
                pass
            self.procs.pop(key, None)   # next launch cold-loads
            raise

    def discard(self, model: str, gpu_ids: list[str]) -> None:
        """Demand-eviction hook: STOP the replica (frees GPU **and** host RAM) —
        not park. An over-committed model grabs the whole GPU, so once another
        model reuses that memory the parked snapshot can't restore (unpark fails)
        and parking would just leak ~the model's worth of host RAM. So evicted
        models are discarded; next access cold-loads. (Merely-idle models still
        park via deactivate — fast restore when nothing else needs the GPU.)"""
        with self._gpu_guard(gpu_ids):
            proc = self.procs.pop((model, tuple(gpu_ids)), None)
            if proc is not None:
                try:
                    proc.stop()
                except Exception:        # noqa: BLE001
                    pass

    def shutdown(self) -> None:
        for p in self.procs.values():
            p.stop()
