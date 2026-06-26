"""Capstone: run the WHOLE platform on a real GPU and prove fast scale-to-zero.

Wires the real GpuLauncher (cuda-checkpoint park/unpark of live vLLM serving
units) into the autoscaler + scheduler + gateway, then drives one lifecycle:

  cold start (≈57s)  →  serve  →  idle→scale-to-zero (park, GPU freed)  →
  request → fast restore (≈9s, unpark)  →  serve (output must match)

Prints measured cold-load vs restore seconds and nvidia-smi GPU memory at each
step. Run on a pod (single process vLLM):

  VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. \
      /root/csvenv/bin/python scripts/_platform_demo.py Qwen/Qwen2.5-3B
"""
import subprocess
import sys

from embers.autoscaler import Autoscaler
from embers.gateway import Router
from embers.gpu_backend import GpuLauncher
from embers.scheduler import GPU, Scheduler


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def gpu_mem_used() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True)
    return out.stdout.strip().splitlines()[0]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
    prompt = "The capital of France is"

    sched = Scheduler([GPU("g0", 40000)])
    router = Router()
    # high port base — RunPod proxies low ports (8000/8001/8888) on localhost.
    launcher = GpuLauncher(port_base=19000)
    clock = FakeClock()
    auto = Autoscaler(sched, router, launch=launcher.launch, clock=clock,
                      on_deactivate=launcher.deactivate)
    auto.register_model(model, vram_mb=20000, idle_ttl=300)

    print(f"[demo] model={model}  gpu_mem_used (idle)={gpu_mem_used()}")

    # 1. cold start ---------------------------------------------------------
    backend = auto.handle_request(model)
    print(f"[demo] COLD LOAD: {launcher.last_seconds:.1f}s  "
          f"gpu_mem_used={gpu_mem_used()}")
    first = backend.complete(prompt, max_tokens=8, temperature=0.0)
    print(f"[demo] served: {first!r}")

    # 2. idle → scale to zero (park, free the GPU) --------------------------
    clock.advance(301)
    auto.tick()
    print(f"[demo] SCALED TO ZERO (parked)  gpu_mem_used={gpu_mem_used()}  "
          f"<- should drop toward 0")

    # 3. request → fast restore (unpark) ------------------------------------
    backend = auto.handle_request(model)
    print(f"[demo] FAST RESTORE: {launcher.last_seconds:.1f}s  "
          f"gpu_mem_used={gpu_mem_used()}")
    second = backend.complete(prompt, max_tokens=8, temperature=0.0)
    print(f"[demo] served after restore: {second!r}  match={second == first}")

    print(f"\n[demo] cold_loads={launcher.cold_loads} restores={launcher.restores}")
    print(f"[demo] >>> cold {launcher.last_seconds and ''}"
          f"vs restore: see the two timings above")
    launcher.shutdown()


if __name__ == "__main__":
    main()
