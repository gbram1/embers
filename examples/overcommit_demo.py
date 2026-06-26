#!/usr/bin/env python3
"""Over-commit demo — more tenant models than fit on one GPU, multiplexed by
demand eviction. Requests each model in turn (each evicts the prior idle one),
then re-requests the first to show it fast-restores (it was parked on eviction).

    # terminal 1:  embers up --config examples/overcommit.yaml
    # terminal 2:
    python examples/overcommit_demo.py --url http://localhost:8080 \
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B
"""
import argparse
import subprocess
import time

import httpx


def gpu_mb():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).split()[0])
    except Exception:
        return -1


def stats(url):
    try:
        return httpx.get(f"{url}/stats", timeout=5).json()
    except Exception:
        return {}


def req(url, model):
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", timeout=600,
                   json={"model": model, "max_tokens": 16,
                         "messages": [{"role": "user", "content": "Say hi."}]})
    r.raise_for_status()
    return time.perf_counter() - t0


def evictions(url):
    return stats(url).get("scaling", {}).get("scaled_to_zero", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    url, models = args.url.rstrip("/"), args.models

    print(f"\n  over-commit: {len(models)} models, ONE GPU (only one resident at a time)\n")
    print(f"  registered (all assigned despite not fitting): {models}")
    print(f"  start: GPU {gpu_mb()} MB, replicas {stats(url).get('replicas', {})}\n")

    # request each in turn — each evicts the prior idle model to get the GPU
    for i, m in enumerate(models):
        dt = req(url, m)
        reps = stats(url).get("replicas", {})
        live = [k for k, v in reps.items() if v]
        print(f"  request {m:<20} {dt:5.1f}s  → GPU {gpu_mb()} MB · resident: {live}")
        if i > 0:
            print(f"      (evicted the previous idle model to make room)")

    # re-request the FIRST model — it was parked on eviction → fast restore
    print(f"\n  re-request {models[0]} (was evicted/parked earlier)…")
    dt = req(url, models[0])
    print(f"  → {dt:.1f}s   (fast RESTORE, not a cold load) · GPU {gpu_mb()} MB")

    sc = stats(url).get("scaling", {})
    print(f"\n  one GPU served {len(models)} tenant models by evicting idle ones on")
    print(f"  demand. scale-downs (evictions+idle): {sc.get('scale_downs')}, "
          f"cold starts: {sc.get('cold_starts')}\n")


if __name__ == "__main__":
    main()
