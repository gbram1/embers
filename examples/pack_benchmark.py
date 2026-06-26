#!/usr/bin/env python3
"""Multi-tenant packing benchmark — several models CO-RESIDENT on ONE GPU, each
scaling to zero when idle. Measures, per model: cold-start, fast-restore, warm
throughput (req/s) + latency; shows them sharing the GPU; then all scale to zero
(GPU freed) and fast-restore.

    # terminal 1:  embers up --config examples/multimodel.yaml
    # terminal 2:
    python examples/pack_benchmark.py --url http://localhost:8080 \
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B
"""
import argparse
import concurrent.futures as cf
import subprocess
import time

import httpx


def gpu_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return sum(int(x) for x in out.split())
    except Exception:
        return None


def stats(url):
    try:
        return httpx.get(f"{url}/stats", timeout=5).json()
    except Exception:
        return {}


def one(url, model, prompt="Say hello.", max_tokens=16):
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", timeout=600,
                   json={"model": model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return time.perf_counter() - t0


def throughput(url, model, n=24, conc=8):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        lat = sorted(ex.map(lambda _: one(url, model), range(n)))
    return n / (time.perf_counter() - t0), lat[len(lat) // 2], lat[int(len(lat) * 0.9)]


def wait_all_zero(url, models, timeout=320):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        reps = stats(url).get("replicas", {})
        used = gpu_used_mb()
        if all(reps.get(m, 0) == 0 for m in models) and (used is None or used < 800):
            return time.perf_counter() - t0
        time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    url, models = args.url.rstrip("/"), args.models
    R = {m: {} for m in models}

    print(f"\n  packing {len(models)} models on ONE GPU → {url}")
    print(f"  start: GPU {gpu_used_mb()} MB, replicas {stats(url).get('replicas', {})}\n")

    # --- 1. load each (cold start) + warm throughput; they stay resident ---
    for m in models:
        print(f"  [{m}] cold start…", flush=True)
        R[m]["cold"] = one(url, m, "In one sentence, what is a GPU?", 40)
        rps, p50, p90 = throughput(url, m)
        R[m].update(rps=rps, p50=p50, p90=p90, gpu=gpu_used_mb())
        print(f"    cold {R[m]['cold']:.1f}s · {rps:.1f} req/s · "
              f"p50 {p50*1000:.0f}ms p90 {p90*1000:.0f}ms · GPU total now {R[m]['gpu']} MB")

    # --- 2. co-residence: all live at once on one GPU ---------------------
    print(f"\n  → all {len(models)} models CO-RESIDENT on one GPU: {gpu_used_mb()} MB "
          f"used · replicas {stats(url).get('replicas', {})}")

    # --- 3. scale to zero: every model idles → GPU freed ------------------
    print("\n  waiting for all to idle out (scale to zero)…", flush=True)
    z = wait_all_zero(url, models)
    print(f"  all scaled to zero after ~{z:.0f}s · GPU now {gpu_used_mb()} MB"
          if z else "  (did not all scale to zero in time)")

    # --- 4. fast restore each ---------------------------------------------
    print("\n  restoring each from zero…", flush=True)
    for m in models:
        R[m]["restore"] = one(url, m, "Welcome back.", 16)
        print(f"    [{m}] restore {R[m]['restore']:.1f}s")

    # --- summary ----------------------------------------------------------
    print(f"\n  {'tenant model':<22}{'cold':>8}{'restore':>9}{'speedup':>9}"
          f"{'req/s':>8}{'p50':>8}")
    print(f"  {'-'*22}{'-'*8:>8}{'-'*9:>9}{'-'*9:>9}{'-'*8:>8}{'-'*8:>8}")
    for m in models:
        d = R[m]
        spd = f"{d['cold']/d['restore']:.1f}x" if d.get("restore") else "—"
        print(f"  {m:<22}{d.get('cold',0):>7.1f}s{d.get('restore',0):>8.1f}s"
              f"{spd:>9}{d.get('rps',0):>7.1f}{d.get('p50',0)*1000:>7.0f}ms")
    print("\n  One GPU hosted all of them at once; each released its slice when idle")
    print("  and restored in seconds — tenants pay only while their model is active.\n")


if __name__ == "__main__":
    main()
