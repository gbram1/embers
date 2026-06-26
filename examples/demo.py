#!/usr/bin/env python3
"""embers demo — watch the whole cold-start → serve → scale-to-zero → fast-restore
story in one run, with timings and GPU state at each phase.

    python examples/demo.py [--url http://localhost:8080] [--model <name>]

Point it at a running `embers up`. On a real GPU you'll see the headline: a model
goes cold→serving in ~tens of seconds the first time, idles to zero (GPU freed, you
stop paying), then FAST-RESTORES in ~10s on the next request — vs a full cold load.
Works against `--mock` too (to see the flow), but the timings only mean something on
a real GPU.
"""
import argparse
import sys
import time

import httpx


def bar(title):
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}", flush=True)


def stats(url):
    try:
        return httpx.get(f"{url}/stats", timeout=5).json()
    except Exception:
        return {}


def gpu_line(url):
    s = stats(url)
    gpus = s.get("gpus", [])
    used = sum(g.get("used_mb", 0) for g in gpus)
    reps = s.get("replicas", {})
    return f"GPU in use: {used} MB · replicas: {reps or '{}'}"


def chat(url, model, content, max_tokens=24):
    """One non-streaming chat request; returns (text, wall_seconds, usage)."""
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", timeout=600,
                   json={"model": model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": content}]})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], dt, j.get("usage", {})


def wait_scale_to_zero(url, model, timeout=180):
    """Poll until the model is idle-scaled to zero AND its GPU is actually freed.
    (Waiting only for replicas==0 fires mid-park — before the GPU is released —
    so we also require the scheduler to show the GPU reclaimed.)"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = stats(url)
        reps = s.get("replicas", {})
        used = sum(g.get("used_mb", 0) for g in s.get("gpus", []))
        if reps.get(model, 0) == 0 and used == 0:
            return time.perf_counter() - t0
        time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()
    url, model = args.url.rstrip("/"), args.model

    print(f"\n  embers demo → {url}  (model: {model})")
    try:
        served = [m["id"] for m in httpx.get(f"{url}/v1/models", timeout=5).json()["data"]]
    except Exception as e:
        sys.exit(f"  could not reach the gateway at {url}: {e}\n"
                 f"  start it first:  embers up --config examples/demo_platform.yaml")
    if model not in served:
        sys.exit(f"  model {model!r} not served here. available: {served}")
    print(f"  served models: {served}")
    print(f"  starting state — {gpu_line(url)}   (model is cold, 0 replicas)")

    # --- 1. COLD START -----------------------------------------------------
    bar("1. COLD START  — first request spins the model up from nothing")
    text, cold_s, usage = chat(url, model, "In one sentence, what is a GPU?", max_tokens=40)
    print(f"  reply: {text!r}")
    print(f"  tokens: {usage.get('total_tokens', '?')}")
    print(f"  ⏱  cold start → first response: {cold_s:.1f}s")
    print(f"  {gpu_line(url)}")

    # --- 2. WARM SERVE -----------------------------------------------------
    bar("2. WARM SERVE  — now it's loaded, requests are fast")
    for q in ("Say hello.", "Name a color."):
        text, dt, _ = chat(url, model, q, max_tokens=12)
        print(f"  {dt*1000:6.0f} ms  ·  {q:<14} → {text!r}")

    # --- 3. SCALE TO ZERO --------------------------------------------------
    bar("3. SCALE TO ZERO  — idle → GPU freed (you stop paying)")
    print("  waiting for the model to idle out (idle_ttl)…", flush=True)
    idle_s = wait_scale_to_zero(url, model)
    if idle_s is None:
        print("  (didn't scale to zero within the timeout — raise idle_ttl or wait)")
    else:
        print(f"  ⏱  scaled to zero after ~{idle_s:.0f}s idle")
        print(f"  {gpu_line(url)}   ← GPU released")

    # --- 4. FAST RESTORE ---------------------------------------------------
    bar("4. FAST RESTORE  — next request restores the snapshot (no full cold load)")
    text, restore_s, _ = chat(url, model, "Welcome back! Count to three.", max_tokens=16)
    print(f"  reply: {text!r}")
    print(f"  ⏱  scale-from-zero → first response: {restore_s:.1f}s")
    print(f"  {gpu_line(url)}")

    # --- summary -----------------------------------------------------------
    s = stats(url).get("embers", {})
    bar("SUMMARY")
    print(f"  cold start (first ever):     {cold_s:6.1f}s")
    print(f"  fast restore (from zero):    {restore_s:6.1f}s")
    if restore_s > 0 and cold_s > restore_s:
        print(f"  → restore is ~{cold_s / restore_s:.1f}x faster than the cold start")
    if s:
        print(f"  snapshot_hit_rate: {s.get('snapshot_hit_rate')} "
              f"(restores={s.get('restores')}, cold_loads={s.get('cold_loads')})")
    print("\n  That's the embers win: pay nothing while idle, restore in seconds.\n")


if __name__ == "__main__":
    main()
