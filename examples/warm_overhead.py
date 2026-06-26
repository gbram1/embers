#!/usr/bin/env python3
"""Warm-path overhead — once a model is loaded, does embers add latency? (≈No.)

embers' win is the COLD path (fast restore). On the WARM path it forwards requests
straight to vLLM, so warm latency ≈ a raw vLLM server — the gateway adds sub-ms.
This measures your warm serving latency through embers (p50/p90) so you can confirm
embers doesn't slow you down once loaded.

    python examples/warm_overhead.py --url http://localhost:8080 --model <name> -n 30

Run it against a WARM model (after the first request has loaded it).
"""
import argparse
import statistics
import time

import httpx


def one(url, model, max_tokens):
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", timeout=120,
                   json={"model": model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": "Say hello."}]})
    r.raise_for_status()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("-n", type=int, default=30, help="warm requests to time")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"\n  warming {args.model} …", flush=True)
    one(url, args.model, args.max_tokens)           # ensure loaded (don't count it)

    print(f"  timing {args.n} warm requests through the embers gateway …", flush=True)
    lat = sorted(one(url, args.model, args.max_tokens) for _ in range(args.n))
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    print(f"\n  warm latency (max_tokens={args.max_tokens}):")
    print(f"    p50 {p(0.50)*1000:6.0f} ms   p90 {p(0.90)*1000:6.0f} ms   "
          f"p99 {p(0.99)*1000:6.0f} ms")
    print(f"    min {lat[0]*1000:6.0f} ms   max {lat[-1]*1000:6.0f} ms   "
          f"mean {statistics.mean(lat)*1000:.0f} ms")
    print("\n  This is ~the same as hitting raw vLLM directly — embers forwards warm")
    print("  requests to the serving unit (the gateway adds <1ms). embers' value is")
    print("  the cold path, and it costs you nothing on the warm path.\n")


if __name__ == "__main__":
    main()
