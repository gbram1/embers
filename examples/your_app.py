#!/usr/bin/env python3
"""Your app — a stand-in for a small project that calls a model. It's a plain
OpenAI client, so the SAME code works whether you point it at a vanilla vLLM
server OR at embers. That's the point: embers is a drop-in; your app doesn't
change. What changes is the cost of a *cold* wake.

    pip install openai
    python examples/your_app.py --url http://localhost:8080/v1 --model <name>

Run it twice the way a real app hits a scale-to-zero backend:
  * against a vanilla vLLM server that was just (re)started → you pay the full
    cold load (~100s on a 3B) every time it wakes, or you keep it always-on ($$).
  * against `embers up` → first wake cold-loads, later wakes fast-restore (~10s).
"""
import argparse
import time

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--prompt", default="In one sentence, what is a GPU?")
    args = ap.parse_args()

    client = OpenAI(base_url=args.url, api_key="x")   # any key if api_keys=[]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=40,
    )
    dt = time.perf_counter() - t0

    print(f"\n  reply: {resp.choices[0].message.content!r}")
    u = resp.usage
    print(f"  tokens: prompt={u.prompt_tokens} completion={u.completion_tokens}")
    print(f"  ⏱  time to response: {dt:.1f}s")
    print("\n  (this is the number that matters on a COLD wake — run it right after")
    print("   the backend scaled to zero / was restarted to see the real cost.)\n")


if __name__ == "__main__":
    main()
