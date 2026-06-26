#!/usr/bin/env python3
"""The "WITHOUT embers" baseline — what a cold start costs the naive way.

This is what your app pays every time a scale-to-zero backend wakes if you DON'T
have snapshot/restore: load the weights + build the engine from scratch. Run it on
a CUDA GPU (it imports vLLM). It loads a model, times the cold load, serves one
request, then loads it AGAIN to show the cost repeats on every cold wake.

    pip install vllm==0.8.5
    python examples/without_embers.py --model Qwen/Qwen2.5-3B

Compare the printed cold-load time to embers' fast restore (~10s) from `demo.py` /
`your_app.py against embers` — that gap is the whole point of embers.

GPU-ONLY: this will not run on macOS (vLLM needs CUDA). On a Mac, see the flow with
`embers up --mock` instead.
"""
import argparse
import time


def cold_load(model):
    """Load the model fresh and return (engine, seconds) — the cold-start cost."""
    from vllm import LLM
    t0 = time.perf_counter()
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.6,
              enforce_eager=False)
    return llm, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    args = ap.parse_args()

    print(f"\n  WITHOUT embers — cold-loading {args.model} the naive way\n")

    print("  cold load (load weights + build engine from scratch)…", flush=True)
    llm, t = cold_load(args.model)
    from vllm import SamplingParams
    out = llm.generate(["In one sentence, what is a GPU?"],
                       SamplingParams(max_tokens=40))[0].outputs[0].text
    print(f"    ⏱  {t:.1f}s   reply: {out.strip()[:80]!r}…")

    print(f"\n  Without embers, you pay this ~{t:.0f}s on EVERY cold wake:")
    print("    • keep the GPU always-on  → no wait, but you pay for an idle GPU 24/7")
    print(f"    • kill it when idle       → free while idle, but ~{t:.0f}s every wake")
    print("\n  With embers you get both: GPU freed when idle AND ~10s restore on wake.")
    print("  (run `python examples/demo.py` against `embers up` to see the restore.)")
    print("  Note: this load reuses any torch.compile cache on disk; a truly cold")
    print("  first compile is longer (~100s) — embers snapshots past all of it.\n")


if __name__ == "__main__":
    main()
