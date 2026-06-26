"""Rung 2 Path-1 prototype: measure cuda-checkpoint restore vs cold start.

Flow (driven together with _rung2_driver.sh):
  1. Load vLLM (pay full cold init ~56s, once). Generate a baseline token.
  2. Signal "ready" + write PID. Park, polling for /tmp/r2_go (NO cuda calls
     while parked — the driver checkpoints/restores the GPU underneath us).
  3. On /tmp/r2_go: generate again, time to first token. Verify output matches.

The driver evicts GPU memory (cuda-checkpoint) while we're parked and restores it
before signalling go. We measure: post-restore time-to-token. Compare the
restore path (driver's restore time + this token time) against the 56.7s cold
start — that delta is the Rung 2 win.

Single-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0) so ONE pid holds the CUDA
context and cuda-checkpoint can target it directly. (v1 multiprocess puts the GPU
in a child pid — a later complication.)
"""
import json
import os
import time

from vllm import LLM, SamplingParams

PROMPT = "The capital of France is"
SP = SamplingParams(max_tokens=8, temperature=0.0)  # greedy = reproducible


def gen(llm):
    out = llm.generate([PROMPT], SP)
    return out[0].outputs[0].text


def main():
    t0 = time.perf_counter()
    llm = LLM(
        model=os.environ.get("R2_MODEL", "Qwen/Qwen2.5-3B"),
        dtype="auto",
        max_model_len=4096,
        enforce_eager=False,
    )
    cold_init = time.perf_counter() - t0

    baseline = gen(llm)  # warm output to compare against post-restore

    with open("/tmp/r2_pid", "w") as f:
        f.write(str(os.getpid()))
    open("/tmp/r2_ready", "w").close()
    print(f"[probe] ready pid={os.getpid()} cold_init={cold_init:.2f}s "
          f"baseline_out={baseline!r}", flush=True)

    # Park WITHOUT touching CUDA while the driver checkpoints+restores the GPU.
    while not os.path.exists("/tmp/r2_go"):
        time.sleep(0.2)

    t1 = time.perf_counter()
    restored = gen(llm)          # first real CUDA work after restore
    token_after_restore = time.perf_counter() - t1

    result = {
        "cold_init_s": round(cold_init, 3),
        "post_restore_first_token_s": round(token_after_restore, 3),
        "output_matches": restored == baseline,
        "baseline_out": baseline,
        "restored_out": restored,
    }
    with open("/tmp/r2_result.json", "w") as f:
        json.dump(result, f)
    print(f"[probe] RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
