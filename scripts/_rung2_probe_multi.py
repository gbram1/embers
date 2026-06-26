"""Rung 2 rigorous probe: load vLLM once, then serve generations on demand so the
driver can measure restore time across N checkpoint/restore cycles and verify
output correctness every cycle.

Handshake with _rung2_driver_multi.sh via files:
  - writes /tmp/r2_ready + /tmp/r2_pid + /tmp/r2_base once loaded
  - loop: when /tmp/r2_tick appears (atomic rename), generate, write /tmp/r2_ack
    as "<n>|<output>", remove tick. "STOP" tick ends the loop.
"""
import os
import time

from vllm import LLM, SamplingParams

PROMPT = "The capital of France is"
SP = SamplingParams(max_tokens=8, temperature=0.0)  # greedy = reproducible


def gen(llm):
    return llm.generate([PROMPT], SP)[0].outputs[0].text


def main():
    llm = LLM(
        model=os.environ.get("R2_MODEL", "Qwen/Qwen2.5-3B"),
        dtype="auto",
        max_model_len=4096,
        enforce_eager=False,
    )
    baseline = gen(llm)
    with open("/tmp/r2_base", "w") as f:
        f.write(baseline)
    with open("/tmp/r2_pid", "w") as f:
        f.write(str(os.getpid()))
    open("/tmp/r2_ready", "w").close()
    print(f"[probe] ready pid={os.getpid()} baseline={baseline!r}", flush=True)

    while True:
        if not os.path.exists("/tmp/r2_tick"):
            time.sleep(0.05)
            continue
        n = open("/tmp/r2_tick").read().strip()
        if n == "STOP":
            break
        out = gen(llm)  # first CUDA work after the driver's restore
        with open("/tmp/r2_ack", "w") as f:
            f.write(f"{n}|{out}")
        os.remove("/tmp/r2_tick")
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
