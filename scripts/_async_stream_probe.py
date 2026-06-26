"""De-risk probe: does vLLM AsyncLLMEngine (true token streaming) coexist with
cuda-checkpoint park/unpark (the cold-start differentiator)?

Run with the bash driver. This script:
  1. builds an AsyncLLMEngine (single-process)
  2. streams a generation, printing each token delta with a timestamp — proving
     tokens arrive INCREMENTALLY (not all at once)
  3. writes its PID and waits for /tmp/as_go (the driver parks+unparks it)
  4. streams again — proving the engine still works after restore

If step 4 produces correct incremental output after a park/unpark cycle, true
streaming is compatible with the platform's scale-to-zero and we can build it.
"""
import asyncio
import os
import time

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

MODEL = os.environ.get("AS_MODEL", "Qwen/Qwen2.5-3B")
PROMPT = "List three colors, one per line:"


async def stream_once(engine, tag: str) -> str:
    sp = SamplingParams(max_tokens=24, temperature=0.0)
    prev, t0 = "", time.perf_counter()
    n = 0
    async for out in engine.generate(PROMPT, sp, request_id=f"{tag}-{time.time()}"):
        text = out.outputs[0].text
        delta = text[len(prev):]
        prev = text
        if delta:
            n += 1
            print(f"[{tag}] +{time.perf_counter()-t0:5.2f}s tok#{n}: {delta!r}",
                  flush=True)
    return prev


async def main():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, dtype="auto", max_model_len=4096, enforce_eager=False))
    open("/tmp/as_pid", "w").write(str(os.getpid()))
    print(f"[probe] pid={os.getpid()} engine ready", flush=True)

    out1 = await stream_once(engine, "before-park")
    print(f"[probe] before-park output: {out1!r}", flush=True)

    open("/tmp/as_ready", "w").close()
    print("[probe] waiting for driver to park+unpark ...", flush=True)
    while not os.path.exists("/tmp/as_go"):
        await asyncio.sleep(0.2)

    out2 = await stream_once(engine, "after-restore")
    print(f"[probe] after-restore output: {out2!r}", flush=True)
    print(f"[probe] MATCH={out1.strip() == out2.strip()}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
