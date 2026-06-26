"""One genuinely-cold load → first token, in a FRESH process.

This script is spawned anew by bench/harness.py for every run — that is what
makes each run cold (the doc's #1 rule: never reuse a loaded model; a second
LLM() in a live process reuses the CUDA context and warm caches). It does the
real vanilla-vLLM load, then prints exactly one JSON line of timings to STDOUT.
All vLLM logging goes to STDERR, where the harness parses the weight-load line
for the sub-phase split.

Not meant to be run by hand (though you can, for a single-shot sanity check):

    python -m bench.run_once --config bench/configs/naive_local.yaml

Requires a CUDA GPU + vLLM installed (i.e. the rented box, not macOS).
"""

import argparse
import json
import random
import sys
import time

import yaml

from bench.protocol import assert_no_resident_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mock_load(cfg: dict) -> dict:
    """Fake a cold load — NO GPU, NO vLLM. Lets you validate the harness loop
    (subprocess-per-run, JSON flow, stderr parsing, percentiles) on macOS before
    renting a GPU. Numbers are synthetic and mean nothing. Emits a fake vLLM
    weight-load log to stderr so the harness's split-parsing path is exercised.
    """
    weight = random.uniform(15.0, 22.0)      # pretend "weight_load" seconds
    init = random.uniform(30.0, 50.0)        # pretend engine init (the dominant cost)
    time.sleep(0.05)                         # keep it fast; don't actually wait 50s
    print(
        f"INFO Loading model weights took 14.99 GB and {weight:.2f} seconds",
        file=sys.stderr,
    )
    construct = weight + init
    first_token = random.uniform(0.05, 0.15)
    return {
        "construct": construct,
        "first_token": first_token,
        "end_to_end": construct + first_token,
    }


def cold_load(cfg: dict) -> dict:
    """Construct vanilla vLLM cold and generate one token. Returns timings (s).

    vanilla vLLM streams weights directly to the GPU inside the LLM()
    constructor, so weight_read and host_to_gpu are NOT separable here — they're
    bundled in `construct`. The harness recovers the weight-load portion from
    vLLM's stderr log; the remainder of `construct` is engine init (compile,
    CUDA-graph capture, KV-cache alloc, warmup). That weight-vs-init split is the
    whole point of the baseline: it tells us which phase dominates before we
    build anything. Separating read from transfer is OUR job later (phase 1.3).
    """
    # Imported here, after the cold check, so import cost isn't charged to load.
    from vllm import LLM, SamplingParams

    assert_no_resident_model()

    t0 = time.perf_counter()
    llm = LLM(
        model=cfg["model"],
        dtype=cfg.get("dtype", "auto"),
        max_model_len=cfg.get("max_model_len"),
        enforce_eager=cfg.get("enforce_eager", False),
    )
    t_construct = time.perf_counter() - t0

    # First token: the actual inference cost, separate from load/init.
    t1 = time.perf_counter()
    llm.generate(["ping"], SamplingParams(max_tokens=1))
    t_first_token = time.perf_counter() - t1

    return {
        "construct": t_construct,          # weight read + host->gpu + engine init
        "first_token": t_first_token,
        "end_to_end": t_construct + t_first_token,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Single cold vLLM load (fresh process)")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--mock",
        action="store_true",
        help="fake the load (no GPU/vLLM) to test the harness locally",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    timings = mock_load(cfg) if args.mock else cold_load(cfg)
    # The ONLY thing on stdout is this JSON line. Harness reads the last line.
    print(json.dumps(timings), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
