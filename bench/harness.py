"""Cold-start benchmark orchestrator — the project's primary deliverable.

For each of N runs it (1) drops the OS page cache, then (2) spawns a FRESH
`bench.run_once` process to do one genuinely-cold vLLM load. Spawning a new
process per run is non-negotiable: it's the only way the CUDA context and page
cache are actually cold (eng doc §7.2). Looping in one process measures a warm
load and reports a phantom number.

It collects each run's JSON timings from stdout, recovers the weight-load
sub-phase from vLLM's stderr, and reports the distribution (p50/p90/p99 — never
a single number; cold start is long-tailed and p99 is what users feel).

Run the IDENTICAL harness against every config row, including Modal/InferX, for
an apples-to-apples target.

    python -m bench.harness --config bench/configs/naive_local.yaml -n 10 \
        --out bench/results/naive_local.json

Requires a CUDA GPU + vLLM (the rented box). --no-drop-caches for a quick local
smoke test only — the numbers are NOT a valid cold baseline without it.
"""

import argparse
import json
import re
import subprocess
import sys

import yaml

from bench.protocol import CacheDropError, clear_vllm_compile_cache, drop_caches

# vLLM reports the two phases we care about directly in its log:
#   "Model loading took 5.79 GiB and 22.92 seconds"   -> storage -> VRAM
#   "init engine (profile, create kv cache, warmup model) took 79.20 seconds"
#       -> torch.compile + CUDA-graph capture + warmup (the dominant cost)
# Wording drifts across versions, so match loosely. Last match per run wins.
_MODEL_LOAD = re.compile(
    r"model loading took [\d.]+\s*gi?b and ([\d.]+)\s*second", re.IGNORECASE
)
_ENGINE_INIT = re.compile(r"init engine.*?took ([\d.]+)\s*second", re.IGNORECASE)


def parse_phases(stderr: str) -> tuple[float | None, float | None]:
    """Recover (weight_load, engine_init) seconds from vLLM's log. Either may be
    None if the line isn't found (version drift) — we then report what we have."""
    wl = _MODEL_LOAD.findall(stderr)
    ei = _ENGINE_INIT.findall(stderr)
    return (float(wl[-1]) if wl else None, float(ei[-1]) if ei else None)


def run_once(config: str, mock: bool = False) -> dict | None:
    """Spawn one fresh cold run. Returns its timings, enriched with the
    weight-load split, or None if the subprocess failed."""
    cmd = [sys.executable, "-m", "bench.run_once", "--config", config]
    if mock:
        cmd.append("--mock")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(f"\n[harness] run failed (exit {proc.returncode})\n")
        return None

    # run_once prints exactly one JSON line to stdout; take the last non-empty.
    line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    t = json.loads(line)

    # vLLM logs the phase lines to stdout (0.8.5); scan both streams to be safe.
    weight_load, engine_init = parse_phases(proc.stdout + "\n" + proc.stderr)
    if weight_load is not None:
        t["weight_load"] = weight_load          # storage -> VRAM (vLLM bundles read+xfer)
    if engine_init is not None:
        t["engine_init"] = engine_init          # compile + graph capture + warmup
    return t


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


# Reported in this order; weight_load/engine_init only present if the log parsed.
PHASES = ["weight_load", "engine_init", "construct", "first_token", "end_to_end"]


def main() -> None:
    ap = argparse.ArgumentParser(description="True-cold-start benchmark orchestrator")
    ap.add_argument("--config", required=True, help="path to a bench/configs/*.yaml")
    ap.add_argument("-n", type=int, default=10, help="runs (>=5-10 for a distribution)")
    ap.add_argument("--out", help="write raw per-run timings as JSON")
    ap.add_argument(
        "--no-drop-caches",
        action="store_true",
        help="skip page-cache drop (smoke test only — NOT a valid cold baseline)",
    )
    ap.add_argument(
        "--mock",
        action="store_true",
        help="fake the loads (no GPU/vLLM) to validate the harness on macOS; "
        "implies --no-drop-caches",
    )
    args = ap.parse_args()

    # Benchmark-control fields live in the config so each config is self-describing:
    #   keep_compile_cache: persist vLLM's torch.compile cache across runs (Rung 1).
    #       false (default) -> clear it before every run = genuinely-cold init.
    #   warmup: N leading runs that populate caches and are discarded from stats.
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    keep_cc = bool(cfg.get("keep_compile_cache", False))
    warmup = int(cfg.get("warmup", 0))

    drop = not (args.no_drop_caches or args.mock)
    caches_dropped = drop

    # When persisting the compile cache, wipe it ONCE so the warmup run repopulates
    # from scratch and the measured runs all hit a warm cache (steady state).
    if keep_cc and not args.mock:
        clear_vllm_compile_cache()

    def do_run():
        nonlocal drop, caches_dropped
        if not keep_cc and not args.mock:
            clear_vllm_compile_cache()  # true-cold init: recompile every run
        if drop:
            try:
                drop_caches()
            except CacheDropError as e:
                # Common on container GPU hosts (RunPod/Vast). Don't crash — warn
                # loudly, stop trying, and flag the results as warm-cache-biased.
                print(
                    f"\n[harness] WARNING: could not drop page cache ({e}).\n"
                    "[harness] Runs after the first are NOT truly cold — weight_read\n"
                    "[harness] is biased low. Use a bare-metal instance for a clean\n"
                    "[harness] baseline. Continuing without cache drops.\n",
                    file=sys.stderr,
                )
                drop = False
                caches_dropped = False
        return run_once(args.config, mock=args.mock)

    for w in range(warmup):
        t = do_run()
        et = f"{t['end_to_end']:.2f}s" if t else "failed"
        print(f"[harness] warmup {w + 1}/{warmup}: end_to_end={et} (discarded)")

    runs = []
    for i in range(args.n):
        t = do_run()
        if t is None:
            continue
        runs.append(t)
        print(f"[harness] run {i + 1}/{args.n}: end_to_end={t['end_to_end']:.2f}s")

    if not runs:
        sys.exit("[harness] all runs failed — see errors above")

    cold = "true-cold" if caches_dropped else "WARM-CACHE-BIASED"
    cc = "compile-cache-persisted" if keep_cc else "compile-cache-cleared"
    print(f"\n=== {args.config}  (n={len(runs)}, {cold}, {cc}) ===")
    for phase in PHASES:
        vals = [r[phase] for r in runs if phase in r]
        if not vals:
            continue
        print(
            f"{phase:14s} p50={pct(vals, 50):6.2f}s  "
            f"p90={pct(vals, 90):6.2f}s  p99={pct(vals, 99):6.2f}s"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "config": args.config,
                    "caches_dropped": caches_dropped,
                    "keep_compile_cache": keep_cc,
                    "runs": runs,
                },
                f,
                indent=2,
            )
        print(f"\n[harness] raw timings → {args.out}")


if __name__ == "__main__":
    main()
