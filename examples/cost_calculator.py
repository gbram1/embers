#!/usr/bin/env python3
"""Cost & latency comparison — always-on vs naive scale-to-zero vs embers.

Runs ANYWHERE (no GPU): it computes GPU-cost and cold-wake latency from measured
numbers for a given traffic pattern. This is the "why embers saves money" proof.

Three ways to serve a model that's idle most of the time:
  * ALWAYS-ON     — keep the GPU loaded 24/7. Zero wake latency, but you pay for
                    a fully-idle GPU around the clock.
  * NAIVE STZ     — scale to zero (kill the process) when idle. Cheap while idle,
                    but every cold wake pays the FULL model load (~100s) — and the
                    user waits that long.
  * EMBERS        — scale to zero AND fast-restore: cheap while idle, ~10s wake.

    python examples/cost_calculator.py --scenario bursty
    python examples/cost_calculator.py --wakes 48 --active-hours 2 --gpu-cost 0.50

Assumes per-second GPU billing (the scale-to-zero premise: a released GPU isn't
charged). Numbers are illustrative — plug in your own from the benchmarks.
"""
import argparse

SCENARIOS = {
    # (cold wakes/day, hours/day actively generating) — idle the rest of the time
    "dev-tool":     (16, 0.5),   # an internal tool: a few bursts in working hours
    "low-traffic":  (48, 1.0),   # a request every ~30 min, short generations
    "bursty":       (24, 2.0),   # hourly bursts of real use
    "always-busy":  (1, 20.0),   # basically always working — STZ barely helps
}


def gpu_hours(strategy, wakes, active_hours, cold_s, restore_s):
    """GPU-hours billed per day for a strategy (per-second billing)."""
    if strategy == "always-on":
        return 24.0
    load_per_wake = cold_s / 3600.0
    restore_per_wake = restore_s / 3600.0
    if strategy == "naive":
        # every wake pays a full cold load before it can serve
        return active_hours + wakes * load_per_wake
    if strategy == "embers":
        # one cold load ever, then a fast restore per subsequent wake
        return active_hours + load_per_wake + max(0, wakes - 1) * restore_per_wake
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="bursty")
    ap.add_argument("--wakes", type=int, help="cold wakes per day (overrides scenario)")
    ap.add_argument("--active-hours", type=float, help="hours/day actively generating")
    ap.add_argument("--gpu-cost", type=float, default=0.50, help="$/GPU-hour (e.g. A40)")
    ap.add_argument("--cold-load", type=float, default=100.0, help="naive cold load seconds")
    ap.add_argument("--restore", type=float, default=10.0, help="embers restore seconds")
    args = ap.parse_args()

    wakes, active = SCENARIOS[args.scenario]
    if args.wakes is not None:
        wakes = args.wakes
    if args.active_hours is not None:
        active = args.active_hours

    print(f"\n  Traffic: {wakes} cold wakes/day, {active}h/day active  "
          f"(GPU ${args.gpu_cost:.2f}/hr · cold {args.cold_load:.0f}s · restore {args.restore:.0f}s)\n")
    print(f"  {'strategy':<16}{'GPU-hrs/day':>12}{'$/day':>9}{'$/month':>10}"
          f"{'cold-wake wait':>16}")
    print(f"  {'-'*16}{'-'*12:>12}{'-'*9:>9}{'-'*10:>10}{'-'*16:>16}")

    rows = []
    for strat, label, wait in [
        ("always-on", "always-on", "0s"),
        ("naive", "naive scale-to-0", f"{args.cold_load:.0f}s"),
        ("embers", "embers", f"~{args.restore:.0f}s"),
    ]:
        h = gpu_hours(strat, wakes, active, args.cold_load, args.restore)
        day = h * args.gpu_cost
        rows.append((strat, h, day))
        print(f"  {label:<16}{h:>12.1f}{day:>9.2f}{day*30:>10.2f}{wait:>16}")

    on = dict((r[0], r) for r in rows)
    emb_day = on["embers"][2]
    print()
    print(f"  embers vs always-on:  {(1 - emb_day/on['always-on'][2])*100:.0f}% cheaper, "
          f"same fast wake")
    print(f"  embers vs naive STZ:  similar cost, but {args.cold_load/args.restore:.0f}x "
          f"faster cold wake ({args.cold_load:.0f}s → {args.restore:.0f}s)")
    print("\n  embers is the only option that's BOTH cheap-when-idle AND fast-to-wake.\n")


if __name__ == "__main__":
    main()
