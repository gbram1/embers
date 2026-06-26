# Benchmarks — establish the baseline first

Before building anything, get a real, self-measured cold-start
baseline for vanilla vLLM. The phase breakdown tells you which cost dominates
(often engine init, not byte movement) and therefore what's actually worth
building. This *is* the derisking step (eng doc §7, project plan Phase 0/1.1).

## Where this runs

A **rented single GPU** (RunPod / Lambda / Vast), 8B model on one A10/L4/A100.
Not macOS — a *real* run needs CUDA + vLLM; `drop_caches` and `nvidia-smi` are
Linux-only.

## Test the harness locally first (no GPU)

Validate the orchestration — subprocess-per-run, JSON flow, stderr parsing,
percentiles — on your laptop with synthetic loads, so you don't debug plumbing
on paid GPU time. `--mock` skips the GPU/vLLM entirely (and implies
`--no-drop-caches`); the numbers are fake.

```bash
python3 -m venv .venv && .venv/bin/pip install pyyaml
.venv/bin/python -m bench.harness --config bench/configs/naive_local.yaml -n 8 --mock
```

Then drop `--mock` on the GPU box for the real baseline.

## Setup (on the box)

```bash
pip install -e '.[bench]'        # installs vLLM + pyyaml
huggingface-cli login            # if the model is gated
```

## Run the control baseline

```bash
python -m bench.harness \
    --config bench/configs/naive_local.yaml \
    -n 10 \
    --out bench/results/naive_local.json
```

`drop_caches` needs sudo; run as a user that can `sudo tee /proc/sys/vm/drop_caches`
passwordless, or run the harness under `sudo -E`.

## What you get

Each run is a **fresh process** (the only way it's truly cold). Output:

```
weight_load    p50= ...s  p90= ...s  p99= ...s   # read + host->gpu (vLLM bundles these)
engine_init    p50= ...s  p90= ...s  p99= ...s   # compile, CUDA graphs, KV cache, warmup
construct      p50= ...s  ...                     # weight_load + engine_init
first_token    p50= ...s  ...
end_to_end     p50= ...s  ...
```

`weight_load` / `engine_init` are recovered from vLLM's own log line; if vLLM's
wording has drifted and the harness can't find it, you still get the rock-solid
`construct` / `first_token` / `end_to_end` numbers (the split is a bonus).

Two takeaways to look for: **does engine_init dominate?** (if so, snapshot/restore
is the high-leverage build, not faster I/O), and **how noisy is the p99 tail?**

## Then anchor the target

Run the **same harness** against Modal's snapshot example (`modal_target.yaml`)
and/or InferX. That measured number — not a blog "10x" — is the bar to beat.

## Caveats / honest limits of this baseline

- vanilla vLLM streams weights **straight to the GPU**, so `weight_read` and
  `host_to_gpu` are NOT separable here — they're both inside `weight_load`.
  Separating and overlapping them is *our* phase-1.3 work, not the baseline's.
- vLLM v1 runs the engine in child processes; timings are still measured from
  the driver and the weight-load log still reaches stderr, so the harness works
  on both v0 and v1.
