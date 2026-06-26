# embers demo

Watch the whole story in one run: a model goes **cold → serving**, **idles to zero**
(GPU freed — you stop paying), then **fast-restores in seconds** on the next request
instead of a full cold load.

```
1. COLD START   — first request spins the model up from nothing   (~tens of seconds)
2. WARM SERVE   — now loaded, requests are milliseconds
3. SCALE TO ZERO— idle → GPU released to 0 MB (you stop paying)
4. FAST RESTORE — next request restores the snapshot               (~10s, not a cold load)
```

## On a real GPU (the real demo) — RunPod, one command

Rent a single CUDA GPU (driver ≥ 550 for snapshot/restore — e.g. a RunPod A40 / L4 /
A4000). Then, on the pod, from the repo root:

```bash
bash scripts/demo_runpod.sh
```

That builds the env (first run ~5 min), starts `embers up`, and runs the demo client.
You'll see each phase with its timing and the GPU state — the headline being the
**fast restore vs the cold start**.

Want a bigger contrast? Edit `examples/demo_platform.yaml` to use `Qwen/Qwen2.5-3B`
(its cold load is ~100s, so the ~10s restore looks even better).

## The full comparison suite

Want *every* with-vs-without test (cold-start, cost economics, warm overhead,
tensor-parallel, LoRA density, integration)? See **[COMPARISON.md](COMPARISON.md)** —
a matrix of what each proves, where it runs, and the expected numbers. Two you can run
on your laptop right now:

```bash
python cost_calculator.py --scenario bursty     # the $ proof (no GPU)
python your_app.py --url http://localhost:8080/v1 --model demo-3b   # integration (vs `embers up --mock`)
```

## Test it yourself: with vs without embers

The honest split — **what's testable where:**

| | On your Mac (no GPU) | On a rented GPU |
|---|---|---|
| Your app talks to embers (integration) | ✅ `embers up --mock` + `your_app.py` | ✅ |
| The cold-start **speedup** (the real benefit) | ❌ nothing to load → nothing to skip | ✅ |

embers' whole benefit is *skipping the GPU model-load* — which only exists on CUDA.
So integration you can check on a laptop; the speedup needs a GPU.

**Your app is identical either way** — embers is a drop-in OpenAI endpoint. The same
[`examples/your_app.py`](your_app.py) runs against a vanilla vLLM server or against
embers; only the *cold-wake cost* differs.

On a GPU, run both and compare:

```bash
# WITHOUT embers — the naive cold-load cost (repeats on every wake):
pip install vllm==0.8.5
python examples/without_embers.py --model Qwen/Qwen2.5-3B        # prints ~100s per cold load

# WITH embers — cold once, then fast restore:
bash scripts/demo_runpod.sh                                      # prints ~60s cold, ~8s restore
# or point your app at it after it scales to zero:
python examples/your_app.py --url http://localhost:8080/v1 --model Qwen/Qwen2.5-3B
```

The gap — **~100s cold load every wake vs ~8s restore** — is the whole pitch.

## On your laptop (no GPU — see the flow)

You can run the same flow with fake in-process models to see the API and the
scale-to-zero logic (timings won't be real — there's no model load to skip):

```bash
pip install -e '.[serve]'
# a mock config needs explicit GPUs (it can't probe nvidia-smi):
embers up --config deploy/platform.local.yaml --mock          # terminal 1
python examples/demo.py --url http://localhost:8080 --model demo-3b   # terminal 2
```

## What it looks like (real run, A40, Qwen2.5-3B)

```
1. COLD START    cold start → first response: 59.0s     GPU: 8000 MB · replicas: 1
2. WARM SERVE    ~285 ms per request
3. SCALE TO ZERO scaled to zero after ~39s idle         GPU: 0 MB ← GPU released
4. FAST RESTORE  scale-from-zero → first response: 7.9s  GPU: 8000 MB · replicas: 1
   SUMMARY       cold 59.0s · restore 7.9s → ~7.5× faster · snapshot_hit_rate 0.5
```

The headline: the model idles to **0 MB (you stop paying)**, then the next request
restores it in **~8s instead of ~60s** — the cold-start cost paid once, skipped forever.

## What you're looking at

- **Cold start** is the slow part everyone pays on a scale-to-zero serverless GPU —
  loading weights + building the engine (torch.compile, CUDA graphs, KV cache). embers
  snapshots that built state once.
- **Fast restore** skips all of it: it restores the GPU snapshot (cuda-checkpoint),
  so scale-from-zero is ~10s instead of ~100s.
- The demo prints `snapshot_hit_rate` from `/stats` — restores ÷ (restores + cold loads),
  the headline metric.

Run them on your own GPU to get your own numbers.
