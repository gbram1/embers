# With vs without embers — the full comparison suite

Every way embers differs from serving a model the plain way, what each test proves,
how to run it, and what to expect. Numbers are from real A40 runs; plug in your own.

| # | Test | Proves | Where it runs | Script |
|---|------|--------|---------------|--------|
| 1 | Cold-start time | restore ~10s vs cold load ~100s | GPU | `without_embers.py` + `demo.py` |
| 2 | Cost economics | 90%+ cheaper than always-on, 10× faster wake than naive | **anywhere** | `cost_calculator.py` |
| 3 | Warm-path overhead | embers adds ~0 latency once loaded | GPU | `warm_overhead.py` |
| 4 | Big model (tensor-parallel) | the cold→restore win extends to models too big for one GPU | 2-GPU | (below) |
| 5 | LoRA density | 1 base + N adapters on one GPU vs N full models | GPU | (below) |
| 6 | App integration | your code is identical with or without embers | **anywhere (mock)** | `your_app.py` |

The honest split: **integration + economics** you can test on your laptop today;
the **timing wins** need a GPU (embers' benefit is skipping the GPU model-load, which
only exists on CUDA).

---

## 1. Cold-start time — the headline

**Without** (`python examples/without_embers.py --model Qwen/Qwen2.5-3B`): loads the
model with raw vLLM and times it — then loads it *again* to show the cost repeats on
every cold wake. Expect **~60–100s per cold load**.

**With** (`bash scripts/demo_runpod.sh`, or `embers up` + `examples/demo.py`): cold
once, then scale-to-zero → **fast restore ~8–10s** on the next request.

> Real A40 run: cold **59.0s** → restore **7.9s** = **~7.5× faster**. The naive way
> pays the 59s on *every* wake; embers pays it once.

## 2. Cost economics — the $ proof (no GPU needed)

```bash
python examples/cost_calculator.py --scenario bursty
python examples/cost_calculator.py --scenario low-traffic --gpu-cost 0.50
```

Computes GPU-$/day for **always-on** vs **naive scale-to-zero** vs **embers** under a
traffic pattern. Expect (bursty, $0.50/hr GPU):

```
always-on          24.0 GPU-hr/day   $360/mo    0s wake
naive scale-to-0    2.7 GPU-hr/day   $ 40/mo  100s wake   ← users wait every cold req
embers              2.1 GPU-hr/day   $ 31/mo  ~10s wake   ← cheap AND fast
```

embers is the only option that's **both** cheap-when-idle **and** fast-to-wake.

## 3. Warm-path overhead — the credibility control

```bash
python examples/warm_overhead.py --url http://localhost:8080 --model Qwen/Qwen2.5-3B -n 30
```

Once warm, embers forwards straight to vLLM, so warm latency ≈ raw vLLM (the gateway
adds <1ms). This proves **embers doesn't slow you down** when loaded — its cost is
zero on the warm path; the value is entirely the cold path.

## 4. Big model (tensor parallel) — the win at scale

A model too big for one GPU is sharded across N GPUs (`tensor_parallel_size: N`). The
naive cold load is ~the same ~100s; embers snapshots and restores the *multi-rank* GPU
state. Run `embers up` with a `tensor_parallel_size: 2` model on a 2-GPU box and watch
the restore.

> Real 2× A40 run: TP cold load ~103s → **TP restore ~16s**. The naive way reloads all
> shards every wake.

## 5. LoRA density — multi-tenant fine-tuning

**Without**: N fine-tuned models = N × the base model's GPU memory (e.g. 10 customers ×
a 6 GB base = 60 GB → multiple GPUs).
**With** (`adapters:` in the config): one base + N small adapters share **one** GPU's
memory — each request picks its adapter via the `model` field.

> Validated on A40: base + adapter served off one GPU, the adapter applied per-request.
> Cost scales with *bases*, not *fine-tunes*.

## 6. App integration — embers is a drop-in

```bash
# laptop, no GPU — prove your app works against embers:
pip install 'embers[serve]' openai
embers up --config ../deploy/platform.local.yaml --mock          # terminal 1
python examples/your_app.py --url http://localhost:8080/v1 --model demo-3b   # terminal 2
```

`your_app.py` is a plain OpenAI client — the **same code** runs against a vanilla vLLM
server or embers. Switching to embers changes nothing in your app; it only changes what
a cold wake costs.

---

## Suggested order to run it

1. On your laptop now: **#2 cost** (`cost_calculator.py`) and **#6 integration**
   (`your_app.py` against `embers up --mock`).
2. On a 1-GPU pod: **#1 cold-start** (`without_embers.py` then `demo.py`),
   **#3 warm overhead** (`warm_overhead.py`).
3. On a 2-GPU pod: **#4 tensor-parallel** restore.
