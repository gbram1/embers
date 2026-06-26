# embers

**Scale-to-zero LLM serving that runs unprivileged on a single rented GPU — no Kubernetes, no privileged host.** Snapshot a serving-ready GPU and restore it in **~10 seconds**, so a model sits at **zero GPUs (and zero cost) while idle** and still answers the next request fast. OpenAI-compatible, built on vLLM.

Serving an LLM otherwise forces a bad choice: keep GPUs running 24/7 (paying for idle time), or scale to zero and eat a ~100-second cold start on every wake. embers removes the tradeoff.

```
cold start (first ever):   104.0s    ← load weights + build the engine
fast restore (from zero):   10.7s    ← restore the GPU snapshot   (~10× faster)
```
*Measured on a ~$0.40/hr rented A40 (Qwen2.5-3B): cold → serving → idle (GPU freed to 0 MB) → restore.*

### How it compares

The snapshot mechanism — NVIDIA `cuda-checkpoint` (+ CRIU for full process-to-disk) — is the same one **NVIDIA Dynamo Snapshot** uses. The difference is *where it runs*: Dynamo Snapshot is Kubernetes-native, needs a privileged host (`CAP_SYS_ADMIN`), and is single-GPU in its current preview. embers runs **unprivileged on a bare rented pod, with no Kubernetes, and snapshots tensor-parallel (multi-GPU) models today** (validated on 2×A40). Rule of thumb: datacenter K8s cluster → Dynamo; a GPU you rented by the hour → embers.

## Features

- **Scale-to-zero + fast restore** — idle models free the GPU (you stop paying); the next request restores in seconds, not minutes. Uses NVIDIA `cuda-checkpoint`, **unprivileged** (works on rented pods — no `CAP_SYS_ADMIN`).
- **OpenAI-compatible** — drop-in `/v1/chat/completions`, `/v1/completions`, and token-by-token streaming (SSE). Point any OpenAI client at it.
- **Multi-GPU** — data-parallel replicas, and **tensor-parallel** sharding for models too big for one GPU (with multi-rank snapshot/restore).
- **Multi-tenant density** — pack several models onto one GPU (fractional GPU), serve **many LoRA adapters off one base model**, and **over-commit** with cost-aware demand eviction.
- **Distributed** — a control plane that places models across many nodes, with heartbeat health-checks and durable (SQLite) state.
- **Observability + metering** — Prometheus metrics, per-model token accounting, and a live dashboard.
- **326 tests**; hardware-validated on A40 and 2×A40.

## Install

Not on PyPI yet — install from source into a virtualenv:

```bash
git clone https://github.com/gbram1/embers.git
cd embers
python3 -m venv .venv && source .venv/bin/activate
pip install '.[serve]'               # control plane + OpenAI-compatible API
# on a Linux + CUDA box, add the vLLM engine (needs Python < 3.13):
pip install '.[serve,gpu]'
```

Extras: `serve` (platform/API), `gpu` (real vLLM serving, Linux + CUDA), `bench` (cold-start harness), `dev` (tests). `[serve]` is pure-Python and installs anywhere, including macOS, so you can run the control plane in `--mock` mode without a GPU.

## Quickstart

On a CUDA GPU box (driver ≥ 550). Snapshot/restore needs NVIDIA's `cuda-checkpoint`
binary on the box (without it, scale-to-zero still works but wakes are full cold loads):

```bash
sudo curl -fsSL -o /usr/local/bin/cuda-checkpoint \
  https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
sudo chmod +x /usr/local/bin/cuda-checkpoint
```

Then:

```bash
# the repo ships a starter platform.yaml (`embers init` regenerates one)
# edit it — declare your models + their VRAM footprint — then:
embers up --config platform.yaml     # runs the whole platform
```

`platform.yaml`:

```yaml
port: 8080
gpus: auto                 # detect via nvidia-smi, or list [{id, vram_mb}]
models:
  - name: Qwen/Qwen2.5-3B
    vram_mb: 6000
    min_replicas: 0        # 0 = scale fully to zero when idle
    idle_ttl: 300          # seconds idle → scale to zero (GPU freed)
```

A cold model spins up on the first request; an idle one scales to zero (GPU freed); the next request restores it in ~10s instead of cold-loading ~100s.

**No GPU handy?** `embers up --mock` runs the whole platform with in-process fake models — same API, same control loop — for local exploration.

## Use it from your app

It's a drop-in OpenAI endpoint — point any OpenAI client at it and select a model by name:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="x")  # any key if api_keys=[]
client.chat.completions.create(
    model="Qwen/Qwen2.5-3B",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
)
```

```bash
curl localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B","messages":[{"role":"user","content":"hi"}]}'
curl localhost:8080/v1/models     # every servable model (even scaled-to-zero)
curl localhost:8080/metrics       # Prometheus
```

Or embed it programmatically:

```python
import embers
p = embers.Platform(embers.load_config("platform.yaml"))
p.serve()                          # serves /v1/* + /metrics + /stats, blocks
```

## See it in one run

`examples/demo.py` drives the whole story — **cold start → warm serve → scale to zero (GPU freed) → fast restore** — printing each phase's timing. On a rented single GPU (driver ≥ 550), from the repo root:

```bash
bash scripts/demo_runpod.sh        # builds the env, starts embers up, runs the demo
```

More examples (multi-model packing, a with/without comparison, a cost calculator) and laptop-mock instructions are in [`examples/`](examples/).

## How it works

The first time a model serves, embers does a normal vLLM load and **captures a snapshot** of the serving-ready GPU state — weights, CUDA context, KV-cache allocation, CUDA graphs. When the model scales to zero, the GPU is freed but the snapshot is kept. The next request **restores the snapshot directly**, skipping the engine init (torch.compile, CUDA-graph capture, warmup) that dominates a cold start — cold start is recomputation, not byte movement.

Every restore is gated by a **dependency fingerprint** (actual weight bytes + GPU model + driver/CUDA version + vLLM version + config). Match → restore; mismatch → cold-load and re-capture a fresh snapshot. A snapshot is **never** restored across a different GPU or version — serving stale state would mean silent wrong output.

## Benchmarks

Cold start is **engine init, not data movement** — weight load is ~2s; init is the rest. So the wins come from *skipping* init, not from faster I/O. End-to-end cold start (cold → first token), Qwen2.5-3B on a single GPU:

| Configuration | p50 | p90 | p99 |
|---|---|---|---|
| Naive vLLM (control) | 98.6s | 110.3s | 113.5s |
| + persist torch.compile cache | 56.7s | 57.2s | 57.3s |
| + GPU checkpoint/restore | **8.9s** | 9.4s | 9.6s |

Methodology: fresh process per run, OS page cache dropped between runs, ≥ 5–10 runs reported as a distribution (never a single number). The `bench/` harness enforces true-cold measurement.

## Deploying

See **[DEPLOY.md](DEPLOY.md)** for the self-serve guide — prerequisites, Docker, config reference, snapshot privileges, and troubleshooting.

## License

Apache-2.0.
