# Deploying embers on your own GPU

This is the self-serve guide for running embers in production — a single config and one
command (`embers up`) gives you an OpenAI-compatible endpoint with automatic
scale-to-zero and fast snapshot-restore cold starts.

For a no-GPU local taste of the API and control loop, skip to [Mock mode](#mock-mode-no-gpu).
For the design rationale, see [`README.md`](README.md).

---

## 1. Prerequisites

| Requirement | Why |
|---|---|
| **Linux + NVIDIA GPU** | vLLM serves on CUDA. (Apple Silicon / AMD can't run the real path — use mock mode.) |
| **Driver R550+** | `cuda-checkpoint` (the snapshot park/unpark) needs it. Check: `nvidia-smi`. |
| **CUDA 12.4-compatible host** | The pinned stack is `vllm==0.8.5` + `torch 2.6.0+cu124`. A different driver needs a matching vLLM/torch. |
| **`cuda-checkpoint` on PATH** | The fast-restore mechanism. Install below. |
| **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** | So a single PID holds the CUDA context (required for park/unpark to target the right process). |

Where to run it: a rented GPU (RunPod / Lambda / Vast), a bare-metal box, or a cloud VM
(EC2 `g5`, GCP `a2`, etc.). Validated on a single **A40**; start with one 3B–8B model on
one GPU, not multi-GPU 70B.

> **Note on snapshot privileges.** On RunPod-style containers, the GPU-memory park works
> *unprivileged*. Full CRIU process-to-disk needs `CAP_SYS_ADMIN` / a bare-metal or
> privileged host. Without those caps, scale-to-zero still works — it just cold-loads on
> the next request instead of fast-restoring.

Install `cuda-checkpoint`:

```bash
curl -fsSL -o /usr/local/bin/cuda-checkpoint \
  https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
chmod +x /usr/local/bin/cuda-checkpoint
```

---

## 2. Option A — install on the box (recommended to start)

```bash
git clone <repo> && cd embers
python3 -m venv .venv && source .venv/bin/activate

# Pinned stack first (pulls the right torch), then the package + serve deps.
pip install vllm==0.8.5 transformers==4.51.3
pip install -e '.[serve]'

export VLLM_ENABLE_V1_MULTIPROCESSING=0

embers init                          # writes platform.yaml
$EDITOR platform.yaml                # declare your real models (see §4)
embers up --config platform.yaml     # serves the whole platform on :8080
```

`embers up` assembles the gateway + scheduler + autoscaler + cold-start loader, starts a
background control loop, and serves `/v1/*`, `/metrics`, `/stats`. A model with no traffic
stays at **0 replicas** (zero GPU) until the first request.

---

## 3. Option B — Docker (production image)

The repo `Dockerfile` bakes the pinned CUDA-12.4 stack + `cuda-checkpoint`:

```bash
docker build -t embers:gpu .

docker run --gpus all \
  --cap-add SYS_ADMIN --cap-add CHECKPOINT_RESTORE \
  -v $PWD/platform.yaml:/etc/embers/platform.yaml \
  -p 8080:8080 \
  embers:gpu
```

You mount your own `platform.yaml` at `/etc/embers/platform.yaml`. The `ENTRYPOINT` is the
bare CLI and the default `CMD` runs `up --config /etc/embers/platform.yaml`, so you can
also override args (e.g. append `serve <model>` for a single unit).

> Building requires a Docker daemon with GPU passthrough (`nvidia-container-toolkit`).
> Standard RunPod *pods* are themselves containers with no nested Docker — use Option A
> there. A Docker-capable GPU host (bare-metal / Lambda / EC2) is needed for this path.

A Helm chart for Kubernetes lives in [`deploy/helm/embers`](deploy/helm/embers) (set
`mock=false`, `image.repository`, `image.tag`; it grants the snapshot caps off-mock).

---

## 4. The config (`platform.yaml`)

`embers init` writes a starter. Full reference:

```yaml
host: 0.0.0.0
port: 8080                 # the gateway / public API port
api_keys: []               # bearer tokens; empty = open (dev only)
tick_interval: 15          # control-loop period (seconds)
serve_port_base: 19000     # serving units bind 127.0.0.1:serve_port_base+N — keep HIGH

gpus: auto                 # detect via nvidia-smi, OR an explicit list:
# gpus:
#   - {id: gpu0, vram_mb: 46068}

models:
  - name: Qwen/Qwen2.5-3B  # HF id or local path
    vram_mb: 8000          # this model's GPU footprint (for bin-packing)
    min_replicas: 0        # 0 = scale fully to zero when idle
    max_replicas: 2        # ceiling under load
    idle_ttl: 300          # seconds idle before scaling to zero (GPU freed)
```

- **`gpus: auto`** detects every GPU via `nvidia-smi`; the scheduler bin-packs models across them.
- **`min_replicas: 0`** is the scale-to-zero switch — pay nothing while idle.
- **`serve_port_base` must be high** on RunPod (it proxies localhost 8000/8001/8888).

---

## 5. Calling it (what your end users do)

It's a drop-in OpenAI endpoint — point any OpenAI client / LangChain / chat UI at
`http://<box>:8080/v1`. Nothing embers-specific.

```bash
# non-streaming
curl http://<box>:8080/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"Qwen/Qwen2.5-3B","messages":[{"role":"user","content":"hi"}]}'

# streaming (token-by-token SSE)
curl -N http://<box>:8080/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"Qwen/Qwen2.5-3B","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://<box>:8080/v1", api_key="<token-or-anything-if-open>")
for chunk in client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[{"role": "user", "content": "hi"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

If `api_keys` is set, requests need `Authorization: Bearer <token>` (SDK: `api_key=`).

**The lifecycle**, transparent to the caller:

```
request → cold-start (first time, ~57s; captures a snapshot)
        → idle past idle_ttl → scale-to-zero (GPU freed, snapshot kept)
        → next request → fast-restore (~7–10s, not a full cold start)
```

---

## 6. Observability

```bash
embers dashboard --url http://<box>:8080   # GPU util, replicas, scale events, cold-start win
curl http://<box>:8080/v1/models           # every servable model (incl. scaled-to-zero)
curl http://<box>:8080/stats               # JSON platform view
curl http://<box>:8080/metrics             # Prometheus
```

The headline metric is **`snapshot_hit_rate` = restores / (restores + cold_loads)** — how
often a spin-up took the fast path. The `/stats` view also shows per-GPU utilization,
live replica counts, and scale-up/down/to-zero counters.

---

## 7. Correctness guarantee (don't bypass it)

Every restore is gated on a **dependency fingerprint** — the hash of the actual weight
bytes + model/arch + vLLM version + GPU type + driver/CUDA version + config flags. A
snapshot is only restored on an exact match; otherwise embers runs the slow path and
captures a fresh one. This is non-negotiable: serving a stale or cross-GPU snapshot
produces *silent wrong output*. **Never** move a snapshot to a different GPU model or
driver — the fingerprint refuses it by design.

---

## 8. Mock mode (no GPU)

To exercise the full API + control loop on any machine (macOS included), with in-process
fake models — no GPU, no vLLM:

```bash
pip install -e '.[serve]'                          # no vLLM needed
embers up --config deploy/platform.local.yaml --mock
```

Same endpoints, same scale-to-zero logic; responses are placeholder strings. Good for
validating routing/streaming/deploys before renting hardware. (Mock can't use
`gpus: auto` — use an explicit `gpus:` list, as in `deploy/platform.local.yaml`.)

---

## 9. Validated results & current limits

**Validated on real hardware (A40, single GPU):**

- Cold-start ladder, Qwen2.5-3B: naive **98.6s** → torch.compile cache **56.7s** →
  cuda-checkpoint restore **~7–10s**.
- Full platform cold→serve→scale-to-zero→restore, incl. true token streaming and a
  request firing mid-park (waited + served, never 503'd).

**Honest limits (today):**

- **Single GPU / single box** is validated. Multi-GPU bin-packing is implemented and
  `gpus: auto` will place across GPUs, but multi-GPU / multi-node is **not yet
  hardware-validated** — don't promise it for production multi-node.
- **Snapshot privileges** vary by host (see §1) — without them, scale-to-zero falls back
  to cold-load instead of fast-restore (still correct, just slower).
- The stack is **version-pinned** to a CUDA-12.4 host driver; a different driver needs the
  matching vLLM/torch (and the fingerprint will refuse cross-driver restores).

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPUs configured/detected` | `gpus: auto` found nothing — run on a GPU box, or list `gpus:` explicitly. |
| Park doesn't free the GPU | Ensure `VLLM_ENABLE_V1_MULTIPROCESSING=0`; confirm driver ≥550 and `cuda-checkpoint` on PATH. |
| Restores never happen (always cold-load) | Host lacks snapshot privileges, or the fingerprint changed (different driver/weights/flags). |
| 503 on first request | No GPU capacity for placement — lower `vram_mb` or free a GPU. 404 = unknown model name. |
| RunPod: endpoint unreachable | Bind high ports — it proxies localhost 8000/8001/8888. Keep `serve_port_base` ≥ 19000. |
