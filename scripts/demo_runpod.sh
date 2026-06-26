#!/bin/bash
# One-shot embers demo on a fresh RunPod GPU pod (single CUDA GPU, driver >=550).
#
# On the pod, from the repo root:
#     bash scripts/demo_runpod.sh
#
# It builds the venv (first run, ~5 min), starts `embers up`, and runs the demo
# client — which shows cold-start → serve → scale-to-zero → fast-restore with timings.
set -e
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/embers ]; then
  echo "[demo] building venv (first run, ~5 min)…"
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q vllm==0.8.5 transformers==4.51.3 \
      fastapi uvicorn pydantic httpx pyyaml
  .venv/bin/pip install -q -e . --no-deps
  echo "[demo] fetching cuda-checkpoint (snapshot/restore)…"
  curl -fsSL -o /usr/local/bin/cuda-checkpoint \
      https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
  chmod +x /usr/local/bin/cuda-checkpoint
fi

export VLLM_ENABLE_V1_MULTIPROCESSING=0      # one PID holds CUDA, for park/restore
# the model is whatever examples/demo_platform.yaml serves (single source of truth)
MODEL=$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('examples/demo_platform.yaml'))['models'][0]['name'])")

echo "[demo] prefetching $MODEL (so the demo isn't waiting on a download)…"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

echo "[demo] starting 'embers up' …"
nohup .venv/bin/embers up --config examples/demo_platform.yaml >/root/embers_up.log 2>&1 &
UP=$!
for i in $(seq 1 60); do
  curl -sf http://localhost:8080/v1/models >/dev/null 2>&1 && break; sleep 1
done

echo "[demo] running the demo client…"
echo
.venv/bin/python examples/demo.py --url http://localhost:8080 --model "$MODEL"

echo "[demo] (embers up still running, pid $UP — stop it with: kill $UP)"
