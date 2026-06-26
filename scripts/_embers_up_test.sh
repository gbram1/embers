#!/usr/bin/env bash
# Validate `embers up` end-to-end on a real GPU. Runs the platform detached,
# then curls it through the full user-facing flow: cold start, streaming, scale
# to zero (GPU freed), restore. Run ON the pod.
#
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. \
#       bash scripts/_embers_up_test.sh
set -u
PY=/root/csvenv/bin/python
GW=18080
G() { curl -s -m 180 "$@"; }   # generous timeout: first request cold-starts

cd ~/embers
cat > /tmp/platform.yaml <<EOF
host: 127.0.0.1
port: $GW
tick_interval: 5
gpus: auto
models:
  - name: Qwen/Qwen2.5-3B
    vram_mb: 20000
    min_replicas: 0
    idle_ttl: 12
EOF

pkill -f "embers.cli up" 2>/dev/null; pkill -f "embers.cli serve" 2>/dev/null
sleep 1
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. \
    $PY -m embers.cli up -c /tmp/platform.yaml >/tmp/up.log 2>&1 &
echo "platform starting (pid $!)"

for _ in $(seq 1 30); do G http://127.0.0.1:$GW/v1/models >/dev/null && break; sleep 1; done
echo "=== /v1/models ==="; G http://127.0.0.1:$GW/v1/models

echo "=== completion (first request → REAL cold start, ~100s) ==="
echo "gpu before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
G -X POST http://127.0.0.1:$GW/v1/completions -H 'Content-Type: application/json' \
   -d '{"model":"Qwen/Qwen2.5-3B","prompt":"The capital of France is","max_tokens":8}'
echo ""; echo "gpu after cold start: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== streaming request (SSE proxied through HttpBackend) — full consume ==="
NCHUNKS=$(G -N -X POST http://127.0.0.1:$GW/v1/chat/completions -H 'Content-Type: application/json' \
   -d '{"model":"Qwen/Qwen2.5-3B","stream":true,"messages":[{"role":"user","content":"Name three colors."}]}' \
   | grep -c "^data:")
echo "streamed SSE chunks received: $NCHUNKS"
echo "stats now: $(G http://127.0.0.1:$GW/stats)"

echo "=== idle → control loop scales to zero (wait for GPU to actually free) ==="
# the park takes ~20s; wait until nvidia-smi shows the GPU genuinely freed
# (not just replicas=0, which flips at the START of the park) before restoring.
for _ in $(seq 1 20); do
  sleep 3
  MEMN=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  SD=$(G http://127.0.0.1:$GW/stats | $PY -c 'import sys,json;print(json.load(sys.stdin)["scaling"]["scale_downs"])' 2>/dev/null)
  echo "  gpu=${MEMN}MiB scale_downs=$SD"
  [ "${MEMN:-9999}" -lt 1000 ] && echo "  -> GPU freed (park complete)" && break
done

echo "=== next request → FAST RESTORE ==="
t=$(date +%s%N)
G -X POST http://127.0.0.1:$GW/v1/completions -H 'Content-Type: application/json' \
   -d '{"model":"Qwen/Qwen2.5-3B","prompt":"The capital of France is","max_tokens":8}'
echo ""; echo "restore-path request took: $(awk "BEGIN{printf \"%.1f\", ($(date +%s%N)-$t)/1e9}")s"
echo "gpu after restore: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== final /stats ==="
G http://127.0.0.1:$GW/stats | $PY -m json.tool
echo "DONE"
pkill -f "embers.cli up" 2>/dev/null; pkill -f "embers.cli serve" 2>/dev/null
