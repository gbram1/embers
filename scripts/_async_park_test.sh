#!/usr/bin/env bash
# Validate the ASYNC park on real hardware: a request arriving DURING the ~20s
# park must WAIT then serve (HTTP 200), NOT 503 — and the control loop / server
# must stay responsive while the park runs. Run ON the pod.
set -u
PY=/root/csvenv/bin/python
GW=18080
G() { curl -s -m 200 "$@"; }
cd ~/embers
cat > /tmp/platform.yaml <<EOF
host: 127.0.0.1
port: $GW
tick_interval: 3
gpus: auto
models: [{name: Qwen/Qwen2.5-3B, vram_mb: 20000, min_replicas: 0, idle_ttl: 8}]
EOF
pkill -9 -f embers.cli 2>/dev/null; sleep 2
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. $PY -m embers.cli up -c /tmp/platform.yaml >/tmp/up.log 2>&1 &
for _ in $(seq 1 40); do G http://127.0.0.1:$GW/v1/models >/dev/null && break; sleep 1; done

echo "=== cold start (request 1) ==="
G -H 'Content-Type: application/json' -X POST http://127.0.0.1:$GW/v1/completions \
  -d '{"model":"Qwen/Qwen2.5-3B","prompt":"hi","max_tokens":4}' >/dev/null
echo "cold done, gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== wait for park to BEGIN (replicas=0 while GPU still high) ==="
for i in $(seq 1 60); do
  REP=$(G http://127.0.0.1:$GW/stats | $PY -c 'import sys,json;print(json.load(sys.stdin)["replicas"]["Qwen/Qwen2.5-3B"])' 2>/dev/null)
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$REP" = "0" ] && [ "${MEM:-0}" -gt 1000 ]; then echo "  PARK IN PROGRESS (replicas=0 gpu=${MEM}MiB)"; break; fi
  sleep 0.5
done

echo "=== fire request DURING park (must wait+serve, NOT 503) ==="
t=$(date +%s%N)
( G -o /tmp/r2.json -w "%{http_code}" -H 'Content-Type: application/json' -X POST http://127.0.0.1:$GW/v1/completions \
    -d '{"model":"Qwen/Qwen2.5-3B","prompt":"The capital of France is","max_tokens":8}' > /tmp/r2.code ) &
REQPID=$!
sleep 2
echo "  control loop responsive mid-park? /v1/models -> HTTP $(G -o /dev/null -w '%{http_code}' http://127.0.0.1:$GW/v1/models)"
wait $REQPID
DUR=$(awk "BEGIN{printf \"%.1f\", ($(date +%s%N)-$t)/1e9}")
echo "  request-during-park: HTTP $(cat /tmp/r2.code) in ${DUR}s"
echo "  body: $(cat /tmp/r2.json)"
echo "  gpu after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "  final /stats: $(G http://127.0.0.1:$GW/stats)"
echo "DONE_ASYNC"
pkill -9 -f embers.cli 2>/dev/null
