#!/usr/bin/env bash
# Diagnose scale-to-zero (park) on `embers up`: cold-start one model, then watch
# /stats + nvidia-smi every 5s for 90s WITHOUT sending more requests — so we see
# whether the park eventually frees the GPU (slow) or never does (hang).
set -u
PY=/root/csvenv/bin/python
GW=18080
cd ~/embers
cat > /tmp/platform.yaml <<EOF
host: 127.0.0.1
port: $GW
tick_interval: 5
gpus: auto
models: [{name: Qwen/Qwen2.5-3B, vram_mb: 20000, min_replicas: 0, idle_ttl: 10}]
EOF
pkill -9 -f embers.cli 2>/dev/null; sleep 2
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. $PY -m embers.cli up -c /tmp/platform.yaml >/tmp/up.log 2>&1 &
for _ in $(seq 1 30); do curl -s -m 5 http://127.0.0.1:$GW/v1/models >/dev/null && break; sleep 1; done

echo "=== cold start ==="
curl -s -m 180 -X POST http://127.0.0.1:$GW/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B","prompt":"hi","max_tokens":4}' >/dev/null
echo "unit pid(s) + tree:"; pgrep -af "embers.cli serve" | grep -v pgrep
echo "child procs of the unit (EngineCore = multiprocess!):"; ps --ppid "$(pgrep -f 'embers.cli serve'|head -1)" -o pid,cmd 2>/dev/null | head

echo "=== watch park progression (90s, no requests) ==="
for i in $(seq 1 18); do
  S=$(curl -s -m 5 http://127.0.0.1:$GW/stats)
  REP=$(echo "$S" | $PY -c 'import sys,json;d=json.load(sys.stdin);print(d["replicas"],"sd="+str(d["scaling"]["scale_downs"]),"s0="+str(d["scaling"]["scaled_to_zero"]))' 2>/dev/null)
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  echo "[$((i*5))s] $REP gpu=$MEM"
  sleep 5
done
echo "=== park error in log? ==="
grep -nE "\[autoscaler\]|control-loop|cuda-checkpoint|Traceback|park" /tmp/up.log | tail -5
echo "DONE_DIAG"
pkill -9 -f embers.cli 2>/dev/null
