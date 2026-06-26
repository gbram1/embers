#!/usr/bin/env bash
# Drives _rung2_probe.py through a cuda-checkpoint evict+restore cycle and
# reports whether GPU state round-trips for a REAL vLLM serving process, plus
# how long restore takes vs a cold start.
#
# Prereq on the box: vLLM venv (/root/csvenv) + cuda-checkpoint installed.
set +e
CC=/usr/local/bin/cuda-checkpoint
PY=/root/csvenv/bin/python

rm -f /tmp/r2_pid /tmp/r2_ready /tmp/r2_go /tmp/r2_result.json

echo "=== loading vLLM (full cold init, once) ==="
VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY /root/_rung2_probe.py & PROBE=$!

# Wait for serving-ready (model load + warm gen can take a few minutes).
for _ in $(seq 1 600); do [ -f /tmp/r2_ready ] && break; sleep 1; done
if [ ! -f /tmp/r2_ready ]; then echo "FAIL: vLLM never reached ready"; kill $PROBE; exit 1; fi
PID=$(cat /tmp/r2_pid)

MEM_READY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "serving-ready: pid=$PID  gpu_mem=${MEM_READY} MiB  state=$($CC --get-state --pid $PID 2>&1)"

echo "=== EVICT: lock + checkpoint (free the GPU) ==="
te=$(date +%s%N)
$CC --action lock --pid "$PID" --timeout 30000 2>&1
$CC --action checkpoint --pid "$PID" 2>&1
evict=$(awk "BEGIN{printf \"%.3f\", ($(date +%s%N)-$te)/1e9}")
MEM_EVICT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "  evict=${evict}s  gpu_mem ${MEM_READY} -> ${MEM_EVICT} MiB  state=$($CC --get-state --pid $PID 2>&1)"

echo "  (GPU is now free — in a real system, billing stops / GPU reused here)"
sleep 2

echo "=== RESTORE: restore + unlock (bring GPU state back) ==="
tr=$(date +%s%N)
$CC --action restore --pid "$PID" 2>&1
$CC --action unlock --pid "$PID" 2>&1
restore=$(awk "BEGIN{printf \"%.3f\", ($(date +%s%N)-$tr)/1e9}")
MEM_REST=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "  restore=${restore}s  gpu_mem -> ${MEM_REST} MiB  state=$($CC --get-state --pid $PID 2>&1)"

# Release the probe to generate its post-restore token.
touch /tmp/r2_go
wait $PROBE

echo ""
echo "=== RESULT ==="
cat /tmp/r2_result.json 2>/dev/null; echo
COLD=$($PY -c "import json;print(json.load(open('/tmp/r2_result.json'))['cold_init_s'])" 2>/dev/null)
TOK=$($PY -c "import json;print(json.load(open('/tmp/r2_result.json'))['post_restore_first_token_s'])" 2>/dev/null)
echo ""
echo "restore-path  = evict ${evict}s (one-time) ; RESTORE ${restore}s + first_token ${TOK}s"
echo "cold-start    = ${COLD}s (init) + first_token"
echo ">>> compare restore (~${restore}s) against Rung-1 cold start 56.7s"
