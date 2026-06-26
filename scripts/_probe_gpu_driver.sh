#!/usr/bin/env bash
# Drives _probe_gpu.py through a cuda-checkpoint checkpoint+restore cycle and
# reports whether GPU memory round-trips correctly in THIS container.
set +e
CC=/usr/local/bin/cuda-checkpoint

rm -f /tmp/cc_pid /tmp/cc_go
python3 /root/_probe_gpu.py & PYPID=$!
for _ in $(seq 1 90); do [ -f /tmp/cc_pid ] && break; sleep 1; done
PID=$(cat /tmp/cc_pid 2>/dev/null)
if [ -z "$PID" ]; then echo "FAIL: probe never started"; exit 1; fi

echo "probe PID=$PID  state: $($CC --get-state --pid "$PID" 2>&1)"
MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "gpu mem before: ${MEM_BEFORE} MiB"

# Correct sequence: lock -> checkpoint -> restore -> unlock.
echo "--- lock (quiesce CUDA) ---"
$CC --action lock --pid "$PID" --timeout 10000 2>&1; echo "  exit=$?  state: $($CC --get-state --pid "$PID" 2>&1)"
echo "--- checkpoint (evict GPU->host) ---"
$CC --action checkpoint --pid "$PID" 2>&1; echo "  exit=$?  state: $($CC --get-state --pid "$PID" 2>&1)"
MEM_CKPT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "gpu mem after checkpoint: ${MEM_CKPT} MiB  (should drop toward 0 if it worked)"

echo "--- restore (host->GPU) ---"
$CC --action restore --pid "$PID" 2>&1; echo "  exit=$?  state: $($CC --get-state --pid "$PID" 2>&1)"
echo "--- unlock (resume CUDA) ---"
$CC --action unlock --pid "$PID" 2>&1; echo "  exit=$?  state: $($CC --get-state --pid "$PID" 2>&1)"

touch /tmp/cc_go
wait $PYPID; RC=$?
# Real pass requires BOTH: checksum survived AND memory actually left the GPU.
DROPPED=$([ "${MEM_CKPT:-999}" -lt "$((MEM_BEFORE/2))" ] && echo yes || echo no)
echo "=== PART A: checksum_ok=$([ $RC -eq 0 ] && echo yes || echo no)  gpu_mem_evicted=$DROPPED ==="
if [ $RC -eq 0 ] && [ "$DROPPED" = yes ]; then
    echo ">>> PASS — cuda-checkpoint genuinely round-tripped GPU state in this container"
else
    echo ">>> INCONCLUSIVE/FAIL — see states above"
fi
