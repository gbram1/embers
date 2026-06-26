#!/usr/bin/env bash
# Drives _async_stream_probe.py: start the AsyncLLMEngine, let it stream once,
# then cuda-checkpoint park+unpark it, then let it stream again. Reports whether
# true streaming (incremental tokens) survives a park/unpark cycle. Run ON a pod.
set -u
PY=/root/csvenv/bin/python
CC=/usr/local/bin/cuda-checkpoint

cd ~/embers
pkill -9 -f _async_stream_probe 2>/dev/null; sleep 1
rm -f /tmp/as_pid /tmp/as_ready /tmp/as_go
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. $PY scripts/_async_stream_probe.py &
PROBE=$!

for _ in $(seq 1 400); do [ -f /tmp/as_ready ] && break; sleep 1; done
[ -f /tmp/as_ready ] || { echo "FAIL: engine never ready"; kill $PROBE; exit 1; }
PID=$(cat /tmp/as_pid)
# AsyncLLMEngine holds the GPU in a worker process — checkpoint the pid that
# actually owns GPU memory, not necessarily the main python pid.
GPUPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "=== engine main pid=$PID  GPU-holding pid=$GPUPID ==="
PID=${GPUPID:-$PID}
echo "gpu before park: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== PARK (lock + checkpoint) ==="
$CC --action lock --pid "$PID" --timeout 30000; echo "  lock rc=$?"
$CC --action checkpoint --pid "$PID"; echo "  checkpoint rc=$?  state=$($CC --get-state --pid $PID 2>&1)"
echo "gpu after park: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)  (want ~0)"

echo "=== UNPARK (restore + unlock) ==="
$CC --action restore --pid "$PID"; echo "  restore rc=$?"
$CC --action unlock --pid "$PID"; echo "  unlock rc=$?  state=$($CC --get-state --pid $PID 2>&1)"
echo "gpu after unpark: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

touch /tmp/as_go            # let the probe stream again
wait $PROBE
echo "DRIVER_DONE"
