#!/usr/bin/env bash
# Rigorous Rung 2: load vLLM once, then run N evict→restore cycles, timing each
# restore and verifying output every cycle. Reports p50/p90/p99 of restore time.
set +u
CC=/usr/local/bin/cuda-checkpoint
PY=/root/csvenv/bin/python
N=${1:-10}

rm -f /tmp/r2_pid /tmp/r2_ready /tmp/r2_tick /tmp/r2_ack /tmp/r2_base /tmp/r2_sorted
VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY /root/_rung2_probe_multi.py \
    2>&1 | grep -vE "Processed prompts|safetensors checkpoint|it/s\]" &

for _ in $(seq 1 600); do [ -f /tmp/r2_ready ] && break; sleep 1; done
if [ ! -f /tmp/r2_ready ]; then echo "FAIL: vLLM never ready"; exit 1; fi
PID=$(cat /tmp/r2_pid); BASE=$(cat /tmp/r2_base)
echo "ready pid=$PID  baseline=\"$BASE\""

: > /tmp/r2_sorted
ok=0
for i in $(seq 1 "$N"); do
    $CC --action lock --pid "$PID" --timeout 30000 >/dev/null 2>&1
    $CC --action checkpoint --pid "$PID" >/dev/null 2>&1
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    t=$(date +%s%N)
    $CC --action restore --pid "$PID" >/dev/null 2>&1
    $CC --action unlock --pid "$PID" >/dev/null 2>&1
    r=$(awk "BEGIN{printf \"%.3f\", ($(date +%s%N)-$t)/1e9}")
    echo "$r" >> /tmp/r2_sorted

    echo "$i" > /tmp/r2_tick.tmp && mv /tmp/r2_tick.tmp /tmp/r2_tick
    for _ in $(seq 1 100); do [ -f /tmp/r2_ack ] && break; sleep 0.1; done
    out=$(cut -d'|' -f2- /tmp/r2_ack 2>/dev/null); rm -f /tmp/r2_ack
    m=NO; [ "$out" = "$BASE" ] && { m=yes; ok=$((ok+1)); }
    echo "cycle $i: evict→${mem}MiB  restore=${r}s  match=$m"
done
echo "STOP" > /tmp/r2_tick
wait 2>/dev/null

echo ""
echo "=== Rung 2 restore time (N=$N) ==="
echo "correct outputs: $ok/$N"
$PY -c "
xs=sorted(float(x) for x in open('/tmp/r2_sorted'))
def pct(p):
    k=(len(xs)-1)*p/100; f=int(k); c=min(f+1,len(xs)-1); return xs[f]+(xs[c]-xs[f])*(k-f)
print(f'restore  p50={pct(50):.3f}s  p90={pct(90):.3f}s  p99={pct(99):.3f}s  min={xs[0]:.3f}s  max={xs[-1]:.3f}s')
import json; json.dump({'restore_s':xs,'n':len(xs)}, open('/tmp/r2_multi.json','w'))
"
