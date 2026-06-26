#!/usr/bin/env bash
# Validate TRUE streaming on real hardware via `embers up`:
#  - real AsyncLLMEngine completion streaming (incremental SSE)
#  - chat streaming (exercises the get_tokenizer/apply_chat_template path)
#  - scale-to-zero with the GPU-holding-child park, then restore
# Run ON the pod.
set -u
PY=/root/csvenv/bin/python
GW=18080
J='-H Content-Type:application/json'
cd ~/embers
cat > /tmp/platform.yaml <<EOF
host: 127.0.0.1
port: $GW
tick_interval: 4
gpus: auto
models: [{name: Qwen/Qwen2.5-3B, vram_mb: 20000, min_replicas: 0, idle_ttl: 10}]
EOF
pkill -9 -f embers.cli 2>/dev/null; sleep 2
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. $PY -m embers.cli up -c /tmp/platform.yaml >/tmp/up.log 2>&1 &
for _ in $(seq 1 40); do curl -s -m5 http://127.0.0.1:$GW/v1/models >/dev/null && break; sleep 1; done

echo "=== cold start (completion) ==="
curl -s -m 200 $J -X POST http://127.0.0.1:$GW/v1/completions \
  -d '{"model":"Qwen/Qwen2.5-3B","prompt":"hi","max_tokens":4}' >/dev/null
echo "gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "=== STREAMING completion (true incremental SSE) ==="
curl -s -N -m 60 $J -X POST http://127.0.0.1:$GW/v1/completions \
  -d '{"model":"Qwen/Qwen2.5-3B","stream":true,"prompt":"Count: one two","max_tokens":12}' \
  | grep -c "^data:" | sed 's/^/  SSE chunks: /'

echo "=== STREAMING chat (tests chat-template path) ==="
CHAT=$(curl -s -N -m 60 $J -X POST http://127.0.0.1:$GW/v1/chat/completions \
  -d '{"model":"Qwen/Qwen2.5-3B","stream":true,"messages":[{"role":"user","content":"Name one color."}]}')
echo "  chat SSE chunks: $(echo "$CHAT" | grep -c '^data:')"
echo "  chat assistant text: $(echo "$CHAT" | PYTHONPATH=. $PY -c 'import sys; from embers.streaming import parse_sse_text; print("".join(parse_sse_text(sys.stdin.read().splitlines(), chat=True)))' 2>&1 | head -1)"

echo "=== idle → scale to zero (child-pid park), wait for GPU to free ==="
for _ in $(seq 1 20); do
  sleep 3
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  echo "  gpu=${MEM}MiB"
  [ "${MEM:-9999}" -lt 1000 ] && echo "  -> GPU freed (park ok)" && break
done

echo "=== request after scale-to-zero (restore) ==="
t=$(date +%s%N)
curl -s -m 200 $J -X POST http://127.0.0.1:$GW/v1/completions \
  -d '{"model":"Qwen/Qwen2.5-3B","prompt":"hi","max_tokens":4}' >/dev/null
echo "  restore took $(awk "BEGIN{printf \"%.1f\", ($(date +%s%N)-$t)/1e9}")s, gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "  final /stats: $(curl -s http://127.0.0.1:$GW/stats)"
echo "STREAM_DONE"
pkill -9 -f embers.cli 2>/dev/null
