#!/usr/bin/env bash
# Run the real cold-start baseline on a rented CUDA GPU, from your Mac, in one
# command. It: (1) rsyncs your code up, (2) sets up a venv + installs vLLM on
# the box (first time only), (3) runs the harness for real, (4) pulls results
# back to bench/results/ here.
#
# Usage:
#   ./scripts/remote.sh -H <ip> -P <port> [options]
#
# Required:
#   -H <ip>        host/IP of the GPU box
#   -P <port>      SSH port (RunPod/Vast give you a non-22 port)
# Options:
#   -i <keyfile>   SSH private key (default: ssh-agent / ~/.ssh defaults)
#   -u <user>      SSH user (default: root)
#   -c <config>    bench config (default: bench/configs/naive_local.yaml)
#   -n <runs>      runs (default: 10)
#   -d <dir>       remote dir (default: ~/embers)
#
# A gated model (Llama) needs a HF token — export it before running:
#   export HF_TOKEN=hf_xxx ; ./scripts/remote.sh -H ... -P ...
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="" PORT="" KEY="" USER="root"
CONFIG="bench/configs/naive_local.yaml" RUNS=10 RDIR="~/embers"
RVENV="/root/csvenv"   # clean vLLM venv on the box (kept outside $RDIR)

while getopts "H:P:i:u:c:n:d:" opt; do
    case "$opt" in
        H) HOST="$OPTARG" ;;
        P) PORT="$OPTARG" ;;
        i) KEY="$OPTARG" ;;
        u) USER="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        n) RUNS="$OPTARG" ;;
        d) RDIR="$OPTARG" ;;
        *) exit 2 ;;
    esac
done

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "error: -H <ip> and -P <port> are required. See: head -30 $0" >&2
    exit 2
fi

# Keepalives so a half-open connection (seen post-run) is detected and the
# client exits instead of hanging: 4 missed 15s probes -> drop after ~60s.
KEEPALIVE=(-o ServerAliveInterval=15 -o ServerAliveCountMax=4)
SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new "${KEEPALIVE[@]}")
[ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")
# scp uses -P (capital) for port, not ssh's lowercase -p.
SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new "${KEEPALIVE[@]}")
[ -n "$KEY" ] && SCP_OPTS+=(-i "$KEY")
TARGET="$USER@$HOST"
NAME="$(basename "${CONFIG%.yaml}")"

echo "==> 1/4  rsync code → $TARGET:$RDIR"
rsync -az --delete \
    --exclude .venv --exclude .git --exclude target --exclude bench/results \
    -e "ssh ${SSH_OPTS[*]}" ./ "$TARGET:$RDIR/"

echo "==> 2/4  ensure vLLM venv on the box (first run takes a few minutes)"
# Install into a CLEAN venv ($RVENV), never the system python: layering pip
# installs over the template's python produces ABI conflicts (flashinfer/tvm
# built for a different torch). The pinned versions match a CUDA 12.4 driver —
# on a newer-driver box, bump them to a modern coherent vLLM release.
ssh "${SSH_OPTS[@]}" "$TARGET" "bash -lc '
    set -e
    cd $RDIR
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
        echo \"no GPU visible on this box\" >&2; exit 1; }
    if ! $RVENV/bin/python -c \"import vllm\" 2>/dev/null; then
        python3 -m venv $RVENV
        $RVENV/bin/pip install -q -U pip
        $RVENV/bin/pip install -q vllm==0.8.5 transformers==4.51.3 pyyaml
    fi
'"

echo "==> 3/4  run baseline ($RUNS cold runs, config=$NAME)"
# Forward HF_TOKEN only if non-empty. Exporting an EMPTY token makes huggingface
# build an "Authorization: Bearer " header that httpx rejects, so unset it for
# anonymous (ungated) access. Run as root so drop_caches works.
if [ -n "${HF_TOKEN:-}" ]; then
    HF_LINE="export HF_TOKEN='$HF_TOKEN' HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'"
else
    HF_LINE="unset HF_TOKEN HUGGING_FACE_HUB_TOKEN"
fi
ssh "${SSH_OPTS[@]}" "$TARGET" "bash -lc '
    cd $RDIR
    $HF_LINE
    mkdir -p bench/results
    $RVENV/bin/python -m bench.harness --config $CONFIG -n $RUNS \
        --out bench/results/$NAME.json
'"

echo "==> 4/4  pull results → bench/results/$NAME.json"
mkdir -p bench/results
scp "${SCP_OPTS[@]}" "$TARGET:$RDIR/bench/results/$NAME.json" "bench/results/$NAME.json"

echo
echo "Done. Local copy: bench/results/$NAME.json"
echo "Remember to STOP the GPU pod in your provider's dashboard — you're billed hourly."
