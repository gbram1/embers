#!/usr/bin/env bash
# Run everything that can be tested locally WITHOUT a GPU:
#   1. Python unit + integration tests (with coverage)
#   2. The full benchmark loop in --mock mode (no GPU/vLLM)
#
# The real cold-start baseline still needs a rented CUDA GPU — see bench/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "No .venv — run ./scripts/setup.sh first." >&2
    exit 1
fi

echo "==> 1/2  python unit + integration tests (pytest + coverage)"
.venv/bin/python -m pytest tests \
    --cov=embers --cov=bench --cov-report=term-missing -q

echo
echo "==> 2/2  benchmark loop, mock mode (no GPU)"
.venv/bin/python -m bench.harness \
    --config bench/configs/naive_local.yaml -n 5 --mock

echo
echo "All local checks passed. Real baseline: ./scripts/setup.sh on a CUDA box,"
echo "then 'pip install -e .[bench]' and drop --mock."
