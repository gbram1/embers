#!/usr/bin/env bash
# One-time local dev setup (macOS-friendly, no GPU needed).
# Creates a venv and installs the Python deps used by the harness + tests.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "==> creating .venv"
    "$PY" -m venv .venv
fi

echo "==> installing python deps (pyyaml + pytest)"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q pyyaml pytest

echo
echo "Done. Run the local tests with:  ./scripts/test.sh"
echo "(vLLM is NOT installed — that's only for the real baseline on a CUDA box.)"
