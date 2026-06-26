#!/usr/bin/env bash
# Step 0 feasibility probe for Rung 2 (GPU snapshot/restore).
#
# Run this ON a rented GPU box, as root. It answers the gating question: does
# cuda-checkpoint + CRIU work in THIS environment, or do we need a
# privileged/bare-metal host?
#
# Two parts, reported independently:
#   A. cuda-checkpoint  — can we round-trip GPU memory (the GPU half)?
#   B. CRIU             — can we checkpoint/restore a process (the process half)?
#
# NOTE: no `set -e` — we WANT to run past failures and report them; a failure
# here is a result, not a crash.
set -uo pipefail

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "=== Step 0: Rung 2 feasibility probe ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1
echo "x64: $(uname -m)   (cuda-checkpoint is x86_64-only)"

# ---------------------------------------------------------------------------
say "--- installing criu + cuda-checkpoint (if needed) ---"
if ! command -v criu >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq criu >/dev/null 2>&1 \
        && echo "criu: installed" || echo "criu: apt install FAILED"
else
    echo "criu: already present ($(criu --version 2>&1 | head -1))"
fi

CC=""
if command -v cuda-checkpoint >/dev/null 2>&1; then
    CC=$(command -v cuda-checkpoint)
elif [ -x /usr/local/bin/cuda-checkpoint ]; then
    CC=/usr/local/bin/cuda-checkpoint
else
    if curl -fsSL -o /usr/local/bin/cuda-checkpoint \
        https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint 2>/dev/null
    then
        chmod +x /usr/local/bin/cuda-checkpoint
        CC=/usr/local/bin/cuda-checkpoint
        echo "cuda-checkpoint: downloaded"
    else
        echo "cuda-checkpoint: download FAILED"
    fi
fi
[ -n "$CC" ] && echo "cuda-checkpoint: $CC" && "$CC" --help 2>&1 | head -20

CC_OK=fail
CRIU_OK=fail

# ---------------------------------------------------------------------------
say "=== Part A: cuda-checkpoint GPU memory round-trip ==="
if [ -z "$CC" ]; then
    echo "SKIP — cuda-checkpoint not available"
else
    cat >/tmp/cc_probe.py <<'PY'
import torch, time, os, sys
t = torch.arange(4096, dtype=torch.float32, device="cuda") * 3.0
torch.cuda.synchronize()
before = float(t.sum().item())
open("/tmp/cc_pid", "w").write(str(os.getpid()))
# hold the GPU allocation until told to verify
while not os.path.exists("/tmp/cc_verify"):
    time.sleep(0.5)
torch.cuda.synchronize()
after = float(t.sum().item())
ok = abs(after - before) < 1e-3
print(f"[probe] checksum before={before} after={after} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
    rm -f /tmp/cc_pid /tmp/cc_verify
    python3 /tmp/cc_probe.py & PYPID=$!
    for _ in $(seq 1 60); do [ -f /tmp/cc_pid ] && break; sleep 1; done
    PID=$(cat /tmp/cc_pid 2>/dev/null || true)

    if [ -z "$PID" ]; then
        echo "FAIL — probe process never allocated GPU memory (torch/cuda issue)"
    else
        echo "probe PID=$PID  state: $("$CC" --get-state --pid "$PID" 2>&1)"
        MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        # Required sequence: lock -> checkpoint -> restore -> unlock (NOT a bare
        # checkpoint: that errors with "cannot be performed in the present state").
        "$CC" --action lock --pid "$PID" --timeout 10000 >/dev/null 2>&1
        "$CC" --action checkpoint --pid "$PID" 2>&1
        echo "  state after checkpoint: $("$CC" --get-state --pid "$PID" 2>&1)"
        MEM_CKPT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "  gpu mem: ${MEM_BEFORE} -> ${MEM_CKPT} MiB (should drop toward 0)"
        "$CC" --action restore --pid "$PID" >/dev/null 2>&1
        "$CC" --action unlock --pid "$PID" >/dev/null 2>&1
    fi
    touch /tmp/cc_verify
    wait "${PYPID:-0}" 2>/dev/null; VERIFY=$?
    # Real pass requires BOTH: data survived AND memory actually left the GPU
    # (checksum alone gives a false positive when checkpoint silently no-ops).
    if [ "${VERIFY:-1}" -eq 0 ] && [ "${MEM_CKPT:-999}" -lt "$(( ${MEM_BEFORE:-0} / 2 ))" ]; then
        CC_OK=ok
    fi
    echo "Part A result: cuda-checkpoint = $CC_OK"
fi

# ---------------------------------------------------------------------------
say "=== Part B: CRIU process checkpoint/restore ==="
# Capabilities are the definitive gate: without CAP_SYS_ADMIN +
# CAP_CHECKPOINT_RESTORE, CRIU cannot dump regardless of the binary.
echo "--- container capabilities (the real gate) ---"
CAPS=$(capsh --decode=$(grep CapEff /proc/self/status | awk '{print $2}') 2>/dev/null)
for c in cap_sys_admin cap_checkpoint_restore cap_sys_ptrace; do
    echo "$CAPS" | grep -qi "$c" && echo "  $c: YES" || echo "  $c: NO (CRIU blocked)"
done
echo "--- criu check (kernel feature support) ---"
criu check 2>&1 | tail -8 || true

if command -v criu >/dev/null 2>&1; then
    setsid bash -c 'i=0; while true; do i=$((i+1)); sleep 1; done' \
        </dev/null >/tmp/criu_proc.log 2>&1 & SPID=$!
    sleep 1
    rm -rf /tmp/criu_img && mkdir -p /tmp/criu_img
    echo "--- criu dump PID=$SPID ---"
    criu dump -t "$SPID" -D /tmp/criu_img --shell-job 2>&1 | tail -8
    if ls /tmp/criu_img/*.img >/dev/null 2>&1; then
        echo "  dump OK (images written). Attempting restore..."
        criu restore -D /tmp/criu_img --shell-job -d 2>&1 | tail -8
        if [ $? -eq 0 ]; then CRIU_OK=ok; echo "  restore OK"; fi
    else
        echo "  dump produced NO images -> FAILED (almost certainly missing"
        echo "  CAP_SYS_ADMIN/CHECKPOINT_RESTORE in this container)"
    fi
    kill "$SPID" 2>/dev/null
    pkill -f 'i=\$((i+1))' 2>/dev/null
    echo "Part B result: criu = $CRIU_OK"
fi

# ---------------------------------------------------------------------------
say "================== VERDICT =================="
echo "cuda-checkpoint (GPU half):  $CC_OK"
echo "CRIU            (process):   $CRIU_OK"
echo
if [ "$CC_OK" = ok ] && [ "$CRIU_OK" = ok ]; then
    echo ">> BOTH WORK. Cheap containers (this host) are enough for full"
    echo "   cuda-checkpoint + CRIU snapshot/restore. Proceed to Step 1 here."
elif [ "$CC_OK" = ok ] && [ "$CRIU_OK" != ok ]; then
    echo ">> GPU toggle works, CRIU blocked. Two paths:"
    echo "   (a) APP-LEVEL fallback (serialize device mem in-process, no CRIU) —"
    echo "       viable on THIS cheap container."
    echo "   (b) bare-metal/privileged host for full CRIU process checkpoint."
elif [ "$CC_OK" != ok ] && [ "$CRIU_OK" = ok ]; then
    echo ">> Unusual: CRIU ok but GPU toggle failed. Check driver/cuda-checkpoint"
    echo "   version and re-run; GPU half is required."
else
    echo ">> NEITHER works here. Rung 2 needs a bare-metal or --privileged host"
    echo "   (Lambda bare-metal / Crusoe / Latitude.sh). RunPod-style unprivileged"
    echo "   containers cannot do it."
fi
echo "============================================="
