"""Allocate a known GPU buffer, hold it, then verify it survived a
cuda-checkpoint checkpoint+restore driven externally. Writes its PID to
/tmp/cc_pid; waits for /tmp/cc_go; then re-checks the checksum."""
import os
import sys
import time

import torch

t = torch.arange(4096, dtype=torch.float32, device="cuda") * 3.0
torch.cuda.synchronize()
before = float(t.sum().item())
with open("/tmp/cc_pid", "w") as f:
    f.write(str(os.getpid()))
print(f"[probe] pid={os.getpid()} gpu-mem allocated, checksum={before}", flush=True)

while not os.path.exists("/tmp/cc_go"):
    time.sleep(0.3)

torch.cuda.synchronize()
after = float(t.sum().item())
ok = abs(after - before) < 1e-3
print(f"[probe] after restore checksum={after} -> {'OK' if ok else 'MISMATCH'}", flush=True)
sys.exit(0 if ok else 1)
