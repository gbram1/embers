"""Integration: run the whole harness in --mock mode (spawns real run_once
subprocesses, parses their output, computes percentiles, writes JSON)."""

import json
import subprocess
import sys

import yaml


def test_harness_mock_full_loop(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"model": "m"}))
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "-m", "bench.harness", "--config", str(cfg),
         "-n", "4", "--mock", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARM-CACHE-BIASED" in proc.stdout  # mock implies no cache drop
    assert "end_to_end" in proc.stdout

    data = json.loads(out.read_text())
    assert data["keep_compile_cache"] is False
    assert len(data["runs"]) == 4
    assert all("end_to_end" in r for r in data["runs"])


def test_harness_mock_warmup_and_keep_cache(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(
        {"model": "m", "keep_compile_cache": True, "warmup": 1}))
    proc = subprocess.run(
        [sys.executable, "-m", "bench.harness", "--config", str(cfg),
         "-n", "3", "--mock"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "warmup 1/1" in proc.stdout       # warmup run happened
    assert "compile-cache-persisted" in proc.stdout
    assert proc.stdout.count("run ") >= 3    # 3 measured runs reported
