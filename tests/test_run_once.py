"""Tests for bench.run_once — config loading and the mock load path (no GPU)."""

import json
import subprocess
import sys

import pytest
import yaml

from bench import run_once


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"model": "m", "max_model_len": 2048}))
    cfg = run_once.load_config(str(p))
    assert cfg["model"] == "m"
    assert cfg["max_model_len"] == 2048


def test_mock_load_shape():
    t = run_once.mock_load({"model": "m"})
    assert set(t) == {"construct", "first_token", "end_to_end"}
    assert t["end_to_end"] == pytest.approx(t["construct"] + t["first_token"])
    assert all(v > 0 for v in t.values())


def test_mock_load_emits_weight_log(capsys):
    run_once.mock_load({"model": "m"})
    err = capsys.readouterr().err
    assert "Loading model weights took" in err  # harness parses this from stderr


def test_main_mock_in_process(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"model": "m"}))
    monkeypatch.setattr("sys.argv",
                        ["run_once", "--config", str(cfg), "--mock"])
    run_once.main()
    line = [x for x in capsys.readouterr().out.splitlines() if x.strip()][-1]
    assert "end_to_end" in json.loads(line)


def test_run_once_mock_end_to_end(tmp_path):
    """Integration: spawn the real module with --mock, expect one JSON line."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"model": "m"}))
    proc = subprocess.run(
        [sys.executable, "-m", "bench.run_once", "--config", str(cfg), "--mock"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    line = [x for x in proc.stdout.splitlines() if x.strip()][-1]
    t = json.loads(line)
    assert "end_to_end" in t
