"""Tests for bench.protocol — cache control & GPU guards, with subprocess mocked
so they run anywhere (no root, no nvidia-smi)."""

import subprocess

import pytest

from bench import protocol


def test_clear_compile_cache_removes_dir(tmp_path, monkeypatch):
    cache = tmp_path / ".cache" / "vllm" / "torch_compile_cache"
    cache.mkdir(parents=True)
    (cache / "graph.bin").write_text("x")
    monkeypatch.setattr("os.path.expanduser",
                        lambda p: p.replace("~", str(tmp_path)))
    protocol.clear_vllm_compile_cache()
    assert not cache.exists()


def test_clear_compile_cache_missing_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser",
                        lambda p: p.replace("~", str(tmp_path)))
    protocol.clear_vllm_compile_cache()  # must not raise when absent


def test_drop_caches_raises_on_failure(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "denied"))
    with pytest.raises(protocol.CacheDropError):
        protocol.drop_caches()


def test_drop_caches_ok(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    protocol.drop_caches()  # no raise


def test_assert_no_resident_model_pass(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "12\n", ""))
    protocol.assert_no_resident_model()  # 12 MiB < threshold


def test_assert_no_resident_model_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "8000\n", ""))
    with pytest.raises(RuntimeError):
        protocol.assert_no_resident_model()
