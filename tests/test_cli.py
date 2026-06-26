"""Tests for the `embers` CLI arg parsing and dispatch (no server bound)."""

import pytest

from embers import cli, gateway, server


def test_serve_mock_passes_no_engine_kwargs(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "serve",
                        lambda model, **kw: captured.update(model=model, **kw))
    monkeypatch.setattr("sys.argv", ["embers", "serve", "m", "--mock"])
    cli.main()
    assert captured["model"] == "m"
    assert captured["mock"] is True
    # mock path must NOT pass vLLM engine kwargs
    assert "max_model_len" not in captured
    assert "gpu_memory_utilization" not in captured


def test_serve_real_passes_engine_kwargs(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "serve",
                        lambda model, **kw: captured.update(model=model, **kw))
    monkeypatch.setattr("sys.argv",
                        ["embers", "serve", "Qwen/Qwen2.5-3B",
                         "--max-model-len", "2048", "--gpu-memory-utilization", "0.8"])
    cli.main()
    assert captured["mock"] is False
    assert captured["max_model_len"] == 2048
    assert captured["gpu_memory_utilization"] == 0.8
    assert captured["dtype"] == "auto"


def test_no_eager_load_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "serve",
                        lambda model, **kw: captured.update(**kw))
    monkeypatch.setattr("sys.argv",
                        ["embers", "serve", "m", "--mock", "--no-eager-load"])
    cli.main()
    assert captured["eager_load"] is False


def test_missing_subcommand_errors(monkeypatch):
    monkeypatch.setattr("sys.argv", ["embers"])
    with pytest.raises(SystemExit):
        cli.main()


def test_serve_requires_model(monkeypatch):
    monkeypatch.setattr("sys.argv", ["embers", "serve"])
    with pytest.raises(SystemExit):
        cli.main()


def test_gateway_parses_backends(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "serve_gateway",
                        lambda backends, **kw: captured.update(b=backends, **kw))
    monkeypatch.setattr("sys.argv",
                        ["embers", "gateway",
                         "--backend", "a=http://h1:8000",
                         "--backend", "a=http://h2:8000",
                         "--backend", "b=http://h3:8000",
                         "--port", "9090", "--api-key", "k1"])
    cli.main()
    assert captured["b"] == [("a", "http://h1:8000"),
                             ("a", "http://h2:8000"),
                             ("b", "http://h3:8000")]
    assert captured["port"] == 9090
    assert captured["api_keys"] == {"k1"}


def test_gateway_rejects_malformed_backend(monkeypatch):
    monkeypatch.setattr(gateway, "serve_gateway", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv",
                        ["embers", "gateway", "--backend", "no-equals-sign"])
    with pytest.raises(SystemExit):
        cli.main()


def test_gateway_no_keys_means_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "serve_gateway",
                        lambda backends, **kw: captured.update(**kw))
    monkeypatch.setattr("sys.argv",
                        ["embers", "gateway", "--backend", "a=http://h:8000"])
    cli.main()
    assert captured["api_keys"] is None  # auth disabled when no keys given


def test_schedule_places_and_prints_plan(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv",
                        ["embers", "schedule",
                         "--gpu", "g0:24000", "--gpu", "g1:24000",
                         "--model", "alpha:6000", "--model", "beta:6000:2",
                         "--policy", "best-fit"])
    cli.main()
    out = capsys.readouterr().out
    assert "policy: best_fit" in out
    assert "alpha" in out and "beta" in out


def test_schedule_reports_unplaceable(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv",
                        ["embers", "schedule",
                         "--gpu", "g0:10000", "--model", "huge:20000"])
    cli.main()
    out = capsys.readouterr().out
    assert "UNPLACED" in out


def test_schedule_requires_a_gpu(monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["embers", "schedule", "--model", "m:1000"])
    with pytest.raises(SystemExit):
        cli.main()


def test_init_writes_starter_config(tmp_path, monkeypatch, capsys):
    out = tmp_path / "platform.yaml"
    monkeypatch.setattr("sys.argv", ["embers", "init", "-o", str(out)])
    cli.main()
    assert out.exists()
    import yaml
    cfg = yaml.safe_load(out.read_text())     # generated config is valid YAML
    assert "models" in cfg and "gpus" in cfg


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch):
    out = tmp_path / "platform.yaml"
    out.write_text("existing")
    monkeypatch.setattr("sys.argv", ["embers", "init", "-o", str(out)])
    with pytest.raises(SystemExit):
        cli.main()


def test_up_builds_and_serves_platform(tmp_path, monkeypatch):
    import yaml
    cfg = tmp_path / "platform.yaml"
    cfg.write_text(yaml.safe_dump({
        "gpus": [{"id": "g0", "vram_mb": 24000}],
        "models": [{"name": "m", "vram_mb": 6000}]}))
    served = {}
    # stub serve() so the test doesn't bind a socket
    monkeypatch.setattr("embers.platform.Platform.serve",
                        lambda self: served.update(models=list(self.autoscaler.models)))
    monkeypatch.setattr("sys.argv",
                        ["embers", "up", "-c", str(cfg), "--mock"])
    cli.main()
    assert served["models"] == ["m"]


def test_dashboard_fetches_and_renders(monkeypatch, capsys):
    import httpx

    class _R:
        def json(self):
            return {"embers": {"cold_loads": 1, "restores": 4,
                                  "invalidations": 0, "snapshot_hit_rate": 0.8}}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _R())
    monkeypatch.setattr("sys.argv",
                        ["embers", "dashboard", "--url", "http://x:8080"])
    cli.main()
    assert "80% hit-rate" in capsys.readouterr().out
