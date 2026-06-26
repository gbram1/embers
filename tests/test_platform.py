"""Tests for the runnable platform — config, GPU detection, assembly, the served
app (mock), and the control loop. All GPU-free."""

import threading
import time

import yaml
from fastapi.testclient import TestClient

from embers.platform import (
    Platform,
    PlatformConfig,
    detect_gpus,
    load_config,
)
from embers.scheduler import GPU


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def write_cfg(tmp_path, extra=None):
    cfg = {
        "port": 9000,
        "gpus": [{"id": "g0", "vram_mb": 24000}],
        "models": [{"name": "m", "vram_mb": 6000, "idle_ttl": 300}],
    }
    if extra:
        cfg.update(extra)
    p = tmp_path / "platform.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


# --- config & GPU detection -----------------------------------------------

def test_load_config_explicit_gpus(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert cfg.port == 9000
    assert cfg.gpus[0].total_mb == 24000
    assert cfg.models[0].name == "m"
    assert cfg.models[0].idle_ttl == 300


def test_load_config_auto_gpus(tmp_path):
    cfg = load_config(write_cfg(tmp_path, {"gpus": "auto"}))
    assert cfg.gpus == "auto"


def test_detect_gpus_parses_nvidia_smi():
    fake = "0, 24000\n1, 49152\n"
    gpus = detect_gpus(query=lambda: fake)
    assert [(g.id, g.total_mb) for g in gpus] == [("gpu0", 24000), ("gpu1", 49152)]


# --- assembly --------------------------------------------------------------

def make_platform(**over):
    cfg = PlatformConfig(
        models=load_config_models(),
        gpus=[GPU("g0", 24000), GPU("g1", 24000)],
        port=9000, **over)
    return Platform(cfg, mock=True, clock=FakeClock())


def load_config_models():
    from embers.platform import ModelConfig
    return [ModelConfig("m", 6000, idle_ttl=300)]


def test_platform_registers_models_and_gpus():
    p = make_platform()
    assert "m" in p.autoscaler.models
    assert len(p.scheduler.gpus) == 2
    assert p.autoscaler.state()["m"] == 0      # starts cold


def test_platform_app_serves_and_cold_starts():
    p = make_platform()
    c = TestClient(p.app)
    assert c.get("/v1/models").json()["data"][0]["id"] == "m"
    r = c.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert r.status_code == 200
    assert p.autoscaler.cold_starts == 1       # request drove the cold start


def test_platform_stats_and_metrics_endpoints():
    p = make_platform()
    c = TestClient(p.app)
    c.post("/v1/completions", json={"model": "m", "prompt": "x"})
    assert "gpus" in c.get("/stats").json()
    assert "embers_requests_total" in c.get("/metrics").text


def test_platform_scale_to_zero_via_tick():
    clock = FakeClock()
    p = Platform(PlatformConfig(models=load_config_models(),
                                gpus=[GPU("g0", 24000)]), mock=True, clock=clock)
    c = TestClient(p.app)
    c.post("/v1/completions", json={"model": "m", "prompt": "x"})
    assert p.autoscaler.state()["m"] == 1
    clock.advance(301)
    p.tick()
    assert p.autoscaler.state()["m"] == 0      # idle → scaled to zero


def test_no_gpus_raises():
    import pytest
    with pytest.raises(RuntimeError):
        Platform(PlatformConfig(models=load_config_models(), gpus=[]), mock=True)


# --- control loop ----------------------------------------------------------

def test_control_loop_ticks_then_stops():
    p = make_platform(tick_interval=0.02)
    ticks = {"n": 0}
    p.autoscaler.tick = lambda: ticks.__setitem__("n", ticks["n"] + 1)
    p.start_control_loop()
    time.sleep(0.1)
    p.shutdown()                                # sets stop + joins
    assert ticks["n"] >= 2                      # loop ran several times
    assert not p._loop_thread.is_alive()        # and stopped cleanly


def test_api_key_auth_enforced(tmp_path):
    p = Platform(PlatformConfig(models=load_config_models(),
                                gpus=[GPU("g0", 24000)], api_keys=["secret"]),
                 mock=True)
    c = TestClient(p.app)
    assert c.post("/v1/completions",
                  json={"model": "m", "prompt": "x"}).status_code == 401
    assert c.post("/v1/completions", headers={"Authorization": "Bearer secret"},
                  json={"model": "m", "prompt": "x"}).status_code == 200


def test_resolve_gpus_mock_auto_needs_no_nvidia_smi():
    # mock + gpus:auto must NOT call nvidia-smi (mock runs on no-GPU machines)
    from embers.platform import _resolve_gpus
    gpus = _resolve_gpus("auto", mock=True)
    assert len(gpus) == 1 and gpus[0].total_mb > 0


def test_resolve_gpus_explicit_passthrough():
    from embers.platform import _resolve_gpus
    from embers.scheduler import GPU
    gpus = _resolve_gpus([GPU("gpu0", 24000)], mock=False)
    assert [g.id for g in gpus] == ["gpu0"]


def test_resolve_gpus_real_auto_missing_nvidia_smi_clear_error():
    # real mode + auto + nvidia-smi absent → a CLEAR error, not a raw traceback
    import embers.platform as P
    orig = P.detect_gpus
    P.detect_gpus = lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
    try:
        import pytest
        with pytest.raises(RuntimeError, match="nvidia-smi|--mock"):
            P._resolve_gpus("auto", mock=False)
    finally:
        P.detect_gpus = orig


def test_mock_platform_with_auto_gpus_builds():
    # the embers-init default (gpus: auto) must work in --mock on a no-GPU box
    from embers.platform import Platform, PlatformConfig, ModelConfig
    p = Platform(PlatformConfig(models=[ModelConfig("m", 6000)], gpus="auto"), mock=True)
    assert "m" in p.controlplane.served_models()
