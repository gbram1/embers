"""Integration: the gateway's /metrics and /stats endpoints reflect real traffic
and the platform snapshot."""

from fastapi.testclient import TestClient

from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.metrics import Registry, platform_snapshot
from embers.server import ModelUnit


def local(model: str):
    u = ModelUnit(model, mock=True)
    u.load()
    return LocalBackend(u)


def test_metrics_endpoint_counts_requests():
    r = Router()
    r.register(local("m"))
    reg = Registry()
    c = TestClient(create_gateway_app(r, registry=reg))
    for _ in range(3):
        c.post("/v1/completions", json={"model": "m", "prompt": "x"})
    text = c.get("/metrics").text
    assert 'embers_requests_total{endpoint="completions",model="m"} 3' in text
    assert "embers_request_latency_seconds_count" in text


def test_metrics_present_even_without_explicit_registry():
    r = Router()
    r.register(local("m"))
    c = TestClient(create_gateway_app(r))     # registry auto-created
    c.post("/v1/completions", json={"model": "m", "prompt": "x"})
    assert c.get("/metrics").status_code == 200


def test_stats_endpoint_returns_platform_snapshot():
    r = Router()
    r.register(local("m"))

    class FakeLoader:
        cold_loads, restores, invalidations = 1, 3, 0

    c = TestClient(create_gateway_app(
        r, snapshot_fn=lambda: platform_snapshot(loader=FakeLoader())))
    body = c.get("/stats").json()
    assert body["embers"]["snapshot_hit_rate"] == 0.75


def test_no_stats_endpoint_without_snapshot_fn():
    r = Router()
    r.register(local("m"))
    c = TestClient(create_gateway_app(r))
    assert c.get("/stats").status_code == 404
