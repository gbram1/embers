"""Integration: the full scale-to-zero loop with autoscaler + scheduler +
gateway. Request cold-starts a model, gateway serves it, idle scales it to zero
(GPU freed, gateway can't route), next request cold-starts again. Phases 4→3→2→1."""

from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit
from tests.test_autoscaler import FakeClock


def build():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000), GPU("g1", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    a = Autoscaler(sched, router, launch, clock=clock)
    gw = TestClient(create_gateway_app(router))
    return a, clock, sched, gw


def test_full_scale_to_zero_lifecycle():
    a, clock, sched, gw = build()
    a.register_model("m", 6000, idle_ttl=300)

    # 1. cold: gateway can't serve a model with no replicas yet
    assert gw.post("/v1/completions",
                   json={"model": "m", "prompt": "x"}).status_code in (404, 503)

    # 2. a request through the autoscaler cold-starts it
    a.handle_request("m")
    r = gw.post("/v1/completions", json={"model": "m", "prompt": "hello"})
    assert r.status_code == 200
    assert r.json()["model"] == "m"
    assert sched.total_free_mb() == 48000 - 6000      # GPU in use

    # 3. idle → control loop scales to zero, frees the GPU
    clock.advance(301)
    a.tick()
    assert sched.total_free_mb() == 48000             # GPU reclaimed
    assert gw.post("/v1/completions",
                   json={"model": "m", "prompt": "x"}).status_code in (404, 503)

    # 4. new traffic cold-starts it again
    a.handle_request("m")
    assert gw.post("/v1/completions",
                   json={"model": "m", "prompt": "again"}).status_code == 200
    assert a.cold_starts == 2


def test_two_models_share_the_cluster():
    a, clock, sched, gw = build()
    a.register_model("a", 16000)
    a.register_model("b", 16000)
    a.handle_request("a")
    a.handle_request("b")          # each lands on its own 24GB GPU
    assert sorted(m["id"] for m in gw.get("/v1/models").json()["data"]) == ["a", "b"]
    for model in ("a", "b"):
        assert gw.post("/v1/completions",
                       json={"model": model, "prompt": "x"}).status_code == 200
