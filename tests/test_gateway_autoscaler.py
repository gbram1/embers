"""Integration: gateway routing THROUGH the autoscaler — cold-start on request,
in-flight tracking, and HTTP error mapping. Phases 2↔4 wired for real traffic."""

from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit
from tests.test_autoscaler import FakeClock


def build():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    auto = Autoscaler(sched, router, launch, clock=clock)
    gw = TestClient(create_gateway_app(router, autoscaler=auto))
    return auto, clock, gw


def test_request_cold_starts_through_gateway():
    auto, _, gw = build()
    auto.register_model("m", 6000)
    assert auto.state()["m"] == 0
    r = gw.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert r.status_code == 200
    assert auto.state()["m"] == 1          # gateway request triggered the cold start
    assert auto.cold_starts == 1


def test_unknown_model_404_via_autoscaler():
    auto, _, gw = build()
    assert gw.post("/v1/completions",
                   json={"model": "ghost", "prompt": "x"}).status_code == 404


def test_no_capacity_503_via_autoscaler():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 1000)])   # too small for the model
    router = Router()
    auto = Autoscaler(sched, router, lambda m, g: None, clock=clock)
    auto.register_model("big", 6000)
    gw = TestClient(create_gateway_app(router, autoscaler=auto))
    assert gw.post("/v1/completions",
                   json={"model": "big", "prompt": "x"}).status_code == 503


def test_inflight_released_after_request():
    auto, clock, gw = build()
    auto.register_model("m", 6000, idle_ttl=300)
    gw.post("/v1/completions", json={"model": "m", "prompt": "x"})
    assert auto.inflight("m") == 0          # end_request ran in finally
    clock.advance(301)
    auto.tick()
    assert auto.state()["m"] == 0           # idle + no in-flight → parked


def test_chat_also_routes_through_autoscaler():
    auto, _, gw = build()
    auto.register_model("m", 6000)
    r = gw.post("/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "yo"}]})
    assert r.status_code == 200
    assert auto.cold_starts == 1
