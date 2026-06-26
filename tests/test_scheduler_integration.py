"""Integration: scheduler places models → builds a gateway Router → the gateway
routes real requests to the placed (mock) units. Phases 3→2→1 end to end."""

from fastapi.testclient import TestClient

from embers.gateway import LocalBackend, create_gateway_app
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit


def test_placed_models_are_routable_through_gateway():
    s = Scheduler([GPU("g0", 24000), GPU("g1", 24000)], policy="best-fit")
    s.place("alpha", 6000)
    s.place("beta", 6000, replicas=2)

    def backend_for(model, gpu_id):
        unit = ModelUnit(model, mock=True)
        unit.load()
        return LocalBackend(unit)

    router = s.to_router(backend_for)
    c = TestClient(create_gateway_app(router))

    assert sorted(m["id"] for m in c.get("/v1/models").json()["data"]) == \
        ["alpha", "beta"]

    for model in ("alpha", "beta"):
        r = c.post("/v1/completions", json={"model": model, "prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["model"] == model

    # beta has 2 replicas → gateway load-balances across them
    assert len(router.replicas("beta")) == 2


def test_overcommit_is_rejected_not_silently_dropped():
    s = Scheduler([GPU("g0", 10000)])
    s.place("a", 6000)
    # a second 6000 model doesn't fit 4000 free — must raise, not overcommit
    from embers.scheduler import NoCapacity
    import pytest
    with pytest.raises(NoCapacity):
        s.place("b", 6000)
    assert s.gpu_state() == [("g0", 6000, 10000)]  # unchanged
