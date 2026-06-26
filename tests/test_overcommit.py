"""Over-commit / demand eviction — pack MORE model footprint than fits on a GPU;
when a request needs room, evict the least-recently-used idle resident model
(park it, free its GPU), then serve. The density play."""

import pytest

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import Router
from embers.scheduler import GPU, NoCapacity, Scheduler


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ReadyBackend:
    def __init__(self, model):
        self.name = model
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def chat(self, *a, **k):
        return "ok"

    def complete(self, *a, **k):
        return "ok"


def make(gpu_mb=20000):
    clock = Clock()
    sched = Scheduler([GPU("g0", gpu_mb)])
    router = Router()
    parked = []

    def launch(model, gpu_ids):
        return ReadyBackend(model)

    a = Autoscaler(sched, router, launch, clock=clock,
                   on_deactivate=lambda m, g: parked.append(m))
    return a, clock, sched, parked


# --- autoscaler demand eviction -------------------------------------------

def test_request_evicts_idle_model_to_make_room():
    # GPU holds ~2 models (8000 each in 20000). Register 3, request all → the
    # 3rd evicts the LRU idle one instead of 503-ing.
    a, clock, sched, parked = make(gpu_mb=20000)
    for m in ("a", "b", "c"):
        a.register_model(m, 8000, idle_ttl=300)

    a.handle_request("a"); clock.t = 1
    a.handle_request("b"); clock.t = 2          # a, b resident (16000/20000)
    assert a.state() == {"a": 1, "b": 1, "c": 0}

    a.handle_request("c")                        # no room → evict LRU idle (a)
    assert a.state()["c"] == 1                   # c got served
    assert a.evictions == 1
    assert "a" in parked                         # a was the LRU idle → evicted
    assert a.state()["a"] == 0                   # a freed
    assert len([g for g in sched.gpus if g.runs("c")]) == 1


def test_does_not_evict_a_busy_model():
    a, clock, sched, parked = make(gpu_mb=20000)
    for m in ("a", "b", "c"):
        a.register_model(m, 8000, idle_ttl=300)
    a.begin_request("a"); clock.t = 1            # a has an IN-FLIGHT request
    a.handle_request("b"); clock.t = 2
    # request c: a is older but in-flight → must evict b, not a
    a.handle_request("c")
    assert a.state()["a"] == 1                    # a kept (in-flight)
    assert "b" in parked and a.state()["b"] == 0  # b (idle) evicted instead


def test_no_eviction_when_nothing_idle_raises():
    a, clock, sched, _ = make(gpu_mb=12000)       # only ~1 model fits
    a.register_model("a", 8000)
    a.register_model("b", 8000)
    a.begin_request("a")                          # a resident + in-flight
    with pytest.raises(Exception):                # b can't fit, a is busy → 503
        a.handle_request("b")


# --- control-plane over-commit assignment ---------------------------------

def node(node_id="n0", gpu_mb=20000):
    sched = Scheduler([GPU(f"{node_id}-g0", gpu_mb)])
    a = Autoscaler(sched, Router(), lambda m, g: ReadyBackend(m), clock=Clock())
    return NodeAgent(node_id, a)


def test_overcommit_assigns_more_than_fits():
    cp = ControlPlane([node(gpu_mb=20000)], overcommit=True)
    cp.assign("a", 8000)
    cp.assign("b", 8000)
    cp.assign("c", 8000)                          # 24000 > 20000 — allowed
    assert {cp._owner[m] for m in ("a", "b", "c")} == {"n0"}


def test_without_overcommit_refuses_more_than_fits():
    cp = ControlPlane([node(gpu_mb=20000)], overcommit=False)
    cp.assign("a", 8000)
    cp.assign("b", 8000)
    with pytest.raises(NoCapacity):
        cp.assign("c", 8000)                      # would exceed the GPU → refused


def test_overcommit_still_refuses_model_too_big_for_any_gpu():
    cp = ControlPlane([node(gpu_mb=20000)], overcommit=True)
    with pytest.raises(NoCapacity):
        cp.assign("huge", 30000)                  # can't EVER fit one unit


def test_eviction_discards_not_parks():
    # demand eviction must use on_evict (STOP, free host RAM) — NOT on_deactivate
    # (park) — because an over-committed model's snapshot can't restore after
    # another model reuses its GPU memory.
    clock = Clock()
    sched = Scheduler([GPU("g0", 20000)])
    parked, discarded = [], []
    a = Autoscaler(sched, Router(), lambda m, g: ReadyBackend(m), clock=clock,
                   on_deactivate=lambda m, g: parked.append(m),
                   on_evict=lambda m, g: discarded.append(m))
    for m in ("a", "b", "c"):
        a.register_model(m, 8000, idle_ttl=300)
    a.handle_request("a"); clock.t = 1
    a.handle_request("b"); clock.t = 2
    a.handle_request("c")                          # evicts LRU idle (a)
    assert discarded == ["a"]                      # a was DISCARDED (stopped)...
    assert "a" not in parked                       # ...not parked
    assert a.evictions == 1


def test_idle_scale_to_zero_still_parks():
    # a model that merely idles out (no pressure) still PARKS (fast restore),
    # only pressure-evicted models discard.
    clock = Clock()
    sched = Scheduler([GPU("g0", 20000)])
    parked, discarded = [], []
    a = Autoscaler(sched, Router(), lambda m, g: ReadyBackend(m), clock=clock,
                   on_deactivate=lambda m, g: parked.append(m),
                   on_evict=lambda m, g: discarded.append(m))
    a.register_model("a", 8000, idle_ttl=10)
    a.handle_request("a")
    clock.t = 100                                  # idle past ttl
    a.tick()                                       # scale-to-zero
    assert parked == ["a"] and discarded == []     # idle → park, not discard
