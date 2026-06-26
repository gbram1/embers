"""Smarter eviction policy — cost/size/frequency-aware victim choice vs pure LRU.
A cache-replacement problem (GPU = cache, model = object, cold-load = miss)."""

import pytest

from embers.autoscaler import (
    EVICTION_POLICIES, Autoscaler, ManagedModel,
    _cost_aware_victim, _lru_victim,
)
from embers.gateway import Router
from embers.scheduler import GPU, Scheduler


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class ReadyBackend:
    def __init__(self, model):
        self.name = model

    @property
    def ready(self):
        return True

    def chat(self, *a, **k):
        return "ok"

    def complete(self, *a, **k):
        return "ok"


def mm(model, *, vram_mb=8000, last_active=0.0, total_requests=0, last_load=0.0):
    m = ManagedModel(model, vram_mb, last_active=last_active)
    m.total_requests = total_requests
    m.last_load_seconds = last_load
    return m


# --- the scoring functions in isolation -----------------------------------

def test_lru_evicts_oldest():
    cands = [mm("a", last_active=5), mm("b", last_active=1), mm("c", last_active=9)]
    assert _lru_victim(cands, now=10).model == "b"      # idle longest


def test_cost_aware_keeps_frequent_expensive_small_model():
    now = 100
    # hot: small, frequently hit, expensive to reload → KEEP
    hot = mm("hot", vram_mb=4000, last_active=90, total_requests=500, last_load=90)
    # cold: big, rarely hit, cheap to reload → EVICT
    cold = mm("cold", vram_mb=20000, last_active=88, total_requests=2, last_load=20)
    assert _cost_aware_victim([hot, cold], now).model == "cold"


def test_cost_aware_differs_from_lru():
    now = 100
    # the LRU victim (oldest) is actually the precious one; cost-aware spares it
    precious_but_old = mm("precious", vram_mb=4000, last_active=10,
                          total_requests=1000, last_load=100)
    fresh_but_cheap = mm("cheap", vram_mb=24000, last_active=95,
                         total_requests=1, last_load=10)
    cands = [precious_but_old, fresh_but_cheap]
    assert _lru_victim(cands, now).model == "precious"        # LRU would kill it
    assert _cost_aware_victim(cands, now).model == "cheap"    # cost-aware saves it


# --- wired into the autoscaler's demand eviction --------------------------

def autoscaler(policy):
    sched = Scheduler([GPU("g0", 20000)])
    parked = []
    a = Autoscaler(sched, Router(), lambda m, g: ReadyBackend(m), clock=Clock(),
                   on_evict=lambda m, g: parked.append(m), eviction_policy=policy)
    return a, parked


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        autoscaler("magic")


def test_cost_aware_eviction_end_to_end():
    a, evicted = autoscaler("cost_aware")
    for name in ("a", "b"):
        a.register_model(name, 8000, idle_ttl=999)
    a.register_model("c", 8000, idle_ttl=999)
    # make "a" precious (many requests), "b" disposable (one request); both idle
    for _ in range(50):
        a.clock.t += 1
        a.handle_request("a")
    a.clock.t += 1
    a.handle_request("b")                       # a, b resident; a hot, b cold
    a.clock.t += 1
    a.handle_request("c")                       # needs room → evicts the cheap one
    assert evicted == ["b"]                     # NOT "a" (LRU would've picked a-vs-b by age)
    assert a.eviction_policy == "cost_aware"


def test_registry_has_both_policies():
    assert set(EVICTION_POLICIES) == {"lru", "cost_aware"}
