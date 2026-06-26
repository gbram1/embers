"""Multi-GPU / multi-replica autoscaler hardening — the scenarios a real
multi-GPU box exercises but a single-GPU validation never does.

The central invariant: a model's replicas live on DISTINCT GPUs, so every
scale-down and reap must act on the *specific* GPU of the replica it drops —
not merely 'a' GPU the model happens to occupy. (Regression guard: an earlier
`_gpu_of(model)` returned the first placement's GPU, so dropping replica #3
parked replica #1's process and evicted the wrong slot — a silent GPU leak
invisible with a single replica.)
"""

import pytest

from embers.autoscaler import Autoscaler
from embers.gateway import Router
from embers.scheduler import GPU, Scheduler


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class GpuBackend:
    """A backend that remembers which GPU(s) it was launched on, with a flippable
    readiness flag (to simulate a specific replica crashing). These tests are all
    data-parallel (one GPU per replica), so `gpu_id` is the single primary GPU."""
    def __init__(self, model, gpu_ids):
        self.name = model
        self.gpu_ids = gpu_ids
        self.gpu_id = gpu_ids[0]
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def complete(self, *a):
        return "ok"

    def chat(self, *a):
        return "ok"


def make(gpus=3, gpu_mb=24000):
    clock = FakeClock()
    sched = Scheduler([GPU(f"g{i}", gpu_mb) for i in range(gpus)])
    router = Router()
    launched = []          # every backend ever launched, in order
    parked = []            # (model, gpu_id) the park hook was asked to free

    def launch(model, gpu_ids):
        b = GpuBackend(model, gpu_ids)
        launched.append(b)
        return b

    def on_deactivate(model, gpu_ids):
        parked.append((model, gpu_ids[0]))     # DP: one GPU per replica

    a = Autoscaler(sched, router, launch, clock=clock, on_deactivate=on_deactivate)
    return a, clock, router, sched, launched, parked


def _running_gpus(sched, model):
    return {g.id for g in sched.gpus if g.runs(model)}


# --- spread across distinct GPUs ------------------------------------------

def test_scale_up_spreads_replicas_across_distinct_gpus():
    a, clock, router, sched, launched, _ = make(gpus=3)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    a.handle_request("m")                       # 1 replica (cold)
    for _ in range(30):                         # load → wants ceil(31/10)=4, cap 3
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3
    # each replica on a distinct GPU
    assert len({b.gpu_id for b in launched}) == 3
    assert len(_running_gpus(sched, "m")) == 3
    assert len(router.replicas("m")) == 3
    assert sched.total_free_mb() == 3 * 24000 - 3 * 6000


def test_scale_up_capped_by_distinct_gpu_count():
    # 2 GPUs but load wants 3 replicas → caps at 2 (replicas need distinct GPUs),
    # no crash, no partial/duplicate placement.
    a, clock, router, sched, launched, _ = make(gpus=2)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=1)
    for _ in range(10):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 2
    assert a.scale_ups == 2
    assert len({b.gpu_id for b in launched}) == 2


# --- scale-down acts on the DROPPED replica's own GPU (the bug) -----------

def test_scale_down_parks_and_evicts_the_dropped_replicas_own_gpu():
    a, clock, router, sched, launched, parked = make(gpus=3)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    a.handle_request("m")
    for _ in range(30):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3

    mm = a.models["m"]
    doomed_gpu = mm.backends[-1].gpu_ids[0]      # _scale_down_locked pops the last
    other_gpus = {r.gpu_ids[0] for r in mm.backends[:-1]}

    for _ in range(15):                          # next window: load wants 2
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 2

    # the park hook + the scheduler eviction both targeted the dropped replica's
    # OWN gpu — not the model's first placement.
    assert ("m", doomed_gpu) in parked
    assert doomed_gpu not in _running_gpus(sched, "m")
    assert _running_gpus(sched, "m") == other_gpus
    assert sched.total_free_mb() == 3 * 24000 - 2 * 6000   # exactly 2 replicas' worth


def test_scale_to_zero_from_many_replicas_frees_every_gpu():
    a, clock, router, sched, launched, parked = make(gpus=3)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10, idle_ttl=300)
    a.handle_request("m")
    for _ in range(30):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3
    up_gpus = {b.gpu_id for b in launched}

    clock.advance(301)                           # go idle
    a.tick()
    assert a.state()["m"] == 0
    assert router.models() == []
    assert sched.total_free_mb() == 3 * 24000    # every GPU reclaimed
    # all three distinct GPUs were parked (none parked twice, none missed)
    assert {g for _, g in parked} == up_gpus
    assert len(parked) == 3


# --- reap targets the DEAD replica's own GPU (the bug) --------------------

def test_reap_frees_the_dead_replicas_own_gpu_not_the_first():
    a, clock, router, sched, launched, _ = make(gpus=3)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    a.handle_request("m")
    for _ in range(30):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3

    mm = a.models["m"]
    dead = mm.backends[1]                         # kill the MIDDLE replica
    dead_gpu = dead.gpu_ids[0]
    survivors = {mm.backends[0].gpu_ids[0], mm.backends[2].gpu_ids[0]}
    dead.backend._ready = False

    a._reap(mm)                                   # reap in isolation (no rescale)
    assert a.reaped == 1
    assert a.state()["m"] == 2
    # exactly the dead replica's GPU was freed; the live ones are untouched
    assert dead_gpu not in _running_gpus(sched, "m")
    assert _running_gpus(sched, "m") == survivors
    assert dead.backend not in router.replicas("m")


# --- round-robin balances across the multi-GPU replicas -------------------

def test_round_robin_cycles_all_multi_gpu_replicas():
    a, clock, router, sched, launched, _ = make(gpus=3)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    a.handle_request("m")
    for _ in range(30):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3

    picks = [router.pick("m") for _ in range(6)]
    # every distinct replica (and thus every GPU) is hit evenly over 6 picks
    assert {p.gpu_id for p in picks} == {b.gpu_id for b in launched}
    assert len(set(picks[:3])) == 3              # first cycle hits all three once


# --- warm floor of >1 replica (HA) ----------------------------------------

def test_min_replicas_floor_above_one_warms_across_gpus_and_holds():
    a, clock, router, sched, launched, _ = make(gpus=3)
    a.register_model("m", 6000, min_replicas=2, max_replicas=3, idle_ttl=300)
    # registration warms the floor: 2 replicas, distinct GPUs, both routable
    assert a.state()["m"] == 2
    assert len({b.gpu_id for b in launched}) == 2
    assert len(router.replicas("m")) == 2

    clock.advance(400)                           # go well past idle_ttl
    a.tick()
    assert a.state()["m"] == 2                    # holds at the floor, not 0 or 1
    assert a.scaled_to_zero == 0
    assert len(_running_gpus(sched, "m")) == 2


# --- full lifecycle: up to many, down to one, no leaks --------------------

def test_full_multi_gpu_lifecycle_leaves_scheduler_consistent():
    a, clock, router, sched, launched, parked = make(gpus=4)
    a.register_model("m", 5000, max_replicas=4, requests_per_replica=10)
    a.handle_request("m")
    for _ in range(40):                           # scale to 4
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 4
    assert sched.total_free_mb() == 4 * 24000 - 4 * 5000

    a.handle_request("m")                         # light load → back to floor 1
    a.tick()
    assert a.state()["m"] == 1
    # scheduler accounting matches the router exactly: one GPU still runs it
    assert len(_running_gpus(sched, "m")) == 1
    assert len(router.replicas("m")) == 1
    assert sched.total_free_mb() == 4 * 24000 - 1 * 5000
    assert a.scale_downs == 3                     # dropped exactly three replicas
