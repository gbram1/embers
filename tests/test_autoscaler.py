"""Tests for the Phase 4 autoscaler — deterministic via a fake clock. Covers
scale-from-zero (cold start), scale-to-zero on idle, load-scaling, bounds,
capacity exhaustion, eviction wiring, and counters."""

import pytest

from embers.autoscaler import Autoscaler
from embers.gateway import LocalBackend, Router
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make(gpus=2, gpu_mb=24000):
    clock = FakeClock()
    sched = Scheduler([GPU(f"g{i}", gpu_mb) for i in range(gpus)])
    router = Router()
    launches = []

    def launch(model, gpu_id):
        launches.append((model, gpu_id))
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    a = Autoscaler(sched, router, launch, clock=clock)
    return a, clock, router, sched, launches


# --- scale from zero (the cold start) -------------------------------------

def test_unknown_model_raises():
    a, *_ = make()
    with pytest.raises(KeyError):
        a.handle_request("nope")


def test_request_cold_starts_from_zero():
    a, clock, router, sched, launches = make()
    a.register_model("m", 6000)            # min_replicas=0 → starts cold
    assert a.state()["m"] == 0
    backend = a.handle_request("m")        # first request triggers spin-up
    assert backend is not None
    assert a.state()["m"] == 1
    assert a.cold_starts == 1
    assert len(launches) == 1              # launch (the cold start) happened once
    assert router.models() == ["m"]


def test_second_request_reuses_replica():
    a, *_ = make()
    a.register_model("m", 6000)
    a.handle_request("m")
    a.handle_request("m")
    assert a.state()["m"] == 1             # no second cold start
    assert a.cold_starts == 1


# --- scale to zero on idle ------------------------------------------------

def test_idle_model_scales_to_zero():
    a, clock, router, sched, _ = make()
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")
    assert sched.total_free_mb() == 48000 - 6000   # GPU consumed
    clock.advance(301)                              # go idle
    a.tick()
    assert a.state()["m"] == 0
    assert a.scaled_to_zero == 1
    assert router.models() == []                    # gone from gateway
    assert sched.total_free_mb() == 48000           # GPU freed


def test_active_model_not_scaled_to_zero():
    a, clock, _, _, _ = make()
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")
    clock.advance(100)                              # still within ttl
    a.tick()
    assert a.state()["m"] == 1


def test_cold_start_again_after_scale_to_zero():
    a, clock, *_ = make()
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")
    clock.advance(301); a.tick()
    assert a.state()["m"] == 0
    a.handle_request("m")                            # second cold start
    assert a.state()["m"] == 1
    assert a.cold_starts == 2


# --- load scaling & bounds ------------------------------------------------

def test_scales_up_under_load_capped_at_max():
    a, clock, _, _, _ = make(gpus=4)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    a.handle_request("m")                            # 1 replica, cold
    for _ in range(100):                             # heavy load this window
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3                       # ceil(101/10)=11 → capped at 3


def test_scales_back_down_when_load_drops():
    a, clock, _, _, _ = make(gpus=4)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=10)
    for _ in range(50):
        a.handle_request("m")
    a.tick()
    assert a.state()["m"] == 3
    a.handle_request("m")                            # light load next window
    a.tick()
    assert a.state()["m"] == 1                       # back to floor


def test_unrequested_model_stays_at_zero_across_ticks():
    # regression: a registered model nobody has requested must NOT be spun up by
    # the control loop just because it isn't "idle" yet.
    a, clock, _, _, _ = make()
    a.register_model("never-used", 6000, min_replicas=0, idle_ttl=300)
    a.tick()                                   # not idle yet, but no traffic
    a.tick()
    assert a.state()["never-used"] == 0        # still cold — nobody asked for it


def test_running_model_not_flapped_to_zero_between_windows():
    # an active model with a quiet tick window keeps ≥1 until it actually idles
    a, clock, _, _, _ = make()
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")                       # 1 replica
    clock.advance(10)                           # recent, not idle
    a.tick()                                    # zero requests this window
    assert a.state()["m"] == 1                  # held warm, not scaled to 0


def test_min_replicas_floor_kept_warm():
    a, clock, router, _, _ = make()
    a.register_model("m", 6000, min_replicas=1, idle_ttl=300)
    assert a.state()["m"] == 1                       # warmed on register
    clock.advance(1000); a.tick()
    assert a.state()["m"] == 1                       # idle but floor keeps 1 warm


# --- capacity exhaustion ---------------------------------------------------

def test_scale_up_stops_at_capacity_without_crashing():
    a, clock, _, sched, _ = make(gpus=1, gpu_mb=10000)
    a.register_model("m", 6000, max_replicas=3, requests_per_replica=1)
    for _ in range(10):
        a.handle_request("m")
    a.tick()
    # only one 6000 replica fits in a single 10000 GPU → stuck at 1, no crash
    assert a.state()["m"] == 1


def test_cold_start_with_no_capacity_raises_on_pick():
    a, _, _, _, _ = make(gpus=1, gpu_mb=1000)
    a.register_model("big", 6000)
    with pytest.raises(Exception):  # NoCapacity -> nothing to pick -> error
        a.handle_request("big")


class FlakyBackend:
    """A backend whose readiness can be flipped to simulate a crash."""
    def __init__(self, model):
        self.name = model
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def complete(self, *a):
        return "ok"

    def chat(self, *a):
        return "ok"


def test_reap_replaces_dead_replica():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000), GPU("g1", 24000)])
    router = Router()
    made = []

    def launch(model, gpu_id):
        b = FlakyBackend(model)
        made.append(b)
        return b

    a = Autoscaler(sched, router, launch, clock=clock)
    a.register_model("m", 6000, min_replicas=1)   # keep 1 warm
    assert a.state()["m"] == 1
    made[0]._ready = False                          # the replica "crashes"
    a.tick()                                        # reap + re-place to floor
    assert a.reaped == 1
    assert a.state()["m"] == 1                      # back to 1 healthy replica
    assert made[-1] is not made[0]                  # a fresh replica was launched
    assert router.replicas("m")[0] is made[-1]      # gateway routes to the new one


def test_reap_treats_throwing_probe_as_dead():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000), GPU("g1", 24000)])
    router = Router()

    class ExplodingBackend:
        name = "m"
        mode = "ok"

        @property
        def ready(self):
            if self.mode == "boom":
                raise RuntimeError("connection refused")
            return True

        def complete(self, *a):
            return "x"

        def chat(self, *a):
            return "x"

    a = Autoscaler(sched, router, lambda m, g: ExplodingBackend(), clock=clock)
    a.register_model("m", 6000, min_replicas=0)
    a.handle_request("m")                           # 1 replica (ready)
    router.replicas("m")[0].mode = "boom"           # its health probe now throws
    a.tick()                                        # treated as dead → reaped
    assert a.reaped >= 1


def test_park_failure_still_frees_scheduler_slot():
    # regression: if on_deactivate (park) raises, the GPU slot must still be
    # freed so the next cold-start can place — never strand the slot.
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    def boom_deactivate(model, gpu_id):
        raise RuntimeError("cuda-checkpoint checkpoint failed")

    a = Autoscaler(sched, router, launch, clock=clock,
                   on_deactivate=boom_deactivate)
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")
    assert sched.total_free_mb() == 24000 - 6000
    clock.advance(301)
    a.tick()                                  # park raises, but slot still frees
    assert a.state()["m"] == 0
    assert a.scale_downs == 1                  # completed despite park failure
    assert sched.total_free_mb() == 24000      # GPU slot reclaimed
    # and the next request can cold-start again (slot was free)
    a.handle_request("m")
    assert a.state()["m"] == 1


def test_on_deactivate_hook_called_on_scale_to_zero():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    parked = []
    a = Autoscaler(sched, router, launch, clock=clock,
                   on_deactivate=lambda m, gids: parked.append((m, gids)))
    a.register_model("m", 6000, idle_ttl=300)
    a.handle_request("m")
    clock.advance(301)
    a.tick()                                  # scale to zero → deactivate hook
    assert parked == [("m", ["g0"])]          # park called with the placed GPU group
    assert sched.total_free_mb() == 24000     # GPU still freed after parking
