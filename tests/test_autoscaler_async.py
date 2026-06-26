"""Async-park tests with a REAL ThreadPoolExecutor + a slow park, exercising the
behaviour mocks-inline can't: tick() doesn't block on the ~20s park, and a
request arriving mid-park WAITS (then unparks) instead of 503ing."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from embers.autoscaler import Autoscaler
from embers.gateway import LocalBackend, Router
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit
from tests.test_autoscaler import FakeClock


def build(park_delay=0.0, on_deactivate=None):
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    ex = ThreadPoolExecutor(max_workers=2)
    a = Autoscaler(sched, router, launch, clock=clock,
                   on_deactivate=on_deactivate, executor=ex)
    return a, clock, sched, ex


def wait_idle(a, timeout=5):
    end = time.time() + timeout
    while a.pending_transitions() and time.time() < end:
        time.sleep(0.01)


def test_tick_does_not_block_on_slow_park():
    started = threading.Event()
    release = threading.Event()

    def slow_park(model, gpu_id):
        started.set()
        release.wait(5)            # simulate a 20s cuda-checkpoint park

    a, clock, sched, ex = build(on_deactivate=slow_park)
    try:
        a.register_model("m", 6000, idle_ttl=10)
        a.handle_request("m")
        clock.advance(11)
        t0 = time.perf_counter()
        a.tick()                   # must return immediately, NOT wait for park
        assert time.perf_counter() - t0 < 0.5
        assert started.wait(2)     # the park is running in the background
        # GPU slot not freed yet — park still in progress
        assert sched.total_free_mb() == 24000 - 6000
        release.set()              # let the park finish
        wait_idle(a)
        assert sched.total_free_mb() == 24000          # slot freed after park
        assert a.state()["m"] == 0
    finally:
        release.set()
        ex.shutdown(wait=True)


def test_request_during_park_waits_then_serves_not_503():
    in_park = threading.Event()
    finish_park = threading.Event()

    def slow_park(model, gpu_id):
        in_park.set()
        finish_park.wait(5)

    a, clock, sched, ex = build(on_deactivate=slow_park)
    try:
        a.register_model("m", 6000, idle_ttl=10)
        a.handle_request("m")
        clock.advance(11)
        a.tick()                       # submits the (slow) park
        assert in_park.wait(2)         # park is in progress, holds the per-model lock

        # a request arrives mid-park, from another thread — it must WAIT, not 503
        result = {}

        def do_request():
            try:
                result["backend"] = a.handle_request("m")
            except Exception as e:     # noqa: BLE001
                result["error"] = e

        clock.advance(1)               # not idle anymore
        req = threading.Thread(target=do_request)
        req.start()
        time.sleep(0.2)
        assert not result              # still blocked on the per-model lock (park)

        finish_park.set()              # park completes → request proceeds
        req.join(5)
        assert "error" not in result
        assert result.get("backend") is not None    # served, never 503'd
        wait_idle(a)
        assert a.state()["m"] == 1     # the waiting request cold-started it back
    finally:
        finish_park.set()
        ex.shutdown(wait=True)


def test_shutdown_stops_executor():
    a, clock, sched, ex = build()
    a.register_model("m", 6000)
    a.shutdown()                       # must not raise; executor stopped
    assert ex._shutdown
