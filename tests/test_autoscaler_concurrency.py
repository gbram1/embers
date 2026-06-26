"""Traffic-safety tests for the autoscaler: single-flight cold-start under
concurrent requests, and the in-flight guard that prevents parking a model
that's still serving."""

import threading
import time

from embers.autoscaler import Autoscaler
from embers.gateway import LocalBackend, Router
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit
from tests.test_autoscaler import FakeClock


def build(launch_delay=0.0):
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000), GPU("g1", 24000)])
    router = Router()
    launches = []
    lock = threading.Lock()

    def launch(model, gpu_id):
        if launch_delay:
            time.sleep(launch_delay)        # simulate a slow cold start
        with lock:
            launches.append(model)
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    a = Autoscaler(sched, router, launch, clock=clock)
    return a, clock, launches


def test_concurrent_requests_cold_start_once():
    """N threads hit a cold model at once → exactly ONE launch (single-flight),
    not N replicas."""
    a, _, launches = build(launch_delay=0.05)
    a.register_model("m", 6000, max_replicas=5)

    errors = []

    def hit():
        try:
            assert a.handle_request("m") is not None
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert launches == ["m"]          # one cold start despite 8 concurrent hits
    assert a.state()["m"] == 1


def test_inflight_blocks_scale_to_zero():
    a, clock, _ = build()
    a.register_model("m", 6000, idle_ttl=300)
    a.begin_request("m")              # in-flight, not yet ended
    assert a.inflight("m") == 1
    clock.advance(301)                # idle by the clock...
    a.tick()
    assert a.state()["m"] == 1        # ...but NOT parked — request still running


def test_scale_to_zero_after_request_ends():
    a, clock, _ = build()
    a.register_model("m", 6000, idle_ttl=300)
    a.begin_request("m")
    a.end_request("m")               # request done
    assert a.inflight("m") == 0
    clock.advance(301)
    a.tick()
    assert a.state()["m"] == 0       # now safe to scale to zero


def test_end_request_never_goes_negative():
    a, _, _ = build()
    a.register_model("m", 6000)
    a.end_request("m")               # unmatched end
    assert a.inflight("m") == 0


def test_begin_request_tracks_multiple_in_flight():
    a, _, _ = build()
    a.register_model("m", 6000)
    a.begin_request("m")
    a.begin_request("m")
    assert a.inflight("m") == 2
    a.end_request("m")
    assert a.inflight("m") == 1
