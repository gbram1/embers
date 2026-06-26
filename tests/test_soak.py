"""Concurrent-load soak — hammer the REAL control plane (autoscaler locks,
single-flight cold-start, in-flight guard, scale-to-zero) from many threads with
a real ThreadPoolExecutor, while a ticker scales models up/down underneath. This
is the mock-mode stress test: no GPU, but the locking/threading is the real code.

Invariants after the storm settles:
  * no request raised / 503'd mid-park (handle_request waits on the per-model lock)
  * every in-flight counter returns to 0 (begin/end balanced)
  * the scheduler has no leaked placements (used == live replicas' footprint)
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from embers.autoscaler import Autoscaler
from embers.scheduler import GPU, Scheduler
from embers.gateway import Router


class Clock:
    """Monotonic-ish clock advanced by the test threads (so idle_ttl can elapse)."""
    def __init__(self):
        self.t = 0.0
        self._l = threading.Lock()

    def __call__(self):
        return self.t

    def tick(self, dt=1.0):
        with self._l:
            self.t += dt


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


def test_concurrent_soak_no_races_no_leaks():
    MODELS = ["a", "b", "c"]
    clock = Clock()
    sched = Scheduler([GPU(f"g{i}", 24000) for i in range(6)])
    router = Router()

    launches = {"n": 0}
    launch_lock = threading.Lock()

    def launch(model, gpu_ids):
        with launch_lock:
            launches["n"] += 1
        return ReadyBackend(model)

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="soak-xition")
    a = Autoscaler(sched, router, launch, clock=clock,
                   on_deactivate=lambda m, g: None, executor=pool)
    for m in MODELS:
        a.register_model(m, 4000, min_replicas=0, max_replicas=3,
                         idle_ttl=2, requests_per_replica=5)

    errors = []
    stop = threading.Event()

    def worker(wid):
        try:
            for i in range(60):
                model = MODELS[(wid + i) % len(MODELS)]
                try:
                    backend = a.begin_request(model)   # cold-start/wait, never 503
                    assert backend is not None
                    backend.chat([], 8, 0.0)
                finally:
                    a.end_request(model)
                if i % 7 == 0:
                    clock.tick(1.0)                     # advance time → idle/scale churn
        except Exception as e:                          # noqa: BLE001
            errors.append(e)

    def ticker():
        while not stop.is_set():
            try:
                a.tick()                                # scale up/down concurrently
            except Exception as e:                      # noqa: BLE001
                errors.append(e)

    tk = threading.Thread(target=ticker, daemon=True)
    tk.start()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(8)))
    stop.set()
    tk.join(timeout=5)

    # let any in-flight reconcile finish, then settle
    pool.shutdown(wait=True)

    assert not errors, f"races/exceptions under load: {errors[:3]}"

    # every in-flight counter balanced back to zero
    for m in MODELS:
        assert a.inflight(m) == 0, f"{m} leaked in-flight: {a.inflight(m)}"

    # scheduler has no leaked placements: used VRAM == live replicas' footprint
    for m in MODELS:
        mm = a.models[m]
        running = [g for g in sched.gpus if g.runs(m)]
        assert len(running) == mm.replicas, (
            f"{m}: scheduler shows {len(running)} placements but {mm.replicas} replicas")

    # router and autoscaler agree on replica counts
    for m in MODELS:
        assert len(router.replicas(m)) == a.models[m].replicas


def test_single_flight_under_thundering_herd():
    # many simultaneous first-requests for a cold model → exactly ONE cold-start.
    clock = Clock()
    sched = Scheduler([GPU("g0", 24000)])
    router = Router()
    launches = []
    lock = threading.Lock()

    def launch(model, gpu_ids):
        with lock:
            launches.append(model)
        return ReadyBackend(model)

    a = Autoscaler(sched, router, launch, clock=clock)
    a.register_model("m", 4000, max_replicas=3)

    start = threading.Barrier(16)
    errs = []

    def hit():
        try:
            start.wait()
            b = a.begin_request("m")
            assert b is not None
            a.end_request("m")
        except Exception as e:        # noqa: BLE001
            errs.append(e)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda _: hit(), range(16)))

    assert not errs
    assert len(launches) == 1, f"thundering herd caused {len(launches)} cold-starts (want 1)"
