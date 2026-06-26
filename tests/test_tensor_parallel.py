"""Tensor-parallel serving — one model sharded across N GPUs as a SINGLE unit
(distinct from data parallelism, which is N independent replicas on N GPUs).

Invariant: a TP unit is one serving process / one router backend / one replica,
but it reserves, launches, parks, and frees its `tensor_parallel_size` GPUs as an
atomic group.
"""

import pytest

from embers.autoscaler import Autoscaler
from embers.gateway import Router
from embers.gpu_backend import GpuLauncher, GpuServingProcess, _devices
from embers.scheduler import GPU, Scheduler


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Unit:
    def __init__(self, model, gpu_ids):
        self.name = model
        self.gpu_ids = gpu_ids
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def complete(self, *a):
        return "ok"

    def chat(self, *a):
        return "ok"


def make(gpus=4, gpu_mb=24000):
    clock = FakeClock()
    sched = Scheduler([GPU(f"gpu{i}", gpu_mb) for i in range(gpus)])
    router = Router()
    launched, parked = [], []

    def launch(model, gpu_ids):
        launched.append(gpu_ids)
        return Unit(model, gpu_ids)

    def on_deactivate(model, gpu_ids):
        parked.append(gpu_ids)

    a = Autoscaler(sched, router, launch, clock=clock, on_deactivate=on_deactivate)
    return a, clock, sched, router, launched, parked


def _running(sched, model):
    return [g.id for g in sched.gpus if g.runs(model)]


# --- a TP unit is ONE replica spanning N GPUs -----------------------------

def test_tp_unit_is_one_replica_spanning_n_gpus():
    a, clock, sched, router, launched, _ = make(gpus=4)
    a.register_model("big", 8000, max_replicas=1, tensor_parallel_size=2)
    a.handle_request("big")                       # cold-start ONE unit
    assert a.state()["big"] == 1                  # ONE serving replica…
    assert len(router.replicas("big")) == 1       # …one backend in the router
    assert len(_running(sched, "big")) == 2       # …occupying TWO GPUs
    assert sched.total_free_mb() == 4 * 24000 - 2 * 8000   # two shards reserved
    assert len(launched[0]) == 2                  # launch got a 2-GPU group


def test_tp_unit_scale_to_zero_frees_all_its_gpus():
    a, clock, sched, router, launched, parked = make(gpus=4)
    a.register_model("big", 8000, max_replicas=1, idle_ttl=300,
                     tensor_parallel_size=2)
    a.handle_request("big")
    assert len(_running(sched, "big")) == 2
    clock.advance(301)
    a.tick()                                      # idle → scale to zero
    assert a.state()["big"] == 0
    assert sched.total_free_mb() == 4 * 24000     # BOTH shards freed
    assert len(parked[0]) == 2                    # park hook got the 2-GPU group


def test_tp_unit_needs_all_its_gpus_atomically():
    # tensor_parallel_size=2 but only 1 GPU → can't place the group → no unit,
    # no partial placement, request 503s cleanly.
    a, clock, sched, *_ = make(gpus=1)
    a.register_model("big", 8000, tensor_parallel_size=2)
    with pytest.raises(Exception):                # NoReadyBackend (nothing placed)
        a.handle_request("big")
    assert _running(sched, "big") == []           # nothing half-placed


def test_two_tp_units_use_disjoint_gpu_groups():
    # data + tensor parallel: 2 units × tp=2 → 4 GPUs, groups disjoint.
    a, clock, sched, router, launched, _ = make(gpus=4)
    a.register_model("big", 8000, max_replicas=2, requests_per_replica=1,
                     tensor_parallel_size=2)
    a.handle_request("big")
    for _ in range(5):                            # load → wants 2 units
        a.handle_request("big")
    a.tick()
    assert a.state()["big"] == 2                  # two serving units
    assert len(_running(sched, "big")) == 4       # 2 units × 2 GPUs
    g0, g1 = (r.gpu_ids for r in a.models["big"].backends)
    assert set(g0).isdisjoint(g1)                 # disjoint GPU groups
    assert len(g0) == len(g1) == 2


# --- launcher pins the whole group + passes the TP flag -------------------

def test_devices_joins_a_gpu_group_to_cuda_visible_devices():
    assert _devices(["gpu0", "gpu1"]) == "0,1"
    assert _devices(["gpu2"]) == "2"
    assert _devices(["gpu1", "gpu3"]) == "1,3"
    assert _devices([]) is None


def test_launcher_default_make_builds_a_tp_process():
    gl = GpuLauncher()
    proc = gl._make("big", 19000, ["gpu0", "gpu1"])
    assert proc.cuda_device == "0,1"              # pinned to both GPUs
    assert proc.tensor_parallel_size == 2


def test_tp_process_spawns_with_flag_and_pins_devices():
    spawned = {}

    def spawn(args):
        spawned["args"] = args

        class P:
            pid = 1
        return P()

    p = GpuServingProcess("big", port=9000, cuda_device="0,1",
                          tensor_parallel_size=2, spawn=spawn,
                          ready_check=lambda: True, sleep=lambda s: None)
    p.start()
    assert "--tensor-parallel-size" in spawned["args"]
    assert spawned["args"][spawned["args"].index("--tensor-parallel-size") + 1] == "2"
    assert p._child_env()["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_single_gpu_process_omits_the_tp_flag():
    spawned = {}

    def spawn(args):
        spawned["args"] = args

        class P:
            pid = 1
        return P()

    p = GpuServingProcess("m", port=9000, cuda_device="0", tensor_parallel_size=1,
                          spawn=spawn, ready_check=lambda: True, sleep=lambda s: None)
    p.start()
    assert "--tensor-parallel-size" not in spawned["args"]   # DP: no TP flag


# --- config + serve plumbing ----------------------------------------------

def test_tp_unit_parks_and_fast_restores_like_dp():
    # multi-rank TP park/restore is hardware-validated (cuda-checkpoint locks all
    # ranks, checkpoints all → both GPUs free; restore → serves). So a TP unit
    # parks on scale-down and FAST-restores on the next launch, same as DP.
    procs = []

    class StubTP:
        def __init__(self, model, port, tp):
            self.model, self.port = model, port
            self.tensor_parallel_size = tp
            self.parked = False
            self.parks = self.unparks = 0

        @property
        def url(self):
            return f"http://127.0.0.1:{self.port}"

        def start(self):
            return 100.0

        def park(self):
            self.parked = True
            self.parks += 1

        def unpark(self):
            self.parked = False
            self.unparks += 1
            return 16.0

        def stop(self):
            pass

    gl = GpuLauncher(make_process=lambda m, port, gpu_ids: procs.append(
        StubTP(m, port, len(gpu_ids))) or procs[-1])
    gl.launch("big", ["gpu0", "gpu1"])            # TP unit (tp=2), cold load
    gl.deactivate("big", ["gpu0", "gpu1"])        # scale to zero → park all ranks
    assert procs[0].parks == 1                    # parked (not killed)
    assert procs[0].parked is True
    gl.launch("big", ["gpu0", "gpu1"])            # next launch → fast unpark
    assert gl.restores == 1 and gl.cold_loads == 1
    assert gl.last_seconds == 16.0


def test_config_carries_tensor_parallel_size():
    from embers.platform import ModelConfig
    assert ModelConfig("big", 8000, tensor_parallel_size=2).tensor_parallel_size == 2
    assert ModelConfig("small", 6000).tensor_parallel_size == 1   # DP default


def test_model_unit_threads_tp_to_engine_kwargs():
    from embers.server import ModelUnit
    u = ModelUnit("big", tensor_parallel_size=2)
    assert u.engine_kwargs["tensor_parallel_size"] == 2
