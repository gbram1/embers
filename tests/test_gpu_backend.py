"""Tests for the real-hardware GPU backend — subprocess/HTTP/cuda-checkpoint
boundaries injected, so the park/unpark lifecycle is verified on macOS."""

import pytest

from embers.gpu_backend import GpuLauncher, GpuServingProcess


class FakeProc:
    def __init__(self):
        self.pid = 4242
        self.terminated = False

    def terminate(self):
        self.terminated = True


def make_process(ready_after=2):
    """A GpuServingProcess with everything faked; ready flips True after N polls.
    Records cuda-checkpoint commands issued."""
    cmds = []
    state = {"polls": 0}

    def spawn(args):
        cmds.append(("spawn", args))
        return FakeProc()

    def run(args):
        cmds.append(("run", args))

    def ready():
        state["polls"] += 1
        return state["polls"] >= ready_after

    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    p = GpuServingProcess("m", port=8001, spawn=spawn, run=run,
                          ready_check=ready, sleep=lambda s: None, clock=clock,
                          gpu_pid_finder=lambda pid: [pid])  # one GPU pid == main pid
    return p, cmds


def test_start_spawns_and_waits_for_ready():
    p, cmds = make_process(ready_after=3)
    secs = p.start()
    assert p.pid == 4242
    assert secs > 0
    spawn = [c for c in cmds if c[0] == "spawn"][0][1]
    assert spawn[1:4] == ["-m", "embers.cli", "serve"]   # spawn[0] is the python
    assert "--port" in spawn and "8001" in spawn


def test_start_times_out_if_never_ready():
    p, _ = make_process(ready_after=10_000)
    with pytest.raises(TimeoutError):
        p.start(timeout=1)


def test_park_targets_gpu_holding_child_pid():
    # AsyncLLMEngine holds the GPU in a child — park must checkpoint THAT pid
    p, cmds = make_process()
    p.start()
    p._gpu_pid_finder = lambda main: [9999]     # pretend the GPU child is 9999
    p.park()
    runs = [c[1] for c in cmds if c[0] == "run"]
    # every cuda-checkpoint command targets the child pid, not the main pid
    for r in runs:
        assert "9999" in r
    assert all(str(p.pid) != r[r.index("--pid") + 1] for r in runs)


def test_unpark_uses_same_pid_as_park():
    p, cmds = make_process()
    p.start()
    p._gpu_pid_finder = lambda main: [9999]
    p.park()
    p.unpark()
    runs = [c[1] for c in cmds if c[0] == "run"]
    assert all("9999" in r for r in runs)       # park + unpark all hit 9999


def test_find_gpu_pids_falls_back_to_main_without_nvidia_smi():
    # on a box with no nvidia-smi, resolve to the main pid (single-process case)
    p, _ = make_process()
    p.pid = 4242
    p._gpu_pid_finder = GpuServingProcess._find_gpu_pids.__get__(p)
    assert p._gpu_pid_finder(4242) == [4242]     # nvidia-smi absent → main pid


def test_park_issues_lock_then_checkpoint():
    p, cmds = make_process()
    p.start()
    p.park()
    assert p.parked is True
    runs = [c[1] for c in cmds if c[0] == "run"]
    actions = [a[a.index("--action") + 1] for a in runs]
    assert actions == ["lock", "checkpoint"]


def test_unpark_issues_restore_then_unlock():
    p, cmds = make_process()
    p.start()
    p.park()
    p.unpark()
    assert p.parked is False
    runs = [c[1] for c in cmds if c[0] == "run"]
    actions = [a[a.index("--action") + 1] for a in runs]
    assert actions == ["lock", "checkpoint", "restore", "unlock"]


def test_stop_terminates_process():
    p, _ = make_process()
    p.start()
    proc = p.proc
    p.stop()
    assert proc.terminated


def test_tensor_parallel_park_locks_all_ranks_before_checkpointing_any():
    # TP unit: 2 rank pids. The proven recipe is lock ALL ranks (quiesce) BEFORE
    # checkpointing any (they share NCCL collectives), then restore all, unlock all.
    p, cmds = make_process()
    p.start()
    p._gpu_pid_finder = lambda main: [101, 202]    # two GPU rank pids (TP)
    p.park()
    p.unpark()
    runs = [c[1] for c in cmds if c[0] == "run"]
    seq = [(a[a.index("--action") + 1], a[a.index("--pid") + 1]) for a in runs]
    assert seq == [
        ("lock", "101"), ("lock", "202"),          # lock BOTH first
        ("checkpoint", "101"), ("checkpoint", "202"),
        ("restore", "101"), ("restore", "202"),    # restore both
        ("unlock", "101"), ("unlock", "202"),      # then unlock both
    ]


# --- GpuLauncher (autoscaler-facing) --------------------------------------

class StubProcess:
    """Minimal stand-in for GpuServingProcess in launcher tests."""
    def __init__(self, model, port):
        self.model, self.port = model, port
        self.parked = False
        self.terminated = False
        self.started = self.parks = self.unparks = 0

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.started += 1
        return 50.0          # pretend cold load ~50s

    def park(self):
        self.parked = True
        self.parks += 1

    def unpark(self):
        self.parked = False
        self.unparks += 1
        return 9.0           # pretend restore ~9s

    def stop(self):
        self.terminated = True


def launcher():
    return GpuLauncher(make_process=lambda m, port, g: StubProcess(m, port))


def test_first_launch_cold_loads():
    gl = launcher()
    b = gl.launch("m", ["g0"])
    assert gl.cold_loads == 1 and gl.restores == 0
    assert b.name == "m"
    assert gl.procs[("m", ("g0",))].started == 1
    assert gl.last_seconds == 50.0


def test_deactivate_parks_then_launch_unparks():
    gl = launcher()
    gl.launch("m", ["g0"])           # cold load
    gl.deactivate("m", ["g0"])       # park (scale to zero)
    assert gl.procs[("m", ("g0",))].parked is True
    gl.launch("m", ["g0"])           # parked → fast unpark
    assert gl.restores == 1
    assert gl.cold_loads == 1       # no second cold load
    assert gl.last_seconds == 9.0


def test_distinct_models_get_distinct_ports():
    gl = launcher()
    gl.launch("a", ["g0"])
    gl.launch("b", ["g0"])
    assert gl.procs[("a", ("g0",))].port != gl.procs[("b", ("g0",))].port


def test_deactivate_unknown_model_is_noop():
    launcher().deactivate("ghost", ["g0"])   # must not raise


def test_repeated_park_unpark_cycles():
    gl = launcher()
    gl.launch("m", ["g0"])
    for _ in range(3):
        gl.deactivate("m", ["g0"])
        gl.launch("m", ["g0"])
    assert gl.cold_loads == 1 and gl.restores == 3


# --- multi-GPU / multi-replica launcher (replicas are distinct processes) --

def test_two_replicas_of_a_model_are_distinct_processes_on_distinct_gpus():
    gl = launcher()
    b0 = gl.launch("m", ["gpu0"])
    b1 = gl.launch("m", ["gpu1"])
    # each replica is its own cold-loaded process on its own port — NOT aliased
    assert gl.cold_loads == 2
    assert ("m", ("gpu0",)) in gl.procs and ("m", ("gpu1",)) in gl.procs
    assert gl.procs[("m", ("gpu0",))] is not gl.procs[("m", ("gpu1",))]
    assert b0.base_url != b1.base_url      # distinct serving units


def test_deactivate_parks_only_the_targeted_replica():
    gl = launcher()
    gl.launch("m", ["gpu0"])
    gl.launch("m", ["gpu1"])
    gl.deactivate("m", ["gpu0"])             # park only the gpu0 replica
    assert gl.procs[("m", ("gpu0",))].parked is True
    assert gl.procs[("m", ("gpu1",))].parked is False   # gpu1 still serving
    # and relaunching gpu0 restores THAT replica (fast), gpu1 untouched
    gl.launch("m", ["gpu0"])
    assert gl.restores == 1
    assert gl.cold_loads == 2              # the two initial cold loads, no more


def test_replica_pins_to_its_physical_gpu_via_cuda_visible_devices():
    # the real default make derives CUDA_VISIBLE_DEVICES from the gpu_id, so a
    # replica lands on the scheduler's chosen GPU — not always GPU 0.
    p = GpuServingProcess("m", port=9000, cuda_device="1")
    env = p._child_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_launcher_serve_host_pins_binding_for_off_box_reach():
    # default binds localhost; serve_host=0.0.0.0 makes units reachable off-box
    assert GpuLauncher()._make("m", 19000, ["gpu0"]).host == "127.0.0.1"
    assert GpuLauncher(serve_host="0.0.0.0")._make("m", 19000, ["gpu0"]).host == "0.0.0.0"


def test_serve_host_passed_to_spawn_args():
    spawned = {}

    def spawn(args):
        spawned["args"] = args

        class P:
            pid = 1
        return P()

    p = GpuServingProcess("m", port=19000, host="0.0.0.0", spawn=spawn,
                          ready_check=lambda: True, sleep=lambda s: None)
    p.start()
    a = spawned["args"]
    assert a[a.index("--host") + 1] == "0.0.0.0"


def test_cuda_device_maps_logical_gpu_id_to_physical_index():
    from embers.gpu_backend import _cuda_device
    assert _cuda_device("gpu0") == "0"
    assert _cuda_device("gpu3") == "3"
    assert _cuda_device("g2") == "2"
    assert _cuda_device(None) is None
    assert _cuda_device("mig-uuid") is None   # no trailing digits → don't pin


class FlakyUnparkProcess(StubProcess):
    """Unpark raises once (simulating a dead/corrupt parked process)."""
    def unpark(self):
        raise RuntimeError("cuda-checkpoint restore failed")


def test_cc_raises_on_nonzero_returncode():
    import subprocess

    p, _ = make_process()
    p.start()

    class R:
        returncode = 1
        stderr = b"the operation cannot be performed in the present state"
    p._run = lambda args: R()
    with pytest.raises(RuntimeError, match="cuda-checkpoint"):
        p.park()


def test_deactivate_park_failure_kills_and_drops_and_raises():
    class BoomPark(StubProcess):
        def park(self):
            raise RuntimeError("checkpoint failed")

    procs = []
    gl = GpuLauncher(make_process=lambda m, port, g: procs.append(
        BoomPark(m, port)) or procs[-1])
    gl.launch("m", ["g0"])
    with pytest.raises(RuntimeError):
        gl.deactivate("m", ["g0"])
    assert gl.park_failures == 1
    assert procs[0].terminated          # killed (so its GPU frees on death)
    assert ("m", ("g0",)) not in gl.procs  # dropped → next launch cold-loads
    gl.launch("m", ["g0"])                # cold-loads fresh, no crash
    assert gl.cold_loads == 2


def test_unpark_failure_falls_back_to_cold_load():
    procs = []

    def make(model, port, gpu_id):
        p = FlakyUnparkProcess(model, port)
        procs.append(p)
        return p

    gl = GpuLauncher(make_process=make)
    gl.launch("m", ["g0"])             # cold load (proc 0)
    gl.deactivate("m", ["g0"])         # park proc 0
    backend = gl.launch("m", ["g0"])  # unpark FAILS → must cold-load a fresh proc
    assert gl.unpark_failures == 1
    assert gl.cold_loads == 2        # fell back to a second cold load
    assert gl.restores == 0
    assert len(procs) == 2           # a new process was created
    assert procs[0].terminated       # the dead parked process was stopped
    assert backend.name == "m"       # still returns a working backend


def test_per_model_engine_config_passed_to_serve_args():
    # fractional GPU: a model's gpu_memory_utilization + max_model_len reach the
    # serving subprocess so several models can be packed on one GPU.
    from embers.gpu_backend import GpuLauncher
    gl = GpuLauncher(engine_by_model={
        "m": {"gpu_memory_utilization": 0.15, "max_model_len": 2048}})
    proc = gl._make("m", 19000, ["gpu0"])
    assert proc.gpu_memory_utilization == 0.15 and proc.max_model_len == 2048
    spawned = {}

    def spawn(args):
        spawned["args"] = args

        class P:
            pid = 1
        return P()
    proc._spawn = spawn
    proc._ready = lambda: True
    proc._sleep = lambda s: None
    proc.start()
    a = spawned["args"]
    assert a[a.index("--gpu-memory-utilization") + 1] == "0.15"
    assert a[a.index("--max-model-len") + 1] == "2048"


def test_same_gpu_ops_are_serialized():
    # a park and a cold-load on the SAME physical GPU must not overlap (else
    # vLLM's startup memory-profile breaks when the park frees memory mid-load).
    import threading
    import time as _t
    events, guard = [], threading.Lock()

    def rec(e):
        with guard:
            events.append(e)

    class SlowProc:
        def __init__(self, model, port):
            self.model, self.port, self.parked = model, port, False

        @property
        def url(self):
            return f"http://127.0.0.1:{self.port}"

        def start(self):
            rec("start_begin"); _t.sleep(0.15); rec("start_end"); return 1.0

        def park(self):
            rec("park_begin"); _t.sleep(0.15); rec("park_end"); self.parked = True

        def stop(self):
            pass

    procs = {}

    def make(m, port, g):
        p = SlowProc(m, port); procs[(m, tuple(g))] = p; return p

    gl = GpuLauncher(make_process=make)
    gl.launch("B", ["gpu0"])                       # B running on gpu0
    t1 = threading.Thread(target=lambda: gl.launch("A", ["gpu0"]))   # cold-load A
    t2 = threading.Thread(target=lambda: gl.deactivate("B", ["gpu0"]))  # park B
    t1.start(); t2.start(); t1.join(); t2.join()

    sb, se = events.index("start_begin"), events.index("start_end")
    pb, pe = events.index("park_begin"), events.index("park_end")
    assert se < pb or pe < sb, f"GPU ops overlapped: {events}"
