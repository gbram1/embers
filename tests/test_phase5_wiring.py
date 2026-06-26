"""Phase 5↔4 integration: the ColdStartLoader IS the autoscaler's launch
callback. The whole project's payoff — scale-to-zero frees the GPU but keeps the
snapshot, so the *next* cold start takes the fast restore path, not a full reload.

GPU ops are faked (return mock serving units); on a pod they become the
cuda-checkpoint capture/restore from scripts/_rung2_*."""

from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.fingerprint import Fingerprint
from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.loader import ColdStartLoader
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit
from tests.test_autoscaler import FakeClock


def fixed_fp(model, gpu):
    return Fingerprint(
        weights_hash=f"hash-{model}", model_version=model,
        engine_version="vllm-0.8.5", gpu_type="NVIDIA A40",
        driver_cuda_version="580", dtype="bf16", max_seq_len=4096,
        tensor_parallel=1, captured_batch_shapes=(1, 2, 4),
    )


def backend(model: str) -> LocalBackend:
    u = ModelUnit(model, mock=True)
    u.load()
    return LocalBackend(u)


def build():
    clock = FakeClock()
    sched = Scheduler([GPU("g0", 24000), GPU("g1", 24000)])
    router = Router()

    # GPU ops faked: cold_load and restore both yield a ready mock backend.
    loader = ColdStartLoader(
        fingerprint_fn=fixed_fp,
        cold_load=lambda m, g: backend(m),
        capture=lambda m, g, b: f"snapshot://{m}",
        restore=lambda m, g, snap: backend(m),
    )
    # THE WIRING: the loader's launch is the autoscaler's launch callback.
    auto = Autoscaler(sched, router, launch=loader.launch, clock=clock)
    gw = TestClient(create_gateway_app(router))
    return auto, loader, clock, sched, gw


def test_first_cold_start_is_slow_path():
    auto, loader, clock, sched, gw = build()
    auto.register_model("m", 6000, idle_ttl=300)
    auto.handle_request("m")                 # scale from zero
    assert loader.cold_loads == 1            # took the SLOW path (no snapshot yet)
    assert loader.restores == 0
    assert gw.post("/v1/completions",
                   json={"model": "m", "prompt": "x"}).status_code == 200


def test_scale_to_zero_keeps_snapshot_then_fast_restore():
    auto, loader, clock, sched, gw = build()
    auto.register_model("m", 6000, idle_ttl=300)

    auto.handle_request("m")                 # 1st cold start → slow path + capture
    assert loader.cold_loads == 1

    clock.advance(301)
    auto.tick()                              # idle → scale to zero, GPU freed
    assert auto.state()["m"] == 0
    assert sched.total_free_mb() == 48000    # GPU reclaimed
    assert loader.store.get("m") is not None  # but the snapshot survives

    auto.handle_request("m")                 # 2nd cold start → FAST restore
    assert loader.restores == 1
    assert loader.cold_loads == 1            # no second full reload
    assert gw.post("/v1/completions",
                   json={"model": "m", "prompt": "again"}).status_code == 200


def test_many_scale_cycles_only_one_cold_load():
    auto, loader, clock, sched, gw = build()
    auto.register_model("m", 6000, idle_ttl=300)
    for _ in range(5):
        auto.handle_request("m")
        clock.advance(301)
        auto.tick()
    # exactly one true cold load; every subsequent spin-up was a fast restore
    assert loader.cold_loads == 1
    assert loader.restores == 4
