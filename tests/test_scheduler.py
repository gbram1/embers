"""Unit tests for the Phase 3 scheduler — GPU accounting, bin-packing policies,
placement, capacity, rollback, eviction, replica spreading."""

import pytest

from embers.scheduler import (
    GPU,
    NoCapacity,
    Scheduler,
    best_fit,
    first_fit,
    worst_fit,
)


# --- GPU accounting --------------------------------------------------------

def test_gpu_free_used_fits_runs():
    g = GPU("g0", 1000)
    assert g.free_mb == 1000 and g.used_mb == 0
    g.placed["m"] = 600
    assert g.used_mb == 600 and g.free_mb == 400
    assert g.fits(400) and not g.fits(401)
    assert g.runs("m") and not g.runs("other")


# --- policies --------------------------------------------------------------

def gpus():
    a, b, c = GPU("a", 1000), GPU("b", 1000), GPU("c", 1000)
    a.placed["x"] = 200   # free 800
    b.placed["x"] = 700   # free 300
    c.placed["x"] = 500   # free 500
    return [a, b, c]

def test_first_fit_takes_first_that_fits():
    assert first_fit(gpus(), 300, "new").id == "a"

def test_best_fit_takes_tightest():
    # need 300 → free: a800 b300 c500 → tightest is b (300)
    assert best_fit(gpus(), 300, "new").id == "b"

def test_worst_fit_takes_loosest():
    assert worst_fit(gpus(), 300, "new").id == "a"  # most free

def test_policy_returns_none_when_nothing_fits():
    assert best_fit(gpus(), 900, "new") is None

def test_policy_skips_gpu_already_running_model():
    # 'x' runs on all three → placing another 'x' replica finds nothing
    assert first_fit(gpus(), 100, "x") is None


# --- placement & capacity --------------------------------------------------

def test_place_records_and_consumes():
    s = Scheduler([GPU("g0", 1000)])
    [p] = s.place("m", 400)
    assert p.gpu_id == "g0"
    assert s.gpu_state() == [("g0", 400, 1000)]
    assert s.total_free_mb() == 600

def test_place_raises_when_no_fit():
    s = Scheduler([GPU("g0", 1000)])
    s.place("big", 800)
    with pytest.raises(NoCapacity):
        s.place("another", 800)

def test_best_fit_packs_densely():
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)], policy="best-fit")
    s.place("m1", 600)            # a: 600 used (free 400), b free 1000
    [p] = s.place("m2", 300)     # best-fit → a (free 400, tightest) not b
    assert p.gpu_id == "a"

def test_replicas_spread_across_distinct_gpus():
    s = Scheduler([GPU("a", 1000), GPU("b", 1000), GPU("c", 1000)])
    placements = s.place("m", 300, replicas=3)
    assert sorted(p.gpu_id for p in placements) == ["a", "b", "c"]

def test_replicas_rollback_on_partial_failure():
    # only 2 GPUs can host but 3 replicas requested → all-or-nothing
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)])
    with pytest.raises(NoCapacity):
        s.place("m", 300, replicas=3)
    # rollback: nothing placed, full capacity restored
    assert s.placements() == []
    assert s.total_free_mb() == 2000


# --- multi-model packing (policy behaviour under contention) ---------------

def test_best_fit_colocates_onto_partially_used_gpu_before_spilling():
    # best-fit should densely pack: a second model lands on the GPU a first model
    # already partly fills (tightest), keeping the empty GPU free for big jobs.
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)], policy="best-fit")
    s.place("m1", 600)               # a: free 400; b: free 1000
    [p] = s.place("m2", 300)         # tightest fit that holds 300 → a (400), not b
    assert p.gpu_id == "a"
    assert s.total_free_mb() == 100 + 1000   # a packed to 900; b still empty

def test_worst_fit_spreads_models_across_empty_gpus():
    # worst-fit should spread: each new model goes to the emptiest GPU.
    s = Scheduler([GPU("a", 1000), GPU("b", 1000), GPU("c", 1000)],
                  policy="worst-fit")
    assert s.place("m1", 200)[0].gpu_id == "a"   # all equal → first
    assert s.place("m2", 200)[0].gpu_id == "b"   # a now 800, b/c 1000 → b
    assert s.place("m3", 200)[0].gpu_id == "c"   # c is the loosest remaining
    # three models, three GPUs — maximally spread
    assert {g.id for g in s.gpus if g.placed} == {"a", "b", "c"}

def test_distinct_models_exhaust_shared_vram():
    # two models packed onto the same GPUs until VRAM runs out — the third
    # placement has nowhere to fit even though every GPU still runs something.
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)], policy="best-fit")
    s.place("m1", 800, replicas=2)   # fills a and b to 800 each (distinct GPUs)
    with pytest.raises(NoCapacity):
        s.place("m2", 300)           # 200 free on each GPU → 300 fits nowhere
    assert s.total_free_mb() == 400  # accounting intact after the failed place

def test_best_fit_multi_replica_packs_tightest_distinct_gpus():
    # 2 replicas of a model, best-fit, across GPUs of differing fullness: each
    # replica takes the tightest *distinct* GPU that still fits.
    s = Scheduler([GPU("a", 1000), GPU("b", 1000), GPU("c", 1000)],
                  policy="best-fit")
    s.place("filler", 700)                       # a: free 300; b,c: free 1000
    ps = s.place("m", 300, replicas=2)           # r1→a (tightest 300), r2→b
    assert sorted(p.gpu_id for p in ps) == ["a", "b"]
    assert len({p.gpu_id for p in ps}) == 2      # distinct GPUs


# --- eviction --------------------------------------------------------------

def test_evict_frees_capacity_and_allows_replace():
    s = Scheduler([GPU("g0", 1000)])
    s.place("m", 800)
    with pytest.raises(NoCapacity):
        s.place("n", 800)
    removed = s.evict("m")
    assert removed[0].model == "m"
    assert s.total_free_mb() == 1000
    s.place("n", 800)            # now fits
    assert s.placements()[0].model == "n"

def test_evict_specific_gpu_only():
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)])
    s.place("m", 300, replicas=2)        # on a and b
    s.evict("m", gpu_id="a")
    running = {p.gpu_id for p in s.placements()}
    assert running == {"b"}


# --- gateway integration ---------------------------------------------------

def test_to_router_registers_a_backend_per_placement():
    s = Scheduler([GPU("a", 1000), GPU("b", 1000)])
    s.place("m", 300, replicas=2)

    class Stub:
        def __init__(self, model, gpu):
            self.name = model
            self.gpu = gpu

    router = s.to_router(lambda model, gpu: Stub(model, gpu))
    assert router.models() == ["m"]
    assert len(router.replicas("m")) == 2
    assert {b.gpu for b in router.replicas("m")} == {"a", "b"}
