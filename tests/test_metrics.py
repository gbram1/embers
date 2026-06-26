"""Tests for Phase 6 metrics primitives, platform snapshot, and dashboard."""

from embers.metrics import (
    Counter,
    Gauge,
    Histogram,
    Registry,
    platform_snapshot,
    render_dashboard,
)


# --- primitives ------------------------------------------------------------

def test_counter_accumulates_per_label():
    c = Counter("reqs")
    c.inc(model="a")
    c.inc(2, model="a")
    c.inc(model="b")
    assert c.value(model="a") == 3
    assert c.value(model="b") == 1
    assert c.value(model="missing") == 0


def test_gauge_sets_latest():
    g = Gauge("util")
    g.set(10, gpu="g0")
    g.set(20, gpu="g0")
    assert g.value(gpu="g0") == 20


def test_histogram_percentiles():
    h = Histogram("lat")
    for v in range(1, 101):           # 1..100
        h.observe(v, model="m")
    s = h.stats(model="m")
    assert s["count"] == 100
    assert s["sum"] == 5050
    assert s["p50"] == 50.5
    assert 90 <= s["p90"] <= 91


def test_histogram_empty_is_zeroed():
    assert Histogram("lat").stats(model="none")["count"] == 0


def test_registry_get_or_create_is_stable():
    r = Registry()
    assert r.counter("x") is r.counter("x")     # same instance reused


def test_prometheus_render_has_type_and_samples():
    r = Registry()
    r.counter("embers_requests_total", "reqs").inc(model="a")
    r.histogram("lat").observe(0.2, model="a")
    text = r.render_prometheus()
    assert "# TYPE embers_requests_total counter" in text
    assert 'embers_requests_total{model="a"} 1' in text
    assert 'quantile="0.5"' in text
    assert "lat_count" in text


# --- platform snapshot & dashboard ----------------------------------------

class FakeSched:
    def gpu_state(self):
        return [("g0", 6000, 24000), ("g1", 0, 24000)]


class FakeAuto:
    cold_starts = 5
    scale_ups = 7
    scale_downs = 4
    scaled_to_zero = 3

    def state(self):
        return {"m": 1}


class FakeLoader:
    cold_loads = 1
    restores = 4
    invalidations = 1


def test_snapshot_aggregates_all_organs():
    snap = platform_snapshot(FakeSched(), FakeAuto(), FakeLoader())
    assert snap["gpus"][0]["util_pct"] == 25
    assert snap["cluster_util_pct"] == 12          # 6000/48000
    assert snap["replicas"] == {"m": 1}
    assert snap["scaling"]["scaled_to_zero"] == 3
    assert snap["embers"]["snapshot_hit_rate"] == 0.8   # 4/(4+1)


def test_snapshot_handles_missing_components():
    assert platform_snapshot() == {}
    only = platform_snapshot(loader=FakeLoader())
    assert "embers" in only and "gpus" not in only


def test_snapshot_hit_rate_zero_when_no_launches():
    class Empty:
        cold_loads = restores = invalidations = 0
    assert platform_snapshot(loader=Empty())["embers"]["snapshot_hit_rate"] == 0.0


def test_render_dashboard_shows_key_numbers():
    out = render_dashboard(platform_snapshot(FakeSched(), FakeAuto(), FakeLoader()))
    assert "cluster utilisation: 12%" in out
    assert "80% hit-rate" in out
    assert "scaled-to-zero" in out
