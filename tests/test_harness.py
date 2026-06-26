"""GPU-free unit tests for the benchmark harness logic.

Covers the pure functions — percentile math and the vLLM weight-load log
parser — so the orchestration can be trusted before it ever touches a GPU.
Run: pytest (or scripts/test.sh).
"""

from bench.harness import parse_phases, pct


class TestPct:
    def test_p50_is_median(self):
        assert pct([1, 2, 3, 4, 5], 50) == 3

    def test_p0_and_p100_are_min_max(self):
        xs = [5, 1, 3, 2, 4]
        assert pct(xs, 0) == 1
        assert pct(xs, 100) == 5

    def test_interpolates_between_points(self):
        # p90 of 0..9 falls between 8 and 9.
        assert pct(list(range(10)), 90) == 8.1

    def test_unsorted_input_ok(self):
        assert pct([9, 0, 5], 50) == 5

    def test_single_element(self):
        assert pct([42.0], 50) == 42.0
        assert pct([42.0], 99) == 42.0

    def test_two_elements_interpolate(self):
        assert pct([0.0, 10.0], 50) == 5.0


class TestParsePhases:
    # Real vLLM 0.8.5 log lines from the A4000 baseline run.
    REAL_LOG = (
        "INFO 06-19 15:27:21 [gpu_model_runner.py:1347] Model loading took "
        "5.7916 GiB and 22.924035 seconds\n"
        "INFO 06-19 15:28:40 [core.py:159] init engine (profile, create kv "
        "cache, warmup model) took 79.20 seconds\n"
    )

    def test_extracts_both_phases(self):
        wl, ei = parse_phases(self.REAL_LOG)
        assert wl == 22.924035
        assert ei == 79.20

    def test_gb_variant(self):
        wl, _ = parse_phases("Model loading took 14.99 GB and 18.42 seconds")
        assert wl == 18.42

    def test_last_match_wins(self):
        log = (
            "Model loading took 1.0 GiB and 9.99 seconds\n"
            "Model loading took 1.0 GiB and 12.50 seconds\n"
        )
        wl, _ = parse_phases(log)
        assert wl == 12.50

    def test_missing_lines_return_none(self):
        assert parse_phases("nothing relevant here") == (None, None)

    def test_weight_only_when_init_absent(self):
        wl, ei = parse_phases("Model loading took 5.0 GiB and 2.0 seconds")
        assert wl == 2.0 and ei is None

    def test_init_only_when_weight_absent(self):
        wl, ei = parse_phases("init engine (...) took 79.20 seconds")
        assert wl is None and ei == 79.20

    def test_case_insensitive(self):
        wl, _ = parse_phases("MODEL LOADING TOOK 5.0 GIB AND 3.0 SECONDS")
        assert wl == 3.0
