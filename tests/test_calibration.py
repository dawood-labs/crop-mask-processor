from cropmask.calibration import (
    MIN_INFORMATIVE_BYTES,
    MIN_RATIO,
    SAFETY_MARGIN,
    MemoryCalibrator,
)
from cropmask.resources import WORKER_BASELINE_BYTES

MB = 1024**2
BIG = MIN_INFORMATIVE_BYTES * 2


def test_configured_value_is_used_until_enough_samples():
    cal = MemoryCalibrator(initial_ratio=8.0)
    cal.observe(BIG, WORKER_BASELINE_BYTES + BIG)  # ratio 1.0
    assert cal.ratio == 8.0


def test_estimate_tightens_once_measurements_arrive():
    cal = MemoryCalibrator(initial_ratio=8.0)
    for _ in range(6):
        cal.observe(BIG, WORKER_BASELINE_BYTES + 2 * BIG)  # ratio 2.0
    assert cal.ratio == 2.0 * SAFETY_MARGIN
    assert cal.ratio < 8.0  # the machine can now run more districts at once


def test_a_single_expensive_district_raises_the_estimate_for_everyone():
    cal = MemoryCalibrator(initial_ratio=2.0)
    for _ in range(6):
        cal.observe(BIG, WORKER_BASELINE_BYTES + BIG // 2)   # cheap, ratio 0.5
    cheap = cal.ratio
    cal.observe(BIG, WORKER_BASELINE_BYTES + 9 * BIG)            # ratio 9.0
    assert cal.ratio > cheap
    assert cal.ratio == 9.0 * SAFETY_MARGIN


def test_estimate_never_drops_below_the_floor():
    cal = MemoryCalibrator(initial_ratio=8.0)
    for _ in range(10):
        cal.observe(BIG, WORKER_BASELINE_BYTES + 1)  # essentially free
    assert cal.ratio == MIN_RATIO


def test_estimate_always_includes_the_worker_baseline():
    cal = MemoryCalibrator(initial_ratio=1.0)
    assert cal.estimate(0) == WORKER_BASELINE_BYTES
    assert cal.estimate(10 * MB) == WORKER_BASELINE_BYTES + 10 * MB


def test_bad_observations_are_ignored():
    cal = MemoryCalibrator(initial_ratio=3.0)
    for _ in range(10):
        cal.observe(0, 123)
        cal.observe(BIG, 0)
    assert cal.ratio == 3.0


def test_small_districts_do_not_skew_the_ratio():
    """A tiny district's peak is baseline, not data.

    Regression: on the 2025 run a 17 MB district measured 17x purely from
    baseline noise, which then had the 635 MB district reserving 16.8 GiB.
    """
    cal = MemoryCalibrator(initial_ratio=2.0)
    for _ in range(10):
        # 1 MB of input, 400 MB of interpreter: an apparent ratio in the hundreds.
        cal.observe(1 * MB, WORKER_BASELINE_BYTES + 300 * MB)
    assert cal.ratio == 2.0
    assert cal.estimate(600 * MB) < WORKER_BASELINE_BYTES + 2.5 * 600 * MB
