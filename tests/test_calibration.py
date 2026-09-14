from cropmask.calibration import MIN_RATIO, SAFETY_MARGIN, MemoryCalibrator
from cropmask.resources import WORKER_BASELINE_BYTES

MB = 1024**2


def test_configured_value_is_used_until_enough_samples():
    cal = MemoryCalibrator(initial_ratio=8.0)
    cal.observe(100 * MB, WORKER_BASELINE_BYTES + 100 * MB)  # ratio 1.0
    assert cal.ratio == 8.0


def test_estimate_tightens_once_measurements_arrive():
    cal = MemoryCalibrator(initial_ratio=8.0)
    for _ in range(6):
        cal.observe(100 * MB, WORKER_BASELINE_BYTES + 200 * MB)  # ratio 2.0
    assert cal.ratio == 2.0 * SAFETY_MARGIN
    assert cal.ratio < 8.0  # the machine can now run more districts at once


def test_a_single_expensive_district_raises_the_estimate_for_everyone():
    cal = MemoryCalibrator(initial_ratio=2.0)
    for _ in range(6):
        cal.observe(100 * MB, WORKER_BASELINE_BYTES + 50 * MB)   # cheap, ratio 0.5
    cheap = cal.ratio
    cal.observe(100 * MB, WORKER_BASELINE_BYTES + 900 * MB)      # ratio 9.0
    assert cal.ratio > cheap
    assert cal.ratio == 9.0 * SAFETY_MARGIN


def test_estimate_never_drops_below_the_floor():
    cal = MemoryCalibrator(initial_ratio=8.0)
    for _ in range(10):
        cal.observe(100 * MB, WORKER_BASELINE_BYTES + 1)  # essentially free
    assert cal.ratio == MIN_RATIO


def test_estimate_always_includes_the_worker_baseline():
    cal = MemoryCalibrator(initial_ratio=1.0)
    assert cal.estimate(0) == WORKER_BASELINE_BYTES
    assert cal.estimate(10 * MB) == WORKER_BASELINE_BYTES + 10 * MB


def test_bad_observations_are_ignored():
    cal = MemoryCalibrator(initial_ratio=3.0)
    for _ in range(10):
        cal.observe(0, 123)
        cal.observe(100 * MB, 0)
    assert cal.ratio == 3.0
