"""Learn the real memory cost of a district while the run is in progress.

A static ``memory_per_input_byte`` has to be pessimistic, because shapefile
size alone does not predict peak RSS: on the 2025 data the growth term ranged
from 0.05x the input (CHINIOT, simple rings) to 6.7x (BADIN, very intricate
ones). A pessimistic constant is safe but wastes the machine - it reserves RAM
that is never touched and so admits fewer districts than the box can run.

Every finished district reports its actual peak RSS, so the scheduler can
replace the guess with a measurement. This class keeps the running worst-case
ratio and hands back an estimate that is the *larger* of the measured
worst case (with a safety margin) and a floor, so the estimate can tighten as
evidence arrives but never collapses to an unsafe value on a lucky sample.
"""

from __future__ import annotations

import logging
import threading

from .resources import WORKER_BASELINE_BYTES

log = logging.getLogger(__name__)

#: Multiply the worst measured ratio by this before trusting it.
SAFETY_MARGIN = 1.6

#: Never trust a measured ratio below this - a handful of simple districts
#: must not convince us that a complex one will also be cheap.
MIN_RATIO = 1.0

#: Districts that must finish before measurements outrank the configured value.
MIN_SAMPLES = 5

#: Ignore districts smaller than this when fitting the ratio.
#:
#: Peak RSS for a small district is almost entirely the fixed worker baseline,
#: so dividing the leftover by a tiny input produces a huge, meaningless ratio.
#: On the 2025 run a 17 MB district measured 17x purely from baseline noise,
#: which then had the 635 MB district reserving 16.8 GiB it could never use.
#: Only districts large enough for the growth term to dominate are informative.
MIN_INFORMATIVE_BYTES = 50 * 1024**2


class MemoryCalibrator:
    """Thread-safe running estimate of peak RSS per byte of input."""

    def __init__(self, initial_ratio: float, baseline: int = WORKER_BASELINE_BYTES):
        self._configured = float(initial_ratio)
        self._baseline = int(baseline)
        self._worst = 0.0
        self._samples = 0
        self._lock = threading.Lock()

    # -- reporting ------------------------------------------------------
    def observe(self, input_bytes: int, peak_rss_bytes: float) -> None:
        """Record what a district actually cost.

        Districts below :data:`MIN_INFORMATIVE_BYTES` are counted but not
        fitted: their peak is baseline, not data, and the ratio they imply is
        noise that would be extrapolated onto the largest district in the run.
        """
        if input_bytes <= 0 or peak_rss_bytes <= 0:
            return
        if input_bytes < MIN_INFORMATIVE_BYTES:
            return
        growth = max(0.0, peak_rss_bytes - self._baseline)
        ratio = growth / input_bytes
        with self._lock:
            self._samples += 1
            if ratio > self._worst:
                previous, self._worst = self._worst, ratio
                if self._samples > MIN_SAMPLES and ratio > self._configured:
                    log.info(
                        "memory calibration: worst observed ratio rose %.2f -> %.2f",
                        previous, ratio,
                    )

    # -- consuming ------------------------------------------------------
    @property
    def ratio(self) -> float:
        """The ratio the scheduler should budget with right now."""
        with self._lock:
            if self._samples < MIN_SAMPLES:
                return self._configured
            measured = max(MIN_RATIO, self._worst * SAFETY_MARGIN)
            # Evidence may tighten the estimate or loosen it, but a measured
            # value above the configured one always wins: that is real data
            # saying the configured number was too optimistic.
            return measured if measured > self._configured else max(MIN_RATIO, measured)

    def estimate(self, input_bytes: int) -> int:
        return self._baseline + int(input_bytes * self.ratio)

    def summary(self) -> str:
        with self._lock:
            if self._samples < MIN_SAMPLES:
                return f"configured {self._configured:.1f}x ({self._samples} sample(s))"
            return (
                f"{self.ratio:.2f}x from {self._samples} sample(s) "
                f"(worst measured {self._worst:.2f}x, configured {self._configured:.1f}x)"
            )
