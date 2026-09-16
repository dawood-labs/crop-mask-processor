"""Make a slow district explain itself.

Pathological geometry does not announce itself. On the 2017 data a single
218,000-acre sugarcane polygon turned one district's cotton erase from seconds
into an estimated 104 minutes, and the only visible symptom was a run that sat
at 65/66 with one core busy and no log output at all.

Two things here stop that from being silent again:

* ``timed`` wraps an individual GEOS operation and, when it crosses
  ``SLOW_SECONDS``, logs which district and stage it belonged to and how many
  vertices were involved. The vertex counts are only computed on that slow path,
  so the cost on the normal path is one clock read.
* ``StageClock`` accumulates wall time per pipeline stage so the report can
  show where each district's time actually went.

A worker process runs one district at a time, so the context is a plain module
global rather than a contextvar: it must be visible from the helper threads a
district starts, and contextvars are not copied into executor threads.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import numpy as np
import shapely

log = logging.getLogger(__name__)

#: A single GEOS call slower than this is logged with its inputs.
SLOW_SECONDS = 5.0

_context = ""


def set_context(text: str) -> None:
    global _context
    _context = text


def context() -> str:
    return _context


def _vertices(geom) -> int:
    try:
        if isinstance(geom, np.ndarray):
            return int(shapely.get_num_coordinates(geom).sum()) if len(geom) else 0
        return int(shapely.get_num_coordinates(geom))
    except Exception:
        return -1


@contextmanager
def timed(operation: str, **inputs):
    """Log the operation, and the size of its inputs, if it runs long."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        if elapsed >= SLOW_SECONDS:
            sizes = ", ".join(f"{name} {_vertices(g):,} verts" for name, g in inputs.items())
            log.warning("SLOW %s took %.1fs in %s (%s)",
                        operation, elapsed, _context or "?", sizes)


class StageClock:
    """Wall-clock seconds per named stage, for the report."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - started)
