"""Machine-aware sizing: how many workers, and how much RAM each may use.

The pipeline must never be OOM-killed, so every decision here is deliberately
conservative. The two inputs we trust are the number of usable CPUs and the
amount of *available* (not total) RAM, both read at runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import psutil

log = logging.getLogger(__name__)

GIB = 1024**3


def usable_cpus() -> int:
    """CPU count that respects cgroup quotas and CPU affinity.

    A container limited to 2 CPUs still reports the host's core count through
    ``os.cpu_count()``, which would make us oversubscribe badly.
    """
    counts = []

    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        counts.append(os.cpu_count() or 1)

    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as fh:
            quota, period = fh.read().split()
            if quota != "max":
                counts.append(max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            counts.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    return max(1, min(counts))


def available_memory() -> int:
    """Bytes of RAM we may actually use, honouring a cgroup memory limit."""
    avail = psutil.virtual_memory().available

    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw == "max":
                continue
            limit = int(raw)
            with open(usage_path, "r", encoding="utf-8") as fh:
                usage = int(fh.read().strip())
            # Absurd sentinel values mean "unlimited" on cgroup v1.
            if 0 < limit < (1 << 62):
                avail = min(avail, max(0, limit - usage))
        except (OSError, ValueError):
            continue

    return avail


@dataclass
class ResourcePlan:
    cpus: int
    available_bytes: int
    budget_bytes: int
    workers: int
    bytes_per_worker: int

    def describe(self) -> str:
        return (
            f"{self.cpus} CPU / {self.available_bytes / GIB:.1f} GiB available -> "
            f"{self.workers} worker(s), budget {self.budget_bytes / GIB:.1f} GiB "
            f"({self.bytes_per_worker / GIB:.1f} GiB each)"
        )


def plan_resources(
    requested_workers: int | None,
    memory_fraction: float,
    min_worker_memory: int,
    cpu_oversubscribe: float = 1.25,
) -> ResourcePlan:
    """Decide the worker count from CPU *and* RAM, whichever binds first.

    Workers are not CPU-bound for their whole life - each one spends time
    downloading its district from GCS and uploading the results, during which
    its core is idle. ``cpu_oversubscribe`` slightly overcommits the cores so
    that window is filled by another district instead of being wasted. Actual
    concurrency is still capped by the memory budget in the scheduler, so
    overcommitting cores cannot overcommit RAM.
    """
    cpus = usable_cpus()
    avail = available_memory()
    budget = int(avail * memory_fraction)

    cpu_cap = max(1, int(cpus * cpu_oversubscribe)) if cpus > 1 else 1
    mem_cap = max(1, budget // min_worker_memory)

    workers = min(cpu_cap, mem_cap)
    if requested_workers:
        if requested_workers > cpu_cap:
            log.warning(
                "workers=%d requested but only %d usable CPU(s); oversubscribing",
                requested_workers, cpus,
            )
        if requested_workers > mem_cap:
            log.warning(
                "workers=%d requested but RAM budget only supports %d; "
                "raise memory_fraction or accept the OOM risk",
                requested_workers, mem_cap,
            )
        workers = requested_workers

    workers = max(1, workers)
    plan = ResourcePlan(
        cpus=cpus,
        available_bytes=avail,
        budget_bytes=budget,
        workers=workers,
        bytes_per_worker=budget // workers,
    )
    log.info("Resource plan: %s", plan.describe())
    return plan


#: Roughly what a worker process costs before it touches any data: the Python
#: interpreter plus GEOS/PROJ/GDAL and the boundary layer. Measured at ~250 MB
#: on the 2025 runs; rounded up.
WORKER_BASELINE_BYTES = 400 * 1024**2


def estimate_district_memory(input_bytes: int, factor: float) -> int:
    """Peak RAM a district is expected to need, in bytes.

    Peak RSS is a fixed per-worker baseline plus a term that scales with the
    input. Measured on the 2025 data, the growth term ranged from ~0.05x the
    input size (CHINIOT, simple geometry) up to ~6.7x (BADIN, very intricate
    rings), so ``factor`` defaults well above the worst case observed.
    """
    return WORKER_BASELINE_BYTES + int(input_bytes * factor)


def total_memory() -> int:
    """Bytes of RAM this process may use in total, honouring a cgroup limit."""
    total = psutil.virtual_memory().total
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw == "max":
                continue
            limit = int(raw)
            if 0 < limit < (1 << 62):
                total = min(total, limit)
        except (OSError, ValueError):
            continue
    return total


def memory_pressure() -> float:
    """Fraction of usable memory currently in use (0.0 - 1.0).

    This must measure the same pool the budget is drawn from. Reading the
    host's totals inside a container makes the admission ceiling dead in
    exactly the deployment that needs it: a 48 GiB cgroup on a 512 GiB host
    can be at its limit while host-wide pressure reads 8%.
    """
    total = total_memory()
    if total <= 0:
        return 0.0
    return 1.0 - (available_memory() / total)
