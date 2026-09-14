"""Orchestration: plan the work, run it in parallel, stay inside the RAM budget.

Districts are embarrassingly parallel, but they are also wildly uneven - the
2025 data ranges from an empty 100-byte placeholder to a 635 MB district. Handing
that to a fixed-size pool either wastes cores on the small ones or runs several
huge ones at once and gets the process OOM-killed.

So admission is driven by an explicit memory budget rather than by a worker
count alone: a district is only submitted when its estimated peak fits in the
RAM still unclaimed by the districts already running. Big districts are
scheduled first (worst-fit-decreasing), which keeps the tail short, and a
district too large to share the machine simply runs on its own.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

from . import gcs
from .calibration import MemoryCalibrator
from .config import Config
from .discovery import DistrictTask, build_index, select_tasks
from .io_layers import read_boundaries
from .pipeline import DistrictResult
from .state import filter_pending, load_completed
from .resources import (
    GIB,
    ResourcePlan,
    estimate_district_memory,
    memory_pressure,
    plan_resources,
)
from .worker import init_worker, run_district

log = logging.getLogger(__name__)

#: Stop admitting new districts above this system-wide memory usage.
PRESSURE_CEILING = 0.88


@dataclass
class RunOutcome:
    results: list[DistrictResult]
    anomalies: list[dict]
    plan: ResourcePlan
    seconds: float
    #: Districts skipped because a previous run already finished them.
    resumed: list = field(default_factory=list)

    @property
    def records(self) -> list[dict]:
        """Report rows for this run plus everything carried over from before."""
        rows = [r for res in self.results for r in res.records]
        rows += [r for done in self.resumed for r in done.records]
        return rows

    @property
    def timings(self) -> list[dict]:
        rows = [
            {
                "province": r.province,
                "district": r.district,
                "seconds": round(r.seconds, 1),
                "peak_rss_mb": round(r.peak_rss_mb, 1),
                "outputs": len(r.outputs),
                "error": r.error or "",
                "source": "this run",
            }
            for r in self.results
        ]
        rows += [
            {
                "province": d.province,
                "district": d.district,
                "seconds": round(d.seconds, 1),
                "peak_rss_mb": round(d.peak_rss_mb, 1),
                "outputs": "",
                "error": "",
                "source": "resumed",
            }
            for d in self.resumed
        ]
        return rows


def prepare(cfg: Config) -> tuple[list[DistrictTask], list[dict], Path]:
    """List the inputs, resolve the boundary and build the district work list."""
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info("Staging boundary shapefile ...")
    boundary_local = gcs.fetch_to_local(
        cfg.boundary_uri, work_dir / "boundary", cfg.credentials_json
    )
    _, lookup = read_boundaries(boundary_local, cfg.boundary_field, cfg.metric_crs)
    log.info("Boundary: %d unique district names", len(lookup))

    log.info("Listing %s ...", cfg.input_uri)
    blobs = gcs.list_blobs(cfg.input_uri, cfg.credentials_json)
    log.info("Found %d objects", len(blobs))

    root_prefix = gcs.GcsPath.parse(cfg.input_uri).prefix
    tasks_by_key, anomalies = build_index(blobs, root_prefix, set(lookup))
    log.info("Resolved %d district folders", len(tasks_by_key))

    tasks = select_tasks(tasks_by_key, cfg.only_provinces, cfg.only_districts, cfg.limit)
    total_gb = sum(t.size_bytes for t in tasks) / GIB
    log.info("Selected %d district(s), %.2f GiB of input", len(tasks), total_gb)

    return tasks, anomalies, boundary_local


def run(cfg: Config) -> RunOutcome:
    started = time.time()
    tasks, anomalies, boundary_local = prepare(cfg)

    resumed: list = []
    if not cfg.overwrite:
        completed = load_completed(cfg.output_uri, cfg.credentials_json)
        if completed:
            tasks, resumed = filter_pending(tasks, completed)
            log.info(
                "Resuming: %d district(s) already finished, %d left to do "
                "(pass --overwrite to redo everything)",
                len(resumed), len(tasks),
            )

    if not tasks and resumed:
        log.info("Every selected district was already processed.")
        empty_plan = plan_resources(
            cfg.workers, cfg.memory_fraction, cfg.min_worker_memory,
            cfg.cpu_oversubscribe,
        )
        return RunOutcome([], anomalies, empty_plan, time.time() - started, resumed)

    if not tasks:
        log.warning("Nothing to process.")
        empty_plan = plan_resources(
            cfg.workers, cfg.memory_fraction, cfg.min_worker_memory,
            cfg.cpu_oversubscribe,
        )
        return RunOutcome([], anomalies, empty_plan, 0.0, resumed)

    plan = plan_resources(
        cfg.workers, cfg.memory_fraction, cfg.min_worker_memory, cfg.cpu_oversubscribe
    )

    # Largest first: the heavy districts get their memory reservation while the
    # budget is still empty, and the run does not end with one giant district
    # holding a single core while every other one idles.
    ordered = sorted(tasks, key=lambda t: t.size_bytes, reverse=True)
    calibrator = MemoryCalibrator(cfg.memory_per_input_byte)
    log.info(
        "Largest district %.2f GiB input -> %.1f GiB estimated peak",
        ordered[0].size_bytes / GIB,
        estimate_district_memory(ordered[0].size_bytes, cfg.memory_per_input_byte) / GIB,
    )

    results = _execute(cfg, ordered, calibrator, plan, boundary_local)
    log.info("Memory model at end of run: %s", calibrator.summary())

    elapsed = time.time() - started
    return RunOutcome(results, anomalies, plan, elapsed, resumed)


def _execute(
    cfg: Config,
    ordered: list[DistrictTask],
    calibrator: MemoryCalibrator,
    plan: ResourcePlan,
    boundary_local: Path,
) -> list[DistrictResult]:
    if plan.workers == 1:
        return _execute_serial(cfg, ordered, boundary_local)

    pending = list(ordered)
    results: list[DistrictResult] = []
    workers = plan.workers
    breaks = 0

    while True:
        try:
            # _pool_loop appends into `results`; it returns nothing, because a
            # returned list would be lost when BrokenProcessPool unwinds out of
            # it - which previously made the retry restart the whole run.
            _pool_loop(
                cfg, pending, calibrator, plan, workers, boundary_local, results
            )
            return results
        except BrokenProcessPool:
            breaks += 1
            # A worker died outright - almost always the kernel OOM killer.
            # Halve the pool and pick up whatever is left.
            done = {(r.province, r.district) for r in results}
            before = len(pending)
            pending = [t for t in pending if t.key not in done]
            progressed = len(pending) < before

            if not progressed and breaks >= 3:
                # Three breaks with nothing completing in between is not memory
                # pressure - the workers are failing at start-up (a bad import,
                # a missing dependency). Shrinking the pool cannot fix that, and
                # retrying just burns time.
                log.error(
                    "Worker pool broke %d times without completing a district - "
                    "workers are failing to start. Abandoning %d district(s).",
                    breaks, len(pending),
                )
                results += [_crash_result(t, "worker failed to start") for t in pending]
                return results

            if workers <= 1:
                log.error(
                    "Worker pool broke with a single worker; %d district(s) abandoned",
                    len(pending),
                )
                results += [
                    _crash_result(t, "worker process died (out of memory?)")
                    for t in pending
                ]
                return results
            workers = max(1, workers // 2)
            log.error(
                "Worker pool broke - %d district(s) done, retrying the remaining "
                "%d with %d worker(s)",
                len(results), len(pending), workers,
            )


def _pool_loop(
    cfg: Config,
    pending: list[DistrictTask],
    calibrator: MemoryCalibrator,
    plan: ResourcePlan,
    workers: int,
    boundary_local: Path,
    out: list[DistrictResult],
) -> None:
    budget = plan.budget_bytes
    queue = list(pending)
    total = len(out) + len(queue)
    completed = len(out)

    # "spawn" keeps GDAL/GEOS state out of the child processes and lets us
    # recycle workers, which returns memory to the OS between big districts.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(cfg, str(boundary_local), cfg.log_level,
                  str(Path(cfg.work_dir) / "cropmask.log")),
        max_tasks_per_child=cfg.max_tasks_per_child or None,
    ) as pool:
        inflight: dict = {}
        reserved: dict = {}   # task key -> RAM currently reserved for it
        claimed = 0

        while queue or inflight:
            # ---- admit as much work as the budget and the pool allow ----
            while queue and len(inflight) < workers:
                if inflight and memory_pressure() > PRESSURE_CEILING:
                    log.warning(
                        "memory pressure %.0f%% - pausing admission",
                        memory_pressure() * 100,
                    )
                    break

                # Re-estimate on every admission: the calibrator tightens as
                # measurements come in, so later districts are budgeted from
                # what this dataset actually costs rather than from the
                # pessimistic configured constant.
                pick = None
                cost = 0
                for task in queue:
                    cost = calibrator.estimate(task.size_bytes)
                    if claimed + cost <= budget:
                        pick = task
                        break
                if pick is None:
                    if inflight:
                        break  # wait for RAM to come back
                    # Nothing running and nothing fits: give the machine to the
                    # largest district on its own.
                    pick = queue[0]
                    cost = calibrator.estimate(pick.size_bytes)
                    log.warning(
                        "%s / %s estimated at %.1f GiB exceeds the %.1f GiB budget - "
                        "running it alone",
                        pick.province, pick.district, cost / GIB, budget / GIB,
                    )

                queue.remove(pick)
                reserved[pick.key] = cost
                claimed += cost
                inflight[pool.submit(run_district, pick)] = pick
                log.info(
                    "-> %s / %s (%.0f MB in, %.1f GiB reserved, %.1f/%.1f GiB claimed, "
                    "%d running)",
                    pick.province, pick.district, pick.size_bytes / 1e6,
                    cost / GIB, claimed / GIB, budget / GIB, len(inflight),
                )

            if not inflight:
                continue

            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED,
                           timeout=cfg.district_timeout or None)
            if not done:
                log.error("No district finished within %ss - continuing to wait",
                          cfg.district_timeout)
                continue

            broke: BrokenProcessPool | None = None
            for future in done:
                task = inflight.pop(future)
                claimed -= reserved.pop(task.key, 0)
                completed += 1
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    # Keep draining: the other futures in this batch already
                    # hold finished work, and dropping them would have the
                    # retry recompute districts that are actually done.
                    broke = exc
                    continue
                except Exception as exc:
                    # Anything else is this district's problem, not the run's.
                    result = _crash_result(task, f"{type(exc).__name__}: {exc}")
                out.append(result)
                if (cfg.adaptive_memory and result.peak_rss_mb
                        and result.peak_is_own_work):
                    calibrator.observe(task.size_bytes, result.peak_rss_mb * 1024**2)
                if result.error:
                    log.error("[%d/%d] %s / %s FAILED: %s", completed, total,
                              task.province, task.district, result.error)
                else:
                    log.info(
                        "[%d/%d] %s / %s ok in %.1fs (peak %.0f MB)",
                        completed, total, task.province, task.district,
                        result.seconds, result.peak_rss_mb,
                    )
            if broke is not None:
                raise broke


def _execute_serial(
    cfg: Config, ordered: list[DistrictTask], boundary_local: Path
) -> list[DistrictResult]:
    """Single-process fallback: easier to profile and to debug a crash."""
    log.info("Running serially (1 worker)")
    init_worker(cfg, str(boundary_local), cfg.log_level,
                str(Path(cfg.work_dir) / "cropmask.log"))
    results = []
    for n, task in enumerate(ordered, 1):
        log.info("[%d/%d] %s / %s", n, len(ordered), task.province, task.district)
        results.append(run_district(task))
    return results


def _crash_result(task: DistrictTask, reason: str) -> DistrictResult:
    res = DistrictResult(province=task.province, district=task.district)
    res.error = reason
    res.records = [{
        "province": task.province, "district": task.district,
        "crop": "ALL", "status": "error", "reason": reason,
    }]
    return res
