"""What runs inside a worker process for a single district.

Each district is fully self-contained: stage its inputs from GCS onto local
scratch, process them, push the outputs back, then delete the scratch. Peak
local disk therefore stays at roughly one district per worker, not the whole
dataset.
"""

from __future__ import annotations

import logging
import os
import resource
import tempfile
from pathlib import Path

import psutil

from . import gcs
from .config import Config
from .discovery import DistrictTask
from .io_layers import read_boundaries
from .pipeline import DistrictResult, process_district
from .state import write_marker

log = logging.getLogger(__name__)

# Per-process caches, populated once by the pool initializer.
_CFG: Config | None = None
_BOUNDARY = None
_LOOKUP: dict[str, list[int]] | None = None


def init_worker(
    cfg: Config,
    boundary_local_path: str,
    log_level: str,
    log_file: str | None = None,
) -> None:
    """Pool initializer: load the boundary once per process, not once per task."""
    global _CFG, _BOUNDARY, _LOOKUP

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        # Workers are where the interesting failures happen, so their
        # tracebacks have to reach the same file as the parent's log rather
        # than only the inherited stderr, which a piped or detached run loses.
        # Each record is one small append, which the kernel does not interleave.
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-7s  [w%(process)d]  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("pyogrio", "fiona", "urllib3", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # GEOS/PROJ are single-threaded here; BLAS threads would only fight the
    # other workers for cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    _CFG = cfg
    _BOUNDARY, _LOOKUP = read_boundaries(
        boundary_local_path, cfg.boundary_field, cfg.metric_crs
    )
    log.debug("worker ready: %d boundary polygons", len(_BOUNDARY))


def _peak_rss_mb() -> float:
    """Peak RSS of this process since it started, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_district(task: DistrictTask) -> DistrictResult:
    """Stage, process and publish one district. Never raises."""
    assert _CFG is not None and _LOOKUP is not None, "worker not initialised"
    cfg = _CFG

    work = Path(tempfile.mkdtemp(prefix=f"{task.district.replace(' ', '_')}_",
                                 dir=cfg.work_dir))
    staged = work / "in"
    produced = work / "out"

    try:
        src = gcs.GcsPath.parse(cfg.input_uri)
        gcs.download_many(
            bucket=src.bucket,
            names=task.blobs(),
            dest_root=staged,
            strip_prefix=src.prefix,
            threads=cfg.transfer_threads,
            credentials_json=cfg.credentials_json,
        )

        result = process_district(
            task=task,
            local_root=staged,
            out_root=produced,
            boundary_geoms=_BOUNDARY,
            boundary_lookup=_LOOKUP,
            cfg=cfg,
        )

        if produced.exists():
            dest = gcs.GcsPath.parse(cfg.output_uri)
            gcs.upload_dir(
                produced, dest,
                threads=cfg.transfer_threads,
                credentials_json=cfg.credentials_json,
            )
            # Report GCS locations, not the scratch paths that are about to vanish.
            for rec in result.records:
                if rec.get("output_path"):
                    rel = Path(rec["output_path"]).relative_to(produced).as_posix()
                    rec["output_path"] = f"{dest}/{rel}"

        result.peak_rss_mb = _peak_rss_mb()
        write_marker(result, cfg.output_uri, cfg.work_dir, cfg.credentials_json)
        return result

    except Exception as exc:
        log.exception("%s / %s failed", task.province, task.district)
        res = DistrictResult(province=task.province, district=task.district)
        res.error = f"{type(exc).__name__}: {exc}"
        res.peak_rss_mb = _peak_rss_mb()
        res.records = [{
            "province": task.province,
            "district": task.district,
            "crop": "ALL",
            "status": "error",
            "reason": res.error,
        }]
        return res
    finally:
        gcs.rmtree_quiet(work)
        # Hand freed arenas back to the OS so a big district does not leave the
        # worker permanently inflated for the districts that follow it.
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2
