"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import gcs
from .config import Config
from .report import build_report
from .resources import GIB
from .runner import run

log = logging.getLogger("cropmask")


def setup_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("pyogrio", "fiona", "urllib3", "google", "gcsfs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cropmask",
        description="FAO crop-mask de-overlap pipeline (GCS in, GCS out).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", help="YAML config file")

    g = p.add_argument_group("data")
    g.add_argument("--year", type=int,
                   help="season to process; fills {year} in the configured URIs")
    g.add_argument("--input-uri", help="gs://bucket/prefix holding the crop folders")
    g.add_argument("--output-uri", help="gs://bucket/prefix to write results under")
    g.add_argument("--boundary-uri", help="district boundary shapefile")
    g.add_argument("--boundary-field", help="district-name column in the boundary file")
    g.add_argument("--credentials-json", help="service-account key file")

    g = p.add_argument_group("scaling")
    g.add_argument("--workers", type=int, help="0/omitted = auto-detect from CPU and RAM")
    g.add_argument("--memory-fraction", type=float,
                   help="fraction of available RAM the run may use")
    g.add_argument("--memory-per-input-byte", type=float,
                   help="RAM assumed per byte of input (calibration factor)")

    g = p.add_argument_group("selection")
    g.add_argument("--only-provinces", nargs="*", help="restrict to these provinces")
    g.add_argument("--only-districts", nargs="*", help="restrict to these districts")
    g.add_argument("--limit", type=int, help="process only N districts (smallest first)")

    g = p.add_argument_group("behaviour")
    g.add_argument("--keep-intermediates", action="store_true", default=None,
                   help="also write the per-step QA layers")
    g.add_argument("--overwrite", action="store_true", default=None,
                   help="reprocess districts a previous run already finished")
    g.add_argument("--work-dir", help="local scratch directory")
    g.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    g.add_argument("--dry-run", action="store_true",
                   help="list the planned work and exit without processing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "dry_run") and v is not None}

    try:
        cfg = Config.load(args.config, **overrides)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    work_dir = Path(cfg.work_dir)
    setup_logging(cfg.log_level, work_dir / "cropmask.log")

    log.info("=" * 72)
    log.info("FAO crop-mask pipeline%s", f"  -  {cfg.year} season" if cfg.year else "")
    log.info("  input    : %s", cfg.input_uri)
    log.info("  output   : %s", cfg.output_uri)
    log.info("  boundary : %s", cfg.boundary_uri)
    log.info("  CRS      : %s (metric) -> %s (output)", cfg.metric_crs, cfg.output_crs)
    log.info("=" * 72)

    if args.dry_run:
        return _dry_run(cfg)

    started = time.time()
    outcome = run(cfg)

    report_path = work_dir / cfg.report_name
    build_report(
        records=outcome.records,
        anomalies=outcome.anomalies,
        timings=outcome.timings,
        out_path=report_path,
        run_meta=_run_meta(cfg, outcome),
    )

    dest = gcs.GcsPath.parse(cfg.output_uri).child(cfg.report_name)
    try:
        gcs.upload_many([(report_path, dest.prefix)], dest.bucket,
                        credentials_json=cfg.credentials_json)
        log.info("Report uploaded to %s", dest)
    except Exception as exc:
        log.error("Could not upload the report (kept locally at %s): %s",
                  report_path, exc)

    _summarise(outcome, report_path, time.time() - started)
    # Per-crop failures are recorded on the rows, not on the district result,
    # so a run where every district failed its boundary lookup would otherwise
    # exit 0 and read as success to any wrapper script.
    failed = any(r.error for r in outcome.results) or any(
        rec.get("status") == "error" for rec in outcome.records
    )
    return 1 if failed else 0


def _dry_run(cfg: Config) -> int:
    from .runner import prepare

    tasks, anomalies, _ = prepare(cfg)
    log.info("-" * 72)
    for task in sorted(tasks, key=lambda t: t.size_bytes, reverse=True):
        log.info(
            "%9.1f MB  %-10s %-24s  crops: %s",
            task.size_bytes / 1e6, task.province, task.district,
            ", ".join(sorted(task.crops)),
        )
    log.info("-" * 72)
    log.info("%d district(s), %.2f GiB total",
             len(tasks), sum(t.size_bytes for t in tasks) / GIB)
    for item in anomalies:
        log.warning("anomaly: %s", item)
    return 0


def _run_meta(cfg: Config, outcome) -> dict:
    meta = {k: str(v) for k, v in cfg.to_dict().items() if k != "credentials_json"}
    meta.update({
        "elapsed_minutes": f"{outcome.seconds / 60:.1f}",
        "workers_used": str(outcome.plan.workers),
        "cpus_detected": str(outcome.plan.cpus),
        "memory_budget_gib": f"{outcome.plan.budget_bytes / GIB:.1f}",
        "districts_processed": str(len(outcome.results)),
    })
    return meta


def _summarise(outcome, report_path: Path, elapsed: float) -> None:
    records = outcome.records
    kept = sum(1 for r in records if r.get("status") == "kept")
    dropped = sum(1 for r in records if r.get("status") == "dropped")
    errors = sum(1 for r in records if r.get("status") == "error")
    acres = sum(r.get("final_acres", 0) or 0 for r in records if r.get("status") == "kept")

    slowest = sorted(outcome.results, key=lambda r: r.seconds, reverse=True)[:5]
    peak = max((r.peak_rss_mb for r in outcome.results), default=0.0)

    log.info("-" * 72)
    log.info("Layers written  : %d", kept)
    log.info("Dropped layers  : %d", dropped)
    log.info("Errors          : %d", errors)
    log.info("Total acreage   : %.2f", acres)
    log.info("Peak worker RSS : %.0f MB", peak)
    if outcome.resumed:
        log.info("Resumed         : %d district(s) carried over from a previous run",
                 len(outcome.resumed))
    log.info("Report          : %s", report_path)
    log.info("Elapsed         : %.1f min", elapsed / 60)
    if slowest:
        log.info("Slowest districts:")
        for r in slowest:
            log.info("    %6.1fs  %-10s %s  (peak %.0f MB)",
                     r.seconds, r.province, r.district, r.peak_rss_mb)
    log.info("-" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
