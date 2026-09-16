#!/usr/bin/env python3
"""Run whole districts locally with stage timings, and check them against a
previous run's results.

Nothing is uploaded. Each district is staged, processed exactly as a worker
would process it, and its per-crop stage timings and final acreage are printed.
If a previous run left a completion marker for the district, its final acreage
is shown alongside so a code change can be checked for identical output.

    python scripts/profile_district.py --year 2017 GUJRANWALA "RAHIM YAR KHAN"
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("districts", nargs="+")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config/default.yaml"))
    ap.add_argument("--work-dir", default=str(Path.home() / "cropmask_profile"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from cropmask import gcs
    from cropmask.config import Config
    from cropmask.constants import CROP_ORDER
    from cropmask.discovery import build_index, norm_name
    from cropmask.io_layers import read_boundaries
    from cropmask.pipeline import process_district
    from cropmask.state import load_completed

    work = Path(args.work_dir) / str(args.year)
    work.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config, year=args.year, work_dir=str(work))

    boundary = gcs.fetch_to_local(cfg.boundary_uri, work / "boundary", cfg.credentials_json)
    bgeoms, lookup = read_boundaries(boundary, cfg.boundary_field, cfg.metric_crs)
    blobs = gcs.list_blobs(cfg.input_uri, cfg.credentials_json)
    tasks, _ = build_index(blobs, gcs.GcsPath.parse(cfg.input_uri).prefix, set(lookup))
    by_name = {norm_name(t.district): t for t in tasks.values()}
    try:
        previous = load_completed(cfg.output_uri, cfg.credentials_json)
    except Exception:
        previous = {}

    stages = ["seconds_read", "seconds_input_union", "seconds_erase",
              "seconds_clip", "seconds_dissolve", "seconds_write"]
    for name in args.districts:
        task = by_name.get(norm_name(name))
        if task is None:
            print(f"{name}: not found")
            continue
        staged = work / "in" / norm_name(name).replace(" ", "_")
        if not staged.exists():
            src = gcs.GcsPath.parse(cfg.input_uri)
            gcs.download_many(src.bucket, task.blobs(), staged, src.prefix,
                              cfg.transfer_threads, cfg.credentials_json)

        out = Path(tempfile.mkdtemp(prefix="out_", dir=work))
        started = time.time()
        result = process_district(task, staged, out, bgeoms, lookup, cfg)
        total = time.time() - started

        old = previous.get((norm_name(task.province), norm_name(task.district)))
        old_acres = {r["crop"]: r.get("final_acres", 0.0) for r in (old.records if old else [])}
        old_secs = f"{old.seconds:.0f}s" if old else "n/a"

        print(f"\n=== {args.year} {task.province} / {task.district}   total {total:.1f}s"
              f"   (previous run: {old_secs}) ===")
        print(f"{'crop':11s} {'max verts':>10s} " + " ".join(f"{s[8:]:>11s}" for s in stages)
              + f" {'final ac':>13s} {'previous ac':>13s}  match")
        for crop in CROP_ORDER:
            rec = next((r for r in result.records if r["crop"] == crop), None)
            if rec is None:
                continue
            prev = old_acres.get(crop)
            match = "" if prev is None else ("YES" if abs(prev - rec["final_acres"]) < 0.01 else "NO")
            print(f"{crop:11s} {rec.get('max_input_vertices', 0):>10,} "
                  + " ".join(f"{rec.get(s, 0.0):>11.1f}" for s in stages)
                  + f" {rec['final_acres']:>13,.2f} {'' if prev is None else f'{prev:,.2f}':>13s}  {match}")
        if result.error:
            print(f"  ERROR: {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
