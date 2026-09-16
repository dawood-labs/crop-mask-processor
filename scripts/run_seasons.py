#!/usr/bin/env python3
"""Process several seasons back to back, unattended.

For each season, in order:

1. Preflight: count districts and list shapefiles missing a sidecar. A missing
   .prj is logged here rather than discovered halfway through the run; the
   pipeline still refuses to guess a CRS, so the affected crop is reported as an
   error in that season's workbook.
2. Run the pipeline, logging to <log-dir>/run_<year>.log, with a CPU and memory
   sampler writing <log-dir>/util_<year>.csv.
3. Append a one-line result to <log-dir>/seasons_summary.txt.

A season that finishes with errors does not stop the queue. Seasons are run one
at a time on purpose: a single season already occupies every core.

    python scripts/run_seasons.py 2018 2019 2020 --overwrite-years 2025 2025
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def preflight(year: int, config: str) -> tuple[int, list[str]]:
    from cropmask import gcs
    from cropmask.config import Config

    cfg = Config.load(config, year=year)
    blobs = gcs.list_blobs(cfg.input_uri, cfg.credentials_json)
    stems: dict[str, set[str]] = collections.defaultdict(set)
    for name, _ in blobs:
        leaf = name.rsplit("/", 1)[-1]
        if "." in leaf:
            stem, ext = name.rsplit(".", 1)
            stems[stem].add(ext.lower())

    shapefiles = {s: e for s, e in stems.items() if "shp" in e}
    problems = []
    for stem, exts in sorted(shapefiles.items()):
        missing = [x for x in ("shx", "dbf", "prj") if x not in exts]
        if missing:
            rel = stem.split("/", 3)[-1] if stem.count("/") >= 3 else stem
            problems.append(f"missing .{'/.'.join(missing)}: {rel}")
    return len(shapefiles), problems


def summarise(log_path: Path) -> str:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    def grab(label):
        m = re.findall(rf"{label}\s*:\s*(.+)", text)
        return m[-1].strip() if m else "?"
    crashed = "Traceback (most recent call last)" in text and "Layers written" not in text
    status = "CRASHED" if crashed else "done"
    return (f"{status} | layers {grab('Layers written')} | dropped {grab('Dropped layers')} | "
            f"errors {grab('Errors')} | acres {grab('Total acreage')} | "
            f"elapsed {grab('Elapsed')} | resumed {grab('Resumed')}")


def wait_for_other_runs() -> None:
    """Never start a season while another cropmask run holds the machine."""
    announced = False
    while True:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        others = [
            line for line in out.splitlines()
            if (parts := line.split()) and len(parts) >= 3
            and "python" in parts[1] and parts[2].endswith("bin/cropmask")
            and int(parts[0]) != os.getpid()
        ]
        if not others:
            return
        if not announced:
            print(f"{now()}  waiting for a running cropmask job to finish", flush=True)
            announced = True
        time.sleep(30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("years", nargs="+", type=int)
    ap.add_argument("--overwrite-years", nargs="*", type=int, default=[],
                    help="seasons to reprocess from scratch instead of resuming")
    ap.add_argument("--config", default=str(REPO / "config/default.yaml"))
    ap.add_argument("--log-dir", default=str(Path.home() / "cropmask_runs"))
    args = ap.parse_args()

    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    summary = logs / "seasons_summary.txt"
    sampler = logs / "sampler.sh"

    for year in args.years:
        wait_for_other_runs()
        print(f"\n{now()}  ===== {year} =====", flush=True)

        try:
            count, problems = preflight(year, args.config)
            print(f"{now()}  preflight: {count} shapefiles, {len(problems)} with missing sidecars",
                  flush=True)
            for p in problems:
                print(f"    {p}", flush=True)
        except Exception as exc:
            count, problems = 0, [f"preflight failed: {exc}"]
            print(f"{now()}  preflight failed: {exc}", flush=True)

        log_path = logs / f"run_{year}.log"
        cmd = ["cropmask", "-c", args.config, "--year", str(year)]
        if year in args.overwrite_years:
            cmd.append("--overwrite")

        started = time.time()
        with open(log_path, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
        sampler_proc = None
        if sampler.exists():
            sampler_proc = subprocess.Popen(
                ["bash", str(sampler), str(proc.pid), str(logs / f"util_{year}.csv")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        code = proc.wait()
        if sampler_proc:
            sampler_proc.wait(timeout=60)

        minutes = (time.time() - started) / 60
        line = (f"{now()}  {year}: exit {code} in {minutes:.1f} min | {summarise(log_path)}"
                f" | preflight issues {len(problems)}")
        print(line, flush=True)
        with open(summary, "a") as fh:
            fh.write(line + "\n")
            for p in problems:
                fh.write(f"        {p}\n")

    print(f"\n{now()}  all seasons finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
