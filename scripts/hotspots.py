#!/usr/bin/env python3
"""Where did a season's time go? Rank the slowest steps so they can be fixed.

Reads the per-stage seconds that every district records in the workbook's
Detail sheet, plus the SLOW warnings the run logged, and prints:

* total seconds per stage across the season - the place to optimise first
* the slowest individual district x crop x stage steps, with the size of the
  largest input polygon, which is the usual explanation
* the slowest districts overall, from the Timings sheet
* every SLOW single GEOS call from the log

    python scripts/hotspots.py ~/cropmask_runs/run_2018.log /tmp/cropmask/2018/acreage_report_2018.xlsx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

STAGES = ["seconds_read", "seconds_input_union", "seconds_erase",
          "seconds_clip", "seconds_dissolve", "seconds_write"]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log_path, report_path = Path(sys.argv[1]), Path(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    pd.set_option("display.width", 200)

    detail = pd.read_excel(report_path, sheet_name="Detail")
    have = [s for s in STAGES if s in detail.columns]
    timed = detail.dropna(subset=have, how="all") if have else detail.iloc[0:0]

    if not timed.empty:
        totals = timed[have].sum().sort_values(ascending=False)
        grand = totals.sum()
        print("=== seconds per stage, whole season (summed over districts) ===")
        for stage, secs in totals.items():
            bar = "#" * int(40 * secs / grand) if grand else ""
            print(f"  {stage[8:]:12s} {secs:9.0f}s  {100 * secs / grand:5.1f}%  {bar}")

        long = timed.melt(
            id_vars=["province", "district", "crop", "max_input_vertices"],
            value_vars=have, var_name="stage", value_name="seconds",
        ).sort_values("seconds", ascending=False).head(top)
        long["stage"] = long["stage"].str[8:]
        print(f"\n=== {top} slowest steps ===")
        print(long.to_string(index=False))
    else:
        print("no per-stage timings in this report (districts processed before timings existed)")

    try:
        timings = pd.read_excel(report_path, sheet_name="Timings")
        slow = timings.sort_values("seconds", ascending=False).head(top)
        print(f"\n=== {top} slowest districts ===")
        print(slow[["province", "district", "seconds", "peak_rss_mb"]].to_string(index=False))
    except ValueError:
        pass

    if log_path.exists():
        slow_lines = [ln for ln in log_path.read_text(errors="replace").splitlines()
                      if "SLOW " in ln]
        print(f"\n=== SLOW single GEOS calls in the log: {len(slow_lines)} ===")
        pattern = re.compile(r"SLOW (\S+) took ([\d.]+)s in (.+?) \((.*)\)")
        parsed = sorted(
            (float(m.group(2)), m.group(1), m.group(3), m.group(4))
            for ln in slow_lines if (m := pattern.search(ln))
        )[::-1]
        for secs, op, where, sizes in parsed[:top]:
            print(f"  {secs:7.1f}s  {op:8s} {where}  ({sizes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
