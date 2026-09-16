"""Excel acreage report: start-vs-final area for every district and crop."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMN_ORDER = [
    "province", "district", "crop", "predicted", "status", "reason",
    "input_polygons", "input_acres", "input_union_acres",
    "after_difference_acres", "after_clip_acres",
    "singlepart_polygons", "singlepart_acres",
    "removed_small_polygons", "acres_removed_small",
    "final_polygons", "final_acres", "acres_lost_total",
    "input_shapefile", "output_path",
    "max_input_vertices", "seconds_read", "seconds_input_union",
    "seconds_erase", "seconds_clip", "seconds_dissolve", "seconds_write",
]


def build_report(
    records: list[dict],
    anomalies: list[dict],
    timings: list[dict],
    out_path: Path,
    run_meta: dict,
) -> Path:
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=COLUMN_ORDER)
    for column in COLUMN_ORDER:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[COLUMN_ORDER].sort_values(["province", "district", "crop"])

    kept = df[df["status"] == "kept"]

    by_crop = pd.DataFrame()
    if not kept.empty:
        # Count province+district pairs: two provinces can share a district name.
        kept = kept.assign(_place=kept["province"].astype(str) + "/" + kept["district"].astype(str))
        by_crop = kept.groupby("crop", as_index=False).agg(
            districts=("_place", "nunique"),
            polygons=("final_polygons", "sum"),
            input_acres=("input_acres", "sum"),
            input_union_acres=("input_union_acres", "sum"),
            final_acres=("final_acres", "sum"),
        )
        # Retention is measured against the dissolved input, not the raw sum:
        # final_acres is taken after a dissolve, so comparing it with a figure
        # that double-counts self-overlap reports loss that never happened.
        by_crop["acres_lost"] = (
            by_crop["input_union_acres"] - by_crop["final_acres"]
        ).round(2)
        by_crop["retained_pct"] = (
            100 * by_crop["final_acres"]
            / by_crop["input_union_acres"].replace(0, pd.NA)
        ).round(2)

    by_district = pd.DataFrame()
    if not kept.empty:
        by_district = (
            kept.pivot_table(
                index=["province", "district"], columns="crop",
                values="final_acres", aggfunc="sum",
            )
            .reset_index()
        )

    sheets = {
        "Detail": df,
        "Summary_by_Crop": by_crop,
        "Final_Acres_by_District": by_district,
        "Deleted_under_threshold": df[df["status"] == "dropped"],
        "Missing_or_Error": df[df["status"].isin(["missing", "error"])],
        "Run_Info": pd.DataFrame(
            sorted(run_meta.items()), columns=["setting", "value"]
        ),
    }
    if timings:
        sheets["Timings"] = pd.DataFrame(timings).sort_values("seconds", ascending=False)
    if anomalies:
        sheets["Input_Anomalies"] = pd.DataFrame(anomalies)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name, index=False)
        _autofit(xl)
    return out_path


def _autofit(writer) -> None:
    for sheet in writer.book.worksheets:
        for column in sheet.columns:
            width = max(
                (len(str(cell.value)) for cell in column if cell.value is not None),
                default=10,
            )
            sheet.column_dimensions[column[0].column_letter].width = min(width + 2, 55)
        sheet.freeze_panes = "A2"
