#!/usr/bin/env python3
"""Time the original implementation against this one, district by district.

Both implementations run on the same district, from the same staged files, in
the same process conditions, one district at a time and single-threaded - so
the ratio measures the algorithm, not the parallelism.

The original implementation does not finish on large districts (on RAHIM YAR
KHAN it had not completed its first difference step after 20 minutes). Each
legacy run therefore gets a time limit. When it is hit, the row reports the
limit as a *lower bound*: "the old code needed more than N seconds", which is
the honest thing to say about a run that never ended.

Usage::

    python scripts/benchmark_legacy.py --year 2025 --timeout 900 \\
        JAMSHORO HYDERABAD GUJRAT SUKKUR BADIN

Needs GOOGLE_APPLICATION_CREDENTIALS. Writes a CSV next to its log.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

METRIC = "EPSG:32642"
SQM_PER_ACRE = 4046.8564224
MIN_POLY_ACRES = 0.5


# ---------------------------------------------------------------------------
# the original implementation, transcribed from fao_crop_pipeline.py
# ---------------------------------------------------------------------------
def _legacy(layers: dict, mask, queue) -> None:
    import geopandas as gpd
    import shapely

    def clean(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        gdf = gdf[gdf.geometry.notna()].copy()
        bad = ~gdf.geometry.is_valid
        if bad.any():
            gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].apply(shapely.make_valid)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        return gdf[~gdf.geometry.is_empty]

    def difference(inp, overlay):
        if len(inp) == 0:
            return inp
        if len(overlay) == 0:
            return inp.copy()
        return clean(gpd.overlay(inp, overlay, how="difference", keep_geom_type=True))

    def frame(geoms):
        return gpd.GeoDataFrame(geometry=list(geoms), crs=METRIC)

    started = time.time()
    rice, cotton = frame(layers["Rice"]), frame(layers["Cotton"])
    cane, maize = frame(layers["Sugarcane"]), frame(layers["Fall Maize"])

    rice_d = difference(rice, maize)
    cane_d = difference(cane, maize)
    cotton_d = difference(cotton, maize)
    rice_d2 = difference(rice_d, cane_d)
    cotton_d2 = difference(cotton_d, cane_d)
    rice_d3 = difference(rice_d2, cotton_d2)

    mask_gdf = frame(mask)
    acres = {}
    for crop, gdf in (("Rice", rice_d3), ("Cotton", cotton_d2),
                      ("Sugarcane", cane_d), ("Fall Maize", maize)):
        if len(gdf) == 0:
            acres[crop] = 0.0
            continue
        clipped = clean(gpd.clip(gdf, mask_gdf, keep_geom_type=True))
        if len(clipped) == 0:
            acres[crop] = 0.0
            continue
        dissolved = clean(clipped[["geometry"]].dissolve().reset_index(drop=True))
        parts = clean(dissolved.explode(index_parts=False).reset_index(drop=True))
        area = parts.geometry.area / SQM_PER_ACRE
        # the legacy rounding, deliberately preserved so the numbers compare
        area = area.round(2)
        acres[crop] = round(float(area[area > MIN_POLY_ACRES].sum()), 2)

    queue.put((time.time() - started, acres))


# ---------------------------------------------------------------------------
# this implementation
# ---------------------------------------------------------------------------
def _current(layers: dict, mask) -> tuple[float, dict]:
    import shapely

    from cropmask import geometry as G

    def concat(*arrays):
        parts = [a for a in arrays if len(a)]
        return np.concatenate(parts) if parts else G.empty_array()

    started = time.time()
    maize = layers["Fall Maize"]
    result = {
        "Fall Maize": maize,
        "Sugarcane": G.erase(layers["Sugarcane"], maize),
        "Cotton": G.erase(layers["Cotton"], concat(maize, layers["Sugarcane"])),
        "Rice": G.erase(layers["Rice"],
                        concat(maize, layers["Sugarcane"], layers["Cotton"])),
    }
    acres = {}
    for crop, geoms in result.items():
        parts = G.dissolve(G.clip(geoms, mask))
        # match the legacy rounding for a like-for-like acreage comparison
        area = np.round(shapely.area(parts) / SQM_PER_ACRE, 2)
        acres[crop] = round(float(area[area > MIN_POLY_ACRES].sum()), 2)
    return time.time() - started, acres


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("districts", nargs="+")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds allowed per legacy run before it is abandoned")
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config/default.yaml"))
    ap.add_argument("--work-dir", default=str(Path.home() / "cropmask_bench"))
    args = ap.parse_args()

    from cropmask import gcs
    from cropmask.config import Config
    from cropmask.discovery import build_index, norm_name
    from cropmask.io_layers import read_boundaries, read_geometries
    from cropmask.pipeline import _staged_relpath
    from cropmask import geometry as G

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config, year=args.year, work_dir=str(work))

    boundary = gcs.fetch_to_local(cfg.boundary_uri, work / "boundary", cfg.credentials_json)
    boundary_geoms, lookup = read_boundaries(boundary, cfg.boundary_field, METRIC)
    blobs = gcs.list_blobs(cfg.input_uri, cfg.credentials_json)
    tasks, _ = build_index(blobs, gcs.GcsPath.parse(cfg.input_uri).prefix, set(lookup))
    by_name = {norm_name(t.district): t for t in tasks.values()}

    out_csv = work / f"benchmark_{args.year}.csv"
    rows = []
    header = (f"{'district':16s} {'input MB':>9s} {'features':>9s} "
              f"{'old (s)':>10s} {'new (s)':>8s} {'speedup':>10s}  acres")
    print(header)
    print("-" * len(header), flush=True)

    ctx = mp.get_context("spawn")
    for name in args.districts:
        task = by_name.get(norm_name(name))
        if task is None:
            print(f"{name:16s} not found", flush=True)
            continue

        staged = work / "in" / norm_name(name).replace(" ", "_")
        if not staged.exists():
            src = gcs.GcsPath.parse(cfg.input_uri)
            gcs.download_many(src.bucket, task.blobs(), staged, src.prefix,
                              cfg.transfer_threads, cfg.credentials_json)

        layers = {
            crop: read_geometries(staged / _staged_relpath(ci.shp_blob, cfg.input_uri), METRIC)
            for crop, ci in task.crops.items()
        }
        for crop in ("Rice", "Cotton", "Sugarcane", "Fall Maize"):
            layers.setdefault(crop, G.empty_array())
        mask = np.asarray([boundary_geoms[i] for i in lookup[norm_name(name)]], dtype=object)
        features = sum(len(v) for v in layers.values())

        new_s, new_acres = _current(layers, mask)

        queue = ctx.Queue()
        proc = ctx.Process(target=_legacy, args=(layers, mask, queue))
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            old_s, old_acres, finished = float(args.timeout), None, False
        else:
            old_s, old_acres = queue.get()
            finished = True

        ratio = old_s / max(new_s, 1e-9)
        speedup = f"{ratio:.1f}x" if finished else f">{ratio:.0f}x"
        old_txt = f"{old_s:.1f}" if finished else f">{old_s:.0f}"
        if finished:
            worst = max(
                abs(old_acres[c] - new_acres[c]) / max(old_acres[c], 1.0)
                for c in new_acres
            )
            match = f"match ({100 * worst:.3f}% max diff)"
        else:
            match = "old run never finished"

        print(f"{name:16s} {task.size_bytes / 1e6:9.1f} {features:9d} "
              f"{old_txt:>10s} {new_s:8.1f} {speedup:>10s}  {match}", flush=True)
        rows.append({
            "district": name, "input_mb": round(task.size_bytes / 1e6, 1),
            "features": features, "old_seconds": round(old_s, 1),
            "old_finished": finished, "new_seconds": round(new_s, 1),
            "speedup": round(ratio, 1), "acreage": match,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
