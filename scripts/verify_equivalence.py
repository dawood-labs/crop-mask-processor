#!/usr/bin/env python3
"""Prove the fast implementation matches the legacy one, on real data.

The optimised pipeline replaces six chained ``geopandas.overlay`` calls and a
full-layer ``dissolve`` with R-tree-filtered array operations. That is only
worth anything if the numbers are unchanged, so this script runs *both*
implementations over an identical spatial subset of a real district and prints
the per-crop acreage side by side.

Usage::

    python scripts/verify_equivalence.py <local_data_root> <district> [window_fraction]

``local_data_root`` is a directory laid out like the GCS input
(``<crop>/<year>/<province>/<district>/*.shp``). A window fraction keeps the
legacy path tractable - it is O(n*m) and will not finish on a whole large
district.
"""

from __future__ import annotations

import glob
import re
import sys
import time
import warnings

import geopandas as gpd
import numpy as np
import shapely

from cropmask import geometry as G
from cropmask.constants import CROP_ORDER, SQM_PER_ACRE

warnings.filterwarnings("ignore")

METRIC = "EPSG:32642"
MIN_POLY_ACRES = 0.5


def norm(value: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(value).strip().lower())
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load(root: str, crop: str, district: str) -> G.GeomArray:
    pattern = f"{root}/{crop}/**/{district}/**/*.shp"
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if "Empty" not in f] or files
    if not files:
        return G.empty_array()
    gdf = gpd.read_file(files[0], columns=[]).to_crs(METRIC)
    return G.clean(np.asarray(gdf.geometry.values, dtype=object))


# --------------------------------------------------------------------------
# legacy implementation, transcribed from fao_crop_pipeline.py
# --------------------------------------------------------------------------
def _gdf(geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs=METRIC)


def _legacy_clean(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(shapely.make_valid)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gdf[~gdf.geometry.is_empty]


def _legacy_difference(inp, overlay):
    if len(inp) == 0:
        return inp
    if len(overlay) == 0:
        return inp.copy()
    return _legacy_clean(gpd.overlay(inp, overlay, how="difference", keep_geom_type=True))


def legacy_run(layers: dict, mask) -> dict:
    rice, cotton, cane, maize = (
        _gdf(layers["Rice"]), _gdf(layers["Cotton"]),
        _gdf(layers["Sugarcane"]), _gdf(layers["Fall Maize"]),
    )
    rice_d = _legacy_difference(rice, maize)
    cane_d = _legacy_difference(cane, maize)
    cotton_d = _legacy_difference(cotton, maize)
    rice_d2 = _legacy_difference(rice_d, cane_d)
    cotton_d2 = _legacy_difference(cotton_d, cane_d)
    rice_d3 = _legacy_difference(rice_d2, cotton_d2)

    deoverlapped = {
        "Rice": rice_d3, "Cotton": cotton_d2,
        "Sugarcane": cane_d, "Fall Maize": maize,
    }
    mask_gdf = _gdf(mask)

    out = {}
    for crop, gdf in deoverlapped.items():
        if len(gdf) == 0:
            out[crop] = (0, 0.0)
            continue
        clipped = _legacy_clean(gpd.clip(gdf, mask_gdf, keep_geom_type=True))
        if len(clipped) == 0:
            out[crop] = (0, 0.0)
            continue
        dissolved = _legacy_clean(clipped[["geometry"]].dissolve().reset_index(drop=True))
        parts = _legacy_clean(dissolved.explode(index_parts=False).reset_index(drop=True)).copy()
        parts["acres"] = (parts.geometry.area / SQM_PER_ACRE).round(2)
        kept = parts[parts["acres"] > MIN_POLY_ACRES]
        out[crop] = (len(kept), round(float(kept["acres"].sum()), 2))
    return out


# --------------------------------------------------------------------------
# optimised implementation
# --------------------------------------------------------------------------
def fast_run(layers: dict, mask) -> dict:
    def cat(*arrays):
        parts = [a for a in arrays if len(a)]
        return np.concatenate(parts) if parts else G.empty_array()

    deoverlapped = {
        "Fall Maize": layers["Fall Maize"],
        "Sugarcane": G.erase(layers["Sugarcane"], layers["Fall Maize"]),
        "Cotton": G.erase(layers["Cotton"], cat(layers["Fall Maize"], layers["Sugarcane"])),
        "Rice": G.erase(
            layers["Rice"],
            cat(layers["Fall Maize"], layers["Sugarcane"], layers["Cotton"]),
        ),
    }
    out = {}
    for crop, geoms in deoverlapped.items():
        parts = G.dissolve(G.clip(geoms, mask))
        kept, acres, _, _ = G.filter_by_area(parts, MIN_POLY_ACRES)
        out[crop] = (len(kept), round(float(acres.sum()), 2))
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, district = sys.argv[1], sys.argv[2]
    fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    layers = {crop: load(root, crop, district) for crop in CROP_ORDER}
    populated = [g for g in layers.values() if len(g)]
    if not populated:
        print(f"no data found for {district} under {root}")
        return 2

    minx, miny, maxx, maxy = shapely.total_bounds(np.concatenate(populated))
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half_w, half_h = (maxx - minx) * fraction / 2, (maxy - miny) * fraction / 2
    window = shapely.box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    print(f"test window: {2 * half_w / 1000:.1f} x {2 * half_h / 1000:.1f} km")

    subset = {}
    for crop, geoms in layers.items():
        if len(geoms) == 0:
            subset[crop] = G.empty_array()
            continue
        hits = shapely.STRtree(geoms).query(window, predicate="intersects")
        subset[crop] = (
            G.clean(shapely.intersection(geoms[hits], window))
            if len(hits) else G.empty_array()
        )
    print("features in window:", {k: len(v) for k, v in subset.items()})

    boundary = gpd.read_file(glob.glob(f"{root}/**/boundary/*.shp", recursive=True)[0])
    boundary = boundary.to_crs(METRIC)
    name_col = next(c for c in boundary.columns if c.lower().startswith("district"))
    mask = np.asarray(
        boundary[boundary[name_col].map(norm) == norm(district)].geometry.values,
        dtype=object,
    )

    start = time.time()
    legacy = legacy_run(subset, mask)
    legacy_seconds = time.time() - start

    start = time.time()
    fast = fast_run(subset, mask)
    fast_seconds = time.time() - start

    print(f"\nlegacy {legacy_seconds:.1f}s   fast {fast_seconds:.1f}s   "
          f"speedup {legacy_seconds / max(fast_seconds, 1e-9):.1f}x\n")
    header = f"{'crop':12s} {'legacy_n':>9s} {'fast_n':>8s} {'legacy_ac':>12s} {'fast_ac':>12s} {'diff_%':>9s}"
    print(header)
    print("-" * len(header))

    ok = True
    for crop in CROP_ORDER:
        ln, la = legacy[crop]
        fn, fa = fast[crop]
        pct = 100 * (fa - la) / la if la else 0.0
        if abs(pct) > 0.01:
            ok = False
        print(f"{crop:12s} {ln:9d} {fn:8d} {la:12.2f} {fa:12.2f} {pct:9.4f}")

    print("\nRESULT:", "MATCH" if ok else "MISMATCH (>0.01%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
