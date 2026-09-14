"""Per-district processing: the actual FAO de-overlap specification.

Legacy ordering, kept for reference::

    Step 1  Rice/Sugarcane/Cotton  -  Fall Maize
    Step 2  Rice_D/Cotton_D        -  Sugarcane_D
    Step 3  Rice_D2                -  Cotton_D2
    Step 4  clip everything to the district polygon
    Step 5  dissolve -> singlepart -> drop <=0.5 ac -> drop layer if <=200 ac

Steps 1-3 are a chained difference, and set algebra collapses the chain::

    Sugarcane_D  = Sugarcane - Maize
    Cotton_D2    = Cotton    - (Maize u Sugarcane)
    Rice_D3      = Rice      - (Maize u Sugarcane u Cotton)

which is a plain crop-priority partition and reaches the same geometry in one
pass per crop instead of six chained overlays. This has been verified against
the legacy implementation on real districts (see scripts/verify_equivalence.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as G
from .constants import CROP_ORDER, PREDICTED_VALUES
from .discovery import DistrictTask, norm_name
from .io_layers import read_geometries, write_intermediate, write_layer

log = logging.getLogger(__name__)


def _staged_relpath(blob_name: str, input_uri: str) -> Path:
    """Where ``download_many`` puts a blob under the staging root.

    Staging mirrors the bucket layout with the input prefix stripped, so the
    crop directory stays in the path and two crops that share a filename stay
    distinct.
    """
    from .gcs import GcsPath

    prefix = GcsPath.parse(input_uri).prefix
    rel = blob_name[len(prefix):] if prefix and blob_name.startswith(prefix) else blob_name
    return Path(rel.lstrip("/"))


def _concat(*arrays: G.GeomArray) -> G.GeomArray:
    parts = [a for a in arrays if len(a)]
    if not parts:
        return G.empty_array()
    return np.concatenate(parts)


@dataclass
class DistrictResult:
    province: str
    district: str
    records: list[dict] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    error: str | None = None


def _blank_record(province: str, district: str, crop: str, source: str = "") -> dict:
    return {
        "province": province,
        "district": district,
        "crop": crop,
        "input_shapefile": source,
        "input_polygons": 0,
        "input_acres": 0.0,
        "after_difference_acres": 0.0,
        "after_clip_acres": 0.0,
        "singlepart_polygons": 0,
        "singlepart_acres": 0.0,
        "removed_small_polygons": 0,
        "acres_removed_small": 0.0,
        "final_polygons": 0,
        "final_acres": 0.0,
        "acres_lost_total": 0.0,
        "predicted": PREDICTED_VALUES.get(crop),
        "status": "",
        "reason": "",
        "output_path": "",
    }


def process_district(
    task: DistrictTask,
    local_root: Path,
    out_root: Path,
    boundary_geoms,
    boundary_lookup: dict[str, list[int]],
    cfg,
) -> DistrictResult:
    """Run the full specification for one district against already-staged files."""
    started = time.time()
    result = DistrictResult(province=task.province, district=task.district)
    metric = cfg.metric_crs

    idx = boundary_lookup.get(norm_name(task.district))
    if not idx:
        log.error("%s: not present in the boundary shapefile - skipped", task.district)
        for crop in CROP_ORDER:
            if crop in task.crops:
                rec = _blank_record(task.province, task.district, crop)
                rec["status"] = "error"
                rec["reason"] = "district name not present in boundary shapefile"
                result.records.append(rec)
        result.seconds = time.time() - started
        return result

    mask = np.asarray([boundary_geoms[i] for i in idx], dtype=object)

    # ---- load the four inputs ------------------------------------------
    layers: dict[str, G.GeomArray] = {}
    records: dict[str, dict] = {}

    for crop in CROP_ORDER:
        source = task.crops[crop].shp_blob if crop in task.crops else ""
        rec = _blank_record(task.province, task.district, crop, source)
        records[crop] = rec

        if crop not in task.crops:
            rec["status"] = "missing"
            rec["reason"] = "no input shapefile for this crop/district"
            layers[crop] = G.empty_array()
            continue

        path = local_root / _staged_relpath(task.crops[crop].shp_blob, cfg.input_uri)
        if not path.exists():
            # Never fall back to a basename search. Crop layers routinely share
            # a filename - 58 of the 66 districts in the 2025 data have at
            # least two crops whose shapefiles are both named after the
            # district - so a basename match silently loads another crop's
            # geometry, and the output looks entirely plausible.
            rec["status"] = "error"
            rec["reason"] = f"staged shapefile not found: {path}"
            log.error("%s/%s: expected staged file missing: %s",
                      task.district, crop, path)
            layers[crop] = G.empty_array()
            continue

        try:
            layers[crop] = read_geometries(path, metric)
        except Exception as exc:
            log.error("%s/%s: read failed: %s", task.district, crop, exc)
            rec["status"] = "error"
            rec["reason"] = f"read error: {exc}"
            layers[crop] = G.empty_array()
            continue

        rec["input_polygons"] = len(layers[crop])
        rec["input_acres"] = G.total_acres(layers[crop])

    # ---- steps 1-3: crop-priority de-overlap ---------------------------
    maize, cane, cotton, rice = (
        layers["Fall Maize"], layers["Sugarcane"], layers["Cotton"], layers["Rice"]
    )

    deoverlapped = {
        "Fall Maize": maize,                                   # never differenced
        "Sugarcane": G.erase(cane, maize),
        "Cotton": G.erase(cotton, _concat(maize, cane)),
        "Rice": G.erase(rice, _concat(maize, cane, cotton)),
    }

    inter_dir = out_root / "_intermediate" / task.province / task.district
    if cfg.keep_intermediates:
        for crop, geoms in deoverlapped.items():
            write_intermediate(geoms, inter_dir, f"{crop}_D", metric)

    for crop, geoms in deoverlapped.items():
        records[crop]["after_difference_acres"] = G.total_acres(geoms)

    # free the raw layers before the clip stage allocates again
    layers.clear()
    del maize, cane, cotton, rice

    # ---- step 4: clip to the district polygon --------------------------
    clipped = {}
    for crop, geoms in deoverlapped.items():
        clipped[crop] = G.clip(geoms, mask)
        records[crop]["after_clip_acres"] = G.total_acres(clipped[crop])
        if cfg.keep_intermediates:
            write_intermediate(clipped[crop], inter_dir, f"{crop}_C", metric)
    deoverlapped.clear()

    # ---- step 5 ---------------------------------------------------------
    for crop in CROP_ORDER:
        rec = records[crop]
        if rec["status"] in ("missing", "error"):
            result.records.append(rec)
            continue
        try:
            path = _finalise(
                clipped[crop], crop, task, out_root, rec, inter_dir, cfg
            )
            if path:
                result.outputs.append(path)
        except Exception as exc:
            log.exception("%s/%s: step 5 failed", task.district, crop)
            rec["status"] = "error"
            rec["reason"] = f"step5 error: {exc}"
        rec["acres_lost_total"] = round(rec["input_acres"] - rec["final_acres"], 2)
        result.records.append(rec)

    kept = [r["crop"] for r in result.records if r["status"] == "kept"]
    result.seconds = time.time() - started
    log.info(
        "%s / %s done in %.1fs - kept: %s",
        task.province, task.district, result.seconds,
        ", ".join(kept) if kept else "(none)",
    )
    return result


def _finalise(geoms, crop, task, out_root: Path, rec: dict, inter_dir: Path, cfg):
    """Dissolve, explode, filter by area, apply the layer threshold, write."""
    parts = G.dissolve(geoms)
    if cfg.keep_intermediates:
        write_intermediate(parts, inter_dir, f"{crop}_C_D_S", cfg.metric_crs)

    rec["singlepart_polygons"] = len(parts)
    rec["singlepart_acres"] = G.total_acres(parts)

    if len(parts) == 0:
        rec["status"] = "dropped"
        rec["reason"] = "no geometry left after difference/clip"
        return None

    kept, acres, n_small, acres_small = G.filter_by_area(parts, cfg.min_polygon_acres)
    rec["removed_small_polygons"] = n_small
    rec["acres_removed_small"] = acres_small
    if cfg.keep_intermediates:
        write_intermediate(kept, inter_dir, f"{crop}_C_D_S_{cfg.min_polygon_acres}", cfg.metric_crs)

    total = round(float(acres.sum()), 2)
    rec["final_polygons"] = len(kept)
    rec["final_acres"] = total

    if len(kept) == 0 or total <= cfg.min_total_acres:
        rec["status"] = "dropped"
        rec["reason"] = f"total {total} acres <= {cfg.min_total_acres} acre threshold"
        rec["final_polygons"] = 0
        return None

    path = final_output_path(out_root, crop, task.province, task.district)
    write_layer(kept, path, PREDICTED_VALUES[crop], cfg.metric_crs, cfg.output_crs)
    rec["status"] = "kept"
    rec["output_path"] = str(path)
    return path


def final_output_path(out_root: Path, crop: str, province: str, district: str) -> Path:
    import re

    safe_crop = crop.replace(" ", "_")
    safe_district = re.sub(r"[^A-Za-z0-9]+", "_", district).strip("_")
    return out_root / crop / province / district / f"{safe_district}_{safe_crop}.shp"
