"""Reading and writing shapefiles as plain geometry arrays."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from pyproj import CRS, Transformer

from . import geometry as G

log = logging.getLogger(__name__)


def _transform(geoms: G.GeomArray, src: CRS, dst: CRS) -> G.GeomArray:
    """Reproject an array of geometries without going through GeoPandas."""
    if src.equals(dst):
        return geoms
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(
        geoms,
        lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1])),
        include_z=False,
    )


def read_geometries(path: str | Path, target_crs: str) -> G.GeomArray:
    """Load only the geometry column of a shapefile, reprojected and repaired.

    Attribute columns are never read: the pipeline discards them anyway, and
    skipping them roughly halves the read time on the larger layers.
    """
    path = str(path)
    info = pyogrio.read_info(path)
    if not info["features"]:
        return G.empty_array()

    src_crs = info.get("crs")
    if not src_crs:
        raise ValueError(f"{path} has no CRS (.prj missing) - cannot reproject")

    geoms = pyogrio.read_dataframe(path, columns=[], read_geometry=True).geometry.values
    geoms = np.asarray(geoms, dtype=object)
    geoms = _transform(geoms, CRS.from_user_input(src_crs), CRS.from_user_input(target_crs))
    return G.clean(geoms)


def read_boundaries(path: str | Path, field: str, target_crs: str):
    """Return ``(geometries, {normalised district name: [indices]})``."""
    from .discovery import norm_name

    path = str(path)
    info = pyogrio.read_info(path)
    if not info.get("crs"):
        raise ValueError("boundary shapefile has no CRS (.prj missing)")

    gdf = pyogrio.read_dataframe(path, columns=[field])
    if field not in gdf.columns:
        available = list(pyogrio.read_info(path)["fields"])
        raise KeyError(
            f"column '{field}' not in boundary shapefile. Available: {available}"
        )

    geoms = np.asarray(gdf.geometry.values, dtype=object)
    geoms = _transform(
        geoms, CRS.from_user_input(info["crs"]), CRS.from_user_input(target_crs)
    )

    # clean() can drop rows, so repair in place to keep names and geometry aligned.
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])

    lookup: dict[str, list[int]] = {}
    for i, value in enumerate(gdf[field].tolist()):
        lookup.setdefault(norm_name(value), []).append(i)

    return geoms, lookup


def write_layer(
    geoms: G.GeomArray,
    path: Path,
    predicted_value: int,
    source_crs: str,
    output_crs: str | None,
) -> None:
    """Write the final shapefile carrying a single int32 ``predicted`` column."""
    import geopandas as gpd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_crs = output_crs or source_crs
    geoms = _transform(
        geoms, CRS.from_user_input(source_crs), CRS.from_user_input(out_crs)
    )

    gdf = gpd.GeoDataFrame(
        {"predicted": np.full(len(geoms), int(predicted_value), dtype="int32")},
        geometry=list(geoms),
        crs=out_crs,
    )
    gdf.to_file(path, engine="pyogrio")


def write_intermediate(
    geoms: G.GeomArray, directory: Path, name: str, crs: str
) -> None:
    """Best-effort QA output. Never allowed to fail a district."""
    if len(geoms) == 0:
        return
    import geopandas as gpd

    try:
        directory.mkdir(parents=True, exist_ok=True)
        safe = name.replace(" ", "_").replace(".", "p")
        gdf = gpd.GeoDataFrame(geometry=list(geoms), crs=crs)
        gdf.to_file(directory / f"{safe}.shp", engine="pyogrio")
    except Exception as exc:
        log.debug("could not write intermediate %s: %s", name, exc)


def read_district_provinces(path: str | Path, district_field: str, province_field: str) -> dict[str, str]:
    """``{normalised district: normalised province}`` from the boundary file.

    Returns an empty mapping when the province column is absent, in which case
    discovery falls back to where most of a district's crops are filed.
    """
    from .discovery import norm_name

    info = pyogrio.read_info(str(path))
    if province_field not in list(info["fields"]):
        log.warning("boundary file has no '%s' column; province folders will be "
                    "resolved by majority", province_field)
        return {}
    table = pyogrio.read_dataframe(str(path), columns=[district_field, province_field],
                                   read_geometry=False)
    return {
        norm_name(d): norm_name(p)
        for d, p in zip(table[district_field], table[province_field])
        if d is not None and p is not None
    }
