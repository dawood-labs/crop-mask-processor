"""Vectorised geometry operations.

The naive way to run this pipeline is to hand whole layers to
``geopandas.overlay`` and ``GeoDataFrame.dissolve``. Both route through GEOS
operations whose cost grows super-linearly with the total vertex count of the
inputs, and on the 2025 data a single large district never finished.

Two observations make the work almost linear instead:

1.  Crop-mask polygons inside one layer barely touch each other (measured:
    ~1.6k intersecting pairs among ~11k features). A full dissolve therefore
    spends nearly all of its time re-noding polygons that are already disjoint.
    Unioning only the *connected components* gives a bit-identical result for a
    fraction of the cost.
2.  A difference only has to consider the erasers that actually touch the
    target. ``T \\ (E1 u E2 u ...) == T \\ (union of erasers intersecting T)``,
    so an R-tree pre-filter removes nearly every candidate pair, and untouched
    features are passed through without any GEOS call at all.

Everything here works on plain numpy object arrays of shapely geometries
rather than GeoDataFrames, which avoids repeated pandas index alignment.
"""

from __future__ import annotations

import logging

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .constants import SQM_PER_ACRE

log = logging.getLogger(__name__)

GeomArray = np.ndarray

POLYGONAL = {"Polygon", "MultiPolygon"}

#: Precision-reduction ladder used to recover from GEOS topology failures.
#:
#: Real crop-mask data contains near-degenerate rings that make GEOS raise
#: ``TopologyException: found non-noded intersection`` part-way through an
#: overlay. Snapping the operands to a fixed grid re-nodes them and the
#: operation succeeds. The steps run from 1 micrometre to 1 centimetre in the
#: metric CRS - far below anything that can move an acreage figure - and the
#: exact, unsnapped attempt is always tried first.
_GRID_LADDER = (None, 1e-6, 1e-4, 1e-2)


def _safe(op, *args, **kwargs):
    """Run a GEOS operation, retrying with increasing grid snapping.

    Falls back to element-wise evaluation as a last resort so that one
    unrecoverable polygon costs a single feature rather than the whole layer.
    """
    last: Exception | None = None
    for grid_size in _GRID_LADDER:
        try:
            if grid_size is None:
                return op(*args, **kwargs)
            return op(*args, grid_size=grid_size, **kwargs)
        except shapely.errors.GEOSException as exc:
            last = exc
            log.debug("%s failed at grid_size=%s: %s", op.__name__, grid_size, exc)

    if args and isinstance(args[0], np.ndarray) and args[0].size > 1:
        log.warning(
            "%s failed on %d features even with grid snapping; "
            "falling back to per-feature evaluation", op.__name__, args[0].size
        )
        return _elementwise(op, *args, **kwargs)

    raise last  # type: ignore[misc]


def _elementwise(op, first: GeomArray, *rest, **kwargs):
    out = np.empty(len(first), dtype=object)
    dropped = 0
    for i, geom in enumerate(first):
        operands = [geom] + [
            r[i] if isinstance(r, np.ndarray) and r.shape == first.shape else r
            for r in rest
        ]
        for grid_size in _GRID_LADDER:
            try:
                out[i] = (
                    op(*operands, **kwargs) if grid_size is None
                    else op(*operands, grid_size=grid_size, **kwargs)
                )
                break
            except shapely.errors.GEOSException:
                out[i] = None
        if out[i] is None:
            dropped += 1
    if dropped:
        log.warning("dropped %d feature(s) GEOS could not process", dropped)
    return out[out != None]  # noqa: E711 - numpy object comparison


def _union_all(geoms: GeomArray):
    """``shapely.union_all`` hardened against topology exceptions."""
    if len(geoms) == 1:
        return geoms[0]
    last: Exception | None = None
    for grid_size in _GRID_LADDER:
        try:
            if grid_size is None:
                return shapely.union_all(geoms)
            return shapely.union_all(geoms, grid_size=grid_size)
        except shapely.errors.GEOSException as exc:
            last = exc

    # Pairwise accumulation: isolates whichever feature GEOS cannot handle.
    log.warning("union of %d features failed; accumulating pairwise", len(geoms))
    acc = geoms[0]
    for geom in geoms[1:]:
        for grid_size in _GRID_LADDER:
            try:
                acc = (
                    shapely.union(acc, geom) if grid_size is None
                    else shapely.union(acc, geom, grid_size=grid_size)
                )
                break
            except shapely.errors.GEOSException:
                continue
        else:
            log.warning("skipped one feature GEOS could not union")
    if acc is None:
        raise last  # type: ignore[misc]
    return acc


def empty_array() -> GeomArray:
    return np.empty(0, dtype=object)


def clean(geoms: GeomArray) -> GeomArray:
    """Repair invalid rings and keep only non-empty polygonal geometry.

    Invalid input is the usual cause of a GEOS ``TopologyException`` halfway
    through a long run, so this runs before every overlay step.
    """
    if len(geoms) == 0:
        return empty_array()

    geoms = geoms[shapely.is_valid_input(geoms) & ~shapely.is_missing(geoms)]
    if len(geoms) == 0:
        return empty_array()

    bad = ~shapely.is_valid(geoms)
    if bad.any():
        log.debug("repairing %d invalid geometries", int(bad.sum()))
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])

    # make_valid can turn a self-intersecting polygon into a GeometryCollection
    # that mixes lines and polygons; keep the polygonal parts only.
    mixed = shapely.get_type_id(geoms) == 7  # GeometryCollection
    if mixed.any():
        geoms = geoms.copy()
        geoms[mixed] = np.array(
            [_polygonal_parts_only(g) for g in geoms[mixed]], dtype=object
        )

    keep = np.isin(shapely.get_type_id(geoms), [3, 6])  # Polygon, MultiPolygon
    geoms = geoms[keep]
    if len(geoms) == 0:
        return empty_array()
    return geoms[~shapely.is_empty(geoms)]


def _polygonal_parts_only(geom):
    """Collapse a GeometryCollection down to its polygonal content."""
    parts = [p for p in shapely.get_parts(geom) if p.geom_type in POLYGONAL]
    if not parts:
        return shapely.Polygon()
    if len(parts) == 1:
        return parts[0]
    return shapely.multipolygons(_flatten_polygons(parts))


def _flatten_polygons(parts):
    out = []
    for p in parts:
        if p.geom_type == "MultiPolygon":
            out.extend(shapely.get_parts(p))
        else:
            out.append(p)
    return out


def total_acres(geoms: GeomArray) -> float:
    """Sum of feature areas in acres. Input must already be in a metric CRS.

    This double-counts any self-overlap inside the layer, which is exactly what
    the legacy report did, so the numbers stay comparable.
    """
    if len(geoms) == 0:
        return 0.0
    return round(float(shapely.area(geoms).sum() / SQM_PER_ACRE), 2)


def union_acres(geoms: GeomArray) -> float:
    """Acreage of the dissolved layer (self-overlap counted once)."""
    if len(geoms) == 0:
        return 0.0
    return round(float(shapely.area(dissolve(geoms)).sum() / SQM_PER_ACRE), 2)


# ---------------------------------------------------------------------------
# difference
# ---------------------------------------------------------------------------
def erase(targets: GeomArray, erasers: GeomArray) -> GeomArray:
    """Remove every eraser from every target (QGIS ``Difference``).

    Only targets whose bounding box meets an eraser are touched; the rest are
    returned as-is. For each affected target only the erasers that actually
    intersect it are unioned, which keeps every GEOS call small.
    """
    if len(targets) == 0 or len(erasers) == 0:
        return targets

    tree = shapely.STRtree(erasers)
    hits = tree.query(targets, predicate="intersects")
    if hits.size == 0:
        return targets

    target_idx, eraser_idx = hits
    out = targets.copy()

    order = np.argsort(target_idx, kind="stable")
    target_idx, eraser_idx = target_idx[order], eraser_idx[order]
    starts = np.flatnonzero(np.r_[True, target_idx[1:] != target_idx[:-1]])
    ends = np.r_[starts[1:], len(target_idx)]

    for s, e in zip(starts, ends):
        t = target_idx[s]
        group = erasers[eraser_idx[s:e]]
        cutter = group[0] if len(group) == 1 else _union_all(group)
        out[t] = _safe(shapely.difference, out[t], cutter)

    return clean(out)


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------
def clip(targets: GeomArray, mask: GeomArray) -> GeomArray:
    """Keep only the part of each target inside ``mask`` (QGIS ``Clip``).

    Features entirely inside the mask skip the intersection entirely, which on
    district boundaries is the overwhelming majority of them.
    """
    if len(targets) == 0:
        return targets
    if len(mask) == 0:
        return empty_array()

    mask_geom = mask[0] if len(mask) == 1 else _union_all(mask)

    tree = shapely.STRtree(np.array([mask_geom], dtype=object))
    touching = np.zeros(len(targets), dtype=bool)
    hits = tree.query(targets, predicate="intersects")
    if hits.size:
        touching[hits[0]] = True

    kept = targets[touching]
    if len(kept) == 0:
        return empty_array()

    inside = shapely.contains_properly(mask_geom, kept)
    out = kept.copy()
    edge = ~inside
    if edge.any():
        out[edge] = _safe(shapely.intersection, kept[edge], mask_geom)

    return clean(out)


# ---------------------------------------------------------------------------
# dissolve
# ---------------------------------------------------------------------------
def dissolve(geoms: GeomArray, max_component_union: int = 100_000) -> GeomArray:
    """Dissolve to singleparts (QGIS ``Dissolve`` + ``Multipart to single``).

    Returns the individual polygons of the dissolved layer. Features that touch
    nothing else are emitted unchanged; each group of mutually touching
    features is unioned on its own.
    """
    if len(geoms) == 0:
        return empty_array()
    if len(geoms) == 1:
        return _explode(geoms)

    tree = shapely.STRtree(geoms)
    pairs = tree.query(geoms, predicate="intersects")
    pairs = pairs[:, pairs[0] != pairs[1]]

    if pairs.size == 0:
        return _explode(geoms)

    n = len(geoms)
    graph = coo_matrix(
        (np.ones(pairs.shape[1], dtype=np.int8), (pairs[0], pairs[1])), shape=(n, n)
    )
    n_comp, labels = connected_components(graph, directed=False)

    counts = np.bincount(labels, minlength=n_comp)
    singleton_mask = counts[labels] == 1

    results = [geoms[singleton_mask]]
    for comp in np.flatnonzero(counts > 1):
        members = geoms[labels == comp]
        if len(members) > max_component_union:
            log.warning(
                "dissolving an unusually large component of %d features", len(members)
            )
        results.append(np.array([_union_all(members)], dtype=object))

    merged = np.concatenate([r for r in results if len(r)])
    return _explode(merged)


def _explode(geoms: GeomArray) -> GeomArray:
    """Multipart -> singlepart."""
    if len(geoms) == 0:
        return empty_array()
    parts = shapely.get_parts(geoms)
    parts = parts[~shapely.is_empty(parts)]
    return parts


# ---------------------------------------------------------------------------
# area filter
# ---------------------------------------------------------------------------
def filter_by_area(geoms: GeomArray, min_acres: float) -> tuple[GeomArray, np.ndarray, int, float]:
    """Drop polygons of ``min_acres`` or less.

    Returns ``(kept, kept_acres, n_dropped, acres_dropped)``.
    """
    if len(geoms) == 0:
        return empty_array(), np.empty(0), 0, 0.0

    acres = np.round(shapely.area(geoms) / SQM_PER_ACRE, 2)
    keep = acres > min_acres
    dropped_acres = round(float(acres[~keep].sum()), 2)
    return geoms[keep], acres[keep], int((~keep).sum()), dropped_acres
