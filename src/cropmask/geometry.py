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
#: overlay (seen on SAHIWAL in the 2025 data). Snapping the operands to a fixed
#: grid re-nodes them and the operation succeeds.
#:
#: The exact, unsnapped attempt always comes first because it is both the
#: fastest and the most precise. Recovery steps are kept few and coarse on
#: purpose: a failing overlay is expensive, so each extra rung costs real time,
#: and snapping finer than a millimetre rarely fixes anything a millimetre
#: cannot. 1 mm and 1 cm in UTM are far below any scale that can move an
#: acreage figure.
_GRID_LADDER = (None, 1e-3, 1e-2)


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

    # Divide and conquer: halve the set until the offending features end up in
    # a group small enough for GEOS to cope with, or are isolated and dropped.
    # Sequential accumulation would be O(n^2) against an ever-growing geometry,
    # which on a large component is slower than the failure it is recovering
    # from.
    log.warning("union of %d features failed; splitting", len(geoms))
    return _union_divide(geoms, last)


def _union_divide(geoms: GeomArray, last: Exception | None):
    if len(geoms) == 1:
        return geoms[0]

    mid = len(geoms) // 2
    halves = []
    for half in (geoms[:mid], geoms[mid:]):
        try:
            halves.append(_union_all(half))
        except shapely.errors.GEOSException:
            halves.append(None)

    left, right = halves
    if left is None and right is None:
        log.warning("dropped %d feature(s) GEOS could not union", len(geoms))
        raise last if last else shapely.errors.GEOSException("union failed")
    if left is None or right is None:
        return left if right is None else right

    for grid_size in _GRID_LADDER:
        try:
            return (
                shapely.union(left, right) if grid_size is None
                else shapely.union(left, right, grid_size=grid_size)
            )
        except shapely.errors.GEOSException:
            continue

    # The two halves cannot be joined at all; keep them side by side rather
    # than losing either.
    return shapely.union_all(np.array([left, right], dtype=object), grid_size=1e-2)


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

    try:
        bad = ~shapely.is_valid(geoms)
    except shapely.errors.GEOSException:
        bad = np.ones(len(geoms), dtype=bool)  # cannot tell: repair everything
    if bad.any():
        log.debug("repairing %d invalid geometries", int(bad.sum()))
        geoms = geoms.copy()
        geoms[bad] = _repair(geoms[bad])
        geoms = geoms[shapely.is_valid_input(geoms) & ~shapely.is_missing(geoms)]
        if len(geoms) == 0:
            return empty_array()

    # make_valid can turn a self-intersecting polygon into a GeometryCollection
    # that mixes lines and polygons; keep the polygonal parts only.
    mixed = shapely.get_type_id(geoms) == 7  # GeometryCollection
    if mixed.any():
        geoms = geoms.copy()
        geoms[mixed] = np.array(
            [_polygonal_parts_only(g) for g in geoms[mixed]], dtype=object
        )
        geoms = geoms[~shapely.is_missing(geoms)]

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


def _repair(geoms: GeomArray) -> GeomArray:
    """``make_valid`` that cannot take a layer down with it.

    ``make_valid`` is itself a GEOS overlay and raises the same topology
    exceptions as any other ("Ring edge missing" was seen on BAHAWALPUR,
    LODHRAN and KHANEWAL in the 2025 data). Repair the batch if possible,
    otherwise repair feature by feature, and fall back to a zero-width buffer -
    the old GEOS trick that re-nodes a ring without going through MakeValid.
    """
    try:
        return shapely.make_valid(geoms)
    except shapely.errors.GEOSException:
        pass

    log.warning("batch make_valid failed on %d feature(s); repairing individually",
                len(geoms))
    out = np.empty(len(geoms), dtype=object)
    for i, geom in enumerate(geoms):
        out[i] = _repair_one(geom)
    return out


def _repair_one(geom):
    try:
        return shapely.make_valid(geom)
    except shapely.errors.GEOSException:
        pass
    for grid_size in (1e-3, 1e-2):
        try:
            return shapely.make_valid(shapely.set_precision(geom, grid_size))
        except shapely.errors.GEOSException:
            continue
    try:
        return shapely.buffer(geom, 0)
    except shapely.errors.GEOSException:
        log.warning("dropped one feature that could not be repaired")
        return None


def _query(tree: "shapely.STRtree", geoms: GeomArray, predicate: str) -> np.ndarray:
    """R-tree query that degrades to bounding boxes rather than failing.

    Evaluating a predicate runs real geometry code and can raise. Bounding-box
    candidates are a superset of the true matches, so falling back to them
    costs a few redundant overlay calls but never changes the result: a pair
    that does not actually intersect contributes nothing to a difference or an
    intersection.
    """
    try:
        return tree.query(geoms, predicate=predicate)
    except shapely.errors.GEOSException as exc:
        log.warning("R-tree '%s' query failed (%s); falling back to bounding boxes",
                    predicate, exc)
        return tree.query(geoms)


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
    hits = _query(tree, targets, "intersects")
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
    hits = _query(tree, targets, "intersects")
    if hits.size:
        touching[hits[0]] = True

    kept = targets[touching]
    if len(kept) == 0:
        return empty_array()

    try:
        inside = shapely.contains_properly(mask_geom, kept)
    except shapely.errors.GEOSException:
        # Cannot prove containment: clip everything, which is always correct
        # and only costs the shortcut.
        inside = np.zeros(len(kept), dtype=bool)
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
    pairs = _query(tree, geoms, "intersects")
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
