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

from . import telemetry
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
#: fastest and the most precise, and only one recovery rung follows it: a
#: failing overlay is expensive, so each extra rung costs real time.
#:
#: 1 mm is the coarsest snap that is safe here. Snapping does not merely move
#: vertices - it pinches shut any neck narrower than the grid, which can split
#: one polygon into two. At 1 cm a 0.9-acre field with a 5 mm waist becomes two
#: 0.45-acre parts, and both are then deleted by the 0.5-acre rule. The area
#: displacement was never the risk; the topology change was.
_GRID_LADDER = (None, 1e-3)


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

    if args and isinstance(args[0], np.ndarray) and args[0].size >= 1:
        log.warning(
            "%s failed on %d features even with grid snapping; "
            "falling back to per-feature evaluation", op.__name__, args[0].size
        )
        return _elementwise(op, *args, **kwargs)

    raise last  # type: ignore[misc]


def _elementwise(op, first: GeomArray, *rest, **kwargs):
    """Evaluate ``op`` one feature at a time, keeping the array length.

    The result is written back into boolean-masked slices by the callers, so it
    must stay the same length as the input; returning only the successes made
    ``clip`` raise ValueError and cost the whole district - the opposite of
    what this fallback exists for. Failures are left as ``None`` and removed by
    the ``clean()`` every caller already runs.
    """
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
    return out


def _union_all(geoms: GeomArray):
    """``shapely.union_all`` hardened against topology exceptions."""
    if len(geoms) == 1:
        return geoms[0]

    heavy = shapely.get_num_coordinates(geoms) >= HEAVY_VERTICES
    if heavy.any() and not heavy.all():
        try:
            return _union_heavy_last(geoms, heavy)
        except shapely.errors.GEOSException:
            pass  # the general path below has the full recovery ladder

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


def _union_heavy_last(geoms: GeomArray, heavy: np.ndarray):
    """Union the light geometries together first, then fold in the heavy ones.

    ``union_all`` merges its inputs as a balanced tree, so one very large
    polygon is re-noded at every level of the tree it passes through - about
    log2(n) times for a group of n. The rice component around GUJRANWALA's
    1.33-million-vertex polygon in 2017 had 126 members and took 93 seconds
    that way. Unioning the 125 small members first and adding the large one
    once took 14 seconds for the same result. Union is associative and
    commutative, so the order changes nothing but the cost.

    Heavy geometries are folded in smallest first, so the largest is processed
    exactly once, at the end.
    """
    acc = _union_all(geoms[~heavy])
    big = geoms[heavy]
    for geom in big[np.argsort(shapely.get_num_coordinates(big), kind="stable")]:
        acc = _union_pair(acc, geom)
    return acc


def _union_pair(a, b):
    last: Exception | None = None
    for grid_size in _GRID_LADDER:
        try:
            return (shapely.union(a, b) if grid_size is None
                    else shapely.union(a, b, grid_size=grid_size))
        except shapely.errors.GEOSException as exc:
            last = exc
    raise last  # type: ignore[misc]


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
        # Returning the surviving half would be a silent, unlogged data loss,
        # and the callers cannot tell a partial union from a complete one. As
        # an eraser it leaves crop overlap; as a district mask it clips every
        # crop to half the district, with all four acreages falling together
        # and looking entirely plausible. Fail instead, so _erase_one falls
        # back to subtracting the erasers one at a time - which is correct.
        lost = mid if left is None else len(geoms) - mid
        log.error("union lost %d of %d feature(s) - refusing to return a "
                  "partial result", lost, len(geoms))
        raise last if last else shapely.errors.GEOSException("partial union")

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


def _make_valid(geoms):
    """Repair with GEOS's "structure" method.

    "structure" rebuilds a polygon as its shells minus its holes. The default
    "linework" method re-nodes every edge and polygonises the result, which is
    both slower and, for some invalid shapes, wrong about area:

    * On the 1,330,827-vertex rice polygon in GUJRANWALA 2017 - invalid only
      because rings touch at single points, the usual raster-vectorisation
      artefact - linework took 275.7 s and structure 62.2 s, producing the same
      polygon to within 0.000000 m2.
    * Where a hole touches its shell along an edge, linework fills the hole back
      in and reports it as crop; two overlapping holes likewise have their
      overlap counted as crop. Structure removes both.

    Across every invalid polygon in the 2024 season (669 of them) the two methods
    disagree on 18 and the whole season's acreage moves by 0.02 acres, so outputs
    made before this change remain comparable.
    """
    return shapely.make_valid(geoms, method="structure", keep_collapsed=False)


def _repair(geoms: GeomArray) -> GeomArray:
    """``make_valid`` that cannot take a layer down with it.

    ``make_valid`` is itself a GEOS overlay and raises the same topology
    exceptions as any other ("Ring edge missing" was seen on BAHAWALPUR,
    LODHRAN and KHANEWAL in the 2025 data). Repair the batch if possible,
    otherwise repair feature by feature, and fall back to a zero-width buffer -
    the old GEOS trick that re-nodes a ring without going through MakeValid.
    """
    try:
        return _make_valid(geoms)
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
        return _make_valid(geom)
    except shapely.errors.GEOSException:
        pass
    for grid_size in (1e-3, 1e-2):
        try:
            return _make_valid(shapely.set_precision(geom, grid_size))
        except shapely.errors.GEOSException:
            continue
    # There is deliberately no buffer(0) rung here. It is the traditional last
    # resort, but it is not area-preserving: on a self-intersecting ring it
    # applies winding semantics where make_valid preserves the covered area,
    # so a bow-tie comes back at 1.0 instead of 2.0. Worse, the loss cannot be
    # detected - GEOS reports the area of the invalid input as 0.0, so there is
    # no trustworthy figure to compare the repair against.
    #
    # By this point make_valid has failed at full precision and at both grid
    # sizes, so nothing here can be trusted. Drop the feature and say so: the
    # acreage loss is visible in the report, a silently halved polygon is not.
    log.warning("dropped one feature that could not be repaired")
    return None


def _difference_one(target, cutter):
    """Difference of two single geometries, or ``None`` if GEOS cannot do it.

    This is the one call in the pipeline with no cheaper correct fallback: an
    erase that silently does nothing leaves crop overlap in the deliverable,
    which is the exact thing the job exists to remove. So it tries hard -
    grid snapping, then repairing both operands and snapping again - and
    reports failure rather than returning the un-erased target.
    """
    for grid_size in _GRID_LADDER:
        try:
            return (
                shapely.difference(target, cutter) if grid_size is None
                else shapely.difference(target, cutter, grid_size=grid_size)
            )
        except shapely.errors.GEOSException:
            continue

    repaired_target, repaired_cutter = _repair_one(target), _repair_one(cutter)
    if repaired_target is None or repaired_cutter is None:
        return None
    for grid_size in _GRID_LADDER:
        try:
            return (
                shapely.difference(repaired_target, repaired_cutter)
                if grid_size is None
                else shapely.difference(
                    repaired_target, repaired_cutter, grid_size=grid_size
                )
            )
        except shapely.errors.GEOSException:
            continue
    return None


#: Margin, in metres, added around a target's bounding box before erasers are
#: cut down to it. It keeps the target strictly inside the box, so nothing the
#: box does at its own edges can ever touch the target's boundary.
LOCAL_MARGIN = 1.0

#: Above this many erasers on one target, the cutter is assembled from
#: connected components instead of one union over all of them.
COMPONENT_CUTTER_THRESHOLD = 32


def _local_cutters(target, erasers: GeomArray) -> GeomArray:
    """Only the part of each eraser that could possibly touch ``target``.

    A difference only ever depends on the eraser inside the target:

        T - E  ==  T - (E n B)     for any box B containing T

    so every eraser can be cut down to the target's box first. Without this,
    one very large eraser is reprocessed in full for every target it touches.
    In the 2017 data a single 218,000-acre sugarcane polygon (237,602 vertices)
    touched 1,005 cotton fields that each needed a union, and each union
    re-noded the whole polygon: 6.2 seconds apiece, about 104 minutes in total,
    for what is 2.7 ms once the eraser is cut to the field.

    The cut uses a real intersection rather than ``clip_by_rect``. Measured on
    all 8,288 affected targets, ``clip_by_rect`` is 2.5x faster but produced 612
    invalid geometries; ``intersection`` produced none, and both erased the same
    ground to within 4.5 square millimetres.

    If a cut fails or comes back invalid, the whole original eraser is used for
    that one feature. That is slower, never wrong: silently dropping an eraser
    would leave crop overlap in the output.
    """
    if len(erasers) == 0:
        return erasers

    xmin, ymin, xmax, ymax = shapely.bounds(target)
    m = LOCAL_MARGIN
    bxmin, bymin, bxmax, bymax = xmin - m, ymin - m, xmax + m, ymax + m

    eb = shapely.bounds(erasers)
    already_local = (
        (eb[:, 0] >= bxmin) & (eb[:, 1] >= bymin)
        & (eb[:, 2] <= bxmax) & (eb[:, 3] <= bymax)
    )
    if already_local.all():
        return erasers

    box = shapely.box(bxmin, bymin, bxmax, bymax)
    far = erasers[~already_local]
    cut = _safe(shapely.intersection, far, box)

    # Never trust a cut that failed or is invalid: fall back to the whole
    # eraser for that feature.
    usable = np.array([g is not None for g in cut], dtype=bool)
    if usable.any():
        usable[usable] = shapely.is_valid(cut[usable])
    cut[~usable] = far[~usable]

    # Cutting a polygon to a box can leave lines where it only grazes the box
    # edge; keep the polygonal part. A cut that is entirely outside the box
    # erases nothing and is dropped.
    mixed = shapely.get_type_id(cut) == 7
    if mixed.any():
        cut[mixed] = np.array([_polygonal_parts_only(g) for g in cut[mixed]], dtype=object)
    polygonal = np.isin(shapely.get_type_id(cut), [3, 6]) & ~shapely.is_empty(cut)

    out = erasers.copy()
    out[~already_local] = cut
    keep = np.ones(len(erasers), dtype=bool)
    keep[np.flatnonzero(~already_local)[~polygonal]] = False
    return out[keep]


def _cutter(erasers: GeomArray):
    """One geometry covering every eraser, built as cheaply as it can be.

    Erasers mostly do not touch each other, and a union over many disjoint
    polygons spends nearly all its time re-noding geometry that never needed
    merging - the same effect that made whole-layer dissolves unusable. Past a
    threshold, only the connected components are unioned; the components are
    pairwise disjoint, so a MultiPolygon of them is already a valid cutter.
    """
    if len(erasers) == 1:
        return erasers[0]
    if len(erasers) <= COMPONENT_CUTTER_THRESHOLD:
        return _union_all(erasers)

    parts = dissolve(erasers)
    candidate = shapely.multipolygons(list(parts))
    if shapely.is_valid(candidate):
        return candidate
    # Parts that touch along a line would make the MultiPolygon invalid; a real
    # union is correct in that case, just slower.
    return _union_all(parts)


def _erase_one(target, erasers: GeomArray):
    """Erase every eraser from one target, degrading as far as needed."""
    erasers = _local_cutters(target, erasers)
    if len(erasers) == 0:
        return target

    with telemetry.timed("erase", target=target, erasers=erasers):
        try:
            cutter = _cutter(erasers)
        except shapely.errors.GEOSException:
            cutter = None

        if cutter is not None:
            result = _difference_one(target, cutter)
            if result is not None:
                return result

    # The combined cutter is what GEOS choked on; subtract the erasers one at
    # a time instead. Each individual difference is a much simpler operation.
    log.debug("combined erase failed; subtracting %d eraser(s) one by one",
              len(erasers))
    accumulated = target
    failures = 0
    for eraser in erasers:
        step = _difference_one(accumulated, eraser)
        if step is None:
            failures += 1
            continue
        accumulated = step
        if shapely.is_empty(accumulated):
            return accumulated

    if failures:
        # Dropping the feature loses its acreage, which the report will show.
        # Keeping it would leave undetected crop overlap, which the report
        # would not.
        log.warning(
            "dropping 1 feature: %d eraser(s) could not be subtracted from it",
            failures,
        )
        return None
    return accumulated


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


#: Geometries with at least this many vertices are always put on the side of
#: an ``intersects`` query that GEOS prepares.
HEAVY_VERTICES = 5_000


def _intersecting_pairs(queries: GeomArray, candidates: GeomArray) -> np.ndarray:
    """``[i, j]`` for every ``queries[i]`` that intersects ``candidates[j]``.

    An R-tree predicate query prepares the geometry it is asked *with* and tests
    it against the geometries stored *in* the tree. Preparing is what makes a
    test cheap, so a small geometry tested against a very large unprepared one
    walks every vertex of the large one - once per small geometry whose bounding
    box overlaps it. On the 2017 data that made the cotton lookup for RAHIM YAR
    KHAN take 146 seconds against a 237,602-vertex sugarcane polygon whose
    bounding box spans most of the district. Asked the other way round, so the
    large polygon is prepared once, the same lookup takes 2.3 seconds and
    returns the identical 32,363 pairs.

    Simply swapping the direction is not enough, because large polygons turn up
    on either side - as erasers in one district, as targets in another, and on
    both sides of the self-join in ``dissolve``. So the inputs are split by size
    and asked three ways, each with the heavy side doing the asking:

        heavy queries   ->  all candidates
        light queries   ->  light candidates
        light queries   <-  heavy candidates      (asked in reverse)

    The three cover every ordered pair exactly once, and no light geometry is
    ever tested against an unprepared heavy one. ``intersects`` is symmetric,
    so asking in reverse returns the same pairs.
    """
    empty = np.empty((2, 0), dtype=np.intp)
    if len(queries) == 0 or len(candidates) == 0:
        return empty

    q_heavy = shapely.get_num_coordinates(queries) >= HEAVY_VERTICES
    c_heavy = shapely.get_num_coordinates(candidates) >= HEAVY_VERTICES
    parts = []

    heavy_q = np.flatnonzero(q_heavy)
    if len(heavy_q):
        hits = _query(shapely.STRtree(candidates), queries[heavy_q], "intersects")
        parts.append(np.vstack([heavy_q[hits[0]], hits[1]]))

    light_q = np.flatnonzero(~q_heavy)
    if len(light_q):
        light_c = np.flatnonzero(~c_heavy)
        if len(light_c):
            hits = _query(shapely.STRtree(candidates[light_c]), queries[light_q],
                          "intersects")
            parts.append(np.vstack([light_q[hits[0]], light_c[hits[1]]]))

        heavy_c = np.flatnonzero(c_heavy)
        if len(heavy_c):
            hits = _query(shapely.STRtree(queries[light_q]), candidates[heavy_c],
                          "intersects")
            parts.append(np.vstack([light_q[hits[1]], heavy_c[hits[0]]]))

    parts = [part for part in parts if part.size]
    if not parts:
        return empty
    return np.concatenate(parts, axis=1)


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

    hits = _intersecting_pairs(targets, erasers)
    if hits.size == 0:
        return targets

    target_idx, eraser_idx = hits
    out = targets.copy()

    order = np.argsort(target_idx, kind="stable")
    target_idx, eraser_idx = target_idx[order], eraser_idx[order]
    starts = np.flatnonzero(np.r_[True, target_idx[1:] != target_idx[:-1]])
    ends = np.r_[starts[1:], len(target_idx)]

    dropped = 0
    for s, e in zip(starts, ends):
        t = target_idx[s]
        result = _erase_one(out[t], erasers[eraser_idx[s:e]])
        if result is None:
            dropped += 1
        out[t] = result

    if dropped:
        log.warning("%d of %d feature(s) could not be erased and were dropped",
                    dropped, len(targets))
        out = out[out != None]  # noqa: E711 - numpy object comparison

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

    touching = np.zeros(len(targets), dtype=bool)
    hits = _intersecting_pairs(targets, np.array([mask_geom], dtype=object))
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

    pairs = _intersecting_pairs(geoms, geoms)
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

    # Compare the unrounded area. Rounding first makes the effective threshold
    # 0.505 acres, not 0.5: anything in (0.5000, 0.5050] rounds to 0.50 and is
    # dropped even though the spec says to keep it. The bias is one-sided - it
    # only ever deletes land - so on a large district it quietly removes
    # hundreds of acres.
    raw = shapely.area(geoms) / SQM_PER_ACRE
    keep = raw > min_acres
    dropped_acres = round(float(raw[~keep].sum()), 2)
    return geoms[keep], np.round(raw[keep], 2), int((~keep).sum()), dropped_acres
