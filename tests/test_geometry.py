"""The fast primitives must agree with the obvious slow definitions."""

import numpy as np
import pytest
import shapely

from cropmask import geometry as G
from cropmask.constants import SQM_PER_ACRE

ACRE = SQM_PER_ACRE


def box(x0, y0, x1, y1):
    return shapely.box(x0, y0, x1, y1)


def arr(*geoms):
    return np.array(geoms, dtype=object)


# --------------------------------------------------------------------- erase
def test_erase_removes_the_overlapping_part():
    targets = arr(box(0, 0, 10, 10))
    erasers = arr(box(5, 0, 15, 10))
    out = G.erase(targets, erasers)
    assert pytest.approx(shapely.area(out).sum()) == 50.0


def test_erase_passes_through_untouched_features():
    targets = arr(box(0, 0, 1, 1), box(100, 100, 101, 101))
    erasers = arr(box(0, 0, 1, 1))
    out = G.erase(targets, erasers)
    assert len(out) == 1
    assert pytest.approx(shapely.area(out).sum()) == 1.0


def test_erase_unions_several_overlapping_erasers():
    targets = arr(box(0, 0, 10, 10))
    erasers = arr(box(0, 0, 6, 10), box(4, 0, 10, 10))  # together cover everything
    assert len(G.erase(targets, erasers)) == 0


def test_erase_with_no_erasers_is_identity():
    targets = arr(box(0, 0, 1, 1))
    assert G.erase(targets, G.empty_array()) is targets


def test_erase_matches_a_naive_whole_layer_difference():
    rng = np.random.default_rng(0)
    targets = arr(*[box(x, y, x + 3, y + 3) for x, y in rng.integers(0, 40, (30, 2))])
    erasers = arr(*[box(x, y, x + 2, y + 2) for x, y in rng.integers(0, 40, (25, 2))])

    fast = shapely.union_all(G.erase(targets, erasers))
    naive = shapely.difference(
        shapely.union_all(targets), shapely.union_all(erasers)
    )
    assert pytest.approx(fast.area, rel=1e-9) == naive.area


# ---------------------------------------------------------------------- clip
def test_clip_keeps_only_the_part_inside_the_mask():
    out = G.clip(arr(box(0, 0, 10, 10)), arr(box(0, 0, 4, 10)))
    assert pytest.approx(shapely.area(out).sum()) == 40.0


def test_clip_drops_features_entirely_outside():
    out = G.clip(arr(box(50, 50, 60, 60)), arr(box(0, 0, 10, 10)))
    assert len(out) == 0


def test_clip_leaves_fully_contained_features_untouched():
    inner = box(1, 1, 2, 2)
    out = G.clip(arr(inner), arr(box(0, 0, 10, 10)))
    assert len(out) == 1
    assert out[0].equals(inner)


def test_clip_with_empty_mask_yields_nothing():
    assert len(G.clip(arr(box(0, 0, 1, 1)), G.empty_array())) == 0


# ------------------------------------------------------------------ dissolve
def test_dissolve_merges_touching_polygons():
    out = G.dissolve(arr(box(0, 0, 5, 5), box(5, 0, 10, 5)))
    assert len(out) == 1
    assert pytest.approx(shapely.area(out).sum()) == 50.0


def test_dissolve_keeps_disjoint_polygons_separate():
    out = G.dissolve(arr(box(0, 0, 1, 1), box(50, 50, 51, 51)))
    assert len(out) == 2


def test_dissolve_counts_overlap_once():
    out = G.dissolve(arr(box(0, 0, 10, 10), box(5, 5, 15, 15)))
    assert len(out) == 1
    assert pytest.approx(shapely.area(out).sum()) == 175.0


def test_dissolve_matches_union_all_on_a_random_layer():
    rng = np.random.default_rng(7)
    geoms = arr(*[box(x, y, x + 4, y + 4) for x, y in rng.integers(0, 60, (80, 2))])
    fast = G.dissolve(geoms)
    assert pytest.approx(shapely.area(fast).sum(), rel=1e-9) == shapely.union_all(geoms).area
    # every returned part must be a singlepart polygon
    assert all(g.geom_type == "Polygon" for g in fast)


def test_dissolve_explodes_multipolygons():
    multi = shapely.multipolygons([box(0, 0, 1, 1), box(10, 10, 11, 11)])
    assert len(G.dissolve(arr(multi))) == 2


# --------------------------------------------------------------------- clean
def test_clean_repairs_a_self_intersecting_polygon():
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    assert not bowtie.is_valid
    out = G.clean(arr(bowtie))
    assert len(out) and all(shapely.is_valid(out))


def test_clean_drops_non_polygonal_and_empty_geometry():
    out = G.clean(arr(shapely.LineString([(0, 0), (1, 1)]), shapely.Polygon(),
                      box(0, 0, 1, 1)))
    assert len(out) == 1


# ------------------------------------------------------------ area filtering
def test_filter_by_area_applies_a_strict_greater_than():
    side = (0.5 * ACRE) ** 0.5          # exactly 0.5 acre
    big = (2.0 * ACRE) ** 0.5
    kept, acres, dropped, dropped_acres = G.filter_by_area(
        arr(box(0, 0, side, side), box(100, 100, 100 + big, 100 + big)), 0.5
    )
    assert len(kept) == 1 and dropped == 1
    assert pytest.approx(acres.sum(), abs=0.01) == 2.0
    assert pytest.approx(dropped_acres, abs=0.01) == 0.5


def test_total_acres_double_counts_self_overlap_like_the_legacy_report():
    side = ACRE ** 0.5
    geoms = arr(box(0, 0, side, side), box(0, 0, side, side))
    assert pytest.approx(G.total_acres(geoms), abs=0.01) == 2.0
    assert pytest.approx(G.union_acres(geoms), abs=0.01) == 1.0


# ------------------------------------------------------- empty-input contract
@pytest.mark.parametrize("call", [
    lambda: G.erase(G.empty_array(), G.empty_array()),
    lambda: G.clip(G.empty_array(), G.empty_array()),
    lambda: G.dissolve(G.empty_array()),
    lambda: G.clean(G.empty_array()),
])
def test_primitives_accept_empty_input(call):
    assert len(call()) == 0


# ----------------------------------------------- GEOS robustness (regression)
def test_union_falls_back_when_geos_raises(monkeypatch):
    """A TopologyException must not lose the layer.

    Real 2025 data (SAHIWAL) hit ``TopologyException: found non-noded
    intersection`` inside ``union_all``; the grid-snapping ladder recovers it.
    """
    calls = {"n": 0}
    real = shapely.union_all

    def flaky(geoms, **kwargs):
        calls["n"] += 1
        if "grid_size" not in kwargs:
            raise shapely.errors.GEOSException("found non-noded intersection")
        return real(geoms, **kwargs)

    monkeypatch.setattr(shapely, "union_all", flaky)
    out = G.dissolve(arr(box(0, 0, 5, 5), box(4, 0, 10, 5)))
    assert calls["n"] >= 2
    assert len(out) == 1
    assert pytest.approx(shapely.area(out).sum(), rel=1e-6) == 50.0


def test_difference_falls_back_when_geos_raises(monkeypatch):
    real = shapely.difference

    def flaky(a, b, **kwargs):
        if "grid_size" not in kwargs:
            raise shapely.errors.GEOSException("found non-noded intersection")
        return real(a, b, **kwargs)

    monkeypatch.setattr(shapely, "difference", flaky)
    out = G.erase(arr(box(0, 0, 10, 10)), arr(box(5, 0, 15, 10)))
    assert pytest.approx(shapely.area(out).sum(), rel=1e-6) == 50.0
