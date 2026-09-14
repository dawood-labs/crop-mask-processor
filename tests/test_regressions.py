"""Regressions for defects found auditing the first full 2025 run.

Each test names the failure it locks out. They are grouped here rather than
spread across the unit-test files because what they have in common is the
failure mode, not the module: every one of them produced plausible-looking
output instead of an error.
"""

import numpy as np
import pytest
import shapely

from cropmask import geometry as G
from cropmask.constants import SQM_PER_ACRE


def acres_box(acres: float):
    side = (acres * SQM_PER_ACRE) ** 0.5
    return shapely.box(0, 0, side, side)


def arr(*geoms):
    return np.array(geoms, dtype=object)


# --------------------------------------------------------------------------
# area threshold
# --------------------------------------------------------------------------
@pytest.mark.parametrize("true_acres", [0.5001, 0.5020, 0.5049, 0.5051, 0.6])
def test_polygons_above_the_threshold_are_kept(true_acres):
    """Rounding to 2dp before comparing made the real cut-off 0.505 acres.

    The bias was one-sided - it only ever deleted land - so on a district with
    200k output polygons it quietly removed hundreds of acres.
    """
    kept, _, _, _ = G.filter_by_area(arr(acres_box(true_acres)), 0.5)
    assert len(kept) == 1, f"{true_acres} acres is above 0.5 and must be kept"


@pytest.mark.parametrize("true_acres", [0.5, 0.4999, 0.25])
def test_polygons_at_or_below_the_threshold_are_dropped(true_acres):
    kept, _, _, _ = G.filter_by_area(arr(acres_box(true_acres)), 0.5)
    assert len(kept) == 0


def test_dropped_acreage_is_measured_unrounded():
    _, _, n_dropped, dropped_acres = G.filter_by_area(
        arr(acres_box(0.4), acres_box(0.3)), 0.5
    )
    assert n_dropped == 2
    assert dropped_acres == pytest.approx(0.7, abs=0.01)


# --------------------------------------------------------------------------
# failure-path shape
# --------------------------------------------------------------------------
def test_elementwise_fallback_preserves_array_length(monkeypatch):
    """A shorter result cannot be written back into a boolean-masked slice.

    clip() assigns into `out[edge]`, so a fallback that returned only the
    successes raised ValueError and cost all four crops of the district - the
    opposite of what the fallback exists for.
    """
    real = shapely.intersection
    bad = shapely.box(2, 2, 3, 3)

    def one_bad_feature(a, b, **kwargs):
        if a is bad:
            raise shapely.errors.GEOSException("side location conflict")
        return real(a, b, **kwargs)

    monkeypatch.setattr(shapely, "intersection", one_bad_feature)
    geoms = arr(shapely.box(0, 0, 1, 1), bad)
    out = G._elementwise(shapely.intersection, geoms, shapely.box(0, 0, 10, 10))
    assert len(out) == len(geoms), "length must survive a per-feature failure"


def test_clip_survives_a_geos_failure_on_one_feature(monkeypatch):
    real = shapely.intersection
    calls = {"n": 0}

    doomed = shapely.box(8, 0, 12, 5)

    def flaky(a, b, **kwargs):
        calls["n"] += 1
        # the whole-array call fails, and so does one individual feature
        if isinstance(a, np.ndarray) or a is doomed:
            raise shapely.errors.GEOSException("side location conflict")
        return real(a, b, **kwargs)

    monkeypatch.setattr(shapely, "intersection", flaky)
    # two features straddling the mask edge, so neither is contained outright
    targets = arr(shapely.box(-1, 0, 5, 5), doomed)
    out = G.clip(targets, arr(shapely.box(0, 0, 10, 10)))
    assert calls["n"] > 1                      # the vectorised call failed first
    # the unrecoverable feature is lost, but the other one - and the district
    # it belongs to - survives
    assert len(out) == 1
    assert pytest.approx(shapely.area(out).sum()) == 25.0


def test_partial_union_is_refused_rather_than_returned(monkeypatch):
    """Returning the surviving half is a silent data loss.

    As an eraser it leaves crop overlap; as a district mask it clips every crop
    to half the district, with all four acreages falling together and looking
    entirely plausible.
    """
    geoms = arr(*[shapely.box(i, 0, i + 0.5, 1) for i in range(8)])
    real = G._union_all

    def fail_on_the_second_half(g, **kwargs):
        if len(g) and g[0].bounds[0] >= 4:
            raise shapely.errors.GEOSException("found non-noded intersection")
        return real(g, **kwargs)

    monkeypatch.setattr(G, "_union_all", fail_on_the_second_half)
    with pytest.raises(shapely.errors.GEOSException):
        G._union_divide(geoms, None)


def test_erase_degrades_to_one_eraser_at_a_time_when_the_union_fails(monkeypatch):
    """The combined cutter is usually what GEOS chokes on."""
    real_union = shapely.union_all

    def fail_on_groups(g, **kwargs):
        if isinstance(g, np.ndarray) and len(g) > 1:
            raise shapely.errors.GEOSException("found non-noded intersection")
        return real_union(g, **kwargs)

    monkeypatch.setattr(shapely, "union_all", fail_on_groups)
    target = arr(shapely.box(0, 0, 10, 10))
    erasers = arr(shapely.box(0, 0, 4, 10), shapely.box(6, 0, 10, 10))
    out = G.erase(target, erasers)
    assert pytest.approx(shapely.area(out).sum()) == 20.0


def test_unrepairable_feature_is_dropped_not_silently_reshaped(monkeypatch):
    """buffer(0) applies winding semantics and can halve a ring.

    A bow-tie is 2.0 acres under make_valid and 1.0 under buffer(0), and the
    loss is undetectable because GEOS reports the invalid input's area as 0.0.
    Once make_valid has failed at every precision, dropping the feature is the
    only honest option - the lost acreage shows up in the report, a silently
    halved polygon does not.
    """
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    assert shapely.buffer(bowtie, 0).area == pytest.approx(1.0)
    assert shapely.make_valid(bowtie).area == pytest.approx(2.0)
    assert shapely.area(bowtie) == 0.0  # why the loss cannot be detected

    def no_make_valid(g, **kw):
        raise shapely.errors.GEOSException("Ring edge missing")

    monkeypatch.setattr(shapely, "make_valid", no_make_valid)
    assert G._repair_one(bowtie) is None


def test_grid_ladder_does_not_snap_coarser_than_a_millimetre():
    """A 1cm snap pinches shut necks and splits polygons in two.

    A 0.9-acre field with a 5mm waist becomes two 0.45-acre parts, and the
    0.5-acre rule then deletes both.
    """
    assert max(g for g in G._GRID_LADDER if g is not None) <= 1e-3


# --------------------------------------------------------------------------
# an unreadable crop must not silently disable de-overlap
# --------------------------------------------------------------------------
def test_only_crops_below_the_failure_are_invalidated():
    """Erasing with an empty layer is the identity, so a crop that failed to
    load silently stops cutting the crops beneath it - which are then published
    as clean while still carrying its overlap.

    Only crops *below* the failure in CROP_ORDER are affected. HYDERABAD in the
    2025 data ships Rice with no .prj; Rice is last, so it erases nothing and
    the other three layers are genuinely unaffected.
    """
    from cropmask.constants import CROP_ORDER

    assert CROP_ORDER == ["Fall Maize", "Sugarcane", "Cotton", "Rice"]

    # a failure in the last crop blocks nothing
    assert CROP_ORDER[: CROP_ORDER.index("Rice")] == [
        "Fall Maize", "Sugarcane", "Cotton",
    ]
    # a failure in Sugarcane blocks Cotton and Rice, not Fall Maize
    position = CROP_ORDER.index("Sugarcane")
    below = CROP_ORDER[position + 1:]
    assert below == ["Cotton", "Rice"]
    assert "Fall Maize" not in below
