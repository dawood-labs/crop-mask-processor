"""Very large polygons: erasing against them must be fast and exactly right.

The 2017 data has a single 218,000-acre sugarcane polygon with 237,602 vertices.
Every cotton field it touched re-unioned all of it, which turned one district's
cotton erase into an estimated 104 minutes. The fix cuts each eraser down to the
target's bounding box first. These tests pin both halves of that: the result is
identical to erasing against the whole eraser, and the whole eraser is no longer
what the expensive calls see.
"""

import logging

import numpy as np
import pytest
import shapely

from cropmask import geometry as G
from cropmask import telemetry


def arr(*geoms):
    return np.array(geoms, dtype=object)


def swiss_cheese(extent=20_000, holes_per_side=60, hole=40, quad_segs=8):
    """A huge polygon full of holes, like a raster-vectorised crop mask."""
    step = extent / holes_per_side
    rings = [
        shapely.Point(step / 2 + i * step, step / 2 + j * step)
        .buffer(hole, quad_segs=quad_segs).exterior.coords
        for i in range(holes_per_side)
        for j in range(holes_per_side)
    ]
    return shapely.Polygon(
        [(0, 0), (extent, 0), (extent, extent), (0, extent)], holes=rings
    )


GIANT = swiss_cheese()


def exact_erase(target, erasers):
    """The definition, with no optimisation at all."""
    out = target
    for e in erasers:
        out = shapely.difference(out, e)
    return out


def assert_same_ground(a, b, tol=1e-6):
    a = a if a is not None else shapely.Polygon()
    b = b if b is not None else shapely.Polygon()
    assert shapely.symmetric_difference(a, b).area == pytest.approx(0.0, abs=tol)


# --------------------------------------------------------------------------
# exactness
# --------------------------------------------------------------------------
def test_giant_is_the_kind_of_polygon_that_caused_the_problem():
    assert shapely.get_num_coordinates(GIANT) > 100_000
    assert len(GIANT.interiors) == 3_600


@pytest.mark.parametrize("seed", range(6))
def test_erasing_against_a_giant_matches_the_definition(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(500, 19_000, 2)
    target = shapely.box(x, y, x + rng.uniform(200, 900), y + rng.uniform(200, 900))
    small = shapely.box(x + 50, y + 50, x + 120, y + 180)
    erasers = arr(GIANT, small)

    got = G._erase_one(target, erasers)
    assert_same_ground(got, exact_erase(target, erasers))


def test_target_entirely_inside_the_eraser_is_erased_completely():
    target = shapely.box(100, 100, 110, 110)      # sits in solid ground, no hole
    eraser = shapely.box(-5_000, -5_000, 5_000, 5_000)
    got = G._erase_one(target, arr(eraser, shapely.box(105, 105, 108, 108)))
    assert got is None or shapely.is_empty(got) or got.area == pytest.approx(0.0)


def test_eraser_whose_edge_lies_exactly_on_the_target_edge():
    """The box margin exists so a cut can never meet the target's own boundary."""
    target = shapely.box(0, 0, 100, 100)
    eraser = shapely.box(100, -1_000, 5_000, 1_000)   # shares the x=100 edge only
    got = G._erase_one(target, arr(eraser, shapely.box(10, 10, 20, 20)))
    assert_same_ground(got, exact_erase(target, [eraser, shapely.box(10, 10, 20, 20)]))


def test_eraser_that_only_grazes_the_box_is_ignored_not_misapplied():
    target = shapely.box(0, 0, 100, 100)
    grazing = shapely.box(101, -500, 3_000, 500)      # inside the 1m margin only
    got = G._erase_one(target, arr(grazing, shapely.box(40, 40, 60, 60)))
    assert_same_ground(got, exact_erase(target, [grazing, shapely.box(40, 40, 60, 60)]))


def test_many_erasers_use_the_component_cutter_and_stay_exact():
    rng = np.random.default_rng(11)
    target = shapely.box(0, 0, 2_000, 2_000)
    # 400 erasers: overlapping, touching along edges, and disjoint, plus the giant
    pieces = [shapely.box(x, y, x + 30, y + 30) for x, y in rng.uniform(-50, 2_000, (300, 2))]
    pieces += [shapely.box(i * 30, 1_000, i * 30 + 30, 1_030) for i in range(60)]   # edge-touching run
    pieces += [GIANT.buffer(0).intersection(shapely.box(-100, -100, 2_100, 2_100))]
    erasers = arr(*pieces)
    assert len(erasers) > G.COMPONENT_CUTTER_THRESHOLD

    got = G._erase_one(target, erasers)
    assert_same_ground(got, exact_erase(target, erasers), tol=1e-4)


def test_whole_layer_erase_with_a_giant_matches_the_definition():
    rng = np.random.default_rng(3)
    targets = arr(*[
        shapely.box(x, y, x + 300, y + 300) for x, y in rng.uniform(0, 19_000, (150, 2))
    ])
    erasers = arr(GIANT, *[
        shapely.box(x, y, x + 80, y + 80) for x, y in rng.uniform(0, 19_000, (200, 2))
    ])
    got = G.erase(targets, erasers)
    expected = shapely.difference(shapely.union_all(targets), shapely.union_all(erasers))
    assert shapely.union_all(got).area == pytest.approx(expected.area, rel=1e-9)


# --------------------------------------------------------------------------
# a failed cut must fall back to the whole eraser, never drop it
# --------------------------------------------------------------------------
def test_failed_cut_falls_back_to_the_whole_eraser(monkeypatch):
    """Dropping an eraser whose cut failed would leave crop overlap silently."""
    target = shapely.box(1_000, 1_000, 1_500, 1_500)
    big = shapely.box(0, 0, 10_000, 10_000)

    def broken_intersection(a, b, **kw):
        raise shapely.errors.GEOSException("side location conflict")

    monkeypatch.setattr(shapely, "intersection", broken_intersection)
    local = G._local_cutters(target, arr(big))
    assert len(local) == 1 and local[0].equals(big)
    got = G._erase_one(target, arr(big))
    assert got is None or shapely.is_empty(got) or got.area == pytest.approx(0.0)


def test_invalid_cut_falls_back_to_the_whole_eraser(monkeypatch):
    target = shapely.box(1_000, 1_000, 1_500, 1_500)
    big = shapely.box(0, 0, 10_000, 10_000)
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])

    monkeypatch.setattr(
        shapely, "intersection",
        lambda a, b, **kw: np.array([bowtie] * len(a), dtype=object)
        if isinstance(a, np.ndarray) else bowtie,
    )
    local = G._local_cutters(target, arr(big))
    assert len(local) == 1 and local[0].equals(big)


# --------------------------------------------------------------------------
# the expensive calls must no longer see the whole giant
# --------------------------------------------------------------------------
def test_union_never_receives_the_whole_giant(monkeypatch):
    """Deterministic stand-in for a timing test: count what the union is fed."""
    seen = []
    real = G._union_all

    def spy(geoms):
        seen.append(int(shapely.get_num_coordinates(geoms).sum()))
        return real(geoms)

    monkeypatch.setattr(G, "_union_all", spy)
    target = shapely.box(5_000, 5_000, 5_400, 5_400)
    G._erase_one(target, arr(GIANT, shapely.box(5_100, 5_100, 5_200, 5_200)))

    giant_verts = shapely.get_num_coordinates(GIANT)
    assert seen, "expected a union for a target with two erasers"
    assert max(seen) < giant_verts / 100, (
        f"union saw {max(seen):,} vertices; the giant has {giant_verts:,}"
    )


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
def test_slow_operations_are_logged_with_context_and_sizes(monkeypatch, caplog):
    monkeypatch.setattr(telemetry, "SLOW_SECONDS", 0.0)
    telemetry.set_context("Punjab / RAHIM YAR KHAN / Cotton erase")
    with caplog.at_level(logging.WARNING, logger="cropmask.telemetry"):
        with telemetry.timed("erase", target=GIANT, erasers=arr(shapely.box(0, 0, 1, 1))):
            pass
    text = caplog.text
    assert "SLOW erase" in text
    assert "RAHIM YAR KHAN" in text
    assert f"{shapely.get_num_coordinates(GIANT):,} verts" in text


def test_fast_operations_are_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="cropmask.telemetry"):
        with telemetry.timed("erase", target=shapely.box(0, 0, 1, 1)):
            pass
    assert "SLOW" not in caplog.text


def test_stage_clock_accumulates_repeated_stages():
    clock = telemetry.StageClock()
    for _ in range(3):
        with clock.stage("erase"):
            pass
    with clock.stage("clip"):
        pass
    assert set(clock.seconds) == {"erase", "clip"}
    assert all(v >= 0 for v in clock.seconds.values())


# --------------------------------------------------------------------------
# which side of an intersects query the large polygon sits on
# --------------------------------------------------------------------------
def _brute_pairs(queries, candidates):
    hits = shapely.STRtree(candidates).query(queries, predicate="intersects")
    return set(zip(hits[0].tolist(), hits[1].tolist()))


def _mixed_layer(n, seed, heavy_every=7):
    """Boxes, with every Nth one a many-vertex circle so both sizes occur."""
    rng = np.random.default_rng(seed)
    out = []
    for k, (x, y) in enumerate(rng.uniform(0, 5_000, (n, 2))):
        if k % heavy_every == 0:
            out.append(shapely.Point(x, y).buffer(rng.uniform(100, 400), quad_segs=64))
        else:
            out.append(shapely.box(x, y, x + 40, y + 40))
    return np.array(out, dtype=object)


@pytest.mark.parametrize("seed", range(4))
def test_size_aware_pairs_match_a_plain_query_exactly(monkeypatch, seed):
    # Lower the threshold so small fixtures exercise all three query shapes:
    # heavy queries, light-vs-light, and light-vs-heavy asked in reverse.
    monkeypatch.setattr(G, "HEAVY_VERTICES", 200)
    queries = _mixed_layer(600, seed)
    candidates = _mixed_layer(500, seed + 100, heavy_every=5)
    got = G._intersecting_pairs(queries, candidates)
    got_set = set(zip(got[0].tolist(), got[1].tolist()))
    assert len(got_set) == got.shape[1], "a pair was returned twice"
    assert got_set == _brute_pairs(queries, candidates)


def test_size_aware_pairs_on_a_self_join(monkeypatch):
    monkeypatch.setattr(G, "HEAVY_VERTICES", 200)
    geoms = _mixed_layer(800, seed=9)
    got = G._intersecting_pairs(geoms, geoms)
    assert set(zip(got[0].tolist(), got[1].tolist())) == _brute_pairs(geoms, geoms)


def test_dissolve_with_heavy_polygons_still_equals_a_full_union(monkeypatch):
    monkeypatch.setattr(G, "HEAVY_VERTICES", 200)
    geoms = _mixed_layer(700, seed=4)
    got = G.dissolve(geoms)
    assert shapely.area(got).sum() == pytest.approx(shapely.union_all(geoms).area, rel=1e-9)


def test_a_light_geometry_is_never_tested_against_an_unprepared_heavy_one(monkeypatch):
    """The actual invariant behind the 146s -> 2.3s fix.

    Whenever a predicate query runs, if the tree holds a heavy geometry then the
    geometries being asked with must be heavy too - otherwise GEOS walks every
    vertex of the heavy one for each light query."""
    violations = []
    real_query = G._query
    # Fixed here rather than read from the module, so the test still measures
    # the invariant against code that predates the constant.
    heavy = getattr(G, "HEAVY_VERTICES", 5_000)

    def spy(tree, geoms, predicate):
        tree_heavy = shapely.get_num_coordinates(tree.geometries).max() >= heavy
        query_light = shapely.get_num_coordinates(geoms).min() < heavy
        if predicate and tree_heavy and query_light:
            violations.append((int(shapely.get_num_coordinates(tree.geometries).max()),
                               int(shapely.get_num_coordinates(geoms).min())))
        return real_query(tree, geoms, predicate)

    monkeypatch.setattr(G, "_query", spy)
    rng = np.random.default_rng(2)
    targets = arr(*[shapely.box(x, y, x + 200, y + 200)
                    for x, y in rng.uniform(0, 19_000, (300, 2))])
    small = arr(*[shapely.box(x, y, x + 50, y + 50)
                  for x, y in rng.uniform(0, 19_000, (100, 2))])

    G.erase(targets, np.concatenate([arr(GIANT), small]))       # giant as eraser
    G.erase(np.concatenate([arr(GIANT), targets]), small)       # giant as target
    G.dissolve(np.concatenate([arr(GIANT), targets]))           # giant in a self-join
    G.clip(targets, arr(GIANT))                                 # giant as the mask

    assert not violations, (
        f"light geometries were tested against unprepared heavy ones: {violations[:3]}"
    )


# --------------------------------------------------------------------------
# union with one very large member
# --------------------------------------------------------------------------
def _component_around_giant(n=120, seed=21):
    """A giant plus many small polygons overlapping its edge, as in a dissolve."""
    rng = np.random.default_rng(seed)
    small = [shapely.box(x, y, x + 60, y + 60)
             for x, y in rng.uniform(-40, 19_990, (n, 2))]
    return arr(GIANT, *small)


def test_heavy_last_union_equals_a_plain_union():
    geoms = _component_around_giant()
    got = G._union_all(geoms)
    expected = shapely.union_all(geoms)
    assert got.area == pytest.approx(expected.area, rel=1e-12)
    assert shapely.symmetric_difference(got, expected).area == pytest.approx(0.0, abs=1e-3)


def test_giant_takes_part_in_exactly_one_union(monkeypatch):
    """The cost fix: the giant must not be re-noded once per tree level."""
    giant_verts = shapely.get_num_coordinates(GIANT)
    calls_with_giant = []
    real_all, real_pair = shapely.union_all, shapely.union

    def spy_all(g, **kw):
        if int(shapely.get_num_coordinates(g).max()) >= giant_verts:
            calls_with_giant.append("union_all")
        return real_all(g, **kw)

    def spy_pair(a, b, **kw):
        if max(shapely.get_num_coordinates(a), shapely.get_num_coordinates(b)) >= giant_verts:
            calls_with_giant.append("union")
        return real_pair(a, b, **kw)

    monkeypatch.setattr(shapely, "union_all", spy_all)
    monkeypatch.setattr(shapely, "union", spy_pair)
    G._union_all(_component_around_giant())
    assert calls_with_giant == ["union"], calls_with_giant


def test_heavy_last_falls_back_when_the_fold_fails(monkeypatch):
    def broken_union(a, b, **kw):
        raise shapely.errors.GEOSException("found non-noded intersection")

    monkeypatch.setattr(shapely, "union", broken_union)
    geoms = _component_around_giant(n=30)
    got = G._union_all(geoms)            # falls back to the general union_all path
    assert got.area == pytest.approx(shapely.union_all(geoms).area, rel=1e-12)


def test_dissolve_with_a_giant_component_is_unchanged():
    geoms = _component_around_giant(n=200, seed=5)
    got = G.dissolve(geoms)
    assert shapely.area(got).sum() == pytest.approx(shapely.union_all(geoms).area, rel=1e-12)
    assert all(g.geom_type == "Polygon" for g in got)


# --------------------------------------------------------------------------
# repair method
# --------------------------------------------------------------------------
def test_repair_keeps_a_hole_that_touches_the_shell_along_an_edge():
    """The default "linework" repair fills this hole back in and reports it as crop."""
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(0, 2), (4, 2), (4, 6), (0, 6)]          # shares the x=0 edge
    broken = shapely.Polygon(shell, holes=[hole])
    assert not broken.is_valid
    assert shapely.make_valid(broken, method="linework").area == pytest.approx(100.0)
    assert G._repair_one(broken).area == pytest.approx(84.0)


def test_repair_does_not_count_overlapping_holes_as_crop():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    holes = [[(1, 1), (6, 1), (6, 6), (1, 6)], [(4, 4), (9, 4), (9, 9), (4, 9)]]
    broken = shapely.Polygon(shell, holes=holes)
    assert G._repair_one(broken).area == pytest.approx(100 - 46)


def test_repair_of_touching_rings_is_unchanged():
    """The common raster case: both methods must agree."""
    broken = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                             holes=[[(0, 0), (5, 2), (2, 5)]])
    assert G._repair_one(broken).area == pytest.approx(
        shapely.make_valid(broken, method="linework").area
    )
