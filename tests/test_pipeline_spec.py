"""The crop-priority partition must reproduce the legacy chained differences."""

import numpy as np
import pytest
import shapely

from cropmask import geometry as G


def box(x0, y0, x1, y1):
    return shapely.box(x0, y0, x1, y1)


def arr(*geoms):
    return np.array(geoms, dtype=object)


def _concat(*arrays):
    parts = [a for a in arrays if len(a)]
    return np.concatenate(parts) if parts else G.empty_array()


def legacy_chain(rice, cotton, cane, maize):
    """Steps 1-3 exactly as the original script ordered them."""
    rice_d = G.erase(rice, maize)
    cane_d = G.erase(cane, maize)
    cotton_d = G.erase(cotton, maize)
    rice_d2 = G.erase(rice_d, cane_d)
    cotton_d2 = G.erase(cotton_d, cane_d)
    rice_d3 = G.erase(rice_d2, cotton_d2)
    return {"Rice": rice_d3, "Cotton": cotton_d2,
            "Sugarcane": cane_d, "Fall Maize": maize}


def fast_partition(rice, cotton, cane, maize):
    """What pipeline.process_district actually runs."""
    return {
        "Fall Maize": maize,
        "Sugarcane": G.erase(cane, maize),
        "Cotton": G.erase(cotton, _concat(maize, cane)),
        "Rice": G.erase(rice, _concat(maize, cane, cotton)),
    }


def test_partition_matches_the_legacy_chain_on_overlapping_layers():
    maize = arr(box(0, 0, 10, 10))
    cane = arr(box(5, 0, 15, 10))
    cotton = arr(box(8, 0, 20, 10))
    rice = arr(box(0, 0, 25, 10))

    legacy = legacy_chain(rice, cotton, cane, maize)
    fast = fast_partition(rice, cotton, cane, maize)

    for crop in legacy:
        a = shapely.union_all(legacy[crop]) if len(legacy[crop]) else shapely.Polygon()
        b = shapely.union_all(fast[crop]) if len(fast[crop]) else shapely.Polygon()
        assert pytest.approx(a.area, rel=1e-9) == b.area, crop
        assert a.symmetric_difference(b).area == pytest.approx(0.0, abs=1e-9), crop


def test_partition_matches_the_legacy_chain_on_random_layers():
    rng = np.random.default_rng(42)

    def layer(n, size):
        return arr(*[box(x, y, x + size, y + size)
                     for x, y in rng.integers(0, 50, (n, 2))])

    maize, cane, cotton, rice = layer(12, 6), layer(15, 5), layer(18, 7), layer(20, 8)

    legacy = legacy_chain(rice, cotton, cane, maize)
    fast = fast_partition(rice, cotton, cane, maize)

    for crop in legacy:
        a = shapely.union_all(legacy[crop]) if len(legacy[crop]) else shapely.Polygon()
        b = shapely.union_all(fast[crop]) if len(fast[crop]) else shapely.Polygon()
        assert a.symmetric_difference(b).area == pytest.approx(0.0, abs=1e-6), crop


def test_the_four_outputs_no_longer_overlap_each_other():
    rng = np.random.default_rng(3)

    def layer(n):
        return arr(*[box(x, y, x + 7, y + 7)
                     for x, y in rng.integers(0, 40, (n, 2))])

    result = fast_partition(layer(10), layer(10), layer(10), layer(10))
    merged = {k: shapely.union_all(v) if len(v) else shapely.Polygon()
              for k, v in result.items()}

    crops = list(merged)
    for i, a in enumerate(crops):
        for b in crops[i + 1:]:
            overlap = merged[a].intersection(merged[b]).area
            assert overlap == pytest.approx(0.0, abs=1e-6), f"{a} still overlaps {b}"
