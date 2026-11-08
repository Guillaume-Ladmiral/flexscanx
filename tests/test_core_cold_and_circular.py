import numpy as np
import pytest

from flexscanx import scan


def test_stats_monotone_wrt_clustersize_cold():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [5, 1, 5]
    expected = [5.0, 5.0, 5.0]
    res1 = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="COLD",
        K=2,
        clustersize=1,
        simcount=0,
        knn_method="graph",
    )
    res2 = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="COLD",
        K=2,
        clustersize=2,
        simcount=0,
        knn_method="graph",
    )
    s1 = float(res1.get("stats", 0.0))
    s2 = float(res2.get("stats", 0.0))
    assert s2 >= s1 - 1e-12


def _is_knn_prefix(area_1based, centre_1based, coords):
    """True when ``area`` is a distance-ordered prefix of ``centre``'s neighbours.

    Ties at equal distance are broken by area index, which is the engine's rule.
    """
    centre = centre_1based - 1
    members = {a - 1 for a in area_1based}
    if centre not in members:
        return False
    dist = np.linalg.norm(coords - coords[centre], axis=1)
    key = sorted(range(len(coords)), key=lambda i: (dist[i], i))
    return set(key[: len(members)]) == members


def test_circular_restricted_hot_respects_radius_and_order():
    """Circular clusters are KNN prefixes bounded by clusterradius (FXA-169 made this load-bearing).

    Areas 1 and 3 carry the excess.  From centre 3 the distance order is
    (1 at 1.0, 4 at 1.0, 2 at 1.414); the index tie-break puts 1 before 4, so the
    best size-2 prefix is {1, 3}.  A radius of 0.99 excludes every neighbour and
    forces a single-area cluster.
    """
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    row_ptr = [0, 2, 4, 6, 8]
    col_idx = [1, 2, 0, 3, 0, 3, 1, 2]
    observed = [20, 4, 20, 5]
    expected = [5.0, 5.0, 5.0, 5.0]
    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=2,
        simcount=0,
        knn_method="graph",
    )
    res = scan(clusterradius=1.01, **common)
    area = res["area"]
    assert area == [1, 3], area
    assert len(area) <= 2
    assert res["max_dist"] == pytest.approx(1.0)
    assert any(_is_knn_prefix(area, centre, coords) for centre in area)

    with pytest.warns(UserWarning, match="clusterradius excludes nearly all neighbors"):
        tight = scan(clusterradius=0.99, **common)
    assert len(tight["area"]) == 1, tight["area"]
    assert tight["stats"] > 0.0
