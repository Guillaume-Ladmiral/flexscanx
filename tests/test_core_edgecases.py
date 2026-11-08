import numpy as np
import pytest

from flexscanx import scan


def test_zero_expected_rows_are_rejected_before_scanning():
    # Zero-exposure Poisson rows are outside the public likelihood contract.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [0, 5, 0]
    expected = [0.0, 5.0, 0.0]  # zero expected in neighbors
    with pytest.raises(ValueError, match="strictly positive"):
        scan(
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            K=2,
            clustersize=2,
            simcount=0,
            knn_method="graph",
        )


def _unit_square_fixture():
    # N=4 unit square; orthogonal neighbours are exactly 1.0 apart, diagonals ~1.414.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    row_ptr = [0, 2, 4, 6, 8]
    col_idx = [1, 2, 0, 3, 0, 3, 1, 2]
    return coords, row_ptr, col_idx


def test_radius_strict_no_out_of_universe_inclusions():
    """A finite clusterradius bounds the centre universe (FXA-169 made this load-bearing).

    Areas 1 and 4 carry a strong excess and area 2 a moderate one (so it passes
    the RESTRICTED gate).  Every neighbour is exactly 1.0 away, so a radius of
    0.99 leaves each centre alone in its universe and the reported cluster must
    be a single area even though the adjacency would allow growth.  A radius of
    1.01 admits the orthogonal neighbours and the cluster {1, 2, 4} forms
    through area 2.
    """
    coords, row_ptr, col_idx = _unit_square_fixture()
    observed = [20, 12, 5, 20]
    expected = [5.0, 5.0, 5.0, 5.0]
    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        simcount=0,
        knn_method="graph",
        knn_mode="K_REFILL",
    )
    with pytest.warns(UserWarning, match="clusterradius excludes nearly all neighbors"):
        strict = scan(clusterradius=0.99, **common)
    assert strict["area"], "the planted excess must be detected"
    assert len(strict["area"]) == 1, strict["area"]
    assert strict["stats"] > 0.0
    assert strict["max_dist"] == 0.0
    assert all(1 <= a <= 4 for a in strict["area"])

    loose = scan(clusterradius=1.01, **common)
    assert loose["area"] == [1, 2, 4], loose["area"]
    assert loose["stats"] > strict["stats"]
    # The diagonal pair (1, 4) is 1.414 apart, beyond the radius from either of
    # them; it is reachable only through the orthogonal neighbour 2, and the
    # reported max_dist may be up to twice the radius.
    assert loose["max_dist"] == pytest.approx(2**0.5)
    assert loose["max_dist"] <= 2 * 1.01


def test_krefill_center_ub_gating_matches_refill_secondary_search():
    # Reproducer for Phase 3 audit finding:
    # secondary exclusion removes a top-K neighbor of center 4, K_REFILL admits area 1,
    # and center-level UB must consider that refilled candidate instead of only the fixed K list.
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ]
    )
    row_ptr = [0, 3, 4, 5, 6, 6]
    col_idx = [1, 2, 3, 0, 0, 0]
    observed = [27, 4, 19, 22, 5]
    expected = [
        7.734015714465792,
        5.476211553993605,
        1.7864442085330015,
        4.478026656592159,
        7.565033026773892,
    ]

    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        simcount=0,
        secondary=1,
        knn_method="graph",
        knn_mode="K_REFILL",
        knn_refill_force=True,
        n_jobs_centers=1,
    )

    res_on = scan(enable_center_ub_gating=True, **common)
    res_off = scan(enable_center_ub_gating=False, **common)

    assert res_on["area"] == [3]
    assert res_on["area"] == res_off["area"]
    assert res_on["secondaries"][0]["area"] == [1, 4]
    assert res_on["secondaries"][0]["area"] == res_off["secondaries"][0]["area"]
    assert res_on["secondaries"][0]["stats"] == res_off["secondaries"][0]["stats"]


def test_restricted_poisson_high_counts_do_not_collapse_to_empty_cluster():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [900, 800, 800]
    expected = [800.0, 800.0, 800.0]

    res_restricted = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=2,
        clustersize=2,
        simcount=0,
        knn_method="graph",
    )
    res_original = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="ORIGINAL",
        clustertype="HOT",
        K=2,
        clustersize=2,
        simcount=0,
        knn_method="graph",
    )

    assert res_original["stats"] > 0.0
    assert res_restricted["stats"] > 0.0
    assert 1 in res_restricted["area"]
