import numpy as np
import pytest

import flexscanx as p


def test_clustersize_one_poisson():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [10, 5, 5]
    expected = [5.0, 5.0, 5.0]
    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="ORIGINAL",
        clustertype="HOT",
        clustersize=1,
        ralpha=0.2,
        K=3,
    )
    assert isinstance(res, dict)
    assert "area" in res and len(res["area"]) == 1


def test_extreme_zero_inputs():
    # Poisson expected counts are strictly positive: a zero denominator makes
    # the likelihood model undefined and must be rejected at the public boundary.
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]
    with pytest.raises(ValueError, match="strictly positive"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[0, 0],
            expected=[0.0, 0.0],
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            clustersize=2,
            ralpha=0.2,
            K=2,
        )
    # Binomial zeros
    res_b = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=[0, 0],
        population=[1, 1],
        model="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        ralpha=0.2,
        K=2,
    )
    assert float(res_b.get("stats", 0.0)) == 0.0


def test_mc_seed_effect_poisson():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 2, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    a = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=99,
        seed=123,
    )
    b = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=99,
        seed=456,
    )
    same_seed = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=99,
        seed=123,
    )
    assert same_seed["rank"] == a["rank"]
    assert same_seed["pval"] == a["pval"]
    assert (b["rank"], b["pval"]) != (a["rank"], a["pval"])


def test_k_equals_n_and_radius():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 2, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    # K==N, small radius so cluster cannot grow
    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=4,
        ralpha=0.2,
        K=4,
        clusterradius=0.5,
    )
    assert "area" in res and len(res["area"]) == 1


def test_series_both_mode_poisson():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = np.array([[10, 6, 1, 1], [8, 4, 2, 2]])
    exp = np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
    out = p.scan_series(
        observed=obs,
        expected=exp,
        coords=x,
        adjacency=(row_ptr, col_idx),
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="BOTH",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=0,
    )
    assert "per_time" in out and isinstance(out["per_time"], list) and len(out["per_time"]) == 2
