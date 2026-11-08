import numpy as np

import flexscanx._core as core
from flexscanx import scan

# ---------------------------------------------------------------------------
# Binomial middle-p numerical correctness at extreme p values (S2-1 regression)
# ---------------------------------------------------------------------------


def test_binomial_middle_p_upper_extreme_p_near_zero():
    """binomial_middle_p_upper well above the mean (p near 0) should be near 0."""
    # X ~ Bin(1000, 1e-6): mean = 0.001. P(X > 5) + 0.5*P(X=5) is negligible.
    # Note: midp_upper(0) is NOT tiny — it equals P(X>0)+0.5*P(X=0) ≈ 0.5, so use k >> mean.
    result = core.binomial_middle_p_upper(5, 1000, 1e-6)
    assert 0.0 <= result <= 1.0
    assert result < 1e-10  # negligible upper tail far above the mean


def test_binomial_middle_p_lower_extreme_p_near_one():
    """binom_cdf complement fix: recurrence underflow prevented for large p."""
    # Before the fix: binom_cdf(49, 50, 0.99) returns 0 due to underflow (pmf0=0.01^50≈0).
    # After fix: complement path binom_cdf(49,50,0.99) = 1 - binom_cdf(0,50,0.01) ≈ 0.395.
    # The midp identity lo+hi==1 verifies the fix (tested more precisely in _lower_plus_upper).
    n = 1000
    p = 0.9999
    # k = n (all observations): lower-tail middle-p = F(n) - 0.5*PMF(n) ≈ 1 - 0.5*p^n ≈ 1
    result_k_n = core.binomial_middle_p_lower(n, n, p)
    assert 0.0 <= result_k_n <= 1.0

    # k = 0 (no observations): lower-tail middle-p = 0.5*P(X=0) = 0.5*(1-p)^n ≈ tiny
    result_k_0 = core.binomial_middle_p_lower(0, n, p)
    assert 0.0 <= result_k_0 <= 1.0
    assert result_k_0 < 1e-10  # (1-0.9999)^1000 = 1e-4000 ≈ 0


def test_binomial_middle_p_lower_plus_upper_equals_one():
    """midp_lower(ax) + midp_upper(ax) must equal 1 for all valid ax."""
    for n, p, k in [(10, 0.3, 5), (100, 0.9, 90), (50, 0.99, 49), (200, 0.01, 2)]:
        lo = core.binomial_middle_p_lower(k, n, p)
        hi = core.binomial_middle_p_upper(k, n, p)
        assert abs(lo + hi - 1.0) < 1e-10, f"n={n} p={p} k={k}: lo+hi={lo+hi}"


def test_poisson_middle_p_lower_plus_upper_equals_one():
    """Poisson midp_lower(ax) + midp_upper(ax) must equal 1."""
    for mean, k in [(1.5, 0), (10.0, 7), (100.0, 99), (0.01, 0)]:
        lo = core.poisson_middle_p_lower(k, mean)
        hi = core.poisson_middle_p_upper(k, mean)
        assert abs(lo + hi - 1.0) < 1e-10, f"mean={mean} k={k}: lo+hi={lo+hi}"


def test_poisson_middle_p_lower_zero_cases():
    """poisson_middle_p_lower(0, mean) must be 0.5*P(X=0) = 0.5*e^{-mean} > 0."""
    import math

    mean = 2.5
    result = core.poisson_middle_p_lower(0, mean)
    expected = 0.5 * math.exp(-mean)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Regression: cache-reset and baseline precompute don't change scan results
# ---------------------------------------------------------------------------


def _make_chain(n):
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr = [0]
    col_idx = []
    for i in range(n):
        nbrs = []
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_scan_results_unchanged_with_cache_llr_true_vs_false():
    """cache_candidate_llr=True must produce identical cluster areas to False."""
    coords, adjacency = _make_chain(10)
    observed = [8, 7, 1, 1, 1, 1, 1, 1, 1, 1]
    expected = [4.0] * 10

    res_cached = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        simcount=0,
        seed=0,
        cache_candidate_llr=True,
    )
    res_nocache = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        simcount=0,
        seed=0,
        cache_candidate_llr=False,
    )
    assert sorted(res_cached["area"]) == sorted(res_nocache["area"])
    assert abs(res_cached["stats"] - res_nocache["stats"]) < 1e-12


def test_scan_results_unchanged_binomial_cache_vs_nocache():
    """Binomial cache_candidate_llr=True == False."""
    coords, adjacency = _make_chain(8)
    observed = [8, 7, 2, 1, 1, 1, 1, 1]
    population = [10] * 8

    res_cached = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        simcount=0,
        seed=0,
        cache_candidate_llr=True,
    )
    res_nocache = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        simcount=0,
        seed=0,
        cache_candidate_llr=False,
    )
    assert sorted(res_cached["area"]) == sorted(res_nocache["area"])
    assert abs(res_cached["stats"] - res_nocache["stats"]) < 1e-12


def test_scan_results_unchanged_cold_spot():
    """COLD scan with cache == without cache."""
    coords, adjacency = _make_chain(10)
    observed = [1, 1, 1, 1, 1, 1, 1, 1, 8, 7]
    expected = [4.0] * 10

    res_cached = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="COLD",
        simcount=0,
        seed=0,
        cache_candidate_llr=True,
    )
    res_nocache = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="COLD",
        simcount=0,
        seed=0,
        cache_candidate_llr=False,
    )
    assert sorted(res_cached["area"]) == sorted(res_nocache["area"])
    assert abs(res_cached["stats"] - res_nocache["stats"]) < 1e-12


def test_stats_monotone_wrt_clustersize_hot():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [1, 6, 0, 0]
    expected = [1.0, 5.0, 1.0, 1.0]
    res1 = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
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
        clustertype="HOT",
        K=3,
        clustersize=2,
        simcount=0,
        knn_method="graph",
    )
    assert isinstance(res1, dict) and isinstance(res2, dict)
    s1 = float(res1.get("stats", 0.0))
    s2 = float(res2.get("stats", 0.0))
    assert s2 >= s1 - 1e-12
