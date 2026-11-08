import numpy as np

import flexscanx._core as core


def test_middle_p_poisson_edges():
    # x < 0 should be 1.0
    assert core.poisson_middle_p_upper(-1, 3.0) == 1.0
    assert core.poisson_middle_p_lower(-1, 3.0) == 0.0
    # simple sanity: upper-tail middle p for mean close to 0
    v = core.poisson_middle_p_upper(0, 0.1)
    assert 0.0 <= v <= 1.0


def test_middle_p_python_exports_use_canonical_names():
    assert hasattr(core, "poisson_middle_p_upper")
    assert hasattr(core, "poisson_middle_p_lower")
    assert hasattr(core, "binomial_middle_p_upper")
    assert hasattr(core, "binomial_middle_p_lower")

    assert not hasattr(core, "poisson_midp_upper")
    assert not hasattr(core, "poisson_midp_lower")
    assert not hasattr(core, "binomial_midp_upper")
    assert not hasattr(core, "binomial_midp_lower")


def test_middle_p_poisson_large_counts_stay_stable():
    upper = core.poisson_middle_p_upper(800, 800.0)
    lower = core.poisson_middle_p_lower(800, 800.0)
    np.testing.assert_allclose(upper, 0.4976499772108532, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lower, 0.5023500227891468, rtol=0, atol=1e-10)
    assert abs((upper + lower) - 1.0) <= 1e-12


def test_toy_scan_flexible_hot():
    # 4-node chain: 1-2-3-4
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # CSR for chain (0-1-2-3)
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]

    handle = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {"ralpha": 0.2, "clustersize": 3}
    res = core.ScanSlice(handle, observed, expected, settings)
    # Check keys existence
    for k in ["area", "n_case", "expected", "Rate_Ratio", "stats"]:
        assert k in res
    # stats should be non-negative
    assert res["stats"] >= 0.0
