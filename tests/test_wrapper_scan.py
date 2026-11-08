import numpy as np

import flexscanx as p


def test_public_api_uses_scan_without_scan_both():
    assert hasattr(p, "scan")
    assert not hasattr(p, "scan_both")


def test_wrapper_scan_flexible_hot():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    res = p.scan(
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
    )
    assert isinstance(res, dict)
    assert "area" in res and len(res["area"]) >= 1
