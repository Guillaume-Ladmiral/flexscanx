import numpy as np

import flexscanx as p


def test_scan_secondary_all_smoke():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]

    # Direct secondary="ALL"
    res1 = p.scan(
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
        secondary="ALL",
    )
    assert isinstance(res1, dict) and "area" in res1


def test_root_public_api_excludes_removed_wrappers():
    assert not hasattr(p, "scan_all")
    assert not hasattr(p, "select_scan_and_plot")
