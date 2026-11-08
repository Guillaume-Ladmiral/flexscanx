import numpy as np

import flexscanx._core as core


def test_flexible_cold_restricted_toy():
    # 4-node chain, designed with low counts to allow cold gating
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [0, 1, 2, 3]
    expected = [5.0, 5.0, 5.0, 5.0]

    h = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {
        "ralpha": 0.2,
        "clustersize": 3,
        "scanmethod": "FLEXIBLE",
        "stattype": "RESTRICTED",
        "clustertype": "COLD",
    }
    res = core.ScanSlice(h, observed, expected, settings)
    assert "area" in res and isinstance(res["area"], list)
    assert res["stats"] >= 0.0


def test_circular_hot_restricted_prefixes():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [10, 6, 1]
    expected = [5.0, 5.0, 5.0]

    h = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {
        "ralpha": 0.2,
        "clustersize": 3,
        "scanmethod": "CIRCULAR",
        "stattype": "RESTRICTED",
        "clustertype": "HOT",
    }
    res = core.ScanSlice(h, observed, expected, settings)
    assert "area" in res and len(res["area"]) >= 1
