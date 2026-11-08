import numpy as np

import flexscanx._core as core


def test_pruning_no_hang_restricted_hot():
    # Construct a case where adding any neighbor cannot improve the LLR,
    # forcing the bound-pruning path to trigger. Previously this could hang.
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    # Center (0) is very hot vs expected; neighbors are weak and fail HOT gating.
    observed = [20, 0, 0]
    expected = [1.0, 10.0, 10.0]

    h = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {
        "ralpha": 0.2,
        "clustersize": 3,
        "scanmethod": "FLEXIBLE",
        "stattype": "RESTRICTED",
        "clustertype": "HOT",
        "model": "POISSON",
    }
    res = core.ScanSlice(h, observed, expected, settings)
    assert isinstance(res, dict)
    assert "area" in res and isinstance(res["area"], list)
    # Best cluster should remain the center alone in this setup
    assert len(res["area"]) == 1


def test_pruning_no_hang_restricted_cold():
    # Mirror case for COLD: center is very cold; neighbors are hot and fail COLD gating.
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [0, 10, 10]
    expected = [10.0, 1.0, 1.0]

    h = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {
        "ralpha": 0.2,
        "clustersize": 3,
        "scanmethod": "FLEXIBLE",
        "stattype": "RESTRICTED",
        "clustertype": "COLD",
        "model": "POISSON",
    }
    res = core.ScanSlice(h, observed, expected, settings)
    assert isinstance(res, dict)
    assert "area" in res and isinstance(res["area"], list)
    assert len(res["area"]) == 1
