import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from flexscanx import scan_series  # noqa: E402


def test_scan_series_geometry_modes_kfixed_and_krefill():
    # T=2, small 4x4 grid; test geometry KNN with K_FIXED and K_REFILL
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    N = coords.shape[0]
    T = 2
    observed = np.vstack([np.full(N, 5, dtype=int), np.full(N, 5, dtype=int)])
    observed[0, 5] = 12
    observed[1, 6] = 12
    expected = np.full((T, N), 5.0, dtype=float)

    # K_FIXED
    res_fixed = scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=None,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=5,
        K=5,
        knn_method="geometry",
        knn_mode="K_FIXED",
        simcount=0,
        seed=11,
    )
    assert isinstance(res_fixed, dict)
    assert "per_time" in res_fixed and len(res_fixed["per_time"]) == T

    # K_REFILL (guardrail won't block: N is small)
    res_refill = scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=None,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=5,
        K=5,
        knn_method="geometry",
        knn_mode="K_REFILL",
        simcount=0,
        seed=11,
    )
    assert isinstance(res_refill, dict)
    assert "per_time" in res_refill and len(res_refill["per_time"]) == T
