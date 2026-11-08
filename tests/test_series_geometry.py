import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from flexscanx import scan_series  # noqa: E402


def test_scan_series_geometry_smoke():
    # T=2, 3x3 grid geometry KNN
    xs, ys = np.meshgrid(np.arange(3), np.arange(3))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    N = coords.shape[0]
    T = 2
    observed = np.vstack([np.full(N, 5, dtype=int), np.full(N, 5, dtype=int)])
    # Plant a small hot spot
    observed[0, 4] = 12
    observed[1, 4] = 12
    expected = np.full((T, N), 5.0, dtype=float)
    res = scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=None,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=4,
        clusterradius=float("inf"),
        ralpha=0.5,
        K=4,
        knn_method="geometry",
        simcount=0,
        seed=7,
    )
    assert isinstance(res, dict)
    assert "per_time" in res and len(res["per_time"]) == T
