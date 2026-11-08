import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # skip if SciPy not installed

from flexscanx import scan  # noqa: E402


def test_scan_geometry_knn_smoke():
    # 3x3 grid coords
    xs, ys = np.meshgrid(np.arange(3), np.arange(3))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    N = coords.shape[0]
    observed = [5] * N
    expected = [5.0] * N
    # Plant a small hot spot at center
    observed[4] = 12
    res = scan(
        coords=coords,
        adjacency=None,  # ignored in geometry mode
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=4,
        clusterradius=float("inf"),
        ralpha=0.5,
        K=4,
        knn_method="geometry",
        knn_mode="K_FIXED",
        simcount=0,
    )
    assert isinstance(res, dict)
    assert "area" in res and isinstance(res["area"], list)
    # Center should be included
    assert 5 in res["area"] or 4 in [a - 1 for a in res["area"]]
