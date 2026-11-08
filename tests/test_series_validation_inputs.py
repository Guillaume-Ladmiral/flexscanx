import numpy as np
import pytest

from flexscanx import scan_series


def test_scan_series_coords_shape_validation():
    # T=1, invalid coords with wrong shape (N x 3)
    obs = np.array([[1, 2, 3]])
    exp = np.array([[5.0, 5.0, 5.0]])
    coords_bad = np.zeros((3, 3), dtype=float)
    row_ptr = [0, 2, 4, 4]
    col_idx = [1, 0, 2, 1]
    with pytest.raises(ValueError, match="coords must be Nx2"):
        scan_series(
            observed=obs,
            expected=exp,
            coords=coords_bad,
            adjacency=(row_ptr, col_idx),
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            clustersize=3,
            ralpha=0.2,
            K=3,
            simcount=0,
        )
