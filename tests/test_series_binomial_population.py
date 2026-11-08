import numpy as np
import pytest

from flexscanx import scan_series


def test_scan_series_binomial_population_strict_and_negative():
    # T=1, N=3; BINOMIAL population validation
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    row_ptr = [0, 1, 3, 3]
    col_idx = [1, 0, 2]
    observed = np.array([[1, 2, 1]], dtype=int)

    # Case 1: fractional population should raise
    pop_frac = np.array([[50.0, 50.5, 50.0]], dtype=float)
    with pytest.raises(ValueError, match="integer values"):
        scan_series(
            observed=observed,
            population=pop_frac,
            coords=coords,
            adjacency=(row_ptr, col_idx),
            model="BINOMIAL",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            clustersize=3,
            ralpha=0.2,
            K=3,
            simcount=0,
        )

    # Case 2: negative population should raise
    pop_neg = np.array([[50.0, -1.0, 50.0]], dtype=float)
    with pytest.raises(ValueError, match="negative"):
        scan_series(
            observed=observed,
            population=pop_neg,
            coords=coords,
            adjacency=(row_ptr, col_idx),
            model="BINOMIAL",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            clustersize=3,
            ralpha=0.2,
            K=3,
            simcount=0,
        )
