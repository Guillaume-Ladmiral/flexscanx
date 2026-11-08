import numpy as np

import flexscanx as p


def test_scan_series_minimal():
    # T=2, N=4 toy series
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = np.array([[10, 6, 1, 1], [8, 4, 2, 2]])
    exp = np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])

    out = p.scan_series(
        observed=obs,
        expected=exp,
        coords=x,
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
    assert "per_time" in out and isinstance(out["per_time"], list) and len(out["per_time"]) == 2


def test_scan_series_accepts_list_denominator_poisson():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = [[10, 6, 1, 1], [8, 4, 2, 2]]
    exp = [[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]]

    out = p.scan_series(
        observed=obs,
        expected=exp,
        coords=x,
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

    assert len(out["per_time"]) == 2


def test_scan_series_warm_start_sequential():
    """warm_start=True with n_jobs_time=1 must complete without error."""
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = np.array([[10, 6, 1, 1], [8, 4, 2, 2], [9, 5, 1, 1]])
    exp = np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])

    out = p.scan_series(
        observed=obs,
        expected=exp,
        coords=x,
        adjacency=(row_ptr, col_idx),
        K=3,
        simcount=0,
        warm_start=True,
        n_jobs_time=1,
    )
    assert "per_time" in out
    assert len(out["per_time"]) == 3
    for entry in out["per_time"]:
        assert "area" in entry
        assert isinstance(entry["area"], list)
