import numpy as np

import flexscanx as p


def test_stability_cold_single_slice():
    # Small line graph
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = [8, 4, 2, 2]
    exp = [5.0, 5.0, 5.0, 5.0]
    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=obs,
        expected=exp,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="COLD",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=0,
    )
    assert isinstance(res, dict)
    assert "area" in res


def test_stability_both_series_repeated():
    # Two time slices; repeated runs to catch any nondeterministic crashes
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = np.array([[10, 6, 1, 1], [8, 4, 2, 2]])
    exp = np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])

    for _ in range(5):
        out = p.scan_series(
            observed=obs,
            expected=exp,
            coords=x,
            adjacency=(row_ptr, col_idx),
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="BOTH",
            clustersize=3,
            ralpha=0.2,
            K=3,
            simcount=0,
            n_jobs_time=1,
        )
        assert isinstance(out, dict)
        assert "per_time" in out and isinstance(out["per_time"], list) and len(out["per_time"]) == 2
