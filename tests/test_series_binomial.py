import numpy as np

import flexscanx as p


def test_scan_series_binomial_minimal():
    # T=2, N=4
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = np.array([[5, 3, 1, 0], [4, 2, 1, 1]], dtype=int)
    pop = np.array([[50, 50, 50, 50], [50, 50, 50, 50]], dtype=float)

    out = p.scan_series(
        observed=obs,
        population=pop,
        coords=x,
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
    assert "per_time" in out and isinstance(out["per_time"], list) and len(out["per_time"]) == 2


def test_scan_series_accepts_list_population_binomial():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    obs = [[5, 3, 1, 0], [4, 2, 1, 1]]
    pop = [[50, 50, 50, 50], [50, 50, 50, 50]]

    out = p.scan_series(
        observed=obs,
        population=pop,
        coords=x,
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

    assert len(out["per_time"]) == 2
