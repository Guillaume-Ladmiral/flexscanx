import numpy as np

import flexscanx as p


def test_binomial_flexible_restricted_hot():
    # Small toy: populations are larger than observed
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [5, 2, 1]
    population = [20.0, 20.0, 20.0]
    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        population=population,
        model="BINOMIAL",
        rantype="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=3,
        ralpha=0.2,
        K=3,
    )
    assert isinstance(res, dict)
    assert "area" in res and len(res["area"]) >= 1
