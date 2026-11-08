import numpy as np

import flexscanx as p


def test_mc_alpha_smoke():
    # Small synthetic example to exercise MC path and mc_alpha plumbing
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [10, 5, 5]
    expected = [5.0, 5.0, 5.0]

    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        ralpha=0.2,
        K=3,
        simcount=250,
        mc_early_stop=True,
        mc_alpha=0.1,  # custom threshold should be accepted by core
        seed=42,
    )
    assert isinstance(res, dict)
    # If MC ran, rank should be between [1, simcount+1]
    if res.get("rank") is not None:
        assert 1 <= int(res["rank"]) <= 251
        pval = float(res.get("pval", 0.0))
        assert 0.0 <= pval <= 1.0
