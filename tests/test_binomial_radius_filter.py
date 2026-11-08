import numpy as np

import flexscanx as p


def _line_graph_coords():
    # Three points on a line: 0 -- 1 -------- 2
    # Distances: d(0,1)=2, d(1,2)=8, d(0,2)=10
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    return coords, (row_ptr, col_idx)


def test_binomial_flexible_original_respects_radius():
    coords, adjacency = _line_graph_coords()
    observed = [2, 6, 2]
    population = [10.0, 10.0, 10.0]

    res = p.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        rantype="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="ORIGINAL",
        clustertype="HOT",
        K=3,
        clustersize=3,
        clusterradius=3.0,  # include neighbor at distance 2, exclude at distance 8
        simcount=0,
        knn_method="graph",
    )
    assert isinstance(res, dict)
    area = set(res.get("area", []))
    # Must not include the far node (3); cluster may be {2} or {1,2} depending on objective
    assert 3 not in area
    assert area.issubset({1, 2})
    assert len(area) >= 1


def test_binomial_flexible_restricted_respects_radius():
    coords, adjacency = _line_graph_coords()
    observed = [2, 6, 2]
    population = [10.0, 10.0, 10.0]

    res = p.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        rantype="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        clusterradius=3.0,
        ralpha=1.0,  # disable gating for this test
        simcount=0,
        knn_method="graph",
    )
    assert isinstance(res, dict)
    area = set(res.get("area", []))
    assert 3 not in area
    assert area.issubset({1, 2})
    assert len(area) >= 1
