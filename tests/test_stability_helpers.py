from __future__ import annotations

import numpy as np

from flexscanx.tuning import (
    StabilityThresholds,
    grid_search_stable,
    is_stable,
    match_clusters_jaccard,
)


def test_match_clusters_jaccard_reports_expected_metrics():
    pool_a = [frozenset([1, 2]), frozenset([5])]
    pool_b = [frozenset([1, 2]), frozenset([5, 6])]

    metrics = match_clusters_jaccard(pool_a, pool_b, coverage_floor=0.5)

    assert metrics.nA == 2
    assert metrics.nB == 2
    assert metrics.count_diff == 0
    assert metrics.mean_jaccard == 0.75
    assert metrics.min_jaccard == 0.5
    assert metrics.coverage == 1.0
    assert metrics.matched == [(0, 0, 1.0), (1, 1, 0.5)]


def test_is_stable_uses_threshold_dataclass():
    thresholds = StabilityThresholds(mean_jaccard=0.8, min_jaccard=0.6, coverage=0.9)
    metrics = match_clusters_jaccard(
        [frozenset([1, 2]), frozenset([5])],
        [frozenset([1, 2]), frozenset([5, 6])],
        coverage_floor=0.5,
    )

    assert not is_stable(metrics, thresholds)

    relaxed = StabilityThresholds(mean_jaccard=0.7, min_jaccard=0.5, coverage=0.9)
    assert is_stable(metrics, relaxed)


def test_grid_search_stable_accepts_binomial_population():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    adjacency = ([0, 1, 3, 5, 6], [1, 0, 2, 1, 3, 2])
    observed = [8, 4, 1, 1]
    population = [50, 50, 50, 50]

    out = grid_search_stable(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        K_grid=[3],
        S_grid=[2],
        model="BINOMIAL",
        clustertype="HOT",
    )

    assert list(out["K"]) == [3]
    assert list(out["S"]) == [2]
