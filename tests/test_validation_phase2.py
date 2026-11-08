import math

import numpy as np
import pytest

from flexscanx import (
    boundary_contrast,
    conductance,
    cv_glm_gain,
    dunn_graph,
    join_counts,
    morans_I_within,
    silhouette_graph,
    trend_level_slope,
)


def grid_4nbr(n_side: int):
    # N = n_side * n_side  # not used
    row_ptr = [0]
    col_idx = []

    def idx(r, c):
        return r * n_side + c

    # build undirected 4-neighbor
    for r in range(n_side):
        for c in range(n_side):
            # i = idx(r, c)  # not used
            nbrs = []
            if r > 0:
                nbrs.append(idx(r - 1, c))
            if r < n_side - 1:
                nbrs.append(idx(r + 1, c))
            if c > 0:
                nbrs.append(idx(r, c - 1))
            if c < n_side - 1:
                nbrs.append(idx(r, c + 1))
            nbrs.sort()
            col_idx.extend(nbrs)
            row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def make_data(n_side=6, lam=10.0, rr_hot=2.0, rr_cold=0.4, seed=123):
    N = n_side * n_side
    row_ptr, col_idx = grid_4nbr(n_side)
    rng = np.random.default_rng(seed)
    expected = np.full(N, lam, dtype=float)
    observed_null = rng.poisson(expected).astype(int)
    # central 2x2 block indices
    center = n_side // 2
    block = []
    for r in range(center - 1, center + 1):
        for c in range(center - 1, center + 1):
            block.append(r * n_side + c)
    membership = np.zeros(N, dtype=int)
    membership[block] = 1
    observed_hot = observed_null.copy()
    for i in block:
        observed_hot[i] = int(rng.poisson(rr_hot * expected[i]))
    observed_cold = observed_null.copy()
    for i in block:
        observed_cold[i] = int(rng.poisson(rr_cold * expected[i]))
    return (
        row_ptr,
        col_idx,
        expected,
        observed_null,
        observed_hot,
        observed_cold,
        membership.tolist(),
    )


def test_graph_compactness_and_autocorr_hot():
    row_ptr, col_idx, expected, _, observed_hot, _, membership = make_data(n_side=6)
    sil = silhouette_graph(row_ptr, col_idx, membership)
    dunn = dunn_graph(row_ptr, col_idx, membership)
    moran_I = morans_I_within(observed_hot, expected, row_ptr, col_idx, membership)
    assert -1.0 <= sil["mean"] <= 1.0
    assert dunn > 0.0
    # For a planted hot block, autocorrelation should be non-negative
    assert moran_I >= -0.25


def test_join_counts_nonnegative():
    row_ptr, col_idx, expected, _, _, _, membership = make_data(n_side=5)
    jc = join_counts(row_ptr, col_idx, membership)
    assert jc["J11"] >= 0 and jc["J10"] >= 0 and jc["J00"] >= 0
    # Sum of join-counts equals number of undirected edges
    E = 0
    for i in range(len(row_ptr) - 1):
        E += row_ptr[i + 1] - row_ptr[i]
    E //= 2
    assert jc["J11"] + jc["J10"] + jc["J00"] == E


def test_cv_glm_gain_hot_and_cold():
    row_ptr, col_idx, expected, _, observed_hot, observed_cold, membership = make_data(n_side=6)
    cv_hot = cv_glm_gain(observed_hot, expected, membership, k=3, seed=7)
    assert math.isfinite(cv_hot["d_dev"]) and cv_hot["d_dev"] > 0.0
    cv_cold = cv_glm_gain(observed_cold, expected, membership, k=3, seed=9)
    assert math.isfinite(cv_cold["d_dev"]) and cv_cold["d_dev"] > 0.0


def test_validation_helpers_reject_length_mismatches():
    row_ptr, col_idx, expected, _, observed_hot, _, membership = make_data(n_side=3)

    with pytest.raises(ValueError, match="membership length"):
        silhouette_graph(row_ptr, col_idx, membership[:-1])

    with pytest.raises(ValueError, match="observed, expected, and membership"):
        cv_glm_gain(observed_hot, expected[:-1], membership, k=3, seed=7)

    with pytest.raises(ValueError, match="observed and expected must have length"):
        morans_I_within(observed_hot[:-1], expected, row_ptr, col_idx, membership)


@pytest.mark.parametrize(
    ("row_ptr", "col_idx", "membership", "message"),
    [
        ([0, 1, 2], [1, 0], [1, 2], "membership must contain only 0 or 1"),
        ([0, 1.5, 2], [1, 0], [1, 0], "integer indices"),
        ([0, 1, 2], [1, 1], [1, 0], "self-loop"),
        ([0, 1, 1], [1], [1, 0], "must be symmetric"),
    ],
)
def test_graph_metrics_reject_invalid_graph_and_membership(row_ptr, col_idx, membership, message):
    for metric in (silhouette_graph, dunn_graph, join_counts, conductance):
        with pytest.raises(ValueError, match=message):
            metric(row_ptr, col_idx, membership)


@pytest.mark.parametrize(
    ("observed", "expected", "model", "message"),
    [
        ([-1, 2], [1.0, 2.0], "POISSON", "non-negative"),
        ([1, 2], [1.0, np.nan], "POISSON", "NaN values"),
        ([1, 2], [2.5, 3.0], "BINOMIAL", "integer values"),
        ([2, 2], [1.0, 3.0], "BINOMIAL", "cannot exceed population"),
        ([1, 2], [1.0, 2.0], "QUASIPOISSON", "model must be one of"),
    ],
)
def test_statistical_graph_metrics_reject_invalid_arrays_and_model(
    observed, expected, model, message
):
    row_ptr, col_idx = [0, 1, 2], [1, 0]
    membership = [1, 0]
    for metric in (boundary_contrast, morans_I_within):
        with pytest.raises(ValueError, match=message):
            metric(observed, expected, row_ptr, col_idx, membership, model=model)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"k": 1}, "k must be >= 2"),
        ({"k": 1.5}, "k must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"model": "QUASIPOISSON"}, "model must be one of"),
        ({"membership": [1, 2, 0, 0]}, "membership must contain only 0 or 1"),
        ({"observed": [1, -1, 2, 1]}, "non-negative"),
        ({"expected": [1.0, np.nan, 2.0, 1.0]}, "NaN values"),
    ],
)
def test_cv_glm_gain_rejects_invalid_contract_inputs(overrides, message):
    kwargs = {
        "observed": [1, 2, 2, 1],
        "expected": [1.0, 2.0, 2.0, 1.0],
        "membership": [1, 1, 0, 0],
        "k": 2,
        "seed": 1,
        "model": "POISSON",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        cv_glm_gain(**kwargs)


@pytest.mark.parametrize(
    ("observed", "population", "message"),
    [
        ([1, 2, 2, 1], [2.5, 3, 3, 2], "integer values"),
        ([3, 2, 2, 1], [2, 3, 3, 2], "cannot exceed population"),
    ],
)
def test_cv_glm_gain_validates_binomial_population(observed, population, message):
    with pytest.raises(ValueError, match=message):
        cv_glm_gain(
            observed,
            population,
            [1, 1, 0, 0],
            k=2,
            seed=1,
            model="BINOMIAL",
        )


def test_trend_level_slope():
    # Build a fake per_time with increasing S (p-values decreasing)
    per_time = []
    base_area = [10, 11, 16]
    ps = [0.5, 0.2, 0.05, 0.01, 0.005]
    for p in ps:
        per_time.append({"area": base_area, "pval": p, "stats": 0.0})
    tr = trend_level_slope(per_time, window=3, use="S", jaccard_threshold=0.3)
    assert tr["slope"] > 0.0
    assert tr["level_last"] > 0.0
