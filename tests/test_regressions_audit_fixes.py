"""Regression tests for post-audit fixes."""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx
from flexscanx.sequence import extract_sequences


def _line_graph(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    coords = np.stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)], axis=1)
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        nbrs: list[int] = []
        if i - 1 >= 0:
            nbrs.append(i - 1)
        if i + 1 < n:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_observed_non_integer_rejected() -> None:
    coords, adjacency = _line_graph(5)
    observed = [1.0, 2.5, 3.0, 4.0, 5.0]
    expected = [5.0] * 5
    try:
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )
    except ValueError as e:
        assert "integer values" in str(e)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for non-integer observed counts")


def test_extract_sequences_with_real_both_series_output() -> None:
    coords, adjacency = _line_graph(5)
    observed = np.asarray(
        [
            [18, 12, 1, 1, 1],
            [1, 1, 1, 12, 18],
        ],
        dtype=int,
    )
    expected = np.full_like(observed, 5.0, dtype=float)

    result = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="BOTH",
        clustersize=2,
        K=3,
        simcount=0,
        secondary="ALL",
    )

    seqs, ids = extract_sequences(result, include_type="both")
    assert seqs.shape[1] == observed.shape[0]
    assert seqs.shape[0] == len(ids)
    assert seqs.dtype == np.int8
    assert np.any(seqs != 0)


def test_secondary_all_with_simcount_zero_still_enumerates() -> None:
    coords, adjacency = _line_graph(6)
    observed = [18, 16, 1, 1, 15, 14]
    expected = [5.0] * 6

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        K=3,
        simcount=0,
        secondary="ALL",
    )
    assert isinstance(res, dict)
    assert "secondaries" in res
    assert len(res["secondaries"]) >= 1


def test_public_stability_and_grid_wrappers_are_callable() -> None:
    coords, adjacency = _line_graph(6)
    observed = [10, 9, 2, 2, 9, 10]
    expected = [5.0] * 6

    stab = flx.stability_across_K(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        K_values=[2, 3],
        clustersize=2,
    )
    assert "pairwise" in stab
    assert len(stab["pairwise"]) == 1

    grid = flx.grid_search_stable(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        K_grid=[2, 3],
        S_grid=[1, 2],
    )
    assert not grid.empty
    assert {"K", "S", "mean_jaccard"}.issubset(set(grid.columns))

    screened = flx.screen_stable_significant(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        K_grid=[2, 3],
        S_grid=[1, 2],
    )
    assert "K" in screened.columns

    obs_tn = np.asarray([observed, observed], dtype=int)
    exp_tn = np.asarray([expected, expected], dtype=float)
    sel_series = flx.find_stable_params_all_series(
        observed=obs_tn,
        expected=exp_tn,
        coords=coords,
        adjacency=adjacency,
        K_start=3,
        S_start=2,
        K_min=2,
        S_min=1,
    )
    assert "best" in sel_series
    assert "K" in sel_series["best"]
    assert "S" in sel_series["best"]


def test_find_stable_params_all_small_graph_defaults_are_valid_and_reusable() -> None:
    coords, adjacency = _line_graph(5)
    observed = [5, 5, 6, 5, 5]
    expected = [5.0] * 5

    sel = flx.find_stable_params_all(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
    )
    best = sel["best"]

    assert 1 <= int(best["S"]) < int(best["K"]) <= len(observed)
    assert (int(best["K"]), int(best["S"])) in sel["metrics"]

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        K=int(best["K"]),
        clustersize=int(best["S"]),
        simcount=0,
    )
    assert isinstance(res, dict)


def test_find_stable_params_all_boundary_n2_defaults_are_valid() -> None:
    coords, adjacency = _line_graph(2)
    observed = [8, 1]
    expected = [4.0, 4.0]

    sel = flx.find_stable_params_all(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
    )

    assert sel["best"] == {"K": 2, "S": 1}


def test_find_stable_params_all_rejects_n1_when_k_must_exceed_s() -> None:
    coords, adjacency = _line_graph(1)

    with pytest.raises(ValueError, match="requires at least 2 areas"):
        flx.find_stable_params_all(
            coords=coords,
            adjacency=adjacency,
            observed=[5],
            expected=[5.0],
        )


def test_find_stable_params_all_series_small_graph_defaults_remain_valid() -> None:
    coords, adjacency = _line_graph(5)
    observed = np.asarray([[5, 5, 6, 5, 5], [5, 6, 5, 5, 5]], dtype=int)
    expected = np.full_like(observed, 5.0, dtype=float)

    sel = flx.find_stable_params_all_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=adjacency,
    )

    best = sel["best"]
    assert 1 <= int(best["S"]) < int(best["K"]) <= observed.shape[1]
    for row in sel["per_time"]:
        assert 1 <= int(row["S"]) < int(row["K"]) <= observed.shape[1]
        assert (int(row["K"]), int(row["S"])) in row["selection"]["metrics"]

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed[0].tolist(),
        expected=expected[0].tolist(),
        K=int(best["K"]),
        clustersize=int(best["S"]),
        simcount=0,
    )
    assert isinstance(res, dict)
