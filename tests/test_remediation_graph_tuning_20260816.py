"""Regression coverage for graph, tuning, and sequence contracts remediated in 2026-08-16."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from flexscanx import _graph_metrics as graph_metrics
from flexscanx.graph import csr_from_geodataframe_contiguity
from flexscanx.helpers import pool_to_geodataframe
from flexscanx.sequence import extract_sequences
from flexscanx.tuning import _normalize_tuning_bounds, grid_search_stable


def _path_csr(n: int) -> tuple[list[int], list[int]]:
    row_ptr = [0]
    col_idx: list[int] = []
    for index in range(n):
        col_idx.extend(neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < n)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def test_silhouette_graph_has_defined_disconnected_graph_conventions() -> None:
    row_ptr, col_idx = [0, 1, 3, 5, 6, 6], [1, 0, 2, 1, 3, 2]

    island = graph_metrics.silhouette_graph(row_ptr, col_idx, [1, 1, 0, 0, 0])
    split_cluster = graph_metrics.silhouette_graph(row_ptr, col_idx, [1, 0, 0, 0, 1])

    assert island == {"mean": 1.0, "values": [1.0, 1.0]}
    assert split_cluster == {"mean": -1.0, "values": [-1.0, -1.0]}


@pytest.mark.skipif(graph_metrics._bc_core is None, reason="numba is unavailable")
def test_boundary_contrast_numba_matches_two_pass_fallback_under_cancellation(monkeypatch) -> None:
    row_ptr, col_idx = _path_csr(6)
    observed = np.asarray([5, 4, 3, 2, 1, 0], dtype=int)
    expected = np.asarray([5 / 5e9, 4 / 4e9, 3 / (3e9 + 1), 2 / 2e9, 1 / 1e9, 1e-9])
    membership = [1, 0, 1, 0, 1, 0]

    accelerated = graph_metrics.boundary_contrast(observed, expected, row_ptr, col_idx, membership)
    monkeypatch.setattr(graph_metrics, "_bc_core", None)
    fallback = graph_metrics.boundary_contrast(observed, expected, row_ptr, col_idx, membership)

    assert accelerated == pytest.approx(fallback, rel=0.0, abs=1e-12)
    assert accelerated["std_delta"] > 0.0


def test_contiguity_fallback_returns_builtin_integer_csr_elements(monkeypatch) -> None:
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    gdf = geopandas.GeoDataFrame({"geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:3857")
    monkeypatch.setitem(sys.modules, "libpysal", None)
    monkeypatch.setitem(sys.modules, "libpysal.weights", None)

    row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf)

    assert all(type(value) is int for value in [*row_ptr, *col_idx])
    assert json.dumps({"row_ptr": row_ptr, "col_idx": col_idx})


def test_small_graph_tuning_defaults_begin_with_a_cluster_size_axis() -> None:
    k_start, s_start, k_min, s_min = _normalize_tuning_bounds(
        N=9, K_start=None, S_start=None, K_min=None, S_min=None, require_K_gt_S=True
    )

    assert (k_start, s_start, k_min, s_min) == (9, 2, 2, 1)


def test_extract_sequences_neutral_conflicts_are_permutation_invariant() -> None:
    hot_cold_hot = {
        "per_time": [
            [
                {"type": "HOT", "area": [2]},
                {"type": "COLD", "area": [2]},
                {"type": "HOT", "area": [2]},
            ]
        ]
    }
    cold_hot_cold = {
        "per_time": [
            [
                {"type": "COLD", "area": [2]},
                {"type": "HOT", "area": [2]},
                {"type": "COLD", "area": [2]},
            ]
        ]
    }

    first, first_ids = extract_sequences(hot_cold_hot)
    second, second_ids = extract_sequences(cold_hot_cold)

    assert first_ids == second_ids == [2]
    assert first.tolist() == second.tolist() == [[0]]


def test_pool_to_geodataframe_maps_string_area_ids_after_integer_normalization() -> None:
    gdf = pd.DataFrame({"area_id": ["1", "2", "3", "4"]})

    result = pool_to_geodataframe(gdf, [frozenset({1, 2}), frozenset({3})], "area_id")

    assert result["PoolCount"].tolist() == [1, 1, 1, 0]
    assert result["PoolMember"].tolist() == [1, 1, 1, 0]


def test_grid_search_stable_rejects_wholly_infeasible_grid_before_building_handles() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    adjacency = _path_csr(3)

    with pytest.raises(ValueError, match="K > S"):
        grid_search_stable(
            coords=coords,
            adjacency=adjacency,
            observed=[2, 2, 2],
            expected=[1.0, 1.0, 1.0],
            K_grid=[2],
            S_grid=[2],
        )


def test_grid_search_stable_populated_schema_matches_documented_columns() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    adjacency = _path_csr(3)

    result = grid_search_stable(
        coords=coords,
        adjacency=adjacency,
        observed=[3, 1, 1],
        expected=[1.0, 1.0, 1.0],
        K_grid=[2],
        S_grid=[1],
    )

    assert result.columns.tolist() == [
        "K",
        "S",
        "pool_size",
        "stats",
        "pval",
        "mean_jaccard",
        "min_jaccard",
        "coverage",
        "count_diff",
    ]
    assert result[["K", "S", "pool_size", "count_diff"]].dtypes.astype(str).tolist() == [
        "int64",
        "int64",
        "int64",
        "int64",
    ]
