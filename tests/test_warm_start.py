"""Tests for scan_series warm_start correctness.

Verifies that warm_start=True produces the same primary cluster results as
warm_start=False (warm_start only affects which centers are tried first, not
which final cluster is selected).
"""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx


def _grid_5x5():
    """Return coords and queen adjacency for a 5x5 grid."""
    n = 5
    coords = np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)
    N = n * n
    from flexscanx.graph import csr_from_dense

    adj_dense = [[0] * N for _ in range(N)]

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        adj_dense[c][idx(ni, nj)] = 1

    row_ptr, col_idx = csr_from_dense(adj_dense)
    return coords, (row_ptr, col_idx), N


@pytest.mark.filterwarnings("ignore")
def test_warm_start_produces_same_primary_cluster_as_cold_start():
    """warm_start=True must yield the same cluster areas as warm_start=False."""
    coords, adjacency, N = _grid_5x5()
    T = 4
    rng = np.random.default_rng(7)
    observed = rng.poisson(5, size=(T, N)).astype(np.int32)
    # Plant a persistent hot spot at center 12 (row 2, col 2)
    observed[:, 12] = 20
    expected = np.full((T, N), 5.0)

    result_cold = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        simcount=0,
        seed=42,
        warm_start=False,
    )
    result_warm = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        simcount=0,
        seed=42,
        warm_start=True,
    )

    for t in range(T):
        cold_area = sorted(result_cold["per_time"][t]["area"])
        warm_area = sorted(result_warm["per_time"][t]["area"])
        assert cold_area == warm_area, (
            f"Time {t}: warm_start=True gave different cluster area "
            f"({warm_area}) vs warm_start=False ({cold_area})"
        )


@pytest.mark.filterwarnings("ignore")
def test_warm_start_priority_centers_are_valid_indices():
    """warm_start must not cause an error even when prev slice has no cluster."""
    coords, adjacency, N = _grid_5x5()
    T = 3
    observed = np.ones((T, N), dtype=np.int32) * 5
    expected = np.full((T, N), 5.0)

    # Should run without error
    result = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        simcount=0,
        seed=0,
        warm_start=True,
    )
    assert len(result["per_time"]) == T
