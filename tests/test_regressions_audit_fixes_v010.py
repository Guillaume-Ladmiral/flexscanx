"""Regression and coverage tests for second-pass audit fixes (v0.0.1 post-remediation).

Covers:
- both_parallel=True produces same results as sequential (S5-2)
- ratio_smoothing='EB' returns finite Rate_Ratio (S5-3)
- calibrate_llr_threshold BINOMIAL simulation branch (S5-4)
- scan_series with knn_method='geometry' (S5-10)
- consensus_sweep secondary exclusion (S5-11)
"""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx
import flexscanx._consensus as consensus_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(n: int):
    """Return coords and adjacency for a simple n-node chain graph."""
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        nbrs = []
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _grid_4x4():
    """Return coords and rook adjacency for a 4x4 grid."""
    n = 4
    coords = np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)
    N = n * n
    from flexscanx.graph import csr_from_dense

    adj = [[0] * N for _ in range(N)]

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    adj[c][idx(ni, nj)] = 1
    row_ptr, col_idx = csr_from_dense(adj)
    return coords, (row_ptr, col_idx), N


# ---------------------------------------------------------------------------
# S5-2: both_parallel correctness
# ---------------------------------------------------------------------------


def test_both_parallel_matches_sequential():
    """scan() both_parallel=True must produce identical HOT and COLD areas to sequential."""
    coords, adjacency = _chain(8)
    observed = [10, 8, 6, 2, 2, 1, 1, 1]
    expected = [4.0] * 8

    res_seq = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="BOTH",
        both_parallel=False,
        simcount=0,
        seed=1,
    )
    res_par = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="BOTH",
        both_parallel=True,
        simcount=0,
        seed=1,
    )

    assert sorted(res_seq["hot"]["area"]) == sorted(res_par["hot"]["area"])
    assert sorted(res_seq["cold"]["area"]) == sorted(res_par["cold"]["area"])
    assert res_seq["primary_type"] == res_par["primary_type"]


def test_both_parallel_series_matches_sequential():
    """scan_series() both_parallel=True must produce identical per-time areas."""
    coords, adjacency = _chain(6)
    T = 3
    observed = np.array(
        [[5, 4, 1, 1, 1, 1], [5, 4, 1, 1, 1, 1], [1, 1, 1, 5, 4, 1]], dtype=np.int32
    )
    expected = np.full((T, 6), 3.0)

    res_seq = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="BOTH",
        both_parallel=False,
        simcount=0,
        seed=2,
    )
    res_par = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustertype="BOTH",
        both_parallel=True,
        simcount=0,
        seed=2,
    )

    for t in range(T):
        seq_t = res_seq["per_time"][t]
        par_t = res_par["per_time"][t]
        assert sorted(seq_t["hot"]["area"]) == sorted(par_t["hot"]["area"]), f"t={t} HOT mismatch"
        assert sorted(seq_t["cold"]["area"]) == sorted(
            par_t["cold"]["area"]
        ), f"t={t} COLD mismatch"


# ---------------------------------------------------------------------------
# S5-3: EB ratio smoothing returns finite Rate_Ratio
# ---------------------------------------------------------------------------


def test_eb_ratio_smoothing_poisson_returns_finite_rate_ratio():
    """ratio_smoothing='EB' must return a finite Rate_Ratio for POISSON."""
    coords, adjacency = _chain(10)
    observed = [12, 10, 2, 1, 1, 1, 1, 1, 1, 1]
    expected = [4.0] * 10

    res_eb = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        ratio_smoothing="EB",
        simcount=0,
    )
    res_none = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        ratio_smoothing="NONE",
        simcount=0,
    )

    rr_eb = res_eb.get("Rate_Ratio")
    rr_none = res_none.get("Rate_Ratio")
    assert rr_eb is not None and np.isfinite(rr_eb), "EB Rate_Ratio must be finite"
    assert rr_none is not None and np.isfinite(rr_none), "NONE Rate_Ratio must be finite"
    # EB shrinks toward global mean; should be between NONE and 1.0 for a hot cluster
    assert 1.0 <= rr_eb, "EB-smoothed rate ratio should be >= 1 for a hot cluster"


def test_eb_ratio_smoothing_binomial_returns_finite_risk_ratio():
    """ratio_smoothing='EB' must return a finite Risk_Ratio for BINOMIAL."""
    coords, adjacency = _chain(8)
    observed = [8, 7, 2, 1, 1, 1, 1, 1]
    population = [10] * 8

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        ratio_smoothing="EB",
        simcount=0,
    )
    rr = res.get("Risk_Ratio")
    assert rr is not None and np.isfinite(rr), "EB Risk_Ratio must be finite for BINOMIAL"


# ---------------------------------------------------------------------------
# S5-4: calibrate_llr_threshold BINOMIAL branch
# ---------------------------------------------------------------------------


def test_calibrate_llr_threshold_binomial_returns_valid_result():
    """calibrate_llr_threshold must work for BINOMIAL model with population= parameter."""
    coords, adjacency = _chain(6)
    observed = [5, 4, 1, 1, 1, 1]
    population = [10] * 6

    result = flx.calibrate_llr_threshold(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        K=4,
        clustersize=3,
        n_sims=19,
        alpha=0.1,
        seed=99,
    )

    assert set(result.keys()) == {"tau", "mu", "sigma", "alpha", "n_sims"}
    assert result["n_sims"] == 19
    assert result["alpha"] == pytest.approx(0.1)
    assert np.isfinite(result["tau"]), "tau must be finite"
    assert np.isfinite(result["mu"]), "mu must be finite"

    independent = flx.calibrate_llr_threshold(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        K=4,
        clustersize=3,
        n_sims=19,
        alpha=0.1,
        seed=100,
    )
    assert any(result[key] != independent[key] for key in ("tau", "mu", "sigma"))


def test_calibrate_llr_threshold_binomial_rejects_expected():
    """calibrate_llr_threshold BINOMIAL must reject expected= with clear error."""
    coords, adjacency = _chain(4)
    with pytest.raises(ValueError, match="population"):
        flx.calibrate_llr_threshold(
            coords=coords,
            adjacency=adjacency,
            observed=[2, 1, 1, 1],
            expected=[10.0, 10.0, 10.0, 10.0],
            model="BINOMIAL",
            K=3,
            clustersize=2,
            n_sims=5,
        )


# ---------------------------------------------------------------------------
# S5-10: scan_series knn_method='geometry'
# ---------------------------------------------------------------------------


def test_scan_series_knn_geometry_mode_runs():
    """scan_series with knn_method='geometry' must complete without error."""
    coords, adjacency, N = _grid_4x4()
    T = 3
    observed = np.ones((T, N), dtype=np.int32) * 3
    observed[:, 5] = 10
    expected = np.full((T, N), 3.0)

    result = flx.scan_series(
        coords=coords,
        adjacency=None,
        observed=observed,
        expected=expected,
        knn_method="geometry",
        K=4,
        simcount=0,
        seed=0,
    )

    assert len(result["per_time"]) == T
    assert result["scan_settings"]["knn_method"] == "geometry"


# ---------------------------------------------------------------------------
# S5-11: consensus_sweep secondary exclusion
# ---------------------------------------------------------------------------


def test_consensus_sweep_only_counts_primary_area(monkeypatch):
    """consensus_sweep must not count secondary cluster areas in freq tally."""
    coords, adjacency, N = _grid_4x4()
    observed = [0] * N
    # Strong hot spot at area index 5 (1-based = 6)
    observed[5] = 20
    expected = [2.0] * N

    class _CoreWithDistinctSecondary:
        @staticmethod
        def ScanSlice(*_args, **_kwargs):
            return {"area": [6], "secondaries": [{"area": [11]}]}

    monkeypatch.setattr(consensus_module, "_require_core", _CoreWithDistinctSecondary)
    monkeypatch.setattr(
        consensus_module, "build_validated_static_handle", lambda *_args, **_kwargs: object()
    )

    result = flx.consensus_sweep(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        clustersize=2,
        grid={"K": [4], "ralpha": [0.2]},
        model="POISSON",
        threshold=0.5,
    )

    assert result["runs"] == 1
    assert len(result["freq"]) == N
    assert result["freq"][5] == 1.0
    assert result["freq"][10] == 0.0
    assert result["consensus"] == [6]
