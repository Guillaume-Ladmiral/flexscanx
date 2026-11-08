import numpy as np
import pytest

import flexscanx as flx


def make_simple_grid(n=15):
    """Create a simple n x n grid with queen adjacency."""
    coords = np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)
    from flexscanx.graph import csr_from_dense

    N = n * n
    adj_dense = [[0] * N for _ in range(N)]

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            # 4-neighbors
            if i > 0:
                adj_dense[c][idx(i - 1, j)] = 1
            if i < n - 1:
                adj_dense[c][idx(i + 1, j)] = 1
            if j > 0:
                adj_dense[c][idx(i, j - 1)] = 1
            if j < n - 1:
                adj_dense[c][idx(i, j + 1)] = 1

    row_ptr, col_idx = csr_from_dense(adj_dense)
    return coords, (row_ptr, col_idx), N


def test_determinism_both_parallel():
    """Test that parallel BOTH scans are deterministic."""
    coords, adjacency, N = make_simple_grid(10)

    # We create a random dataset but keep the seed fixed to test the engine determinism
    rng = np.random.default_rng(42)
    observed = rng.poisson(10, size=N)

    # Add a strong hot spot
    observed[25:35] += 50
    # Add a strong cold spot
    observed[65:75] = 0

    expected = np.full(N, 10.0)

    # Run BOTH scan 5 times and check that results are identical
    results = []
    for _ in range(5):
        res = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            clustertype="BOTH",
            clustersize=15,
            n_jobs_centers=4,
            simcount=0,
        )
        results.append(res)

    base = results[0]
    for r in results[1:]:
        assert r["hot"]["area"] == base["hot"]["area"], "Parallel HOT paths diverged"
        assert r["hot"]["stats"] == base["hot"]["stats"], "Parallel HOT stats diverged"
        assert r["cold"]["area"] == base["cold"]["area"], "Parallel COLD paths diverged"
        assert r["cold"]["stats"] == base["cold"]["stats"], "Parallel COLD stats diverged"


def test_warm_start_preserves_scan_results():
    """A warm-start traversal must preserve every per-time primary result."""
    coords, adjacency, N = make_simple_grid(15)
    T = 10

    # Generate stable time series with a persistent hotspot
    rng = np.random.default_rng(42)
    observed = rng.poisson(10, size=(T, N))
    expected = np.full((T, N), 10.0)

    # Persistent hotspot at index 100-110
    observed[:, 100:110] += 30

    res_cold = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustersize=15,
        warm_start=False,
        simcount=0,
    )
    res_warm = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustersize=15,
        warm_start=True,
        simcount=0,
    )

    def _hot_result(entry):
        if isinstance(entry, dict) and "hot" in entry:
            return entry["hot"]
        return entry

    for t in range(T):
        cold_hot = _hot_result(res_cold["per_time"][t])
        warm_hot = _hot_result(res_warm["per_time"][t])
        assert warm_hot["area"] == cold_hot["area"]
        assert float(warm_hot["stats"]) == pytest.approx(float(cold_hot["stats"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
