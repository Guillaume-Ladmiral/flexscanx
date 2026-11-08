"""
Test cases for audit fixes:
- Negative observed count validation
- Strict expected/population parameter requirements
"""

import numpy as np
import pytest

import flexscanx as flx


def make_simple_grid(n=5):
    """Create a simple n x n grid with queen adjacency."""
    coords = np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)
    # Queen adjacency for a grid
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
            # Diagonals for queen
            if i > 0 and j > 0:
                adj_dense[c][idx(i - 1, j - 1)] = 1
            if i > 0 and j < n - 1:
                adj_dense[c][idx(i - 1, j + 1)] = 1
            if i < n - 1 and j > 0:
                adj_dense[c][idx(i + 1, j - 1)] = 1
            if i < n - 1 and j < n - 1:
                adj_dense[c][idx(i + 1, j + 1)] = 1

    row_ptr, col_idx = csr_from_dense(adj_dense)
    return coords, (row_ptr, col_idx), N


class TestNegativeCountValidation:
    """Test that negative observed counts are properly rejected."""

    def test_scan_rejects_negative_counts(self):
        """scan() should raise ValueError for negative observed counts."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10, 8, -5, 12, 9] + [5] * (N - 5)  # Negative at index 2
        expected = [10.0] * N

        with pytest.raises(ValueError, match="must be non-negative"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_scan_allows_zero_counts(self):
        """scan() should accept zero observed counts."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10, 0, 5, 12, 0] + [5] * (N - 5)  # Zeros are OK
        expected = [10.0] * N

        # Should not raise
        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )
        assert result is not None

    def test_scan_series_rejects_negative_counts(self):
        """scan_series() should raise ValueError for negative observed counts."""
        coords, adjacency, N = make_simple_grid(5)
        T = 3
        observed = np.ones((T, N), dtype=int) * 10
        observed[1, 3] = -2  # Negative at time=1, area=3
        expected = np.ones((T, N), dtype=float) * 10.0

        with pytest.raises(ValueError, match="must be non-negative"):
            flx.scan_series(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )


class TestExpectedPopulationValidation:
    """Test strict expected/population parameter requirements."""

    def test_poisson_requires_expected(self):
        """POISSON model should require 'expected' parameter."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N

        with pytest.raises(ValueError, match="model='POISSON' requires the 'expected' parameter"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                # Missing expected parameter
                model="POISSON",
                simcount=0,
            )

    def test_poisson_rejects_population(self):
        """POISSON model should reject 'population' parameter (use expected)."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N
        expected = [10.0] * N
        population = [100.0] * N

        with pytest.raises(ValueError, match="population.*parameter is not accepted"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                population=population,  # Should be rejected
                model="POISSON",
                simcount=0,
            )

    def test_binomial_requires_population(self):
        """BINOMIAL model should require 'population'."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N

        with pytest.raises(
            ValueError, match="model='BINOMIAL' requires the 'population' parameter"
        ):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                # Missing population
                model="BINOMIAL",
                simcount=0,
            )

    def test_binomial_accepts_population(self):
        """BINOMIAL model should accept 'population' parameter."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N
        population = [100.0] * N

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            simcount=0,
        )
        assert result is not None

    def test_binomial_rejects_expected_parameter(self):
        """BINOMIAL model should reject 'expected'; only 'population' is accepted."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N
        expected = [100.0] * N

        with pytest.raises(ValueError, match="Parameter 'expected' is not accepted for BINOMIAL"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="BINOMIAL",
                simcount=0,
            )

    def test_binomial_rejects_both_expected_and_population(self):
        """BINOMIAL model should reject if both expected and population provided."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N
        expected = [100.0] * N
        population = [100.0] * N

        with pytest.raises(ValueError, match="provide only 'population'"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                population=population,
                model="BINOMIAL",
                simcount=0,
            )

    def test_scan_series_binomial_rejects_expected_parameter(self):
        """scan_series() BINOMIAL should reject 'expected'; only 'population' is accepted."""
        coords, adjacency, N = make_simple_grid(5)
        T = 2
        observed = np.ones((T, N), dtype=int) * 10
        expected = np.ones((T, N), dtype=float) * 100.0

        with pytest.raises(ValueError, match="Parameter 'expected' is not accepted for BINOMIAL"):
            flx.scan_series(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="BINOMIAL",
                simcount=0,
            )


class TestTyping:
    """Test that return types are properly annotated (static analysis)."""

    def test_scan_returns_dict(self):
        """scan() should return a dict with known structure."""
        coords, adjacency, N = make_simple_grid(5)
        observed = [10] * N
        expected = [10.0] * N

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )

        # Check structure matches TypedDict definition
        assert isinstance(result, dict)
        assert "area" in result
        assert "stats" in result
        assert "type" in result
        assert result["type"] in ("HOT", "COLD", "BOTH")

    def test_scan_series_returns_dict_with_per_time(self):
        """scan_series() should return a dict with 'per_time' key."""
        coords, adjacency, N = make_simple_grid(5)
        T = 3
        observed = np.ones((T, N), dtype=int) * 10
        expected = np.ones((T, N), dtype=float) * 10.0

        result = flx.scan_series(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )

        assert isinstance(result, dict)
        assert "per_time" in result
        assert isinstance(result["per_time"], list)
        assert len(result["per_time"]) == T


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
