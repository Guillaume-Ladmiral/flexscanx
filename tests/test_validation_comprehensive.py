"""Comprehensive tests for input validation (Phase 1 audit fixes)."""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx


# Test data fixtures
@pytest.fixture
def simple_lattice():
    """Simple 3x3 lattice for testing."""
    N = 9
    coords = np.array([[i % 3, i // 3] for i in range(N)], dtype=float)

    # Queen adjacency for 3x3 grid
    adj = {}
    for i in range(N):
        x, y = i % 3, i // 3
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 3 and 0 <= ny < 3:
                    neighbors.append(ny * 3 + nx)
        adj[i] = neighbors

    # Convert to CSR
    row_ptr = [0]
    col_idx = []
    for i in range(N):
        col_idx.extend(sorted(adj[i]))
        row_ptr.append(len(col_idx))

    return coords, (row_ptr, col_idx), N


class TestObservedValidation:
    """Tests for observed array validation."""

    def test_observed_with_nan(self, simple_lattice):
        """Observed with NaN should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9], dtype=float)
        expected = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="contains NaN values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_observed_with_inf(self, simple_lattice):
        """Observed with Inf should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, np.inf, 4, 5, 6, 7, 8, 9], dtype=float)
        expected = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="contains infinite values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_observed_with_negative(self, simple_lattice):
        """Observed with negative values should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, -3, 4, 5, 6, 7, 8, 9], dtype=int)
        expected = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="must be non-negative"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_observed_valid(self, simple_lattice):
        """Valid observed array should work."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
        expected = np.ones(N, dtype=float)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )
        assert result is not None


class TestExpectedValidation:
    """Tests for expected array validation (POISSON model)."""

    def test_expected_with_nan(self, simple_lattice):
        """Expected with NaN should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)

        with pytest.raises(ValueError, match="contains NaN values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_expected_with_inf(self, simple_lattice):
        """Expected with Inf should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)

        with pytest.raises(ValueError, match="contains infinite values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_expected_with_zero(self, simple_lattice):
        """Expected with zero should be rejected for POISSON."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)

        with pytest.raises(ValueError, match="must be strictly positive"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_expected_with_negative(self, simple_lattice):
        """Expected with negative should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)

        with pytest.raises(ValueError, match="must be non-negative"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_expected_valid(self, simple_lattice):
        """Valid expected array should work."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            simcount=0,
        )
        assert result is not None


class TestBinomialPopulationValidation:
    """Tests for population array validation (BINOMIAL model)."""

    def test_population_with_nan(self, simple_lattice):
        """Population with NaN should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        with pytest.raises(ValueError, match="contains NaN values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_population_with_inf(self, simple_lattice):
        """Population with Inf should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10.0, 20.0, np.inf, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        with pytest.raises(ValueError, match="contains infinite values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_population_with_zero(self, simple_lattice):
        """Population with zero should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10.0, 20.0, 0.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        with pytest.raises(ValueError, match="must be strictly positive"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_population_with_negative(self, simple_lattice):
        """Population with negative should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10.0, 20.0, -30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        with pytest.raises(ValueError, match="must be non-negative"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_population_with_non_integer(self, simple_lattice):
        """Population with non-integer should raise ValueError (strict validation)."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10.0, 20.0, 30.5, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        with pytest.raises(ValueError, match="must contain integer values"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_population_valid_integers(self, simple_lattice):
        """Valid integer population should work."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
        population = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=int)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            rantype="BINOMIAL",
            simcount=0,
        )
        assert result is not None

    def test_population_valid_floats_with_integer_values(self, simple_lattice):
        """Float array with exact integer values should work."""
        coords, adjacency, N = simple_lattice
        observed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
        population = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=float)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            rantype="BINOMIAL",
            simcount=0,
        )
        assert result is not None


class TestRantypeValidation:
    """Tests for rantype validation logic."""

    def test_poisson_with_invalid_rantype(self, simple_lattice):
        """POISSON with invalid rantype should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="Invalid rantype"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                rantype="BINOMIAL",  # Invalid for POISSON
                simcount=0,
            )

    def test_binomial_with_multinomial_rantype_rejected(self, simple_lattice):
        """BINOMIAL with explicit MULTINOMIAL rantype should be rejected."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=int)

        with pytest.raises(ValueError, match="Invalid rantype"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                rantype="MULTINOMIAL",
                simcount=0,
            )

    def test_binomial_with_valid_rantype(self, simple_lattice):
        """BINOMIAL with BINOMIAL rantype should work."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=int)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            rantype="BINOMIAL",
            simcount=0,
        )
        assert result is not None

    def test_binomial_defaults_rantype_to_binomial(self, simple_lattice):
        """BINOMIAL model should pick the BINOMIAL null when rantype is omitted."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        population = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=int)

        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            simcount=0,
        )
        assert result is not None


class TestLengthMismatch:
    """Tests for array length validation."""

    def test_observed_length_mismatch(self, simple_lattice):
        """Observed length != N should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N + 1, dtype=int)  # Wrong length
        expected = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="observed length mismatch"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )

    def test_expected_length_mismatch(self, simple_lattice):
        """Expected length != N should raise ValueError."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.ones(N + 1, dtype=float)  # Wrong length

        with pytest.raises(ValueError, match="expected/population length mismatch"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                simcount=0,
            )


class TestErrorMessages:
    """Tests for improved error messages."""

    def test_missing_expected_for_poisson(self, simple_lattice):
        """Missing expected for POISSON should have clear error message."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)

        with pytest.raises(ValueError, match="model='POISSON' requires the 'expected' parameter"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                model="POISSON",
                simcount=0,
            )

    def test_missing_population_for_binomial(self, simple_lattice):
        """Missing population for BINOMIAL should have clear error message."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)

        with pytest.raises(
            ValueError, match="model='BINOMIAL' requires the 'population' parameter"
        ):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )

    def test_both_expected_and_population_for_binomial(self, simple_lattice):
        """Providing both expected and population for BINOMIAL should error."""
        coords, adjacency, N = simple_lattice
        observed = np.ones(N, dtype=int)
        expected = np.ones(N, dtype=float)
        population = np.ones(N, dtype=float)

        with pytest.raises(ValueError, match="provide only 'population'"):
            flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                population=population,
                model="BINOMIAL",
                rantype="BINOMIAL",
                simcount=0,
            )
