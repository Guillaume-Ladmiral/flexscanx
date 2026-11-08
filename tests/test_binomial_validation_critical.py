"""Test critical BINOMIAL validation: observed <= population."""

import numpy as np
import pytest


def test_binomial_observed_exceeds_population_single_slice():
    """Test that scan() rejects observed > population for BINOMIAL model."""
    from flexscanx import scan

    N = 10
    coords = np.random.default_rng(12).random((N, 2))
    adjacency = (
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [9, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0],
    )

    observed = np.array([5, 3, 8, 2, 4, 6, 1, 7, 3, 5], dtype=int)
    population = np.array(
        [10, 10, 5, 10, 10, 10, 10, 10, 10, 10], dtype=int
    )  # observed[2]=8 > population[2]=5

    with pytest.raises(ValueError, match=r"observed counts cannot exceed population"):
        scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            population=population,
            model="BINOMIAL",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            K=5,
            clustersize=3,
            simcount=0,
        )


def test_binomial_observed_exceeds_population_series():
    """Test that scan_series() rejects observed > population for BINOMIAL model."""
    from flexscanx import scan_series

    T, N = 3, 10
    rng = np.random.default_rng(44)
    coords = rng.random((N, 2))
    adjacency = (
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [9, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0],
    )

    observed = rng.integers(0, 8, size=(T, N))
    population = np.full((T, N), 10, dtype=int)
    # Inject violation: observed[1, 3] = 12 > population[1, 3] = 10
    observed[1, 3] = 12

    with pytest.raises(ValueError, match=r"observed counts cannot exceed population"):
        scan_series(
            observed=observed,
            population=population,
            coords=coords,
            adjacency=adjacency,
            model="BINOMIAL",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="HOT",
            K=5,
            clustersize=3,
            simcount=0,
        )


def test_binomial_valid_observed_equals_population():
    """Test that observed == population is valid for BINOMIAL model."""
    from flexscanx import scan

    N = 10
    coords = np.random.default_rng(76).random((N, 2))
    adjacency = (
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [9, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0],
    )

    observed = np.array([10, 8, 10, 5, 10, 6, 3, 10, 4, 10], dtype=int)
    population = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=int)

    # Should not raise
    result = scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        population=population,
        model="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=5,
        clustersize=3,
        simcount=0,
    )
    assert result is not None


def test_binomial_population_rejects_values_above_native_integer_range():
    from flexscanx import scan

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    adjacency = ([0, 1, 2], [1, 0])

    with pytest.raises(ValueError, match="exceed int32 range"):
        scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1, 2],
            population=[3_000_000_000, 3_000_000_000],
            model="BINOMIAL",
            K=2,
            clustersize=1,
            simcount=0,
        )


def test_native_binomial_population_rejects_values_above_integer_range():
    from flexscanx import _core, build_static

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    handle = build_static(
        coords=coords,
        adjacency=([0, 1, 2], [1, 0]),
        K=2,
    )

    with pytest.raises(ValueError, match="population.*int range"):
        _core.ScanSlice(
            handle,
            [1, 2],
            [3_000_000_000.0, 3_000_000_000.0],
            {
                "model": "BINOMIAL",
                "scanmethod": "FLEXIBLE",
                "stattype": "RESTRICTED",
                "clustertype": "HOT",
                "clustersize": 1,
                "simcount": 0,
            },
        )
