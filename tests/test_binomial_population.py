import numpy as np
import pytest

import flexscanx as p


def test_binomial_population_fractional_raises():
    """Test that fractional BINOMIAL populations now always raise (strict mode)."""
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]
    observed = [1, 2]
    population = [10.0, 3.7]  # fractional

    # v0.0.1: Always strict - fractional populations raise ValueError
    with pytest.raises(ValueError, match="integer values"):
        _ = p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            population=population,
            model="BINOMIAL",
            rantype="BINOMIAL",
            clustertype="HOT",
            stattype="RESTRICTED",
            scanmethod="FLEXIBLE",
        )


def test_binomial_population_integer_accepted():
    """Test that integer BINOMIAL populations are accepted."""
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]
    observed = [1, 2]
    population = [10.0, 4.0]  # Exact integers - should work

    # v0.0.1: Integer populations work fine
    result = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        population=population,
        model="BINOMIAL",
        rantype="BINOMIAL",
        simcount=0,
    )
    assert "area" in result
