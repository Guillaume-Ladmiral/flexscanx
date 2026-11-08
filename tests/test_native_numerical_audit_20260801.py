"""Independent-reference regressions for the 2026-08-01 native numerical audit."""

from __future__ import annotations

import time
from decimal import Decimal, localcontext

import numpy as np
import pytest

import flexscanx as flx
import flexscanx._core as core

try:
    from scipy import stats as scipy_stats
except ModuleNotFoundError:  # pragma: no cover - exercised by minimal-dependency CI
    scipy_stats = None


@pytest.mark.parametrize(
    ("k", "n", "p"),
    [
        (51_000, 100_000, 0.5),
        (5_500, 10_000, 0.5),
        (97, 1_000_000, 0.0001),
    ],
)
@pytest.mark.skipif(scipy_stats is None, reason="SciPy is required for the independent reference")
def test_binomial_middle_p_uses_an_accurate_independent_tail(k: int, n: int, p: float):
    assert scipy_stats is not None
    expected = scipy_stats.binom.sf(k, n, p) + 0.5 * scipy_stats.binom.pmf(k, n, p)

    upper = core.binomial_middle_p_upper(k, n, p)
    lower = core.binomial_middle_p_lower(k, n, p)

    assert upper == pytest.approx(expected, rel=2e-10, abs=1e-300)
    assert lower + upper == pytest.approx(1.0, abs=2e-14)


def test_binomial_middle_p_runtime_does_not_scale_linearly_with_n():
    started = time.perf_counter()
    upper = core.binomial_middle_p_upper(100_000_000, 200_000_000, 0.5)
    elapsed = time.perf_counter() - started

    assert upper == pytest.approx(0.5, abs=2e-8)
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("k", "mean"),
    [
        (1_000_000_000, 1_000_000_000.0),
        (2_147_483_647, 2_147_483_647.0),
        (1_000_094_868, 1_000_000_000.0),
    ],
)
@pytest.mark.skipif(scipy_stats is None, reason="SciPy is required for the independent reference")
def test_poisson_middle_p_is_accurate_at_high_counts(k: int, mean: float):
    assert scipy_stats is not None
    expected = scipy_stats.poisson.sf(k, mean) + 0.5 * scipy_stats.poisson.pmf(k, mean)

    upper = core.poisson_middle_p_upper(k, mean)
    lower = core.poisson_middle_p_lower(k, mean)

    assert upper == pytest.approx(expected, rel=2e-9, abs=2e-14)
    assert lower + upper == pytest.approx(1.0, abs=2e-14)


def _two_area_llr(model: str) -> float:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    model_data = (
        {"expected": [2_000_000_000.0, 2_000_000_000.0]}
        if model == "POISSON"
        else {"population": [2_000_000_000, 2_000_000_000]}
    )
    result = flx.scan(
        coords=coords,
        adjacency=([0, 1, 2], [1, 0]),
        observed=[1_000_000_001, 999_999_999],
        model=model,
        scanmethod="CIRCULAR",
        stattype="ORIGINAL",
        clustertype="HOT",
        clustersize=1,
        K=2,
        knn_mode="K_FIXED",
        simcount=0,
        **model_data,
    )
    assert result["area"] == [1]
    return float(result["stats"])


def _decimal_llr(model: str) -> float:
    with localcontext() as context:
        context.prec = 80
        n_in = Decimal(1_000_000_001)
        n_out = Decimal(999_999_999)
        m_in = m_out = Decimal(2_000_000_000)
        n_total = n_in + n_out
        m_total = m_in + m_out
        if model == "POISSON":
            value = n_in * ((n_in / m_in) / (n_total / m_total)).ln()
            value += n_out * ((n_out / m_out) / (n_total / m_total)).ln()
        else:
            p_total = n_total / m_total
            value = Decimal(0)
            for count, population in ((n_in, m_in), (n_out, m_out)):
                probability = count / population
                value += population * (
                    probability * (probability / p_total).ln()
                    + (1 - probability) * ((1 - probability) / (1 - p_total)).ln()
                )
        return float(value)


@pytest.mark.parametrize("model", ["POISSON", "BINOMIAL"])
def test_original_llr_retains_small_positive_large_denominator_signal(model: str):
    assert _two_area_llr(model) == pytest.approx(_decimal_llr(model), rel=2e-7, abs=1e-18)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (core.poisson_middle_p_upper, (1, -1.0)),
        (core.poisson_middle_p_lower, (1, float("nan"))),
        (core.binomial_middle_p_upper, (1, -1, 0.5)),
        (core.binomial_middle_p_lower, (1, 2, 1.1)),
    ],
)
def test_direct_middle_p_bindings_reject_invalid_distribution_parameters(function, args):
    with pytest.raises(ValueError):
        function(*args)
