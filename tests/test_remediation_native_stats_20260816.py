"""Focused public regressions for the 2026-08-16 native statistics remediation."""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx
import flexscanx._core as core


def _line_graph() -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """Return a two-area undirected line suitable for a minimal scan."""
    return np.array([[0.0, 0.0], [1.0, 0.0]]), ([0, 1, 2], [1, 0])


def test_poisson_monte_carlo_rejects_mean_above_native_integer_domain() -> None:
    """A Poisson MC request must fail before constructing an int-result sampler."""
    coords, adjacency = _line_graph()

    with pytest.raises(ValueError, match="Poisson mean exceeds the supported int range"):
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1, 0],
            expected=[float(np.iinfo(np.int32).max) + 1.0, 1.0],
            model="POISSON",
            rantype="POISSON",
            simcount=0,
            K=2,
            clustersize=1,
        )


def test_early_stop_checks_the_recommended_199_replicate_run_before_completion() -> None:
    """The Wilson rule is evaluated at documented cadence even when B <= 200."""
    coords, adjacency = _line_graph()
    simcount = 199
    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[1000, 0],
        expected=[1.0, 1.0],
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="ORIGINAL",
        clustertype="HOT",
        rantype="MULTINOMIAL",
        simcount=simcount,
        seed=17,
        mc_early_stop=True,
        mc_alpha=0.05,
        K=2,
        clustersize=1,
        n_jobs_centers=1,
    )

    effective_denominator = int(round(int(result["rank"]) / float(result["pval"])))
    assert effective_denominator < simcount + 1
    assert effective_denominator % 25 == 1


def test_completed_monte_carlo_count_is_reported_through_native_scan_and_series() -> None:
    """Provenance distinguishes the requested budget from completed null replicates."""
    coords, adjacency = _line_graph()
    row_ptr, col_idx = adjacency
    simcount = 199
    scan_kwargs = {
        "model": "POISSON",
        "scanmethod": "CIRCULAR",
        "stattype": "ORIGINAL",
        "clustertype": "HOT",
        "rantype": "MULTINOMIAL",
        "simcount": simcount,
        "seed": 17,
        "mc_early_stop": True,
        "mc_alpha": 0.05,
        "K": 2,
        "clustersize": 1,
        "n_jobs_centers": 1,
    }
    observed = np.asarray([1000, 0], dtype=np.int32)
    expected = np.asarray([1.0, 1.0], dtype=float)

    handle = core.BuildStatic(coords, row_ptr, col_idx, 2, float("inf"))
    raw = core.ScanSlice(handle, observed, expected, scan_kwargs)
    raw_actual = raw["simcount_actual"]
    assert raw_actual < simcount
    assert raw_actual == int(round(int(raw["rank"]) / float(raw["pval"]))) - 1

    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        **scan_kwargs,
    )
    assert result["scan_settings"]["simcount_requested"] == simcount
    assert result["scan_settings"]["simcount_actual"] == raw_actual

    both = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        **(scan_kwargs | {"clustertype": "BOTH"}),
    )
    assert both["scan_settings"]["simcount_requested"] == simcount
    assert both["scan_settings"]["simcount_actual"] == simcount
    assert both["hot"]["scan_settings"]["simcount_actual"] == simcount
    assert both["cold"]["scan_settings"]["simcount_actual"] == simcount

    series = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=np.vstack([observed, np.zeros_like(observed)]),
        expected=np.vstack([expected, expected]),
        **scan_kwargs,
    )
    assert series["scan_settings"]["simcount_requested"] == simcount
    assert series["scan_settings"]["simcount_actual"] == [raw_actual, 25]

    both_series = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=np.vstack([observed, np.zeros_like(observed)]),
        expected=np.vstack([expected, expected]),
        **(scan_kwargs | {"clustertype": "BOTH"}),
    )
    assert both_series["scan_settings"]["simcount_requested"] == simcount
    assert both_series["scan_settings"]["simcount_actual"] == [simcount, simcount]
