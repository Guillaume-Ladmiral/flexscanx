"""Regression coverage for the exact-flexible cost guard (FXA-115)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import flexscanx as flx
from flexscanx._scan_internal import warn_exact_flexible_large_k


def _line_problem(n: int = 14) -> dict[str, object]:
    coords = np.column_stack((np.arange(n, dtype=float), np.zeros(n, dtype=float)))
    row_ptr = [0]
    col_idx: list[int] = []
    for area in range(n):
        col_idx.extend(neighbor for neighbor in (area - 1, area + 1) if 0 <= neighbor < n)
        row_ptr.append(len(col_idx))
    observed = [5] * n
    observed[0] = 10
    return {
        "coords": coords,
        "adjacency": (row_ptr, col_idx),
        "observed": observed,
        "expected": [5.0] * n,
        "K": n,
        "clustersize": n,
        "stattype": "ORIGINAL",
    }


def test_cost_estimate_uses_cluster_depth_and_total_repetitions() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_exact_flexible_large_k("FLEXIBLE", 30, 4, None, None, stattype="ORIGINAL")
    assert not caught, "a shallow K=30, clustersize=4 scan is below the measured guard budget"

    with pytest.warns(UserWarning, match="estimated exact-enumeration work"):
        warn_exact_flexible_large_k("FLEXIBLE", 14, 14, None, None, stattype="ORIGINAL")

    with pytest.warns(UserWarning, match="999 repeated scans"):
        warn_exact_flexible_large_k(
            "FLEXIBLE", 8, 8, None, None, stattype="ORIGINAL", repetitions=999
        )


def test_scan_warns_for_deep_exact_flexible_enumeration() -> None:
    with pytest.warns(UserWarning, match="estimated exact-enumeration work"):
        flx.scan(**_line_problem(), simcount=0, n_jobs_centers=1)


def test_scan_series_warns_once_for_total_deep_enumeration() -> None:
    problem = _line_problem()
    problem["observed"] = [problem["observed"]]
    problem["expected"] = [problem["expected"]]

    with pytest.warns(UserWarning, match="estimated exact-enumeration work") as caught:
        flx.scan_series(**problem, simcount=0, n_jobs_centers=1)
    assert len(caught) == 1


def test_calibration_warns_for_repeated_deep_enumeration() -> None:
    with pytest.warns(UserWarning, match="estimated exact-enumeration work") as caught:
        flx.calibrate_llr_threshold(**_line_problem(), n_sims=1, seed=None)
    assert len(caught) == 1
