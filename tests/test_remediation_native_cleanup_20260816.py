"""Regression contracts for FXA-068, FXA-130, and FXA-145."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import flexscanx as flx

ROOT = Path(__file__).resolve().parents[1]


def _line(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    coords = np.column_stack((np.arange(n, dtype=float), np.zeros(n, dtype=float)))
    row_ptr = [0]
    col_idx: list[int] = []
    for area in range(n):
        col_idx.extend(neighbor for neighbor in (area - 1, area + 1) if 0 <= neighbor < n)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_poisson_result_reports_eb_applied_and_unavailable_fallback() -> None:
    coords, adjacency = _line(10)
    applied = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[12, 10, 2, 1, 1, 1, 1, 1, 1, 1],
        expected=[4.0] * 10,
        ratio_smoothing="EB",
        simcount=0,
    )
    assert applied["scan_settings"]["ratio_smoothing"] == "EB"
    assert applied["ratio_smoothing_applied"] == "EB"

    fallback = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[5] * 10,
        expected=[5.0] * 10,
        stattype="ORIGINAL",
        ratio_smoothing="EB",
        simcount=0,
    )
    assert fallback["scan_settings"]["ratio_smoothing"] == "EB"
    assert fallback["ratio_smoothing_applied"] == "NONE"


def test_binomial_result_reports_eb_applied_and_unavailable_fallback() -> None:
    coords, adjacency = _line(8)
    applied = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[8, 7, 2, 1, 1, 1, 1, 1],
        population=[10] * 8,
        model="BINOMIAL",
        ratio_smoothing="EB",
        simcount=0,
    )
    assert applied["ratio_smoothing_applied"] == "EB"

    fallback = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[2] * 8,
        population=[10] * 8,
        model="BINOMIAL",
        stattype="ORIGINAL",
        ratio_smoothing="EB",
        simcount=0,
    )
    assert fallback["ratio_smoothing_applied"] == "NONE"


def test_dead_debug_header_is_absent_from_maintained_core() -> None:
    assert not (ROOT / "src/flexscanx_core/debug_log.hpp").exists()


def test_alias_sampler_comment_states_fixed_seed_streams_are_not_aligned() -> None:
    source = (ROOT / "src/flexscanx_core/monte_carlo.hpp").read_text(encoding="utf-8")
    assert "not draw-for-draw CRN-aligned" in source
    assert "keep CRN alignment across methods" not in source
