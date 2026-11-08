"""Regression tests for the tuning repairs of the 2026-09-07 run (FXA-185, FXA-186)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import flexscanx as flx
from flexscanx.tuning import _normalize_tuning_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]


def _grid(side: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    n = side * side
    coords = np.array([[i % side, i // side] for i in range(n)], dtype=float)
    row_ptr, col_idx = [0], []
    for i in range(n):
        r, c = divmod(i, side)
        neighbours = []
        if r > 0:
            neighbours.append(i - side)
        if r < side - 1:
            neighbours.append(i + side)
        if c > 0:
            neighbours.append(i - 1)
        if c < side - 1:
            neighbours.append(i + 1)
        col_idx.extend(sorted(neighbours))
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _defaults(n: int) -> tuple[int, int, int, int]:
    return _normalize_tuning_bounds(
        N=n, K_start=None, S_start=None, K_min=None, S_min=None, require_K_gt_S=True
    )


# ---------------------------------------------------------------------------
# FXA-185: the cluster-size axis is actually searched at small N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", [3, 4, 5, 6])
def test_find_stable_params_all_steps_the_cluster_size_axis(side: int) -> None:
    coords, adjacency = _grid(side)
    n = side * side
    rng = np.random.default_rng(side)
    observed = rng.poisson(3.0, n)
    out = flx.find_stable_params_all(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=np.full(n, 3.0),
        model="POISSON",
        clustertype="HOT",
    )
    phases = [step["phase"] for step in out["trace"]]
    assert any(phase.startswith("decS") for phase in phases), phases
    s_values = {step["to"][1] for step in out["trace"]} | {step["from"][1] for step in out["trace"]}
    assert len(s_values) >= 2, s_values
    assert out["best"]["S"] >= 1 and out["best"]["K"] > out["best"]["S"]


@pytest.mark.parametrize("n", list(range(4, 120)))
def test_default_floors_leave_room_below_the_start_values(n: int) -> None:
    k_start, s_start, k_min, s_min = _defaults(n)
    assert k_min <= k_start
    assert s_min <= s_start
    # Whenever the start is above the smallest legal size, at least one step must exist.
    if s_start > 1:
        assert s_min < s_start, (n, s_start, s_min)
    if k_start > 2:
        assert k_min < k_start, (n, k_start, k_min)


def test_multiplicative_step_still_moves_from_two() -> None:
    """round(0.8 * 2) == 2 used to freeze the axis; the step must reach 1 when 1 is feasible."""
    coords, adjacency = _grid(3)
    rng = np.random.default_rng(3)
    observed = rng.poisson(3.0, 9)
    out = flx.find_stable_params_all(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=np.full(9, 3.0),
        model="POISSON",
        clustertype="HOT",
        S_start=2,
        S_min=1,
    )
    visited = {step["to"][1] for step in out["trace"] if step["phase"].startswith("decS")}
    assert 1 in visited


# ---------------------------------------------------------------------------
# FXA-186: the reference documents the implemented default rule
# ---------------------------------------------------------------------------


def test_api_reference_states_the_scaled_min_defaults() -> None:
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "# Default: max(8, 1)" not in text
    assert "# Default: max(5, 1)" not in text
    assert "K_min=None,              # Default: max(2, min(8, round(25% of N)))" in text
    assert "S_min=None,              # Default: max(1, min(5, round(6% of N)))" in text
    # The documented rule is the implemented rule.
    for n in (9, 25, 36, 64, 100):
        _k0, _s0, k_min, s_min = _defaults(n)
        assert k_min == max(2, min(8, round(0.25 * n)))
        assert s_min == max(1, min(5, round(0.06 * n)))
