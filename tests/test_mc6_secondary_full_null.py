"""MC-6 / DEF-STOP: secondary clusters are ranked against the FULL-region null.

Phase 3 of
``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/REMEDIATION_PLAN_2026-06-15.md``
(Option A, paper-faithful). Per DEF-STOP
(``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/findings/00_reference_definitions.md``;
Otani & Takahashi
2021 §2.4) each secondary cluster's significance is evaluated "the same way as the
MLC -- its LR is compared against the same Monte-Carlo null distribution of the
maximum ... as if it were the primary cluster", i.e. the *unconditional full-region*
null.

Through 0.0.2 flexscanx recomputed that null with already-detected areas excluded
(a *conditional* null). Because ``max over (windows that avoid detected) <= max over
(all windows)``, the conditional null's maxima are smaller, so secondary p-values
were anti-conservative versus the paper (report §5 "MC-6 / ENUM-5"; ISSUES_AND_FIXES
A7). The fix ranks each secondary against the full-region primary null by default
(``secondary_null="full"``); ``secondary_null="conditional"`` restores the prior
behaviour.

These pins lock:
  * the MLC/primary is byte-identical under either null (DEF-STOP: the switch only
    affects secondaries),
  * the new default makes the secondary p-value MORE conservative (>= the old
    conditional value) -- a deliberate, correct behaviour change, not a regression,
  * ``secondary_null="conditional"`` reproduces the exact pre-fix p-value, and
  * the new default is deterministic for a fixed seed.

The numeric anchors were captured from the clean 0.0.2 rebuild that introduced the
fix; the conditional values are the genuine pre-fix numbers (that code path is
byte-unchanged -- only its reachability is now gated on ``secondary_null``).

Style mirrors ``tests/test_gate4_both_alpha_split.py`` (small coords + CSR adjacency
driving ``flexscanx.scan``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import flexscanx as flx

# --- planted dataset ----------------------------------------------------------
#
# A 12-node line graph. Cases sit at the null rate 5 everywhere except a STRONG
# primary blob (nodes 2-4, count 22) and a MODERATE secondary blob (nodes 8-10,
# count 16). The primary region dominates the per-replicate null maximum, so
# excluding it (the conditional null) visibly lowers the reference maxima the
# secondary is ranked against -- the MC-6 signature.

_N = 12
_PRIMARY_NODES = (2, 3, 4)  # 0-based; reported 1-based as {3, 4, 5}
_SECONDARY_NODES = (8, 9, 10)  # 0-based; reported 1-based as {9, 10, 11}
_SEED = 12345
_SIMCOUNT = 999

# Captured from the 0.0.2 rebuild (audit_scratch/phase3_mc6_explore.py).
_PRIMARY_AREA_1B = [3, 4, 5]
_PRIMARY_STATS = 14.622015801739849
_SECONDARY_AREA_1B = [9, 10, 11]
_SECONDARY_STATS = 2.501568054672674
# Full-region null (new default, Option A) vs conditional null (prior behaviour).
_SECONDARY_RANK_FULL = 255
_SECONDARY_PVAL_FULL = 255 / (_SIMCOUNT + 1)  # 0.255
_SECONDARY_RANK_CONDITIONAL = 180
_SECONDARY_PVAL_CONDITIONAL = 180 / (_SIMCOUNT + 1)  # 0.180


def _chain(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """An ``n``-node line graph: coords on the x-axis + CSR rook adjacency."""
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = [0], []
    for i in range(n):
        if i > 0:
            col_idx.append(i - 1)
        if i < n - 1:
            col_idx.append(i + 1)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _planted() -> tuple[np.ndarray, tuple[list[int], list[int]], np.ndarray, np.ndarray]:
    coords, adjacency = _chain(_N)
    expected = np.full(_N, 5.0)
    observed = np.full(_N, 5, dtype=int)
    for node in _PRIMARY_NODES:
        observed[node] = 22
    for node in _SECONDARY_NODES:
        observed[node] = 16
    return coords, adjacency, observed, expected


def _scan(secondary_null: str | None = None, **override: Any) -> dict[str, Any]:
    coords, adjacency, observed, expected = _planted()
    kwargs: dict[str, Any] = dict(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="ORIGINAL",
        clustertype="HOT",
        clustersize=4,
        K=6,
        simcount=_SIMCOUNT,
        rantype="MULTINOMIAL",
        seed=_SEED,
        n_jobs_centers=1,
        secondary=1,
    )
    if secondary_null is not None:
        kwargs["secondary_null"] = secondary_null
    kwargs.update(override)
    return flx.scan(**kwargs)  # type: ignore[arg-type]


def _first_secondary(result: dict[str, Any]) -> dict[str, Any]:
    secondaries = result.get("secondaries") or []
    assert secondaries, "expected at least one secondary cluster on the planted data"
    return secondaries[0]


def _areas(cluster: dict[str, Any]) -> list[int]:
    return sorted(int(a) for a in (cluster.get("area") or []))


class TestMC6SecondaryFullRegionNull:
    """The secondary null is the full-region primary null (DEF-STOP / Option A)."""

    def test_default_is_full_region_null(self) -> None:
        """Omitting ``secondary_null`` is identical to ``secondary_null="full"``."""
        default = _first_secondary(_scan())
        full = _first_secondary(_scan("full"))
        assert default["rank"] == full["rank"]
        assert default["pval"] == full["pval"]
        assert default["rank"] == _SECONDARY_RANK_FULL

    def test_mlc_primary_unchanged_by_switch(self) -> None:
        """CRITICAL: the MLC/primary is byte-identical under either null.

        DEF-STOP changes only how secondaries are ranked; if the primary p-value
        moved, the window-search exclusion would be wrongly entangled with the
        null-distribution change.
        """
        full = _scan("full")
        conditional = _scan("conditional")
        for res in (full, conditional):
            assert _areas(res) == _PRIMARY_AREA_1B
            assert res["stats"] == pytest.approx(_PRIMARY_STATS, rel=1e-12)
            assert res["rank"] == 1
            assert res["pval"] == pytest.approx(1 / (_SIMCOUNT + 1))
        assert full["stats"] == conditional["stats"]
        assert full["rank"] == conditional["rank"]
        assert full["pval"] == conditional["pval"]

    def test_full_is_more_conservative_than_conditional(self) -> None:
        """The defining behaviour change: the full-region null yields a LARGER
        (more conservative) secondary p-value than the conditional null."""
        full = _first_secondary(_scan("full"))
        conditional = _first_secondary(_scan("conditional"))
        # Same cluster is detected either way (the window search is unchanged).
        assert _areas(full) == _SECONDARY_AREA_1B == _areas(conditional)
        assert full["stats"] == pytest.approx(conditional["stats"], rel=1e-12)
        # Full-region maxima >= conditional maxima -> full rank/p-value is larger.
        assert full["pval"] >= conditional["pval"]
        assert full["pval"] > conditional["pval"], "expected a strict gap on this data"

    def test_full_region_secondary_pvalue_pinned(self) -> None:
        """New default (Option A): the secondary is ranked against the full-region
        null (== the primary's null), pinned to the 0.0.2-rebuild value."""
        secondary = _first_secondary(_scan("full"))
        assert _areas(secondary) == _SECONDARY_AREA_1B
        assert secondary["stats"] == pytest.approx(_SECONDARY_STATS, rel=1e-12)
        assert secondary["rank"] == _SECONDARY_RANK_FULL
        assert secondary["pval"] == pytest.approx(_SECONDARY_PVAL_FULL)

    def test_conditional_reproduces_prefix_value(self) -> None:
        """Back-compat: ``secondary_null="conditional"`` reproduces the exact
        pre-fix (anti-conservative) secondary p-value."""
        secondary = _first_secondary(_scan("conditional"))
        assert secondary["rank"] == _SECONDARY_RANK_CONDITIONAL
        assert secondary["pval"] == pytest.approx(_SECONDARY_PVAL_CONDITIONAL)

    def test_full_region_secondary_pvalue_is_deterministic(self) -> None:
        """Same seed -> identical secondary p-value under the new default."""
        first = _first_secondary(_scan("full"))
        second = _first_secondary(_scan("full"))
        assert first["rank"] == second["rank"]
        assert first["pval"] == second["pval"]

    def test_invalid_secondary_null_raises(self) -> None:
        with pytest.raises(ValueError, match="secondary_null must be one of"):
            _scan("bogus")
