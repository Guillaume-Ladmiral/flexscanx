"""MC-3 / DEF-BOTH: the BOTH envelope is ranked against ONE combined two-sided null.

Phase 2 of
``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/REMEDIATION_PLAN_2026-06-15.md``
(Option A, paper-faithful). Per DEF-BOTH / DEF-MC-NULLMAX
(``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/findings/00_reference_definitions.md``;
Tango 2008 p.81) the two-sided statistic *deletes the direction indicator*, so the
per-replicate maximum is a single max over windows of either orientation and the
BOTH cluster is ranked against that one combined two-sided null distribution.

flexscanx realises ``clustertype="BOTH"`` as an envelope of two one-sided sub-scans.
Through 0.0.2 the reported primary kept its OWN one-sided p-value (ranked against a
single direction's null), which is anti-conservative for a two-sided claim -- it pays
no multiplicity cost for looking both ways (report §5 "MC-3"; ISSUES_AND_FIXES A12).
The fix forms the per-replicate combined null ``max(hot_sim_max, cold_sim_max)`` and
re-ranks the envelope primary (and its secondaries) against it. Because
``max(hot, cold) >=`` either one-sided max, the BOTH p-value becomes MORE
conservative -- a deliberate, correct behaviour change, not a regression.

These pins lock:
  * the BOTH primary p-value is the two-sided value (>= each one-sided p; strictly
    greater on the planted data),
  * the per-direction one-sided p-values are retained under ``hot``/``cold``,
  * the combined null is exactly the element-wise max of the two directions' nulls,
    and the Dwass rank arithmetic ``rank == 1 + #{combined >= stats}`` holds,
  * BOTH secondaries also use the combined null (>= their one-sided, full-region
    Phase-3 value),
  * single-direction HOT/COLD scans (and their p-values) are byte-identical -- the
    ``sim_max`` emission does not leak into the one-sided path, and ``sim_max`` is
    never exposed on the public result,
  * ``simcount=0`` BOTH is unaffected (primary pval ``None``), and
  * the result is deterministic for a fixed seed.

Numeric anchors captured from the clean 0.0.2 rebuild that introduced the fix.
Style mirrors ``tests/test_mc6_secondary_full_null.py`` and
``tests/test_gate4_both_alpha_split.py`` (small coords + CSR adjacency driving
``flexscanx.scan``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import flexscanx as flx
from flexscanx._result_format import (
    _combined_two_sided_null,
    _rerank_cluster_against_null,
    _strip_sim_max,
)

_SIMCOUNT = 999
_SEED = 20260616


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


# --- MAIN planted dataset (primary only) --------------------------------------
#
# A 14-node line graph at the null rate (expected = observed = 10) everywhere
# except a HOT blob (nodes 3,4 -> 16) and a COLD blob (nodes 9,10 -> 5). The two
# signals are comparable, so the BOTH primary could be either direction; the
# minimum observed count is 5, so there are NO zero-case windows (which would drag
# in the rflexscan-only MC-2 confound). HOT and COLD draw the SAME multinomial null
# replicates (shared seed, CRN), so max(hot,cold) is the per-replicate two-sided
# maximum.

_MAIN_N = 14
_MAIN_HOT_NODES = (3, 4)
_MAIN_COLD_NODES = (9, 10)

# Captured from the 0.0.2 rebuild.
_MAIN_PRIMARY_TYPE = "COLD"
_MAIN_PRIMARY_AREA_1B = [10, 11]
_MAIN_PRIMARY_STATS = 3.6352537597136316
_MAIN_PRIMARY_RANK_TWO_SIDED = 206
_MAIN_PRIMARY_PVAL_TWO_SIDED = 206 / (_SIMCOUNT + 1)  # 0.206
_MAIN_HOT_RANK = 101
_MAIN_HOT_PVAL = 101 / (_SIMCOUNT + 1)  # 0.101
_MAIN_HOT_STATS = 3.4546464981464555
_MAIN_COLD_RANK = 151
_MAIN_COLD_PVAL = 151 / (_SIMCOUNT + 1)  # 0.151
_MAIN_COLD_STATS = 3.6352537597136316


def _main_inputs() -> tuple[np.ndarray, tuple[list[int], list[int]], np.ndarray, np.ndarray]:
    coords, adjacency = _chain(_MAIN_N)
    expected = np.full(_MAIN_N, 10.0)
    observed = np.full(_MAIN_N, 10, dtype=int)
    for node in _MAIN_HOT_NODES:
        observed[node] = 16
    for node in _MAIN_COLD_NODES:
        observed[node] = 5
    return coords, adjacency, observed, expected


def _main_common(**override: Any) -> dict[str, Any]:
    coords, adjacency, observed, expected = _main_inputs()
    kwargs: dict[str, Any] = dict(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="ORIGINAL",
        clustersize=4,
        K=6,
        simcount=_SIMCOUNT,
        rantype="MULTINOMIAL",
        seed=_SEED,
        n_jobs_centers=1,
    )
    kwargs.update(override)
    return kwargs


def _areas(cluster: dict[str, Any]) -> list[int]:
    return sorted(int(a) for a in (cluster.get("area") or []))


class TestMC3CombinedTwoSidedNull:
    """The BOTH primary is ranked against the combined two-sided null (Option A)."""

    def test_primary_pval_is_two_sided_and_pinned(self) -> None:
        res = flx.scan(clustertype="BOTH", **_main_common())
        assert res["type"] == "BOTH"
        assert res["primary_type"] == _MAIN_PRIMARY_TYPE
        assert _areas(res["primary"]) == _MAIN_PRIMARY_AREA_1B
        assert res["primary"]["stats"] == pytest.approx(_MAIN_PRIMARY_STATS, rel=1e-12)
        assert res["primary"]["rank"] == _MAIN_PRIMARY_RANK_TWO_SIDED
        assert res["primary"]["pval"] == pytest.approx(_MAIN_PRIMARY_PVAL_TWO_SIDED)

    def test_primary_pval_ge_each_one_sided(self) -> None:
        """The anti-conservatism fix: the BOTH primary's two-sided p is *always* >= the
        WINNING direction's own one-sided p -- same MLC, and (no early stop) the same
        null length, while ``combined = max(hot, cold) >=`` that direction's null
        elementwise -- so it can only push the reported p up. On this comparable
        hot+cold dataset it also exceeds the LOSING direction's one-sided p, but that
        is data-specific, not a general theorem: the losing direction reports a
        different (smaller-stat) window, so a much stronger HOT signal could leave the
        two-sided p below a weak COLD's one-sided p. The general guarantee is only
        against the winning/primary direction (which is what the old primary p was)."""
        res = flx.scan(clustertype="BOTH", **_main_common())
        two_sided = res["primary"]["pval"]
        # General guarantee: >= the winning/primary direction's own one-sided p.
        winning = res["cold"] if res["primary_type"] == "COLD" else res["hot"]
        assert two_sided > winning["pval"]
        # Data-specific on this comparable dataset: also >= the losing direction's p.
        assert two_sided >= res["hot"]["pval"]
        assert two_sided >= res["cold"]["pval"]

    def test_one_sided_pvalues_retained_under_hot_cold(self) -> None:
        """Transparency: the per-direction one-sided p-values stay under hot/cold."""
        res = flx.scan(clustertype="BOTH", **_main_common())
        assert res["hot"]["rank"] == _MAIN_HOT_RANK
        assert res["hot"]["pval"] == pytest.approx(_MAIN_HOT_PVAL)
        assert res["hot"]["stats"] == pytest.approx(_MAIN_HOT_STATS, rel=1e-12)
        assert res["cold"]["rank"] == _MAIN_COLD_RANK
        assert res["cold"]["pval"] == pytest.approx(_MAIN_COLD_PVAL)
        assert res["cold"]["stats"] == pytest.approx(_MAIN_COLD_STATS, rel=1e-12)

    def test_sim_max_never_exposed(self) -> None:
        """The internal transport array must not leak onto the public result."""
        res = flx.scan(clustertype="BOTH", **_main_common())
        for node in (res, res["primary"], res["hot"], res["cold"]):
            assert "sim_max" not in node
            for secondary in node.get("secondaries") or []:
                assert "sim_max" not in secondary


class TestMC3CombinedNullArithmetic:
    """White-box: the combined null is the element-wise max and the rank arithmetic
    matches ``bindings_scan_result.cpp`` exactly."""

    def _direction_sim_max(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Run the two one-sided sub-scans the BOTH envelope runs internally, with
        ``_emit_sim_max`` set, and return their full-region null-max arrays plus the
        observed COLD (primary) stats/rank."""
        from flexscanx._scan_internal import _build_base_settings
        from flexscanx.api import core

        coords, adjacency, observed, expected = _main_inputs()
        handle = flx.build_static(coords, adjacency, K=6)
        arr_obs = np.asarray(observed, dtype=np.int32)
        arr_exp = np.asarray(expected, dtype=float)
        base = _build_base_settings(
            N=_MAIN_N,
            ralpha=0.2,
            clustersize_eff=4,
            scanmethod="CIRCULAR",
            stattype="ORIGINAL",
            model="POISSON",
            simcount=_SIMCOUNT,
            rantype_u="MULTINOMIAL",
            mc_early_stop=False,
            mc_alpha=0.05,
            mc_progress=None,
            mc_progress_every=None,
            secondary=None,
            secondary_p_stop=None,
            secondary_stats_stop=None,
            default_secondary_p_stop=1.0,
            debug_prune=False,
            enable_center_ub_gating=None,
            cache_candidate_llr=None,
            scheduler_backend=None,
            enable_frame_ub_prune=None,
            mc_use_alias=False,
            ratio_smoothing="NONE",
            seed=_SEED,
            n_jobs_centers=1,
        )
        settings_hot = dict(base, clustertype="HOT", _emit_sim_max=True)
        settings_cold = dict(base, clustertype="COLD", _emit_sim_max=True)
        raw_hot = core.ScanSlice(handle, arr_obs, arr_exp, settings_hot)
        raw_cold = core.ScanSlice(handle, arr_obs, arr_exp, settings_cold)
        hot_sim = np.asarray(raw_hot["sim_max"], dtype=float)
        cold_sim = np.asarray(raw_cold["sim_max"], dtype=float)
        return hot_sim, cold_sim, float(raw_cold["stats"]), int(raw_cold["rank"])

    def test_emitted_arrays_align_and_combine_to_elementwise_max(self) -> None:
        hot_sim, cold_sim, _, _ = self._direction_sim_max()
        assert hot_sim.size == cold_sim.size == _SIMCOUNT
        combined = np.maximum(hot_sim, cold_sim)
        # The combined null dominates both one-sided nulls replicate-by-replicate.
        assert np.all(combined >= hot_sim)
        assert np.all(combined >= cold_sim)
        # ... and equals one of the two on every replicate.
        assert np.all((combined == hot_sim) | (combined == cold_sim))

    def test_public_primary_rank_equals_dwass_against_combined(self) -> None:
        """The public BOTH primary rank/pval == the Dwass rule applied to the
        element-wise-max combined null."""
        hot_sim, cold_sim, cold_stats, cold_rank_one_sided = self._direction_sim_max()
        combined = np.maximum(hot_sim, cold_sim)
        expected_rank = 1 + int(np.count_nonzero(combined >= cold_stats))
        expected_pval = expected_rank / (combined.size + 1)

        res = flx.scan(clustertype="BOTH", **_main_common())
        assert res["primary"]["rank"] == expected_rank
        assert res["primary"]["pval"] == pytest.approx(expected_pval)
        # The combined null can only push the rank up vs the one-sided cold rank.
        assert expected_rank >= cold_rank_one_sided


# --- SECONDARY planted dataset ------------------------------------------------
#
# A 16-node line with TWO cold blobs (so the COLD direction yields a primary AND a
# secondary) plus a mild hot blob that lifts the combined null. The BOTH primary is
# COLD; its secondary is re-ranked against the same combined two-sided null, so its
# p-value is >= the one-sided full-region (Phase-3) secondary p-value.

_SEC_N = 16
_SEC_SEED = 777
_SEC_PRIMARY_AREA_1B = [4, 5]
_SEC_SECONDARY_AREA_1B = [11, 12]
_SEC_SECONDARY_STATS = 2.3095934921268224
_SEC_SECONDARY_RANK_TWO_SIDED = 628
_SEC_SECONDARY_PVAL_TWO_SIDED = 628 / (_SIMCOUNT + 1)  # 0.628
_SEC_SECONDARY_RANK_ONE_SIDED = 453
_SEC_SECONDARY_PVAL_ONE_SIDED = 453 / (_SIMCOUNT + 1)  # 0.453
# secondary_null="conditional" retained-direction value (non-overlapping windows only,
# anti-conservative vs the full-region one-sided 0.453); the BOTH primary IGNORES this.
_SEC_SECONDARY_RANK_COND_ONE_SIDED = 398
_SEC_SECONDARY_PVAL_COND_ONE_SIDED = 398 / (_SIMCOUNT + 1)  # 0.398


def _sec_common(**override: Any) -> dict[str, Any]:
    coords, adjacency = _chain(_SEC_N)
    expected = np.full(_SEC_N, 10.0)
    observed = np.full(_SEC_N, 10, dtype=int)
    observed[3] = observed[4] = 4  # COLD blob A (stronger)
    observed[10] = observed[11] = 5  # COLD blob B (weaker -> secondary)
    observed[14] = 15  # mild hot, contributes to the combined null
    kwargs: dict[str, Any] = dict(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="ORIGINAL",
        clustersize=4,
        K=6,
        simcount=_SIMCOUNT,
        rantype="MULTINOMIAL",
        seed=_SEC_SEED,
        n_jobs_centers=1,
        secondary=2,
    )
    kwargs.update(override)
    return kwargs


class TestMC3SecondaryCombinedNull:
    """BOTH secondaries are also ranked against the combined two-sided null and stay
    consistent with the Phase-3 full-region per-direction null."""

    def test_both_secondary_uses_combined_null_more_conservative(self) -> None:
        both = flx.scan(clustertype="BOTH", **_sec_common())
        cold_solo = flx.scan(clustertype="COLD", **_sec_common())

        assert both["primary_type"] == "COLD"
        assert _areas(both["primary"]) == _SEC_PRIMARY_AREA_1B

        both_secs = both["primary"].get("secondaries") or []
        cold_secs = cold_solo.get("secondaries") or []
        assert both_secs and cold_secs

        both_sec, cold_sec = both_secs[0], cold_secs[0]
        # Same window/statistic (the window search is unchanged); only the reference
        # null differs (combined two-sided vs one-sided full-region).
        assert _areas(both_sec) == _SEC_SECONDARY_AREA_1B == _areas(cold_sec)
        assert both_sec["stats"] == pytest.approx(_SEC_SECONDARY_STATS, rel=1e-12)
        assert cold_sec["stats"] == pytest.approx(both_sec["stats"], rel=1e-12)
        # Two-sided secondary p >= one-sided (full-region) secondary p; strict here.
        assert both_sec["pval"] >= cold_sec["pval"]
        assert both_sec["pval"] > cold_sec["pval"]

    def test_secondary_pvalues_pinned(self) -> None:
        both = flx.scan(clustertype="BOTH", **_sec_common())
        cold_solo = flx.scan(clustertype="COLD", **_sec_common())
        both_sec = (both["primary"]["secondaries"] or [])[0]
        cold_sec = (cold_solo["secondaries"] or [])[0]
        assert both_sec["rank"] == _SEC_SECONDARY_RANK_TWO_SIDED
        assert both_sec["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_TWO_SIDED)
        assert cold_sec["rank"] == _SEC_SECONDARY_RANK_ONE_SIDED
        assert cold_sec["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_ONE_SIDED)

    def test_retained_cold_secondary_keeps_one_sided_pvalue(self) -> None:
        """The envelope re-ranks only the primary copy; the retained cold sub-result
        keeps its one-sided full-region secondary p-value."""
        both = flx.scan(clustertype="BOTH", **_sec_common())
        retained = (both["cold"]["secondaries"] or [])[0]
        assert retained["rank"] == _SEC_SECONDARY_RANK_ONE_SIDED
        assert retained["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_ONE_SIDED)


class TestMC3SecondaryNullOverride:
    """Resolved semantics (MC-3 audit, claim F): the BOTH envelope is a two-sided
    FULL-REGION test by construction (DEF-BOTH deletes the direction indicator), so
    the envelope primary's secondaries are ALWAYS ranked against the combined
    full-region two-sided null -- ``secondary_null="conditional"`` is intentionally
    overridden for the primary. The retained per-direction sub-results (``hot``/
    ``cold``) still honour ``secondary_null`` so the one-sided view is transparent.
    """

    def test_primary_secondary_ignores_conditional_uses_combined_full(self) -> None:
        both_full = flx.scan(clustertype="BOTH", **_sec_common(secondary_null="full"))
        both_cond = flx.scan(clustertype="BOTH", **_sec_common(secondary_null="conditional"))
        pf = (both_full["primary"]["secondaries"] or [])[0]
        pc = (both_cond["primary"]["secondaries"] or [])[0]
        # Same window, and the SAME (full-region combined) null in both modes.
        assert _areas(pf) == _areas(pc) == _SEC_SECONDARY_AREA_1B
        assert pf["rank"] == pc["rank"] == _SEC_SECONDARY_RANK_TWO_SIDED
        assert pf["pval"] == pc["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_TWO_SIDED)

    def test_retained_direction_secondary_honours_conditional(self) -> None:
        """The override is scoped to the envelope primary: the retained one-sided
        sub-result still reflects the user's secondary_null switch."""
        both_full = flx.scan(clustertype="BOTH", **_sec_common(secondary_null="full"))
        both_cond = flx.scan(clustertype="BOTH", **_sec_common(secondary_null="conditional"))
        rf = (both_full["cold"]["secondaries"] or [])[0]
        rc = (both_cond["cold"]["secondaries"] or [])[0]
        assert rf["rank"] == _SEC_SECONDARY_RANK_ONE_SIDED
        assert rf["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_ONE_SIDED)
        assert rc["rank"] == _SEC_SECONDARY_RANK_COND_ONE_SIDED
        assert rc["pval"] == pytest.approx(_SEC_SECONDARY_PVAL_COND_ONE_SIDED)
        # conditional (non-overlapping windows only) is anti-conservative vs full.
        assert rc["pval"] < rf["pval"]


class TestMC3SingleDirectionByteIdentical:
    """The sim_max emission must not perturb single-direction HOT/COLD results."""

    def test_solo_hot_cold_match_pins_and_have_no_sim_max(self) -> None:
        hot_solo = flx.scan(clustertype="HOT", **_main_common())
        cold_solo = flx.scan(clustertype="COLD", **_main_common())
        assert hot_solo["rank"] == _MAIN_HOT_RANK
        assert hot_solo["pval"] == pytest.approx(_MAIN_HOT_PVAL)
        assert hot_solo["stats"] == pytest.approx(_MAIN_HOT_STATS, rel=1e-12)
        assert cold_solo["rank"] == _MAIN_COLD_RANK
        assert cold_solo["pval"] == pytest.approx(_MAIN_COLD_PVAL)
        assert cold_solo["stats"] == pytest.approx(_MAIN_COLD_STATS, rel=1e-12)
        assert "sim_max" not in hot_solo
        assert "sim_max" not in cold_solo

    def test_both_subscans_equal_solo_one_sided(self) -> None:
        """The retained hot/cold sub-results equal the standalone one-sided scans
        (ORIGINAL: the BOTH sub-scans use the same settings as a solo scan)."""
        both = flx.scan(clustertype="BOTH", **_main_common())
        hot_solo = flx.scan(clustertype="HOT", **_main_common())
        cold_solo = flx.scan(clustertype="COLD", **_main_common())
        assert both["hot"]["pval"] == hot_solo["pval"]
        assert both["hot"]["rank"] == hot_solo["rank"]
        assert both["cold"]["pval"] == cold_solo["pval"]
        assert both["cold"]["rank"] == cold_solo["rank"]


class TestMC3EdgeCases:
    """simcount=0 and determinism."""

    def test_simcount_zero_primary_pval_is_none(self) -> None:
        res = flx.scan(clustertype="BOTH", **_main_common(simcount=0))
        assert res["type"] == "BOTH"
        assert res["primary"]["pval"] is None
        assert res["hot"]["pval"] is None
        assert res["cold"]["pval"] is None
        assert "sim_max" not in res["primary"]

    def test_deterministic_for_fixed_seed(self) -> None:
        first = flx.scan(clustertype="BOTH", **_sec_common())
        second = flx.scan(clustertype="BOTH", **_sec_common())
        assert first["primary"]["pval"] == second["primary"]["pval"]
        assert first["primary"]["rank"] == second["primary"]["rank"]
        f_sec = (first["primary"]["secondaries"] or [])[0]
        s_sec = (second["primary"]["secondaries"] or [])[0]
        assert f_sec["pval"] == s_sec["pval"]
        assert f_sec["rank"] == s_sec["rank"]


class TestMC3BothEarlyStopping:
    """BOTH may stop only on its aligned two-sided Monte Carlo evidence."""

    @staticmethod
    def _assert_combined_decision_resolved(result: dict[str, Any], simcount: int) -> None:
        primary = result["primary"]
        rank = int(primary["rank"])
        pval = float(primary["pval"])
        # Dwass arithmetic exposes the effective combined-null length:
        # p = rank / (B_used + 1), with rank = 1 + #{null >= observed}.
        n_dwass = int(round(rank / pval))
        assert pval == pytest.approx(rank / n_dwass)
        if n_dwass == simcount + 1:
            return

        # The Wilson decision interval used for early stopping must be based on
        # this combined two-sided Dwass estimate, not either one-sided subscan.
        z = 1.959963984540054
        z2 = z * z
        denom = 1.0 + z2 / n_dwass
        centre = (pval + z2 / (2.0 * n_dwass)) / denom
        half = z * np.sqrt(pval * (1.0 - pval) / n_dwass + z2 / (4.0 * n_dwass**2)) / denom
        assert not (centre - half <= 0.05 <= centre + half)

    @pytest.mark.parametrize("series", [False, True])
    def test_both_early_stop_uses_combined_two_sided_criterion(self, series: bool) -> None:
        """One-sided subscan decisions cannot terminate a two-sided BOTH test."""
        coords, adjacency = _chain(6)
        observed = np.array([15, 14, 12, 8, 6, 5])
        expected = np.full(6, 10.0)
        kwargs: dict[str, Any] = dict(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="CIRCULAR",
            stattype="ORIGINAL",
            clustertype="BOTH",
            clustersize=3,
            K=6,
            simcount=_SIMCOUNT,
            seed=2,
            mc_early_stop=True,
            mc_alpha=0.05,
            n_jobs_centers=1,
        )
        if series:
            kwargs["observed"] = observed[None, :]
            kwargs["expected"] = expected[None, :]
            result = flx.scan_series(**kwargs)["per_time"][0]
        else:
            result = flx.scan(**kwargs)

        self._assert_combined_decision_resolved(result, _SIMCOUNT)


class TestMC3Helpers:
    """Pure-Python unit tests for the combine/re-rank/strip helpers (no scan)."""

    def test_combined_two_sided_null_is_elementwise_max(self) -> None:
        hot = {"sim_max": [1.0, 5.0, 2.0]}
        cold = {"sim_max": [4.0, 3.0, 2.0]}
        combined = _combined_two_sided_null(hot, cold)
        assert combined is not None
        np.testing.assert_array_equal(combined, np.array([4.0, 5.0, 2.0]))

    def test_combined_two_sided_null_missing_returns_none(self) -> None:
        assert _combined_two_sided_null({"sim_max": [1.0]}, {}) is None
        assert _combined_two_sided_null({}, {"sim_max": [1.0]}) is None
        assert _combined_two_sided_null({"sim_max": []}, {"sim_max": []}) is None

    def test_combined_two_sided_null_rejects_unequal_replicate_counts(self) -> None:
        """A two-sided null cannot use independently stopped one-sided prefixes."""
        hot = {"sim_max": [1.0, 9.0, 2.0, 7.0]}
        cold = {"sim_max": [4.0, 3.0]}
        with pytest.raises(ValueError, match="must have equal lengths"):
            _combined_two_sided_null(hot, cold)

    def test_rerank_cluster_matches_dwass_rule(self) -> None:
        null_max = np.array([1.0, 2.0, 3.0, 4.0])
        cluster = {"stats": 3.0}
        _rerank_cluster_against_null(cluster, null_max)
        # rank = 1 + #{null >= 3.0} = 1 + 2 = 3 (>= counts ties toward the rank).
        assert cluster["rank"] == 3
        assert cluster["pval"] == pytest.approx(3 / (4 + 1))

    def test_strip_sim_max_removes_from_dict_and_secondaries(self) -> None:
        result = {
            "sim_max": [1.0, 2.0],
            "secondaries": [{"sim_max": [3.0]}, {"area": [1]}],
        }
        _strip_sim_max(result)
        assert "sim_max" not in result
        assert "sim_max" not in result["secondaries"][0]
