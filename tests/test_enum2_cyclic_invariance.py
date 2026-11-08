"""ENUM-2 / DEF-ENUM: the FLEXIBLE scan re-visits cyclic subsets but inference is
identical to a brute-force max over all connected subsets.

Phase 4 of
``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/REMEDIATION_PLAN_2026-06-15.md``.
The audit (ENUM-2;
``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/findings/03_enumeration.md``)
found the F-7b "visit-once" DFS in
``scan_flexible.hpp`` regenerates a connected subset once per *cyclic* insertion
order -- it de-duplicates acyclic (chain) orders but not cycles, so the
``scan_flexible.hpp`` claim "generated exactly once ... verified against
brute-force" was false for cyclic induced subgraphs. Phase 4 corrected the comment
to: completeness holds (each connected subset generated >= once), visit-once does
NOT, and the duplicates are inference-neutral because the scan keeps only ``max``
LLR under a strict total-order tie-break and re-evaluating an identical window is
idempotent.

These pins LOCK the inference invariant -- the load-bearing justification that the
comment's only error was the "exactly once" wording, not a correctness bug. They
run on a 2x3 rook grid, whose interior faces are 4-cycles, so the FLEXIBLE DFS
provably re-visits (audit/PART-B measured engine ``dfs_nodes_expanded`` == the
re-visiting reconstruction on this graph). With ``K = N`` and ``clustersize = N``
the FLEXIBLE candidate set is *exactly* the set of all connected node-subsets, so
the engine MLC must equal an independent brute-force argmax over that set.

The brute-force reference is recomputed IN-TEST (not hard-coded): an independent
connected-subset enumeration + a re-implementation of the Poisson/Binomial LLR
(mirroring ``llr_poisson.hpp`` / ``llr_binomial.hpp``) and the engine tie-break
(``tie_break_prefers``: smallest size, then lexicographically smallest area). So
if a future visit-once perf fix ever drops a connected subset, the brute-force max
diverges and this test fails. Numeric stats/p anchors are additionally pinned from
the 0.0.2 rebuild.

Style mirrors ``tests/test_mc3_both_combined_null.py`` and
``tests/test_mc6_secondary_full_null.py`` (small coords + CSR adjacency driving
``flexscanx.scan``).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pytest

import flexscanx as flx

_SEED = 20260617
_SIMCOUNT = 999


# --- a cyclic scan graph: 2x3 rook grid (interior faces are 4-cycles) ---------
#
#   0 - 1 - 2
#   |   |   |
#   3 - 4 - 5
#
# Every induced subgraph that wraps a face has a cycle, so the FLEXIBLE DFS
# regenerates those subsets along multiple insertion orders (ENUM-2). The
# inference must nevertheless match a clean brute-force max.

_GRID_EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
_N = 6


def _grid_adj() -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(_N)}
    for a, b in _GRID_EDGES:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _grid_csr() -> tuple[list[int], list[int]]:
    adj = _grid_adj()
    row_ptr, col_idx = [0], []
    for i in range(_N):
        col_idx.extend(sorted(adj[i]))
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


# Coordinates only matter for the (here unused) radius cap; the FLEXIBLE
# connectivity comes from the CSR adjacency under knn_method="graph".
_COORDS = np.array([[0, 1], [1, 1], [2, 1], [0, 0], [1, 0], [2, 0]], dtype=float)


# --- planted datasets ---------------------------------------------------------
# A HOT blob on the middle column {1, 4} (0-based) lifted above the rate-5 field;
# the COLD MLC is the complementary connected region {0, 2, 3, 4, 5}.

_POISSON_OBS = np.array([4, 14, 4, 4, 13, 4], dtype=int)
_POISSON_EXP = np.full(_N, 5.0)

_BINOM_OBS = np.array([5, 30, 5, 5, 28, 6], dtype=int)
_BINOM_POP = np.full(_N, 50.0)

# Captured from the 0.0.2 rebuild (audit_scratch/enum2_verify.py, PART C/D).
_ANCHORS: dict[tuple[str, str], dict[str, Any]] = {
    ("POISSON", "HOT"): dict(area0=[1, 4], stats=7.767383486899794, rank=7, pval=7 / 1000),
    ("POISSON", "COLD"): dict(
        area0=[0, 2, 3, 4, 5], stats=3.2387344256156343, rank=143, pval=143 / 1000
    ),
    ("BINOMIAL", "HOT"): dict(area0=[1, 4], stats=37.738762125406595, rank=1, pval=1 / 1000),
    ("BINOMIAL", "COLD"): dict(
        area0=[0, 2, 3, 4, 5], stats=15.602073327812747, rank=1, pval=1 / 1000
    ),
}


# --- independent brute-force reference ----------------------------------------


def _poisson_llr(nZ: float, mZ: float, nG: float, mG: float, hot: bool) -> float:
    """Mirror of ``llr_poisson_original_impl`` (with the LLR-5 zero-count guard)."""
    n_out, m_out = nG - nZ, mG - mZ
    if mZ <= 0.0 or m_out <= 0.0:
        return 0.0
    r_in, r_out = nZ / mZ, n_out / m_out
    if hot and not (r_in > r_out):
        return 0.0
    if not hot and not (r_in < r_out):
        return 0.0
    c1 = nZ * math.log(r_in) if nZ > 0.0 else 0.0
    c2 = n_out * math.log(r_out) if n_out > 0.0 else 0.0
    c3 = nG * math.log(nG / mG) if (nG > 0.0 and mG > 0.0) else 0.0
    return c1 + c2 - c3


def _binomial_llr(nZ: float, mZ: float, nG: float, mG: float, hot: bool) -> float:
    """Mirror of ``llr_binomial_original_impl``."""
    n_out, m_out = nG - nZ, mG - mZ
    if mZ <= 0.0 or m_out <= 0.0:
        return 0.0
    p_in, p_out = nZ / mZ, n_out / m_out
    if hot and not (p_in > p_out):
        return 0.0
    if not hot and not (p_in < p_out):
        return 0.0
    c1 = (nZ * math.log(p_in) + (mZ - nZ) * math.log(1.0 - p_in)) if 0.0 < p_in < 1.0 else 0.0
    c2 = (
        (n_out * math.log(p_out) + (m_out - n_out) * math.log(1.0 - p_out))
        if 0.0 < p_out < 1.0
        else 0.0
    )
    p_tot = nG / mG if mG > 0.0 else 0.0
    c0 = (nG * math.log(p_tot) + (mG - nG) * math.log(1.0 - p_tot)) if 0.0 < p_tot < 1.0 else 0.0
    return c1 + c2 - c0


def _connected(nodes: frozenset[int], adj: dict[int, set[int]]) -> bool:
    start = next(iter(nodes))
    seen, stack = {start}, [start]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w in nodes and w not in seen:
                seen.add(w)
                stack.append(w)
    return seen == set(nodes)


def _all_connected_subsets(adj: dict[int, set[int]], clustersize: int) -> set[frozenset[int]]:
    out: set[frozenset[int]] = set()
    nodes = list(adj)
    for k in range(1, clustersize + 1):
        for combo in combinations(nodes, k):
            s = frozenset(combo)
            if _connected(s, adj):
                out.add(s)
    return out


def _brute_force_mlc(
    observed: np.ndarray, denom: np.ndarray, model: str, hot: bool
) -> tuple[float, tuple[int, ...]]:
    """Independent argmax over ALL connected subsets, with the engine tie-break:
    among positive-stat windows at the global max, the smallest size then the
    lexicographically-smallest sorted area."""
    adj = _grid_adj()
    nG, mG = float(observed.sum()), float(denom.sum())
    llr = _poisson_llr if model == "POISSON" else _binomial_llr
    best_stat = 0.0
    best_area: tuple[int, ...] | None = None
    for s in _all_connected_subsets(adj, clustersize=_N):
        nZ = float(sum(observed[i] for i in s))
        mZ = float(sum(denom[i] for i in s))
        stat = llr(nZ, mZ, nG, mG, hot)
        if stat <= 0.0:
            continue
        area = tuple(sorted(s))
        if stat > best_stat:
            best_stat, best_area = stat, area
        elif (
            best_area is not None
            and stat == best_stat
            and (len(area), area)
            < (
                len(best_area),
                best_area,
            )
        ):
            best_area = area
    assert best_area is not None, "planted data must yield a positive-LLR window"
    return best_stat, best_area


# --- scan driver --------------------------------------------------------------


def _scan(model: str, clustertype: str, **override: Any) -> dict[str, Any]:
    if model == "POISSON":
        data: dict[str, Any] = dict(
            observed=_POISSON_OBS, expected=_POISSON_EXP, rantype="MULTINOMIAL"
        )
    else:
        data = dict(observed=_BINOM_OBS, population=_BINOM_POP, rantype="BINOMIAL")
    kwargs: dict[str, Any] = dict(
        coords=_COORDS,
        adjacency=_grid_csr(),
        model=model,
        scanmethod="FLEXIBLE",
        stattype="ORIGINAL",
        clustertype=clustertype,
        clustersize=_N,  # no size cap -> full connected-subset set
        K=_N,  # universe = all nodes -> connectivity is the only constraint
        knn_method="graph",
        knn_mode="K_FIXED",
        simcount=_SIMCOUNT,
        seed=_SEED,
        n_jobs_centers=1,
        **data,
    )
    kwargs.update(override)
    return flx.scan(**kwargs)  # type: ignore[arg-type]


def _area0(cluster: dict[str, Any]) -> tuple[int, ...]:
    """Result areas are 1-based; shift to 0-based to match the brute force."""
    return tuple(sorted(int(a) - 1 for a in (cluster.get("area") or [])))


_CELLS = [("POISSON", "HOT"), ("POISSON", "COLD"), ("BINOMIAL", "HOT"), ("BINOMIAL", "COLD")]


class TestEnum2MlcEqualsBruteForce:
    """The FLEXIBLE MLC equals an independent brute-force max over all connected
    subsets -- re-visits of cyclic subsets are idempotent on the max."""

    @pytest.mark.parametrize(("model", "clustertype"), _CELLS)
    def test_mlc_area_and_stats_match_brute_force(self, model: str, clustertype: str) -> None:
        observed = _POISSON_OBS if model == "POISSON" else _BINOM_OBS
        denom = _POISSON_EXP if model == "POISSON" else _BINOM_POP
        bf_stat, bf_area = _brute_force_mlc(observed, denom, model, hot=clustertype == "HOT")

        res = _scan(model, clustertype)
        assert _area0(res) == bf_area
        assert res["stats"] == pytest.approx(bf_stat, rel=1e-12)

        anchor = _ANCHORS[(model, clustertype)]
        assert list(bf_area) == anchor["area0"]
        assert bf_stat == pytest.approx(anchor["stats"], rel=1e-12)
        assert res["stats"] == pytest.approx(anchor["stats"], rel=1e-12)

    @pytest.mark.parametrize(("model", "clustertype"), _CELLS)
    def test_simcount_zero_mlc_still_matches_brute_force(
        self, model: str, clustertype: str
    ) -> None:
        observed = _POISSON_OBS if model == "POISSON" else _BINOM_OBS
        denom = _POISSON_EXP if model == "POISSON" else _BINOM_POP
        bf_stat, bf_area = _brute_force_mlc(observed, denom, model, hot=clustertype == "HOT")

        res = _scan(model, clustertype, simcount=0)
        assert _area0(res) == bf_area
        assert res["stats"] == pytest.approx(bf_stat, rel=1e-12)
        assert res["pval"] is None  # no MC requested


class TestEnum2PValueEqualsBruteForce:
    """The reported p-value is the Dwass rank of the (brute-force-identical)
    statistic against the per-replicate null max -- duplicates do not perturb it."""

    @pytest.mark.parametrize(("model", "clustertype"), _CELLS)
    def test_pvalue_and_rank_pinned(self, model: str, clustertype: str) -> None:
        res = _scan(model, clustertype)
        anchor = _ANCHORS[(model, clustertype)]
        assert res["rank"] == anchor["rank"]
        assert res["pval"] == pytest.approx(anchor["pval"])

    def test_public_pvalue_is_dwass_rank_of_brute_force_stat(self) -> None:
        """White-box (mirrors test_mc3): emit the per-replicate null, confirm the
        native statistic against brute force independently, then rank the exact
        native value against its null.  A separate float reference can differ by
        an ulp at a tied null value and must not redefine the native Dwass rank."""
        from flexscanx._scan_internal import _build_base_settings
        from flexscanx.api import core

        handle = flx.build_static(_COORDS, _grid_csr(), K=_N, knn_mode="K_FIXED")
        for clustertype in ("HOT", "COLD"):
            bf_stat, _ = _brute_force_mlc(
                _POISSON_OBS, _POISSON_EXP, "POISSON", hot=clustertype == "HOT"
            )
            base = _build_base_settings(
                N=_N,
                ralpha=0.2,
                clustersize_eff=_N,
                scanmethod="FLEXIBLE",
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
            settings = dict(base, clustertype=clustertype, _emit_sim_max=True)
            raw = core.ScanSlice(
                handle,
                np.asarray(_POISSON_OBS, np.int32),
                np.asarray(_POISSON_EXP, float),
                settings,
            )
            sim_max = np.asarray(raw["sim_max"], dtype=float)
            assert sim_max.size == _SIMCOUNT
            assert float(raw["stats"]) == pytest.approx(bf_stat, rel=1e-12)
            expected_rank = 1 + int(np.count_nonzero(sim_max >= float(raw["stats"])))
            assert int(raw["rank"]) == expected_rank
            assert _ANCHORS[("POISSON", clustertype)]["rank"] == expected_rank


class TestEnum2DuplicatesAreOrderInvariant:
    """Re-visits + thread reordering must not change the selected MLC or p-value:
    the strict total-order tie-break makes the max idempotent and deterministic."""

    @pytest.mark.parametrize(("model", "clustertype"), _CELLS)
    def test_identical_across_threads_and_schedulers(self, model: str, clustertype: str) -> None:
        ref = _scan(model, clustertype)
        variants = [
            _scan(model, clustertype, n_jobs_centers=2),
            _scan(model, clustertype, n_jobs_centers=4, scheduler_backend="WORK_STEAL"),
        ]
        for var in variants:
            assert _area0(var) == _area0(ref)
            assert var["stats"] == pytest.approx(ref["stats"], rel=1e-12)
            assert var["rank"] == ref["rank"]
            assert var["pval"] == pytest.approx(ref["pval"])
