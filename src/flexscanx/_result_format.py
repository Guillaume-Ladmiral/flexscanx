from __future__ import annotations

import copy
import math
from typing import Any, Sequence, cast, overload

import numpy as np

from .types import BothResultDict, SeriesEntryBothResultDict


def _attach_names_to_result(result: dict[str, Any], names: Sequence[str] | None) -> None:
    """Attach area names to a scan result tree in-place."""
    if names is None or not isinstance(result, dict) or "area" not in result:
        return
    result["name"] = [names[int(i) - 1] for i in result["area"]]
    if "secondaries" in result and isinstance(result["secondaries"], list):
        for secondary in result["secondaries"]:
            if isinstance(secondary, dict) and "area" in secondary:
                secondary["name"] = [names[int(i) - 1] for i in secondary["area"]]


_RATIO_KEYS = ("Rate_Ratio", "Risk_Ratio")
_RATIO_CI_KEYS = ("Rate_Ratio_CI", "Risk_Ratio_CI")


def _sanitize_ratio_fields(cluster: dict[str, Any]) -> None:
    """Replace non-finite ratio values with ``None`` in one cluster body, in place.

    ``Rate_Ratio``/``Risk_Ratio`` are documented as ``float | None`` and their CIs
    as ``tuple | None``.  A legal but tiny denominator (a denormal ``expected``)
    makes the native ratio overflow to ``inf``, which is not strict-JSON encodable
    and contradicts the ``None`` = degenerate convention (FXA-181).  The
    provenance sub-tree already follows this rule; the cluster body now does too.
    """
    for key in _RATIO_KEYS:
        value = cluster.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            cluster[key] = None
    for key in _RATIO_CI_KEYS:
        value = cluster.get(key)
        if isinstance(value, (tuple, list)) and any(
            isinstance(v, float) and not math.isfinite(v) for v in value
        ):
            cluster[key] = None


def _finalize_direction_result(
    result: dict[str, Any],
    *,
    clustertype: str,
    names: Sequence[str] | None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a single-direction result dict in-place and return it."""
    if not isinstance(result, dict):
        return result

    result.setdefault("type", clustertype)
    _sanitize_ratio_fields(result)
    if "secondaries" in result and isinstance(result["secondaries"], list):
        for secondary in result["secondaries"]:
            if isinstance(secondary, dict):
                secondary.setdefault("type", clustertype)
                _sanitize_ratio_fields(secondary)

    if provenance is not None:
        prov = dict(provenance)
        prov["clustertype"] = clustertype
        actual = _result_simcount_actual(result)
        if actual is not None:
            prov["simcount_actual"] = actual
        # The primary's completed-replicate count lives under scan_settings, paired
        # with simcount_requested, so the top-level transport copy is removed.  Each
        # secondary KEEPS its own ``simcount_actual``: under
        # secondary_null="conditional" with mc_early_stop=True the native side stops
        # every secondary's null independently, so that per-cluster value is the
        # only truthful denominator behind that cluster's pval (FXA-166/FXA-167);
        # it is declared in types.ScanClusterDict and API_REFERENCE.md.
        result.pop("simcount_actual", None)
        result.setdefault("scan_settings", prov)

    _attach_names_to_result(result, names)
    return result


def _result_simcount_actual(result: dict[str, Any]) -> int | None:
    """Return a native completed-replicate count from a result or its provenance."""
    actual = result.get("simcount_actual")
    if isinstance(actual, int) and not isinstance(actual, bool) and actual >= 0:
        return actual
    settings = result.get("scan_settings")
    if isinstance(settings, dict):
        actual = settings.get("simcount_actual")
        if isinstance(actual, int) and not isinstance(actual, bool) and actual >= 0:
            return actual
    return None


def _pop_result_simcount_actual(result: dict[str, Any], *, requested_simcount: int) -> int:
    """Remove and return one series entry's completed MC count.

    A BOTH envelope carries three independently-owned result dictionaries.  All
    must report the same completed count because its combined two-sided null is
    indexed replicate-by-replicate.
    """
    if result.get("type") != "BOTH":
        actual = _result_simcount_actual(result)
        if actual is None and requested_simcount == 0:
            # A zero-budget run has an unambiguous completed count even for a
            # test double or older native result that omits the transport key.
            return 0
        if actual is None:
            raise RuntimeError("native scan result did not report simcount_actual")
        result.pop("simcount_actual", None)
        return actual

    counts: list[int] = []
    for key in ("primary", "hot", "cold"):
        branch = result.get(key)
        if not isinstance(branch, dict):
            raise RuntimeError(f"BOTH result is missing {key!r} completed-replicate provenance")
        actual = _result_simcount_actual(branch)
        if actual is None and requested_simcount == 0:
            actual = 0
        if actual is None:
            raise RuntimeError(f"BOTH {key} result did not report simcount_actual")
        branch.pop("simcount_actual", None)
        counts.append(actual)
    if len(set(counts)) != 1:
        raise RuntimeError(f"BOTH result has unequal completed MC counts: {counts}")
    return counts[0]


def _combined_two_sided_null(
    res_hot: dict[str, Any], res_cold: dict[str, Any]
) -> np.ndarray | None:
    """Combine the two per-direction full-region null-max arrays into the
    paper-correct two-sided null = per-replicate ``max(hot, cold)`` (MC-3).

    Returns ``None`` when either sub-scan ran without Monte Carlo (no ``sim_max``):
    there is then no null to combine and the BOTH primary keeps its one-sided
    p-value (``None``).
    """
    if not isinstance(res_hot, dict) or not isinstance(res_cold, dict):
        return None
    hot_sim = res_hot.get("sim_max")
    cold_sim = res_cold.get("sim_max")
    if hot_sim is None or cold_sim is None:
        return None
    hot_arr = np.asarray(hot_sim, dtype=float)
    cold_arr = np.asarray(cold_sim, dtype=float)
    # The two sub-scans draw the SAME null replicates (shared seed/time_index +
    # counter-based CRN), so replicate b aligns across directions.  Unequal
    # lengths prove an invalid/incomplete BOTH simulation rather than a prefix
    # that is safe to use for data-dependent early stopping.
    if hot_arr.size != cold_arr.size:
        raise ValueError(
            "BOTH Monte Carlo null arrays must have equal lengths; "
            f"got HOT={hot_arr.size}, COLD={cold_arr.size}"
        )
    if hot_arr.size == 0:
        return None
    return np.maximum(hot_arr, cold_arr)


def _rerank_cluster_against_null(cluster: dict[str, Any], null_max: np.ndarray) -> None:
    """Re-rank one cluster's ``stats`` against ``null_max`` in place.

    Uses the Dwass rule, identical to ``bindings_scan_result.cpp``:
    ``rank = 1 + #{null >= stats}``, ``pval = rank / (len + 1)``.
    """
    if not isinstance(cluster, dict) or "stats" not in cluster:
        return
    stats = float(cluster["stats"])
    rank = 1 + int(np.count_nonzero(null_max >= stats))
    cluster["rank"] = rank
    cluster["pval"] = rank / (null_max.size + 1)


def _strip_sim_max(result: dict[str, Any]) -> None:
    """Drop the internal ``sim_max`` transport array from a result dict and its
    secondaries (it is not part of the public result surface)."""
    if not isinstance(result, dict):
        return
    result.pop("sim_max", None)
    for secondary in result.get("secondaries") or []:
        if isinstance(secondary, dict):
            secondary.pop("sim_max", None)


@overload
def _build_both_result(
    res_hot: dict[str, Any],
    res_cold: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> BothResultDict: ...


@overload
def _build_both_result(
    res_hot: dict[str, Any],
    res_cold: dict[str, Any],
    *,
    provenance: None = None,
) -> SeriesEntryBothResultDict: ...


def _build_both_result(
    res_hot: dict[str, Any],
    res_cold: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
) -> BothResultDict | SeriesEntryBothResultDict:
    """Build a standardized BOTH-result envelope from HOT and COLD outputs.

    Per DEF-BOTH / DEF-MC-NULLMAX (Tango 2008, p.81) the two-sided statistic
    deletes the direction indicator, so the BOTH cluster is ranked against ONE
    combined two-sided null = per-replicate ``max(hot_max, cold_max)`` -- not the
    winning direction's one-sided null. flexscanx realises BOTH as an envelope of
    two one-sided sub-scans; when each carries its full-region null-max array
    (``sim_max``), the envelope primary -- and its secondaries (which Phase 3/MC-6
    already rank against their direction's full-region null) -- are re-ranked
    against the element-wise max of the two (MC-3, Option A). Because
    ``max(hot, cold) >=`` either one-sided max, the BOTH p-value becomes MORE
    conservative: this is the correct two-sided test (DEF-BOTH), not a regression.
    The per-direction one-sided p-values are retained under ``hot``/``cold``.
    """
    hot_stats = (
        float(res_hot.get("stats", float("-inf"))) if isinstance(res_hot, dict) else float("-inf")
    )
    cold_stats = (
        float(res_cold.get("stats", float("-inf"))) if isinstance(res_cold, dict) else float("-inf")
    )
    primary_src, primary_type = (res_hot, "HOT") if hot_stats >= cold_stats else (res_cold, "COLD")

    combined = _combined_two_sided_null(res_hot, res_cold)
    # sim_max is an internal transport array; strip it from the retained sub-results
    # now that the combined null is built. primary_src is one of them, so the
    # deepcopy below is already sim_max-free and never reaches the public result.
    _strip_sim_max(res_hot)
    _strip_sim_max(res_cold)

    if combined is not None:
        # The envelope primary is a distinct dict carrying the TWO-SIDED p-value;
        # the one-sided per-direction values must remain under hot/cold, so do not
        # mutate primary_src in place (it is aliased to res_hot/res_cold).
        primary: dict[str, Any] = copy.deepcopy(primary_src)
        _rerank_cluster_against_null(primary, combined)
        for secondary in primary.get("secondaries") or []:
            _rerank_cluster_against_null(secondary, combined)
    else:
        # Keep stable ownership even when Monte Carlo is disabled.  Public
        # ``primary``, ``hot`` and ``cold`` nodes must never alias solely because
        # simcount changed.
        primary = copy.deepcopy(primary_src)

    # The envelope primary is the inference-bearing result for a BOTH scan.  Keep
    # the directional one-sided values explicitly named so reporting layers do
    # not accidentally substitute either for the two-sided primary p-value.
    primary["pval_hot"] = res_hot.get("pval")
    primary["pval_cold"] = res_cold.get("pval")

    out: dict[str, Any] = {
        "type": "BOTH",
        "primary_type": primary_type,
        "primary": primary,
        "hot": res_hot,
        "cold": res_cold,
    }
    if provenance is not None:
        actual_hot = _result_simcount_actual(res_hot)
        actual_cold = _result_simcount_actual(res_cold)
        if actual_hot != actual_cold:
            raise RuntimeError(
                "BOTH result has unequal completed MC counts: "
                f"HOT={actual_hot}, COLD={actual_cold}"
            )
        both_provenance = dict(provenance)
        if actual_hot is not None:
            both_provenance["simcount_actual"] = actual_hot
        out["scan_settings"] = both_provenance
    return cast(BothResultDict | SeriesEntryBothResultDict, out)


def _remap_result_area_ids(result: dict[str, Any], id_map: dict[int, int]) -> dict[str, Any]:
    """Remap row-position area IDs in a result tree into caller-defined IDs."""

    def _map_one(value: Any) -> int:
        area_id = int(value)
        if area_id not in id_map:
            raise ValueError(f"scan result contains unexpected area id {area_id}")
        return id_map[area_id]

    seen: set[int] = set()

    def _visit(obj: Any) -> None:
        if isinstance(obj, dict):
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            if isinstance(obj.get("area"), list):
                obj["area"] = [_map_one(v) for v in obj["area"]]
            for key in ("from_area", "to_area"):
                if obj.get(key) is not None:
                    obj[key] = _map_one(obj[key])
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    _visit(value)
        elif isinstance(obj, list):
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            for value in obj:
                if isinstance(value, (dict, list)):
                    _visit(value)

    _visit(result)
    return result
