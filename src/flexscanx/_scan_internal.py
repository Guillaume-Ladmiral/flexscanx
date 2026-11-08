"""Internal helpers for scan/scan_series/scan_from_geodataframe.

Not part of the public API. Moved here to keep api.py focused on
the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt

from ._core_loader import require_core as _require_core
from ._result_format import _build_both_result, _finalize_direction_result
from ._utils import (
    compute_default_n_jobs,
    mc_alias_env_enabled,
    validate_binomial_population,
    validate_cluster_radius,
    validate_expected_array,
    validate_float_parameter,
    validate_integer_parameter,
    warn_at_caller,
)


def _validate_scan_common(ralpha: float, clustersize: int, clusterradius: float) -> None:
    """Shared basic validation for scan parameters.

    - ralpha in [0,1]
    - clustersize >= 1
    - clusterradius >= 0 or +inf
    """
    ralpha_f = validate_float_parameter(ralpha, "ralpha")
    if not np.isfinite(ralpha_f) or not (0.0 <= ralpha_f <= 1.0):
        raise ValueError(f"ralpha must be in [0,1], got {ralpha}")
    validate_integer_parameter(clustersize, "clustersize", minimum=1)
    validate_cluster_radius(clusterradius)


def _validate_seed(seed: int | None) -> int | None:
    """Validate RNG seed and return normalized int."""
    if seed is None:
        return None
    return validate_integer_parameter(seed, "seed", minimum=0, maximum=2_147_483_647)


def _resolve_clustersize(clustersize: int, K: int | None, *, warn: bool = True) -> int:
    """Resolve effective clustersize against KNN horizon.

    If clustersize > K, cap to K and warn once for deterministic behavior.
    """
    c = int(clustersize)
    if K is None:
        return c
    k = int(K)
    if c <= k:
        return c
    if warn:
        warn_at_caller(f"clustersize ({c}) exceeds K ({k}); capping clustersize to K for this run.")
    return k


_EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET = 1_000_000


def _estimated_exact_flexible_expansions(K: int, clustersize: int, repetitions: int) -> int:
    """Conservatively estimate ordered DFS expansions, capped above the warning budget.

    The centre is fixed, leaving at most ``K - 1`` choices at depth one.  The
    live enumerator can revisit cyclic connected subsets through different
    insertion orders, so the falling-factorial expansion universe is a more
    useful guardrail than either K alone or a visit-once subset count.
    """
    horizon = max(1, int(K))
    depth = max(1, min(int(clustersize), horizon))
    repeats = max(1, int(repetitions))
    per_run_budget = _EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET // repeats
    if per_run_budget < 1:
        return _EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET + 1

    estimate = 1
    term = 1
    for selected_after_center in range(1, depth):
        term *= horizon - selected_after_center
        estimate += term
        if estimate > per_run_budget:
            return _EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET + 1
    return estimate * repeats


def warn_exact_flexible_large_k(
    scanmethod: str,
    K: int | None,
    clustersize: int,
    enable_frame_ub_prune: bool | None,
    enable_center_ub_gating: bool | None,
    *,
    stattype: str = "RESTRICTED",
    repetitions: int = 1,
) -> None:
    """Warn when requested exact FLEXIBLE work exceeds a conservative budget.

    ORIGINAL scans enumerate connected clusters without the RESTRICTED
    middle-p screen. Their practical expansion universe depends jointly on K,
    the effective maximum cluster depth, and the number of repeated scans.
    The pruning heuristics that previously hid this cost are off by default
    because they can return suboptimal clusters (2026-06 audit, CC-1/F-7b).
    """
    if str(scanmethod).upper() != "FLEXIBLE" or str(stattype).upper() != "ORIGINAL" or K is None:
        return
    if bool(enable_frame_ub_prune) or bool(enable_center_ub_gating):
        return  # a possibly suboptimal opt-in pruning heuristic is active
    effective_clustersize = min(int(K), int(clustersize))
    estimate = _estimated_exact_flexible_expansions(K, effective_clustersize, repetitions)
    if estimate > _EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET:
        repeat_text = "1 scan" if int(repetitions) == 1 else f"{int(repetitions)} repeated scans"
        warn_at_caller(
            f"FLEXIBLE ORIGINAL scan with K={int(K)} and effective "
            f"clustersize={effective_clustersize} has estimated exact-enumeration "
            f"work above {_EXACT_FLEXIBLE_EXPANSION_WARN_BUDGET:,} ordered "
            f"expansion units across {repeat_text}. The search is exponential in K "
            "and effective clustersize; reduce either control, or opt in to "
            "the faster but possibly suboptimal pruning heuristics "
            "(enable_frame_ub_prune=True, enable_center_ub_gating=True)."
        )


def warn_pruning_heuristics_enabled(
    scanmethod: str,
    enable_frame_ub_prune: bool | None,
    enable_center_ub_gating: bool | None,
) -> None:
    """Warn once per call that an enabled pruning heuristic can lose the true maximum.

    ``greedy_upper_bound`` is not a valid upper bound (2026-06 audit, re-measured
    2026-09-02: ``enable_frame_ub_prune`` lowered the reported maximum in 144 of
    7,056 scans).  Both options are documented as opt-in speed heuristics that may
    return suboptimal clusters; the warning makes that discoverable at the call
    site (D5).  Circular scans ignore both flags, so they stay silent.
    """
    if str(scanmethod).upper() != "FLEXIBLE":
        return
    enabled = [
        name
        for name, value in (
            ("enable_frame_ub_prune", enable_frame_ub_prune),
            ("enable_center_ub_gating", enable_center_ub_gating),
        )
        if bool(value)
    ]
    if not enabled:
        return
    warn_at_caller(
        f"{' and '.join(enabled)} enabled: the greedy bound behind these opt-in "
        "pruning heuristics is not a valid upper bound, so the scan may return a "
        "suboptimal cluster and its p-value (measured: 144 of 7,056 scans lost the "
        "true maximum with enable_frame_ub_prune). Leave both at their default "
        "(disabled) for an exact scan."
    )


# Mapping of valid rantype values per model.
# BINOMIAL deliberately accepts only BINOMIAL rantype — using MULTINOMIAL or POISSON
# null sampling for a BINOMIAL model is statistically invalid.
_VALID_RANTYPE = {
    "POISSON": {"MULTINOMIAL", "POISSON"},
    "BINOMIAL": {"BINOMIAL"},
}
_VALID_RATIO_SMOOTHING = frozenset({"NONE", "HALDANE", "EB"})
_VALID_SCANMETHOD = frozenset({"FLEXIBLE", "CIRCULAR"})
_VALID_STATTYPE = frozenset({"RESTRICTED", "ORIGINAL"})
_VALID_CLUSTERTYPE = frozenset({"HOT", "COLD", "BOTH"})
_VALID_SCHEDULER_BACKEND = frozenset({"ATOMIC", "WORK_STEAL"})
# Null distribution used to rank secondary clusters (MC-6 / DEF-STOP).
# "FULL" reuses the unconditional full-region primary null (paper-faithful
# default); "CONDITIONAL" recomputes it over non-overlapping windows (prior,
# anti-conservative behaviour, kept for back-compat).
_VALID_SECONDARY_NULL = frozenset({"FULL", "CONDITIONAL"})

# Fixed seed used for the CRN (Common Random Numbers) stream when crn=True and the
# caller does not provide an explicit seed.  Both scan() (seed=None → C++ uses 0xDEADBEEF)
# and scan_series() (crn=True, seed=None → Python uses _CRN_DEFAULT_SEED) would otherwise
# produce different null streams.  A named constant makes the intent explicit and keeps
# both paths auditable.
_CRN_DEFAULT_SEED: int = 0xC0FFEE


def _default_rantype_for_model(model: str) -> str:
    """Return the default Monte Carlo null for the normalized model."""
    return "BINOMIAL" if model == "BINOMIAL" else "MULTINOMIAL"


def _normalize_model_and_rantype(model: str, rantype: str | None) -> tuple[str, str]:
    """Validate model/rantype pair and return normalized uppercase values."""
    model_u = str(model).upper()

    if model_u not in _VALID_RANTYPE:
        raise ValueError(f"Unknown model='{model}'. Valid models: {list(_VALID_RANTYPE.keys())}")

    if rantype is None:
        return model_u, _default_rantype_for_model(model_u)

    rantype_u = str(rantype).upper()

    if rantype_u not in _VALID_RANTYPE[model_u]:
        raise ValueError(
            f"Invalid rantype='{rantype}' for model='{model}'. "
            f"Valid options: {_VALID_RANTYPE[model_u]}"
        )
    return model_u, rantype_u


def _normalize_scanmethod_and_stattype(scanmethod: str, stattype: str) -> tuple[str, str]:
    """Validate scanmethod/stattype and return normalized uppercase values."""
    scanmethod_u = str(scanmethod).upper()
    stattype_u = str(stattype).upper()

    if scanmethod_u not in _VALID_SCANMETHOD:
        raise ValueError(
            f"scanmethod must be one of {sorted(_VALID_SCANMETHOD)}, got {scanmethod!r}"
        )
    if stattype_u not in _VALID_STATTYPE:
        raise ValueError(f"stattype must be one of {sorted(_VALID_STATTYPE)}, got {stattype!r}")
    return scanmethod_u, stattype_u


def _normalize_clustertype(clustertype: str, *, allow_both: bool = True) -> str:
    """Validate and normalize a scan direction."""
    clustertype_u = str(clustertype).upper()
    valid = _VALID_CLUSTERTYPE if allow_both else frozenset({"HOT", "COLD"})
    if clustertype_u not in valid:
        raise ValueError(f"clustertype must be one of {sorted(valid)}, got {clustertype!r}")
    return clustertype_u


def _normalize_scheduler_backend(scheduler_backend: str) -> str:
    """Validate scheduler backend and return normalized uppercase value."""
    scheduler_u = str(scheduler_backend).upper()
    if scheduler_u not in _VALID_SCHEDULER_BACKEND:
        raise ValueError(
            "scheduler_backend must be one of {'ATOMIC', 'WORK_STEAL'}, "
            f"got {scheduler_backend!r}"
        )
    return scheduler_u


def _mc_alias_enabled(mc_use_alias: bool) -> bool:
    """Return whether alias sampling should be enabled."""
    return bool(mc_use_alias or mc_alias_env_enabled())


def _normalize_ratio_smoothing(ratio_smoothing: str) -> str:
    """Validate and normalize the reporting-only ratio smoothing mode."""
    ratio_smoothing_u = str(ratio_smoothing).upper()
    if ratio_smoothing_u not in _VALID_RATIO_SMOOTHING:
        raise ValueError(
            f"ratio_smoothing must be one of {{'NONE', 'HALDANE', 'EB'}}, got {ratio_smoothing!r}"
        )
    return ratio_smoothing_u


def _normalize_secondary_null(secondary_null: str) -> str:
    """Validate and normalize the secondary-cluster null-distribution mode."""
    secondary_null_u = str(secondary_null).upper()
    if secondary_null_u not in _VALID_SECONDARY_NULL:
        raise ValueError(
            f"secondary_null must be one of {{'full', 'conditional'}}, got {secondary_null!r}"
        )
    return secondary_null_u


def _normalize_names(names: Sequence[str] | None, N: int) -> list[str] | None:
    """Validate optional area-name labels against problem size."""
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise TypeError("names must be a sequence of labels, not a single string/bytes object")
    names_list = list(names)
    if len(names_list) != int(N):
        raise ValueError(f"names must have length N ({int(N)}), got {len(names_list)}")
    return names_list


def _normalize_secondary_limit(secondary: int | str | None, N: int) -> int | None:
    """Normalize secondary cluster request to an integer cap."""
    if secondary is None:
        return None
    if isinstance(secondary, str) and secondary.strip().lower() == "all":
        return int(N)
    return validate_integer_parameter(secondary, "secondary", minimum=0)


def _build_base_settings(
    *,
    N: int,
    ralpha: float,
    clustersize_eff: int,
    scanmethod: str,
    stattype: str,
    model: str,
    simcount: int,
    rantype_u: str,
    mc_early_stop: bool,
    mc_alpha: float,
    mc_progress: Any | None,
    mc_progress_every: int | None,
    secondary: int | str | None,
    secondary_p_stop: float | None,
    secondary_stats_stop: float | None,
    default_secondary_p_stop: float | None,
    secondary_null: str = "full",
    debug_prune: bool,
    enable_center_ub_gating: bool | None,
    cache_candidate_llr: bool | None,
    scheduler_backend: str | None,
    enable_frame_ub_prune: bool | None,
    mc_use_alias: bool,
    ratio_smoothing: str,
    seed: int | None,
    n_jobs_centers: int | None,
    extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized settings payload for core.ScanSlice."""
    simcount_i = validate_integer_parameter(simcount, "simcount", minimum=0)
    mc_alpha_f = validate_float_parameter(mc_alpha, "mc_alpha")
    if not np.isfinite(mc_alpha_f):
        raise ValueError(f"mc_alpha must be finite, got {mc_alpha!r}")
    if not 0.0 < mc_alpha_f < 1.0:
        raise ValueError(f"mc_alpha must be in (0,1), got {mc_alpha}")

    settings: dict[str, Any] = {
        "ralpha": float(ralpha),
        "clustersize": int(clustersize_eff),
        "scanmethod": scanmethod,
        "stattype": stattype,
        "model": model,
        "simcount": simcount_i,
        "rantype": rantype_u,
        "mc_early_stop": bool(mc_early_stop),
        "mc_alpha": mc_alpha_f,
    }
    if mc_progress is not None:
        settings["mc_progress"] = mc_progress
    if mc_progress_every is not None:
        settings["mc_progress_every"] = validate_integer_parameter(
            mc_progress_every, "mc_progress_every", minimum=1
        )

    secondary_limit = _normalize_secondary_limit(secondary, N)
    if secondary_limit is not None:
        settings["secondary"] = int(secondary_limit)
    if secondary_p_stop is not None:
        secondary_p_stop_f = validate_float_parameter(secondary_p_stop, "secondary_p_stop")
        if not np.isfinite(secondary_p_stop_f) or not 0.0 <= secondary_p_stop_f <= 1.0:
            raise ValueError(f"secondary_p_stop must be in [0,1], got {secondary_p_stop}")
        settings["secondary_p_stop"] = secondary_p_stop_f
    elif secondary_limit is not None and default_secondary_p_stop is not None:
        settings["secondary_p_stop"] = float(default_secondary_p_stop)
    if secondary_stats_stop is not None:
        secondary_stats_stop_f = validate_float_parameter(
            secondary_stats_stop, "secondary_stats_stop"
        )
        if not np.isfinite(secondary_stats_stop_f):
            raise ValueError(f"secondary_stats_stop must be finite, got {secondary_stats_stop!r}")
        if secondary_stats_stop_f < 0.0:
            raise ValueError(
                f"secondary_stats_stop must be non-negative, got {secondary_stats_stop}"
            )
        settings["secondary_stats_stop"] = secondary_stats_stop_f
    # Validate the secondary null mode unconditionally (so a bad value is rejected
    # even without secondary enumeration); only thread the non-default through to
    # the core, leaving the default settings payload byte-identical to before.
    secondary_null_u = _normalize_secondary_null(secondary_null)
    if secondary_null_u != "FULL":
        settings["secondary_null"] = secondary_null_u
    if debug_prune:
        settings["debug_prune"] = True
    if enable_center_ub_gating is not None:
        settings["enable_center_ub_gating"] = bool(enable_center_ub_gating)
    if cache_candidate_llr is not None:
        settings["cache_candidate_llr"] = bool(cache_candidate_llr)
    if scheduler_backend is not None:
        settings["scheduler_backend"] = _normalize_scheduler_backend(scheduler_backend)
    if enable_frame_ub_prune is not None:
        settings["enable_frame_ub_prune"] = bool(enable_frame_ub_prune)
    if _mc_alias_enabled(mc_use_alias):
        settings["mc_alias"] = True

    ratio_smoothing_u = _normalize_ratio_smoothing(ratio_smoothing)
    if ratio_smoothing_u != "NONE":
        settings["ratio_smoothing"] = ratio_smoothing_u
    if seed is not None:
        settings["seed"] = int(seed)

    if n_jobs_centers is not None:
        settings["n_jobs_centers"] = validate_integer_parameter(
            n_jobs_centers, "n_jobs_centers", minimum=1
        )
    else:
        settings["n_jobs_centers"] = compute_default_n_jobs(N)

    if extra_settings:
        settings.update(extra_settings)
    return settings


def _run_single_direction(
    handle: Any,
    arr_obs: np.ndarray,
    arr_exp: np.ndarray,
    base_settings: dict[str, Any],
    clustertype: str,
    provenance: dict[str, Any],
    names: Sequence[str] | None,
    core_module: Any | None = None,
) -> dict[str, Any]:
    """Run a single-direction scan (HOT or COLD).

    Returns result dict with scan_settings and type annotations.
    """
    core = core_module if core_module is not None else _require_core()

    settings = dict(base_settings)
    settings["clustertype"] = clustertype
    res = core.ScanSlice(handle, arr_obs, arr_exp, settings)

    return _finalize_direction_result(
        res, clustertype=clustertype, names=names, provenance=provenance
    )


def _require_projected_geodataframe(gdf: Any) -> None:
    """Reject GeoDataFrames without a projected CRS."""
    crs = getattr(gdf, "crs", None)
    if crs is None:
        raise ValueError(
            "scan_from_geodataframe requires a projected GeoDataFrame CRS with linear units; "
            "gdf.crs is missing. Set the CRS and reproject before scanning."
        )

    is_geographic = False
    try:
        is_geographic = bool(getattr(crs, "is_geographic", False))
    except (AttributeError, TypeError, ValueError):
        crs_text = str(crs).upper()
        is_geographic = ("4326" in crs_text) or ("GEOGCS" in crs_text) or ("DEGREE" in crs_text)
    if is_geographic:
        raise ValueError(
            "scan_from_geodataframe requires a projected CRS with linear units; "
            f"got geographic CRS {crs!s}. Reproject with GeoDataFrame.to_crs(...) first."
        )


def _extract_geodataframe_data(gdf: Any, id_col: str) -> tuple[np.ndarray, list[int]]:
    """Validate CRS and extract coords and area IDs from a GeoDataFrame."""
    _require_projected_geodataframe(gdf)
    for index, geometry in enumerate(gdf.geometry):
        if geometry is None or bool(getattr(geometry, "is_empty", False)):
            raise ValueError(f"geometry row {index} contains an empty geometry")
        if not bool(getattr(geometry, "is_valid", False)):
            raise ValueError(f"geometry row {index} contains an invalid geometry")
    area_ids = _normalize_geodataframe_area_ids(gdf, id_col)

    # Build coords from centroids
    try:
        cent = gdf.geometry.centroid
        coords = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    except Exception:
        # Fallback: use GeoSeries .x/.y accessors if present
        xs = [geom.centroid.x for geom in gdf.geometry]
        ys = [geom.centroid.y for geom in gdf.geometry]
        coords = np.column_stack([xs, ys])

    return coords, area_ids


def _normalize_geodataframe_area_ids(gdf: Any, id_col: str) -> list[int]:
    """Validate and normalize GeoDataFrame area IDs for public result remapping."""
    if id_col not in getattr(gdf, "columns", []):
        raise ValueError(
            f"id_col '{id_col}' not found in geodataframe columns: {list(gdf.columns)}"
        )

    ids: list[int] = []
    seen: set[int] = set()
    for idx, value in enumerate(gdf[id_col].tolist()):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"id_col '{id_col}' must contain positive integer area IDs; "
                f"row {idx} has value {value!r}"
            )
        area_id = int(value)
        if area_id <= 0:
            raise ValueError(
                f"id_col '{id_col}' must contain positive integer area IDs; "
                f"row {idx} has value {area_id}"
            )
        if area_id in seen:
            raise ValueError(
                f"id_col '{id_col}' must contain unique area IDs; duplicate value {area_id} "
                f"appears at row {idx}"
            )
        seen.add(area_id)
        ids.append(area_id)
    return ids


def _validate_denominator(
    model: str,
    expected: npt.ArrayLike | None,
    population: npt.ArrayLike | None,
    arr_obs: np.ndarray,
    N: int,
) -> np.ndarray:
    """Validate and return the denominator array (expected or population) for a scan.

    Encapsulates the BINOMIAL/POISSON parameter validation shared by scan() and
    scan_series():
    - BINOMIAL: requires 'population', rejects 'expected', validates positive integers,
      checks observed <= population
    - POISSON: requires strictly positive 'expected', rejects 'population'

    Parameters
    ----------
    model : str
        Normalized uppercase model name ("BINOMIAL" or "POISSON").
    expected : array-like or None
        Expected counts (POISSON only).
    population : array-like or None
        Population sizes (BINOMIAL only).
    arr_obs : np.ndarray
        Already-validated observed array (used for observed <= population check).
    N : int
        Expected length of the denominator array.

    Returns
    -------
    np.ndarray
        Validated denominator array as float64.
    """
    if model == "BINOMIAL":
        if expected is not None and population is not None:
            raise ValueError(
                "model='BINOMIAL': provide only 'population'. "
                "Parameter 'expected' is not accepted for BINOMIAL."
            )
        if expected is not None:
            raise ValueError(
                "model='BINOMIAL' requires 'population'. "
                "Parameter 'expected' is not accepted for BINOMIAL."
            )
        if population is None:
            raise ValueError(
                "model='BINOMIAL' requires the 'population' parameter. "
                "Provide an array of integer population sizes "
                "(number of trials) matching observed length."
            )
        arr_exp = validate_binomial_population(population, name="population")
        if (arr_obs > arr_exp).any():
            violations = np.where(arr_obs > arr_exp)[0].tolist()
            raise ValueError(
                f"model='BINOMIAL': observed counts cannot exceed population. "
                f"Violations at indices: {violations[:10]}{'...' if len(violations) > 10 else ''}"
            )
    else:  # POISSON
        if expected is None:
            raise ValueError(
                "model='POISSON' requires the 'expected' parameter. "
                "Provide an array of expected counts (positive floats) matching observed length."
            )
        if population is not None:
            raise ValueError(
                "model='POISSON': 'population' parameter is not accepted. Use 'expected' instead."
            )
        arr_exp = validate_expected_array(expected, name="expected", allow_zero=False)

    if len(arr_exp) != N:
        raise ValueError(f"expected/population length mismatch: got {len(arr_exp)}, expected N={N}")
    return arr_exp


# ---------------------------------------------------------------------------
# scan_series slice context + runner
# ---------------------------------------------------------------------------


@dataclass
class _SeriesSliceContext:
    """Explicit parameter container for scan_series per-time-slice execution.

    Breaks the implicit closure dependency of the former nested _run_one function,
    making all inputs inspectable and the dependency graph explicit.
    """

    obs: np.ndarray  # shape (T, N)
    exp: np.ndarray  # shape (T, N)
    handle: Any
    N: int
    names: list[str] | None
    # Normalized model/scan params
    model_u: str
    rantype_u: str
    scanmethod_u: str
    stattype_u: str
    clustertype: str
    # Scan params
    ralpha: float
    clustersize_eff: int
    simcount: int
    seed_i: int | None
    crn: bool
    warm_start: bool
    # MC params
    mc_early_stop: bool
    mc_alpha: float
    mc_progress: Any
    mc_progress_every: int | None
    mc_use_alias: bool
    # Secondary enumeration
    secondary: int | str | None
    secondary_p_stop: float | None
    secondary_stats_stop: float | None
    secondary_null: str
    # Core toggles
    debug_prune: bool
    enable_center_ub_gating: bool | None
    cache_candidate_llr: bool | None
    scheduler_backend: str | None
    enable_frame_ub_prune: bool | None
    ratio_smoothing: str
    # Parallelism
    n_jobs_centers: int | None
    max_workers: int
    both_parallel: bool
    # Core proxy (passed explicitly to avoid importing api.py's module-level `core`)
    core_module: Any = field(default=None)


def _run_one_slice(t: int, ctx: _SeriesSliceContext, prev_result: object = None) -> dict[str, Any]:
    """Execute one time-slice scan using an explicit _SeriesSliceContext.

    Replaces the former nested _run_one closure in scan_series.
    """
    core = ctx.core_module if ctx.core_module is not None else _require_core()

    n_jobs_eff = ctx.n_jobs_centers
    if ctx.max_workers > 1 or ctx.both_parallel:
        n_jobs_eff = 1

    base_settings = _build_base_settings(
        N=ctx.N,
        ralpha=ctx.ralpha,
        clustersize_eff=ctx.clustersize_eff,
        scanmethod=ctx.scanmethod_u,
        stattype=ctx.stattype_u,
        model=ctx.model_u,
        simcount=ctx.simcount,
        rantype_u=ctx.rantype_u,
        mc_early_stop=ctx.mc_early_stop,
        mc_alpha=ctx.mc_alpha,
        mc_progress=ctx.mc_progress,
        mc_progress_every=ctx.mc_progress_every,
        secondary=ctx.secondary,
        secondary_p_stop=ctx.secondary_p_stop,
        secondary_stats_stop=ctx.secondary_stats_stop,
        default_secondary_p_stop=1.0,
        secondary_null=ctx.secondary_null,
        debug_prune=bool(ctx.debug_prune),
        enable_center_ub_gating=ctx.enable_center_ub_gating,
        cache_candidate_llr=ctx.cache_candidate_llr,
        scheduler_backend=ctx.scheduler_backend,
        enable_frame_ub_prune=ctx.enable_frame_ub_prune,
        mc_use_alias=ctx.mc_use_alias,
        ratio_smoothing=ctx.ratio_smoothing,
        seed=int(ctx.seed_i if ctx.seed_i is not None else _CRN_DEFAULT_SEED),
        n_jobs_centers=n_jobs_eff,
        extra_settings={
            "crn": bool(ctx.crn),
            "time_index": int(t),
        },
    )

    # Warm-start: prioritize centers from previous slice
    if bool(ctx.warm_start) and int(t) > 0 and prev_result is not None:
        from .sequence._utils import _extract_clusters_for_timepoint

        prio_set: set[int] = set()
        for _, areas in _extract_clusters_for_timepoint(prev_result):
            prio_set.update(areas)
        prio = sorted(prio_set)
        if prio:
            # area IDs are already 1-based; C++ binding subtracts 1 → 0-based node index
            base_settings["priority_centers"] = [int(v) for v in prio]

    arr_obs = np.asarray(ctx.obs[t], dtype=np.int32)
    arr_exp = np.asarray(ctx.exp[t], dtype=float)
    ct = ctx.clustertype

    if ct != "BOTH":
        settings = dict(base_settings)
        settings["clustertype"] = ct
        res = core.ScanSlice(ctx.handle, arr_obs, arr_exp, settings)
        return _finalize_direction_result(res, clustertype=ct, names=ctx.names, provenance=None)

    # BOTH: run HOT and COLD with identical RNG config (shared seed/time_index).
    # The two-sided restricted statistic screens each tail at ralpha/2 (Bonferroni
    # split; Tango 2008 p.81) -- so each one-sided sub-scan runs at ralpha/2.
    # No-op for STATTYPE=ORIGINAL (no per-region gate). See scan() for rationale.
    settings_hot = dict(base_settings)
    settings_hot["clustertype"] = "HOT"
    settings_cold = dict(base_settings)
    settings_cold["clustertype"] = "COLD"
    if ctx.stattype_u == "RESTRICTED":
        _half_ralpha = float(base_settings["ralpha"]) / 2.0
        settings_hot["ralpha"] = _half_ralpha
        settings_cold["ralpha"] = _half_ralpha
    # MC-3 (Option A): emit each direction's full-region null-max array so
    # _build_both_result can rank the envelope against the combined two-sided null
    # max(hot_sim_max, cold_sim_max). Both sub-scans share base_settings (hence
    # seed + time_index), so replicate b aligns across directions under CRN.
    settings_hot["_emit_sim_max"] = True
    settings_cold["_emit_sim_max"] = True
    # Each native slice has only a one-sided Wilson stopping criterion.  BOTH
    # inference needs aligned HOT/COLD maxima and a combined two-sided decision,
    # so exhaust the requested budget rather than stop the directions separately.
    settings_hot["mc_early_stop"] = False
    settings_cold["mc_early_stop"] = False
    if ctx.both_parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as ex:
            future_hot = ex.submit(core.ScanSlice, ctx.handle, arr_obs, arr_exp, settings_hot)
            future_cold = ex.submit(core.ScanSlice, ctx.handle, arr_obs, arr_exp, settings_cold)
            raw_hot = future_hot.result()
            raw_cold = future_cold.result()
    else:
        raw_hot = core.ScanSlice(ctx.handle, arr_obs, arr_exp, settings_hot)
        raw_cold = core.ScanSlice(ctx.handle, arr_obs, arr_exp, settings_cold)

    res_hot = _finalize_direction_result(
        raw_hot, clustertype="HOT", names=ctx.names, provenance=None
    )
    res_cold = _finalize_direction_result(
        raw_cold, clustertype="COLD", names=ctx.names, provenance=None
    )
    return cast(dict[str, Any], _build_both_result(res_hot, res_cold))
