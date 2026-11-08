from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING:
    try:
        from ._core import ScanHandle  # type: ignore[attr-defined]

        _HandleT = ScanHandle | None
    except ImportError:
        _HandleT = Any  # type: ignore[assignment,misc]
else:
    _HandleT = Any

import numpy as np
import numpy.typing as npt

from ._contracts import validate_scan_result_contract, validate_series_result_contract
from ._core_loader import LazyCoreProxy
from ._result_format import (
    _build_both_result,
    _pop_result_simcount_actual,
    _remap_result_area_ids,
)
from ._scan_internal import (
    _CRN_DEFAULT_SEED,
    _build_base_settings,
    _extract_geodataframe_data,
    _normalize_clustertype,
    _normalize_model_and_rantype,
    _normalize_names,
    _normalize_scanmethod_and_stattype,
    _resolve_clustersize,
    _run_one_slice,
    _run_single_direction,
    _SeriesSliceContext,
    _validate_denominator,
    _validate_scan_common,
    _validate_seed,
    warn_exact_flexible_large_k,
    warn_pruning_heuristics_enabled,
)
from ._utils import (
    build_validated_static_handle,
    describe_static_handle,
    ensure_static_handle,
    validate_integer_parameter,
    validate_observed_array,
    warn_at_caller,
)
from .types import BothResultDict, ScanResultDict, SeriesResultBase

core = LazyCoreProxy()


def _json_safe_provenance_value(value: Any) -> Any:
    """Return provenance composed only of strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe_provenance_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_provenance_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_provenance_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe_time_label(value: Any) -> Any:
    """Return one ``scan_series`` time label as a strict-JSON scalar.

    ``times`` used to be stored verbatim, so a numpy or pandas time index broke
    ``json.dumps(result, allow_nan=False)`` even though the package guarantees a
    strict-JSON envelope (FXA-172).  Native JSON scalars pass through; numpy
    scalars become their Python equivalents; datetime-like labels (``datetime``,
    ``pandas.Timestamp``, ``numpy.datetime64``) become ISO 8601 strings so the
    label is preserved rather than reduced to an integer count of nanoseconds;
    anything else is stored as ``str(value)``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value))
    if isinstance(value, np.generic):
        return _json_safe_time_label(value.item())
    if isinstance(value, datetime.date):  # datetime.datetime and pandas.Timestamp included
        return value.isoformat()
    return str(value)


def _effective_static_provenance(
    *,
    static_desc: dict | None,
    reused_handle: bool,
    K: int | None,
    clusterradius: float | None,
    knn_method: str | None,
    knn_mode: str | None,
    knn_refill_force: bool | None,
) -> dict[str, Any]:
    """Return truthful effective static settings for result provenance."""
    desc = static_desc if isinstance(static_desc, dict) else {}
    return _json_safe_provenance_value(
        {
            "K": int(desc["K"]) if "K" in desc else (int(K) if K is not None else None),
            "clusterradius": (
                float(desc["radius"])
                if "radius" in desc
                else float("inf") if clusterradius is None else float(clusterradius)
            ),
            "knn_method": (
                None
                if reused_handle
                else str("graph" if knn_method is None else knn_method).strip().lower()
            ),
            "knn_mode": str(desc.get("knn_mode", knn_mode or "K_REFILL")).upper(),
            "knn_refill_force": (
                None
                if reused_handle
                else False if knn_refill_force is None else bool(knn_refill_force)
            ),
            "static": static_desc,
        }
    )


def _build_scan_provenance(
    *,
    base_settings: dict[str, Any],
    static_provenance: dict[str, Any],
    clustertype: str,
    ralpha: float,
    requested_seed: int | None,
    effective_seed: int,
    requested_simcount: int,
    requested_mc_early_stop: bool,
    requested_n_jobs_centers: int | None,
    requested_mc_use_alias: bool,
    both_parallel: bool,
    handle_reused: bool,
) -> dict[str, Any]:
    """Build complete, normalized, replay-oriented scan provenance."""
    mc_alias_effective = bool(base_settings.get("mc_alias", False))
    provenance = {
        "model": str(base_settings["model"]),
        "scanmethod": str(base_settings["scanmethod"]),
        "stattype": str(base_settings["stattype"]),
        "clustertype": str(clustertype),
        "clustersize": int(base_settings["clustersize"]),
        "ralpha": float(ralpha),
        "secondary": base_settings.get("secondary"),
        # ``seed`` remains the directly replayable value; the two explicit keys
        # disambiguate the public request from the stream that actually ran.
        "seed": int(effective_seed),
        "seed_requested": requested_seed,
        "seed_effective": int(effective_seed),
        "n_jobs_centers": int(base_settings["n_jobs_centers"]),
        "n_jobs_centers_requested": requested_n_jobs_centers,
        "simcount": int(requested_simcount),
        "simcount_requested": int(requested_simcount),
        "simcount_actual": None,
        "rantype": str(base_settings["rantype"]),
        "mc_early_stop": bool(base_settings["mc_early_stop"]),
        "mc_early_stop_requested": bool(requested_mc_early_stop),
        "mc_alpha": float(base_settings["mc_alpha"]),
        "mc_progress_every": base_settings.get("mc_progress_every"),
        "debug_prune": bool(base_settings.get("debug_prune", False)),
        "enable_center_ub_gating": bool(base_settings.get("enable_center_ub_gating", False)),
        "cache_candidate_llr": bool(base_settings.get("cache_candidate_llr", True)),
        "enable_frame_ub_prune": bool(base_settings.get("enable_frame_ub_prune", False)),
        "scheduler_backend": str(base_settings.get("scheduler_backend", "ATOMIC")),
        "both_parallel": bool(both_parallel),
        "mc_use_alias": mc_alias_effective,
        "mc_use_alias_requested": bool(requested_mc_use_alias),
        "mc_alias_env_override": mc_alias_effective and not bool(requested_mc_use_alias),
        "ratio_smoothing": str(base_settings.get("ratio_smoothing", "NONE")),
        "secondary_p_stop": base_settings.get("secondary_p_stop"),
        "secondary_stats_stop": base_settings.get("secondary_stats_stop"),
        "secondary_null": str(base_settings.get("secondary_null", "FULL")),
        "handle_reused": bool(handle_reused),
        **static_provenance,
    }
    return _json_safe_provenance_value(provenance)


def scan(
    *,
    coords: np.ndarray | None = None,
    adjacency: tuple[Sequence[int], Sequence[int]] | None = None,
    observed: npt.ArrayLike,
    expected: npt.ArrayLike | None = None,
    population: npt.ArrayLike | None = None,
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "HOT",
    clustersize: int | None = None,
    clusterradius: float | None = None,
    ralpha: float = 0.2,
    K: int | None = None,
    names: Sequence[str] | None = None,
    # Precompute reuse
    handle: _HandleT = None,
    # Additional performance knobs
    secondary: int | str | None = None,
    seed: int | None = None,
    n_jobs_centers: int | None = None,
    simcount: int = 999,
    rantype: str | None = None,
    mc_early_stop: bool = False,
    mc_alpha: float = 0.05,
    mc_progress: Any | None = None,
    mc_progress_every: int | None = None,
    debug_prune: bool = False,
    # Core toggles (optional; default to core defaults if None)
    enable_center_ub_gating: bool | None = None,
    cache_candidate_llr: bool | None = None,
    enable_frame_ub_prune: bool | None = None,
    scheduler_backend: str | None = None,
    # KNN/static build controls
    knn_method: str | None = None,
    knn_mode: str | None = None,
    knn_refill_force: bool | None = None,
    both_parallel: bool = False,
    # Monte Carlo sampling method
    mc_use_alias: bool = False,
    # Ratio smoothing for reported rate estimates (no effect on scan statistic or p-values)
    ratio_smoothing: str = "NONE",
    # Early stop for secondary enumeration
    secondary_p_stop: float | None = None,
    secondary_stats_stop: float | None = None,
    # Null distribution for ranking secondary clusters (MC-6 / DEF-STOP)
    secondary_null: str = "full",
) -> ScanResultDict | BothResultDict:
    """Single-slice spatial scan using the compiled C++ core.

    Supports POISSON and BINOMIAL models; FLEXIBLE and CIRCULAR scan methods;
    RESTRICTED and ORIGINAL statistics; HOT, COLD, and BOTH cluster directions.

    Parameters
    ----------
    coords : ndarray of shape (N, 2), optional
        Projected (x, y) coordinates.  Required when *handle* is not provided.
    adjacency : tuple of (row_ptr, col_idx), optional
        CSR adjacency arrays for the scan graph.  Required when *handle* is not
        provided and *knn_method* is ``'graph'``.
    observed : sequence of int
        Length-N integer count vector.
    expected : sequence of float, optional
        Length-N float expected-count vector (POISSON model).
        Required for POISSON; not accepted for BINOMIAL.
    population : sequence of float, optional
        Length-N integer population vector (BINOMIAL model).
        Required for BINOMIAL; not accepted for POISSON.
    model : str, default "POISSON"
        Statistical model: ``'POISSON'`` or ``'BINOMIAL'``.
    scanmethod : str, default "FLEXIBLE"
        Scan geometry: ``'FLEXIBLE'`` or ``'CIRCULAR'``.
    stattype : str, default "RESTRICTED"
        LLR form: ``'RESTRICTED'`` or ``'ORIGINAL'``.
    clustertype : str, default "HOT"
        Direction: ``'HOT'``, ``'COLD'``, or ``'BOTH'``.
    clustersize : int, optional
        Maximum cluster size S (default 15).  Capped to K when *clustersize* > K;
        the cap warning is emitted only when *clustersize* was passed explicitly
        (the default is capped silently).
    clusterradius : float, optional
        Maximum cluster radius in coordinate units. Omission means unlimited
        for a newly built handle and defers to an existing handle's metadata.
    ralpha : float, default 0.2
        Middle-p threshold for RESTRICTED per-region screening (centre and
        nodes).  Areas whose direction-appropriate marginal middle-p exceeds this
        value are screened out.  For ``clustertype="BOTH"`` the two-sided
        restricted statistic screens each tail at ``ralpha/2`` (Bonferroni split;
        Tango 2008, p.81): the two one-sided sub-scans run at ``ralpha/2`` while
        the merged result still reports the configured ``ralpha``.
    K : int, optional
        KNN horizon — number of nearest neighbours per centre to consider.
        Defaults to ``min(15, N)``.  Must be in ``[1, N]``.
    names : sequence of str, optional
        Length-N area labels attached to cluster ``'name'`` keys in results.
    handle : ScanHandle, optional
        Precomputed static handle from ``build_static()``.  When provided, the
        handle's coordinates and graph are reused and *coords*, *adjacency*,
        *K*, *clusterradius*, *knn_mode* and *knn_refill_force* must either be
        omitted or agree with the handle; a conflicting value raises
        ``ValueError`` rather than being silently discarded.
    secondary : int or "ALL", optional
        Enumerate non-overlapping secondary clusters.  ``None`` (default)
        disables secondary enumeration; ``"ALL"`` returns all; an integer caps
        the count.
    seed : int, optional
        RNG seed in ``[0, 2147483647]`` for reproducible Monte Carlo.
    n_jobs_centers : int, optional
        Worker threads for centre scanning.  Defaults to a problem-size-scaled
        value (see ``compute_default_n_jobs``).
    simcount : int, default 999
        Number of Monte Carlo replications.  Pass 0 to disable MC (no p-value).
    rantype : str, optional
        Null distribution for MC: ``'MULTINOMIAL'`` or ``'POISSON'`` (POISSON
        model); ``'BINOMIAL'`` (BINOMIAL model).  Default resolves per model.
    mc_early_stop : bool, default False
        Stop MC early when the Wilson 95 % CI for the p-value no longer straddles
        *mc_alpha*.  For ``clustertype="BOTH"`` the full requested simulation
        budget is used so HOT/COLD replicate maxima remain aligned and the
        two-sided stopping decision cannot be made from a one-sided sub-scan.
    mc_alpha : float, default 0.05
        Significance threshold used by *mc_early_stop*.
    mc_progress : callable, optional
        Callback invoked during MC.  Receives a dict with keys ``'done'``,
        ``'simcount'``, ``'phat'``, and ``'ci'``.  Providing a callback
        disables GIL-release parallelism and may slow large scans.
    mc_progress_every : int, optional
        Invoke *mc_progress* every this many simulations.  Defaults to the
        internal batch size (~200) when the callback is set.
    debug_prune : bool, default False
        When True the C++ core writes a per-scan diagnostic line to stderr
        with pruning / exploration counters.
    enable_center_ub_gating : bool, optional
        Override the centre-level upper-bound gate (flexible scans).
        Default ``None`` uses the compiled core default (**disabled** since the
        2026-06-11 audit: the greedy bound is heuristic and enabling it may
        return suboptimal clusters).
    cache_candidate_llr : bool, optional
        Override per-candidate LLR caching (flexible scans).
        Default ``None`` uses the compiled core default (enabled).
    enable_frame_ub_prune : bool, optional
        Override per-frame upper-bound pruning (flexible scans).
        Default ``None`` uses the compiled core default (**disabled** since the
        2026-06-11 audit: the greedy bound is not a valid upper bound, so
        enabling it may return suboptimal clusters; opt-in speed heuristic
        only).  Has no effect on circular scans.
    scheduler_backend : str, optional
        Centre-iteration scheduler: ``'ATOMIC'`` (default) or ``'WORK_STEAL'``.
    knn_method : str, optional
        How the KNN adjacency is built: ``'graph'`` uses *adjacency*;
        ``'geometry'`` builds a symmetric KNN graph from *coords* via KD-tree
        (ignores *adjacency* even if supplied). Defaults to ``'graph'`` for a
        newly built handle.
    knn_mode : str, optional
        ``'K_REFILL'`` precomputes an extended sorted list and refills
        when near neighbours are excluded; ``'K_FIXED'`` uses deterministic
        top-K (distance², then index tie-break). Defaults to ``'K_REFILL'`` for
        a newly built handle.
    knn_refill_force : bool, optional
        Force-enable K_REFILL on large N (overrides the built-in guardrail).
        Defaults to False for a newly built handle.
    both_parallel : bool, default False
        When *clustertype* is ``'BOTH'``, run HOT and COLD scans in parallel
        Python threads.  Forces *n_jobs_centers* = 1 per direction to avoid
        thread over-subscription.
    mc_use_alias : bool, default False
        Use the Vose alias method for MULTINOMIAL MC sampling (experimental).
        Can accelerate large *simcount* (> 9999) at the cost of O(N) extra memory
        per simulation.
    ratio_smoothing : str, default "NONE"
        Smoothing for reported rate/risk-ratio estimates (output only; no effect
        on scan statistic or p-values): ``'NONE'``, ``'HALDANE'`` (+0.5/+0.5),
        or ``'EB'`` (empirical-Bayes shrinkage).
    secondary_p_stop : float, optional
        Stop secondary enumeration when the next cluster's MC p-value exceeds
        this threshold.  Effective default is 1.0 (no practical pruning).
    secondary_stats_stop : float, optional
        Stop secondary enumeration when the next cluster's LLR statistic falls
        below this threshold.
    secondary_null : str, default "full"
        Monte-Carlo null distribution each *secondary* cluster's statistic is
        ranked against (no effect on the MLC/primary or when no secondaries are
        requested):

        * ``'full'`` (default, paper-faithful) -- rank every secondary against
          the same unconditional full-region null already computed for the
          primary, i.e. "as if it were the primary cluster" (DEF-STOP; Otani &
          Takahashi 2021 §2.4).  The secondary *window search* still excludes
          already-detected areas (non-overlap); only the reference null is the
          full-region one.
        * ``'conditional'`` -- recompute the null over only the windows that
          avoid detected areas.  These maxima are <= the full-region maxima, so
          secondary p-values are anti-conservative relative to the paper.  This
          was flexscanx's behaviour through 0.0.2; pass it to reproduce the old
          secondary p-values.

    Returns
    -------
    ScanResultDict or BothResultDict
        When *clustertype* is ``'HOT'`` or ``'COLD'``: a ``ScanResultDict``
        with keys ``area``, ``n_case``, ``expected``, ``stats``, ``rank``,
        ``type``, and optionally ``pval``, ``Rate_Ratio``/``Risk_Ratio``,
        ``max_dist``, ``secondaries``, ``scan_settings``.
        When *clustertype* is ``'BOTH'``: a ``BothResultDict`` with sub-keys
        ``hot``, ``cold``, ``primary``, ``primary_type``, ``scan_settings``.
    """
    # Resolve clustersize sentinel: None means "use default of 15"
    clustersize_was_set = clustersize is not None
    clustersize = (
        validate_integer_parameter(clustersize, "clustersize", minimum=1)
        if clustersize is not None
        else 15
    )
    # Basic validation
    _validate_scan_common(
        ralpha, clustersize, float("inf") if clusterradius is None else clusterradius
    )
    seed_requested = _validate_seed(seed)
    # The former native-only default (0xDEADBEEF) could not be supplied through
    # the legal Python seed domain.  Always pass one documented legal default so
    # a result produced from seed=None can be replayed explicitly.
    seed_i = _CRN_DEFAULT_SEED if seed_requested is None else seed_requested

    # Validate and normalize model/rantype
    _model_u, _rantype_u = _normalize_model_and_rantype(model, rantype)
    _scanmethod_u, _stattype_u = _normalize_scanmethod_and_stattype(scanmethod, stattype)
    ct = _normalize_clustertype(clustertype)

    # Validate observed before using its length to validate/reuse a static handle.
    arr_obs_np = validate_observed_array(observed, name="observed")
    if arr_obs_np.ndim != 1:
        raise ValueError(f"observed must be a 1D array, got shape {arr_obs_np.shape}")

    # Build or reuse handle using shared utility
    N_from_obs = int(len(arr_obs_np)) if handle is not None else None
    reused_handle = handle is not None
    handle, N, eff_K, static_desc = ensure_static_handle(
        handle,
        coords,
        adjacency,
        K,
        N_from_obs,
        clusterradius,
        knn_method,
        knn_mode,
        knn_refill_force,
    )
    if eff_K is not None:
        K = eff_K
    names = _normalize_names(names, N)
    warn_on_cap = clustersize_was_set
    clustersize_eff = _resolve_clustersize(clustersize, K, warn=warn_on_cap)

    base_settings = _build_base_settings(
        N=N,
        ralpha=ralpha,
        clustersize_eff=clustersize_eff,
        scanmethod=_scanmethod_u,
        stattype=_stattype_u,
        model=_model_u,
        simcount=simcount,
        rantype_u=_rantype_u,
        mc_early_stop=mc_early_stop,
        mc_alpha=mc_alpha,
        mc_progress=mc_progress,
        mc_progress_every=mc_progress_every,
        secondary=secondary,
        secondary_p_stop=secondary_p_stop,
        secondary_stats_stop=secondary_stats_stop,
        default_secondary_p_stop=1.0,
        secondary_null=secondary_null,
        debug_prune=debug_prune,
        enable_center_ub_gating=enable_center_ub_gating,
        cache_candidate_llr=cache_candidate_llr,
        scheduler_backend=scheduler_backend,
        enable_frame_ub_prune=enable_frame_ub_prune,
        mc_use_alias=mc_use_alias,
        ratio_smoothing=ratio_smoothing,
        seed=seed_i,
        n_jobs_centers=n_jobs_centers,
    )
    warn_pruning_heuristics_enabled(_scanmethod_u, enable_frame_ub_prune, enable_center_ub_gating)
    warn_exact_flexible_large_k(
        _scanmethod_u,
        K,
        clustersize_eff,
        enable_frame_ub_prune,
        enable_center_ub_gating,
        stattype=_stattype_u,
        repetitions=(int(base_settings["simcount"]) + 1) * (2 if ct == "BOTH" else 1),
    )

    if len(arr_obs_np) != N:
        raise ValueError(f"observed length mismatch: got {len(arr_obs_np)}, expected N={N}")
    arr_obs = arr_obs_np
    # Validate and choose denominator based on model
    arr_exp = _validate_denominator(_model_u, expected, population, arr_obs, N)

    static_provenance = _effective_static_provenance(
        static_desc=static_desc,
        reused_handle=reused_handle,
        K=K,
        clusterradius=clusterradius,
        knn_method=knn_method,
        knn_mode=knn_mode,
        knn_refill_force=knn_refill_force,
    )
    provenance = _build_scan_provenance(
        base_settings=base_settings,
        static_provenance=static_provenance,
        clustertype=ct,
        ralpha=float(ralpha),
        requested_seed=seed_requested,
        effective_seed=seed_i,
        requested_simcount=simcount,
        requested_mc_early_stop=mc_early_stop,
        requested_n_jobs_centers=n_jobs_centers,
        requested_mc_use_alias=mc_use_alias,
        both_parallel=both_parallel,
        handle_reused=reused_handle,
    )

    if ct != "BOTH":
        # Single-direction scan
        single = _run_single_direction(
            handle, arr_obs, arr_exp, base_settings, ct, provenance, names, core_module=core
        )
        return validate_scan_result_contract(single, require_scan_settings=True)  # type: ignore[return-value]

    # BOTH: run HOT and COLD separately and merge (two-pass; shared precompute).
    #
    # Per Tango (2008, Jpn. J. Biometrics 29(2):75-95, p.81), the two-sided
    # restricted statistic deletes the direction indicator and screens each region
    # at alpha1/2 -- a Bonferroni split of the per-region significance level across
    # the two tails. flexscanx realises BOTH as an envelope of two one-sided
    # RESTRICTED sub-scans, so each sub-scan must screen at ralpha/2. (No-op for
    # STATTYPE=ORIGINAL, which has no per-region middle-p gate: left at the full
    # ralpha so the sub-scan metadata is not misleading.) The top-level BOTH
    # envelope continues to report the user-configured two-sided level (ralpha);
    # the per-tail screen actually applied (ralpha/2) is recorded on each sub-scan.
    both_settings = dict(base_settings)
    if _stattype_u == "RESTRICTED":
        both_settings["ralpha"] = float(base_settings["ralpha"]) / 2.0
    # MC-3 (Option A): the BOTH envelope is ranked against ONE combined two-sided
    # null = per-replicate max(hot_sim_max, cold_sim_max) (DEF-BOTH/DEF-MC-NULLMAX).
    # Ask each one-sided sub-scan to return its full-region null-max array so
    # _build_both_result can combine them; the two sub-scans share seed/time_index
    # (counter-based CRN), so replicate b aligns across directions.
    both_settings["_emit_sim_max"] = True
    # The native early-stop criterion is one-sided because each ScanSlice sees
    # only HOT or COLD.  Stopping those slices independently and truncating to a
    # common prefix is not a valid stopping rule for the combined two-sided
    # maximum.  Until the native core owns an aligned BOTH controller, exhaust
    # the requested budget for correctness.
    both_settings["mc_early_stop"] = False
    if both_parallel:
        both_settings["n_jobs_centers"] = 1

    provenance = _build_scan_provenance(
        base_settings=both_settings,
        static_provenance=static_provenance,
        clustertype="BOTH",
        ralpha=float(ralpha),
        requested_seed=seed_requested,
        effective_seed=seed_i,
        requested_simcount=simcount,
        requested_mc_early_stop=mc_early_stop,
        requested_n_jobs_centers=n_jobs_centers,
        requested_mc_use_alias=mc_use_alias,
        both_parallel=both_parallel,
        handle_reused=reused_handle,
    )
    both_provenance = dict(provenance)
    if _stattype_u == "RESTRICTED":
        both_provenance["ralpha"] = float(ralpha) / 2.0

    if both_parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_hot = ex.submit(
                _run_single_direction,
                handle,
                arr_obs,
                arr_exp,
                both_settings,
                "HOT",
                both_provenance,
                names,
                core,
            )
            f_cold = ex.submit(
                _run_single_direction,
                handle,
                arr_obs,
                arr_exp,
                both_settings,
                "COLD",
                both_provenance,
                names,
                core,
            )
            res_hot = f_hot.result()
            res_cold = f_cold.result()
    else:
        res_hot = _run_single_direction(
            handle, arr_obs, arr_exp, both_settings, "HOT", both_provenance, names, core_module=core
        )
        res_cold = _run_single_direction(
            handle,
            arr_obs,
            arr_exp,
            both_settings,
            "COLD",
            both_provenance,
            names,
            core_module=core,
        )

    both = _build_both_result(res_hot, res_cold, provenance=provenance)
    return validate_scan_result_contract(both, require_scan_settings=True)  # type: ignore[return-value]


def scan_series(
    *,
    observed: Sequence[Sequence[int]] | np.ndarray,
    expected: Sequence[Sequence[float]] | np.ndarray | None = None,
    population: Sequence[Sequence[float]] | np.ndarray | None = None,
    coords: np.ndarray | None = None,
    adjacency: tuple[Sequence[int], Sequence[int]] | None = None,
    names: Sequence[str] | None = None,
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "HOT",
    clustersize: int | None = None,
    clusterradius: float | None = None,
    ralpha: float = 0.2,
    K: int | None = None,
    simcount: int = 999,
    rantype: str | None = None,
    secondary: int | str | None = None,
    crn: bool = True,
    seed: int | None = None,
    n_jobs_centers: int | None = None,
    handle: Any | None = None,
    mc_early_stop: bool = False,
    mc_alpha: float = 0.05,
    mc_progress: Any | None = None,
    mc_progress_every: int | None = None,
    # Core toggles (optional)
    enable_center_ub_gating: bool | None = None,
    cache_candidate_llr: bool | None = None,
    scheduler_backend: str | None = None,
    # KNN/static
    knn_method: str | None = None,
    knn_mode: str | None = None,
    knn_refill_force: bool | None = None,
    both_parallel: bool = False,
    enable_frame_ub_prune: bool | None = None,
    n_jobs_time: int | None = None,
    warm_start: bool = False,
    times: Sequence | None = None,
    debug_prune: bool = False,
    mc_use_alias: bool = False,
    # Ratio smoothing for reported rate estimates (no effect on scan statistic or p-values)
    ratio_smoothing: str = "NONE",
    secondary_p_stop: float | None = None,
    secondary_stats_stop: float | None = None,
    secondary_null: str = "full",
) -> SeriesResultBase:
    """Time-series spatial scan with static precompute reuse across slices.

    Builds or reuses one ``BuildStatic`` handle and calls the single-slice core
    for each time point.  Supports POISSON and BINOMIAL models.

    Parameters
    ----------
    observed : ndarray of shape (T, N)
        Integer count matrix; rows are time points, columns are areas.
    expected : ndarray of shape (T, N), optional
        Expected-count matrix (POISSON).  Required for POISSON; not accepted for
        BINOMIAL.
    population : ndarray of shape (T, N), optional
        Population-size matrix (BINOMIAL).  Required for BINOMIAL; not accepted
        for POISSON.
    coords : ndarray of shape (N, 2), optional
        Projected (x, y) coordinates.  Required when *handle* is not provided.
    adjacency : tuple of (row_ptr, col_idx), optional
        CSR adjacency for the scan graph.
    names : sequence of str, optional
        Length-N area labels.
    model, scanmethod, stattype, clustertype : str
        Forwarded to the core scanner for every time slice (see ``scan()``).
    clustersize : int, optional
        Maximum cluster size S (default 15).
    clusterradius : float, optional
        Maximum cluster radius. Omission means unlimited for a newly built
        handle and defers to an existing handle's metadata.
    ralpha : float, default 0.2
        Middle-p threshold for RESTRICTED centre pre-screening.
    K : int, optional
        KNN horizon (default ``min(15, N)``).
    simcount : int, default 999
        Monte Carlo replications per time slice (0 disables MC).
    rantype : str, optional
        Null distribution override (see ``scan()``).
    secondary : int or "ALL", optional
        Secondary cluster enumeration per time slice (see ``scan()``).
    crn : bool, default True
        Use Common Random Numbers across time slices.  Stabilises cross-slice
        differences without changing per-slice null distributions.

        **Seed note:** when ``crn=True`` and ``seed=None``, the same documented
        default seed as ``scan()`` anchors the CRN stream. The effective seed is
        recorded and may be passed explicitly to replay the stream.
    seed : int, optional
        RNG seed in ``[0, 2147483647]``.
    n_jobs_centers : int, optional
        Worker threads for centre scanning per time slice.
    handle : ScanHandle, optional
        Precomputed static handle (see ``build_static()``).
    mc_early_stop, mc_alpha, mc_progress, mc_progress_every : various
        Monte Carlo controls (see ``scan()``).
    enable_center_ub_gating, cache_candidate_llr, scheduler_backend : various
        Core performance toggles (see ``scan()``).
    knn_method : str, optional
        KNN adjacency source: ``'graph'`` or ``'geometry'``. Defaults to
        ``'graph'`` for a newly built handle.
    knn_mode : str, optional
        KNN precompute mode: ``'K_REFILL'`` or ``'K_FIXED'``. Defaults to
        ``'K_REFILL'`` for a newly built handle.
    knn_refill_force : bool, optional
        Force K_REFILL on large N. Defaults to False for a newly built handle.
    both_parallel : bool, default False
        Run HOT and COLD in parallel Python threads per time slice (forces
        *n_jobs_centers* = 1 per direction).
    enable_frame_ub_prune : bool, optional
        Per-frame upper-bound pruning for flexible scans (see ``scan()``).
    n_jobs_time : int, optional
        Parallel workers across time slices.  ``None`` (default) is sequential.
        Setting > 1 disables *warm_start* (a warning is emitted).
    warm_start : bool, default False
        Pass cluster centres from time t-1 as priority centres for time t,
        potentially reducing work on slowly-evolving data.  Silently ignored
        when *n_jobs_time* > 1.
    times : sequence, optional
        Custom time labels for ``result['times']``.  Defaults to 0-based integers.
        Labels are stored as strict-JSON scalars: numpy integers and floats become
        Python numbers, and datetime-like labels (``datetime``, ``pandas.Timestamp``,
        ``numpy.datetime64``) become ISO 8601 strings.
    debug_prune : bool, default False
        Emit per-slice pruning diagnostics to stderr.
    mc_use_alias : bool, default False
        Vose alias MC sampling (see ``scan()``).
    ratio_smoothing : str, default "NONE"
        Rate/risk-ratio smoothing (output only; see ``scan()``).
    secondary_p_stop, secondary_stats_stop : float, optional
        Early-stop thresholds for secondary enumeration (see ``scan()``).
    secondary_null : str, default "full"
        Null distribution secondary clusters are ranked against per time slice:
        ``'full'`` (paper-faithful full-region null) or ``'conditional'`` (prior
        anti-conservative behaviour).  See ``scan()``.

    Returns
    -------
    SeriesResultBase
        Dict with keys ``'per_time'`` (list of per-slice result dicts),
        ``'times'`` (time labels), and ``'scan_settings'`` (shared provenance).
    """
    import numpy as _np

    obs_raw = _np.asarray(observed)
    if obs_raw.ndim != 2:
        raise ValueError("observed must be T x N")
    # Resolve clustersize sentinel: None means "use default of 15"
    clustersize_was_set = clustersize is not None
    clustersize = (
        validate_integer_parameter(clustersize, "clustersize", minimum=1)
        if clustersize is not None
        else 15
    )
    n_jobs_time_i = (
        validate_integer_parameter(n_jobs_time, "n_jobs_time", minimum=1)
        if n_jobs_time is not None
        else 1
    )
    simcount = validate_integer_parameter(simcount, "simcount", minimum=0)
    # Warn when warm_start is silently ignored due to parallel time processing
    if warm_start and n_jobs_time_i > 1:
        warn_at_caller(
            "warm_start=True is ignored when n_jobs_time > 1 (parallel time slices). "
            "Set n_jobs_time=1 or n_jobs_time=None to enable warm starts."
        )
    # Basic validation
    _validate_scan_common(
        ralpha, clustersize, float("inf") if clusterradius is None else clusterradius
    )
    seed_requested = _validate_seed(seed)
    seed_i = seed_requested
    T, N = obs_raw.shape
    if T < 1:
        raise ValueError("observed must contain at least one time slice")
    if times is not None and len(times) != T:
        raise ValueError(f"times must have length T ({T}), got {len(times)}")
    names = _normalize_names(names, N)

    # Validate and normalize model/rantype (parity with scan())
    _model_u, _rantype_u = _normalize_model_and_rantype(model, rantype)
    _scanmethod_u, _stattype_u = _normalize_scanmethod_and_stattype(scanmethod, stattype)
    _clustertype_u = _normalize_clustertype(clustertype)

    # Validate observed matrix for NaN/Inf/negative values
    obs = validate_observed_array(obs_raw.reshape(-1), name="observed").reshape(T, N)

    # Validate denominator shape, then delegate model/parameter validation.
    # Cross-model parameter checks must happen BEFORE the routing branch so that
    # scan_series raises the same ValueError as scan() rather than silently
    # ignoring the wrong parameter and then complaining about the missing one.
    if _model_u == "BINOMIAL" and expected is not None:
        raise ValueError(
            "model='BINOMIAL': Parameter 'expected' is not accepted for BINOMIAL. "
            "Use 'population' instead (integer trial counts matching observed shape)."
        )
    if _model_u == "POISSON" and population is not None:
        raise ValueError(
            "model='POISSON': Parameter 'population' is not accepted. "
            "Use 'expected' instead (positive floats matching observed shape)."
        )
    expected_arr = None
    population_arr = None
    raw_denom = population if _model_u == "BINOMIAL" else expected
    if raw_denom is not None:
        denom_raw = _np.asarray(raw_denom)
        if denom_raw.shape != (T, N):
            kind = "population" if _model_u == "BINOMIAL" else "expected"
            raise ValueError(f"{kind} must match observed shape T x N")
        if _model_u == "BINOMIAL":
            population_arr = denom_raw
        else:
            expected_arr = denom_raw
    # Flatten 2D inputs for shared validation; _validate_denominator rejects cross-model
    # parameters (expected= for BINOMIAL, population= for POISSON) with a clear ValueError.
    flat_expected = expected_arr.reshape(-1).tolist() if expected_arr is not None else None
    flat_population = population_arr.reshape(-1).tolist() if population_arr is not None else None
    exp = _validate_denominator(
        _model_u, flat_expected, flat_population, obs.reshape(-1), T * N
    ).reshape(T, N)

    # Build or reuse handle using shared utility
    reused_handle = handle is not None
    handle, N, eff_K, static_desc = ensure_static_handle(
        handle, coords, adjacency, K, N, clusterradius, knn_method, knn_mode, knn_refill_force
    )
    if eff_K is not None:
        K = eff_K
    elif isinstance(static_desc, dict) and "K" in static_desc:
        try:
            K = int(static_desc["K"])
        except (TypeError, ValueError):
            pass
    warn_on_cap = clustersize_was_set
    clustersize_eff = _resolve_clustersize(clustersize, K, warn=warn_on_cap)
    warn_pruning_heuristics_enabled(_scanmethod_u, enable_frame_ub_prune, enable_center_ub_gating)
    warn_exact_flexible_large_k(
        _scanmethod_u,
        K,
        clustersize_eff,
        enable_frame_ub_prune,
        enable_center_ub_gating,
        stattype=_stattype_u,
        repetitions=T * (simcount + 1) * (2 if _clustertype_u == "BOTH" else 1),
    )

    # ``None`` is reproducible under CRN and fresh without CRN; provenance below
    # records the effective value in both cases so neither meaning is hidden.
    if seed_i is None and not crn:
        import secrets

        seed_i = secrets.randbelow(2_147_483_648)
    elif seed_i is None:
        seed_i = _CRN_DEFAULT_SEED

    # Decide parallelism across time slices
    max_workers = n_jobs_time_i
    max_workers = max(1, min(max_workers, T))
    per: list[dict[str, Any]] = []

    # Build an explicit slice context so all inputs are inspectable and the
    # dependency graph is clear (replaces the former implicit closure).
    _ctx = _SeriesSliceContext(
        obs=obs,
        exp=exp,
        handle=handle,
        N=N,
        names=names,
        model_u=_model_u,
        rantype_u=_rantype_u,
        scanmethod_u=_scanmethod_u,
        stattype_u=_stattype_u,
        clustertype=_clustertype_u,
        ralpha=ralpha,
        clustersize_eff=clustersize_eff,
        simcount=simcount,
        seed_i=seed_i,
        crn=crn,
        warm_start=warm_start,
        mc_early_stop=mc_early_stop,
        mc_alpha=mc_alpha,
        mc_progress=mc_progress,
        mc_progress_every=mc_progress_every,
        mc_use_alias=mc_use_alias,
        secondary=secondary,
        secondary_p_stop=secondary_p_stop,
        secondary_stats_stop=secondary_stats_stop,
        secondary_null=secondary_null,
        debug_prune=bool(debug_prune),
        enable_center_ub_gating=enable_center_ub_gating,
        cache_candidate_llr=cache_candidate_llr,
        scheduler_backend=scheduler_backend,
        enable_frame_ub_prune=enable_frame_ub_prune,
        ratio_smoothing=ratio_smoothing,
        n_jobs_centers=n_jobs_centers,
        max_workers=max_workers,
        both_parallel=both_parallel,
        core_module=core,
    )

    if max_workers == 1:
        for t in range(T):
            per.append(_run_one_slice(t, _ctx, per[t - 1] if t > 0 else None))
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            per = list(ex.map(lambda t: _run_one_slice(t, _ctx), range(T)))
    static_provenance = _effective_static_provenance(
        static_desc=static_desc,
        reused_handle=reused_handle,
        K=K,
        clusterradius=clusterradius,
        knn_method=knn_method,
        knn_mode=knn_mode,
        knn_refill_force=knn_refill_force,
    )
    n_jobs_centers_eff = 1 if max_workers > 1 or both_parallel else n_jobs_centers
    provenance_settings = _build_base_settings(
        N=N,
        ralpha=ralpha,
        clustersize_eff=clustersize_eff,
        scanmethod=_scanmethod_u,
        stattype=_stattype_u,
        model=_model_u,
        simcount=simcount,
        rantype_u=_rantype_u,
        mc_early_stop=False if _clustertype_u == "BOTH" else mc_early_stop,
        mc_alpha=mc_alpha,
        mc_progress=mc_progress,
        mc_progress_every=mc_progress_every,
        secondary=secondary,
        secondary_p_stop=secondary_p_stop,
        secondary_stats_stop=secondary_stats_stop,
        default_secondary_p_stop=1.0,
        secondary_null=secondary_null,
        debug_prune=debug_prune,
        enable_center_ub_gating=enable_center_ub_gating,
        cache_candidate_llr=cache_candidate_llr,
        scheduler_backend=scheduler_backend,
        enable_frame_ub_prune=enable_frame_ub_prune,
        mc_use_alias=mc_use_alias,
        ratio_smoothing=ratio_smoothing,
        seed=seed_i,
        n_jobs_centers=n_jobs_centers_eff,
    )
    provenance = _build_scan_provenance(
        base_settings=provenance_settings,
        static_provenance=static_provenance,
        clustertype=_clustertype_u,
        ralpha=float(ralpha),
        requested_seed=seed_requested,
        effective_seed=seed_i,
        requested_simcount=simcount,
        requested_mc_early_stop=mc_early_stop,
        requested_n_jobs_centers=n_jobs_centers,
        requested_mc_use_alias=mc_use_alias,
        both_parallel=both_parallel,
        handle_reused=reused_handle,
    )
    provenance.update(
        {
            "crn": bool(crn),
            "n_jobs_time": int(max_workers),
            "n_jobs_time_requested": n_jobs_time,
            "warm_start": bool(warm_start and max_workers == 1),
            "warm_start_requested": bool(warm_start),
            "T": int(T),
            "N": int(N),
        }
    )
    out = {
        "per_time": per,
        "times": (
            [_json_safe_time_label(label) for label in times]
            if times is not None
            else list(range(T))
        ),
        "scan_settings": provenance,
    }
    validated = validate_series_result_contract(out)
    # Validate the public envelope before reading native-only transport fields,
    # so malformed results retain the documented result-contract diagnostic.
    # A series can early-stop at a different boundary for every slice; the list
    # therefore aligns with per_time rather than pretending one scalar applies.
    provenance["simcount_actual"] = [
        _pop_result_simcount_actual(entry, requested_simcount=simcount) for entry in per
    ]
    return validated  # type: ignore[return-value]


# New helper/wrapper utilities for improved UX


def build_static(
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    K: int | None,
    clusterradius: float = float("inf"),
    *,
    knn_mode: str = "K_REFILL",
    knn_refill_force: bool = False,
) -> Any:
    """Build and return a reusable static precompute handle.

    Validates inputs before delegating to the core static builder.
    """
    return build_validated_static_handle(
        coords,
        adjacency,
        K,
        clusterradius,
        knn_mode=str(knn_mode),
        knn_refill_force=bool(knn_refill_force),
    )


def describe_static(handle: _HandleT) -> dict:
    """Return a dictionary describing a static handle built by ``build_static()``.

    Raises
    ------
    ValueError
        If *handle* is not a static precompute handle (the message names the
        ``handle`` parameter rather than the compiled binding).
    """
    return describe_static_handle(handle)


def scan_from_geodataframe(
    gdf: Any,
    *,
    id_col: str,
    contiguity: str = "queen",
    K: int | None = None,
    clusterradius: float = float("inf"),
    knn_mode: str = "K_REFILL",
    knn_refill_force: bool = False,
    names: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build contiguity, scan, and return results keyed by GeoDataFrame area IDs.

    This wrapper is intentionally strict:
    - `gdf.crs` must be projected (not geographic / degree-based)
    - `id_col` is required and must contain unique positive integer area IDs
    - returned cluster `area`, `from_area`, and `to_area` values are remapped from
      row positions into the `id_col` value space
    - `names` defaults to `id_col` string values when not provided
    """
    if "handle" in kwargs:
        raise TypeError(
            "scan_from_geodataframe: 'handle' is incompatible with the graph built internally "
            "from gdf; call scan(handle=...) directly to reuse a precompute handle."
        )

    from .graph import csr_from_geodataframe_contiguity as _csr_from_gdf

    coords, area_ids = _extract_geodataframe_data(gdf, id_col)

    row_ptr, col_idx = _csr_from_gdf(gdf, method=contiguity)

    names_eff = _normalize_names(names, len(area_ids))
    if names_eff is None:
        names_eff = [str(v) for v in area_ids]

    _INTERNAL_KWARGS = ("coords", "adjacency")
    for _k in _INTERNAL_KWARGS:
        if _k in kwargs:
            raise TypeError(
                f"scan_from_geodataframe: '{_k}' is built internally from gdf; "
                "do not pass it via kwargs."
            )

    result = scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        K=K,  # None lets ensure_static_handle apply the default (min(_DEFAULT_K, N))
        clusterradius=clusterradius,
        knn_method="graph",
        knn_mode=str(knn_mode),
        knn_refill_force=bool(knn_refill_force),
        names=names_eff,
        **kwargs,
    )
    id_map = {i + 1: area_id for i, area_id in enumerate(area_ids)}
    return _remap_result_area_ids(cast(dict[str, Any], result), id_map)
