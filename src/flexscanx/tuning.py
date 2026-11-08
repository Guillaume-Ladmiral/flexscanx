from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import api as _api
from ._stability import (
    StabilityMetrics,
    StabilityThresholds,
    is_stable,
    match_clusters_jaccard,
)
from ._utils import (
    _validate_coords_and_infer_N,
    ensure_static_handle,
    validate_binomial_population,
    validate_expected_array,
    validate_integer_parameter,
    validate_observed_array,
)
from .helpers import cluster_pool as _clusters_to_sets

# Named constants for default parameter heuristics in find_stable_params_all
_DEFAULT_K_START_FRAC: float = 0.25  # K_start = 25% of N
_DEFAULT_S_START_FRAC: float = 0.12  # S_start = 12% of N
_DEFAULT_K_MIN: int = 8  # hard lower bound on K
_DEFAULT_S_MIN: int = 5  # hard lower bound on S
# The default floors scale with N at half the rate of the start values.  Scaling
# S_min by the same 12 % as S_start made the two coincide for every N in
# 17..41, so the cluster-size descent had no room by construction (FXA-185).
_DEFAULT_S_MIN_FRAC: float = 0.06  # S_min = 6% of N, capped at _DEFAULT_S_MIN
_DEFAULT_K_CAP: int = 500  # upper cap on K_start
_DEFAULT_K_FLOOR: int = 15  # minimum default K_start before N-specific clipping
_GRID_COLUMNS = (
    "K",
    "S",
    "pool_size",
    "stats",
    "pval",
    "mean_jaccard",
    "min_jaccard",
    "coverage",
    "count_diff",
)


def _scaled_tuning_floor(default_floor: int, frac: float, N: int, *, minimum: int) -> int:
    """Scale a default lower bound down for small graphs while keeping a floor."""
    return max(int(minimum), min(int(default_floor), int(round(float(frac) * int(N)))))


def _normalize_tuning_bounds(
    *,
    N: int,
    K_start: int | None,
    S_start: int | None,
    K_min: int | None,
    S_min: int | None,
    require_K_gt_S: bool,
) -> tuple[int, int, int, int]:
    """Return a feasible `(K_start, S_start, K_min, S_min)` search domain."""
    if int(N) < 1:
        raise ValueError("coords must contain at least one area")

    strict_gap = 1 if require_K_gt_S else 0
    min_valid_K = 1 + strict_gap
    max_valid_K = int(N)
    max_valid_S = int(N) - strict_gap

    if max_valid_K < min_valid_K or max_valid_S < 1:
        raise ValueError(
            "No valid (K, S) search space: require_K_gt_S=True requires at least 2 areas."
        )

    if K_min is None:
        K_min_eff = _scaled_tuning_floor(
            _DEFAULT_K_MIN, _DEFAULT_K_START_FRAC, N, minimum=min_valid_K
        )
    else:
        K_min_eff = validate_integer_parameter(K_min, "K_min")
    if not (min_valid_K <= K_min_eff <= max_valid_K):
        raise ValueError(
            f"K_min must be in [{min_valid_K}, {max_valid_K}] for N={N}, got {K_min_eff}"
        )

    if S_min is None:
        S_min_eff = _scaled_tuning_floor(_DEFAULT_S_MIN, _DEFAULT_S_MIN_FRAC, N, minimum=1)
    else:
        S_min_eff = validate_integer_parameter(S_min, "S_min")
    if not (1 <= S_min_eff <= max_valid_S):
        raise ValueError(f"S_min must be in [1, {max_valid_S}] for N={N}, got {S_min_eff}")

    min_feasible_K = max(K_min_eff, S_min_eff + strict_gap, min_valid_K)
    if min_feasible_K > max_valid_K:
        raise ValueError(
            "No valid (K, S) search space after applying K_min/S_min and require_K_gt_S."
        )

    if K_start is None:
        K_start_eff = int(
            min(
                max(_DEFAULT_K_FLOOR, round(_DEFAULT_K_START_FRAC * N)),
                _DEFAULT_K_CAP,
            )
        )
    else:
        K_start_eff = validate_integer_parameter(K_start, "K_start")
    K_start_eff = min(max(K_start_eff, min_feasible_K), max_valid_K)

    S_max_at_start = min(max_valid_S, K_start_eff - strict_gap)
    if S_max_at_start < S_min_eff:
        raise ValueError("No valid initial (K, S) state after applying bounds and require_K_gt_S.")

    if S_start is None:
        default_s_start = max(S_min_eff, round(_DEFAULT_S_START_FRAC * N))
        if N >= 4 and S_max_at_start >= 2:
            default_s_start = max(default_s_start, 2)
        S_start_eff = int(min(default_s_start, S_max_at_start))
    else:
        S_start_eff = validate_integer_parameter(S_start, "S_start")
    S_start_eff = min(max(S_start_eff, S_min_eff), S_max_at_start)

    return int(K_start_eff), int(S_start_eff), int(K_min_eff), int(S_min_eff)


def _validate_tuning_pair(*, K: int, S: int, N: int, require_K_gt_S: bool) -> None:
    """Validate that a returned tuning pair is directly reusable by `scan()`."""
    if int(K) < 1 or int(K) > int(N):
        raise ValueError(f"helper produced invalid K={int(K)} for N={int(N)}")
    if int(S) < 1:
        raise ValueError(f"helper produced invalid S={int(S)}; must be >= 1")
    if require_K_gt_S:
        if int(K) <= int(S):
            raise ValueError(
                f"helper produced invalid pair (K={int(K)}, S={int(S)}); require K > S"
            )
    elif int(S) > int(K):
        raise ValueError(f"helper produced invalid pair (K={int(K)}, S={int(S)}); require S <= K")


def _scan_pool(
    *,
    handle: Any,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None,
    population: Sequence[float] | None,
    K: int,
    S: int,
    model: str,
    scanmethod: str,
    stattype: str,
    clustertype: str,
    ralpha: float,
    clusterradius: float,
    seed: int | None,
    n_jobs_centers: int | None,
    secondary_stats_stop: float | None,
) -> list[frozenset[int]]:
    res = _api.scan(
        handle=handle,
        coords=None if handle is not None else coords,
        adjacency=None if handle is not None else adjacency,
        observed=observed,
        expected=expected,
        population=population,
        model=model,
        scanmethod=scanmethod,
        stattype=stattype,
        clustertype=clustertype,
        clustersize=S,
        clusterradius=float(clusterradius),
        ralpha=float(ralpha),
        K=int(K),
        knn_method="graph",  # using provided CSR
        knn_mode="K_FIXED",
        # This is a build-time setting already fixed in ``handle``.  Leaving
        # it unspecified preserves the strict handle-reuse boundary.
        knn_refill_force=None,
        simcount=0,
        secondary="ALL",
        secondary_stats_stop=secondary_stats_stop,
        seed=seed,
        n_jobs_centers=n_jobs_centers,
    )
    return _clusters_to_sets(res)


def find_stable_params_all(
    *,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    K_start: int | None = None,
    S_start: int | None = None,
    K_min: int | None = None,
    S_min: int | None = None,
    stepK: float = 0.8,
    stepS: float = 0.8,
    require_K_gt_S: bool = True,
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "BOTH",
    ralpha: float = 0.2,
    clusterradius: float = float("inf"),
    thresholds: StabilityThresholds | None = None,
    binary_refine: bool = True,
    seed: int | None = None,
    n_jobs_centers: int | None = None,
    secondary_stats_stop: float | None = None,
) -> dict[str, Any]:
    """Select the smallest (K, S) with stable cluster sets when increasing either knob.

    Policy
    - Uses K_FIXED, simcount=0, and the given stopping rule (secondary_stats_stop) consistently.
    - Pools HOT and COLD clusters (no primary/secondary distinction).
    - Top-down coordinate descent with optional binary refinement.

    Returns dict with:
    - 'best': {'K': int, 'S': int}
    - 'trace': list of steps and stability checks
    - 'metrics': per (K,S) caches of pool sizes and (optional) neighbor stability metrics
    """
    coords = np.asarray(coords, dtype=float)
    row_ptr, col_idx = adjacency
    N = int(coords.shape[0])

    stepK_f = float(stepK)
    stepS_f = float(stepS)
    if not np.isfinite(stepK_f) or not 0.0 < stepK_f < 1.0:
        raise ValueError(f"stepK must be finite and in (0,1), got {stepK!r}")
    if not np.isfinite(stepS_f) or not 0.0 < stepS_f < 1.0:
        raise ValueError(f"stepS must be finite and in (0,1), got {stepS!r}")

    K_start, S_start, K_min, S_min = _normalize_tuning_bounds(
        N=N,
        K_start=K_start,
        S_start=S_start,
        K_min=K_min,
        S_min=S_min,
        require_K_gt_S=require_K_gt_S,
    )

    thr = thresholds or StabilityThresholds()
    thr.__post_init__()

    # Handle cache per K
    handles: dict[int, Any] = {}

    def get_handle(K: int) -> Any:
        if K in handles:
            return handles[K]
        h, _, _, _ = ensure_static_handle(
            None,
            coords,
            (row_ptr, col_idx),
            int(K),
            N,
            float(clusterradius),
            "graph",
            "K_FIXED",
            False,
        )
        handles[K] = h
        return h

    # Pool cache per (K,S)
    pools: dict[tuple[int, int], list[frozenset[int]]] = {}

    def get_pool(K: int, S: int) -> list[frozenset[int]]:
        key = (int(K), int(S))
        if key in pools:
            return pools[key]
        if require_K_gt_S and not (int(K) > int(S)):
            pools[key] = []
            return pools[key]
        h = get_handle(K)
        pool = _scan_pool(
            handle=h,
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            population=population,
            K=int(K),
            S=int(S),
            model=model,
            scanmethod=scanmethod,
            stattype=stattype,
            clustertype=clustertype,
            ralpha=float(ralpha),
            clusterradius=float(clusterradius),
            seed=seed,
            n_jobs_centers=n_jobs_centers,
            secondary_stats_stop=secondary_stats_stop,
        )
        pools[key] = pool
        return pool

    def check_stable(KA: int, SA: int, KB: int, SB: int) -> StabilityMetrics:
        A = get_pool(KA, SA)
        B = get_pool(KB, SB)
        return match_clusters_jaccard(A, B, coverage_floor=thr.coverage_floor)

    trace: list[dict[str, Any]] = []
    metrics_map: dict[tuple[int, int], dict[str, Any]] = {}

    # 1) Decrease K with S fixed
    K_curr = int(K_start)
    S_curr = int(S_start)

    def dec_axis_K(K0: int, S: int) -> int:
        K_vals = [int(K0)]
        while True:
            # A purely multiplicative step cannot leave small values (round(0.8*2)
            # == 2); always move by at least one while a smaller feasible value
            # exists, and stop only when nothing below the current value remains.
            K_next = max(min(int(round(stepK_f * K_vals[-1])), K_vals[-1] - 1), int(K_min))
            if require_K_gt_S and K_next <= int(S):
                break
            if K_next >= K_vals[-1]:
                break
            m = check_stable(K_next, S, K_vals[-1], S)
            trace.append(
                {
                    "phase": "decK",
                    "from": (K_vals[-1], S),
                    "to": (K_next, S),
                    "metrics": m,
                }
            )
            if is_stable(m, thr):
                K_vals.append(K_next)
            else:
                if not binary_refine:
                    break
                lo, hi = K_next, K_vals[-1]
                anchor = K_vals[-1]
                # binary refine to smallest stable between lo..hi
                while lo < hi - 1:
                    mid = lo + (hi - lo) // 2
                    if require_K_gt_S and mid <= S:
                        lo = mid
                        continue
                    m_mid = check_stable(mid, S, anchor, S)
                    trace.append(
                        {
                            "phase": "decK-refine",
                            "from": (anchor, S),
                            "to": (mid, S),
                            "metrics": m_mid,
                        }
                    )
                    if is_stable(m_mid, thr):
                        hi = mid
                    else:
                        lo = mid
                K_vals.append(hi)
                break
        return int(min(K_vals))

    K_star = dec_axis_K(K_curr, S_curr)

    # 2) Decrease S with K fixed
    def dec_axis_S(K: int, S0: int) -> int:
        S_vals = [int(S0)]
        while True:
            # See dec_axis_K: step by at least one while a smaller feasible value
            # exists, and stop only at the floor (FXA-185).
            S_next = max(min(int(round(stepS_f * S_vals[-1])), S_vals[-1] - 1), int(S_min))
            if require_K_gt_S and int(K) <= S_next:
                break
            if S_next >= S_vals[-1]:
                break
            m = check_stable(K, S_next, K, S_vals[-1])
            trace.append(
                {
                    "phase": "decS",
                    "from": (K, S_vals[-1]),
                    "to": (K, S_next),
                    "metrics": m,
                }
            )
            if is_stable(m, thr):
                S_vals.append(S_next)
            else:
                if not binary_refine:
                    break
                lo, hi = S_next, S_vals[-1]
                anchor = S_vals[-1]
                while lo < hi - 1:
                    mid = lo + (hi - lo) // 2
                    if require_K_gt_S and int(K) <= mid:
                        lo = mid
                        continue
                    m_mid = check_stable(K, mid, K, anchor)
                    trace.append(
                        {
                            "phase": "decS-refine",
                            "from": (K, anchor),
                            "to": (K, mid),
                            "metrics": m_mid,
                        }
                    )
                    if is_stable(m_mid, thr):
                        hi = mid
                    else:
                        lo = mid
                S_vals.append(hi)
                break
        return int(min(S_vals))

    S_star = dec_axis_S(K_star, S_curr)
    _validate_tuning_pair(K=K_star, S=S_star, N=N, require_K_gt_S=require_K_gt_S)
    get_pool(K_star, S_star)

    best = {"K": int(K_star), "S": int(S_star)}

    # Collect per-(K,S) pool sizes that were computed
    for (K, S), pool in pools.items():
        metrics_map[(int(K), int(S))] = {"pool_size": len(pool)}

    return {"best": best, "trace": trace, "metrics": metrics_map}


def grid_search_stable(
    *,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    K_grid: Sequence[int],
    S_grid: Sequence[int],
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "BOTH",
    ralpha: float = 0.2,
    clusterradius: float = float("inf"),
    simcount: int = 0,
    seed: int | None = None,
    n_jobs_centers: int | None = None,
    secondary_stats_stop: float | None = None,
) -> Any:
    """Evaluate a (K, S) grid and compute stability metrics vs an anchor setting."""
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for grid_search_stable") from e

    k_values = list(K_grid)
    s_values = list(S_grid)
    if not k_values or not s_values:
        raise ValueError("K_grid and S_grid must be non-empty")

    # Pre-build one precompute handle per unique K to avoid redundant O(N·K) work per grid cell.
    coords, N = _validate_coords_and_infer_N(coords)
    ks = sorted(
        {
            validate_integer_parameter(value, f"K_grid[{idx}]", minimum=1, maximum=N)
            for idx, value in enumerate(k_values)
        }
    )
    ss = sorted(
        {
            validate_integer_parameter(value, f"S_grid[{idx}]", minimum=1, maximum=N)
            for idx, value in enumerate(s_values)
        }
    )
    feasible_pairs = [(k, s) for k in ks for s in ss if k > s]
    if not feasible_pairs:
        raise ValueError("grid_search_stable requires at least one feasible grid cell with K > S")

    _handles: dict[int, Any] = {}
    for k in sorted({k for k, _ in feasible_pairs}):
        h, _, _, _ = ensure_static_handle(
            None,
            coords,
            adjacency,
            k,
            N,
            float(clusterradius),
            "graph",
            "K_FIXED",
            False,
        )
        _handles[k] = h

    pools: dict[tuple[int, int], list[frozenset[int]]] = {}
    rows: list[dict[str, Any]] = []

    for k, s in feasible_pairs:
        res = _api.scan(
            handle=_handles[k],
            coords=None,
            adjacency=None,
            observed=observed,
            expected=expected,
            population=population,
            model=model,
            scanmethod=scanmethod,
            stattype=stattype,
            clustertype=clustertype,
            clustersize=int(s),
            clusterradius=float(clusterradius),
            K=int(k),
            ralpha=float(ralpha),
            simcount=simcount,
            secondary="ALL",
            secondary_stats_stop=secondary_stats_stop,
            seed=seed,
            n_jobs_centers=n_jobs_centers,
        )
        pool = _clusters_to_sets(res)
        pools[(int(k), int(s))] = pool
        primary = res.get("primary", res) if isinstance(res, dict) else {}
        stats = primary.get("stats") if isinstance(primary, dict) else None
        pval = primary.get("pval") if isinstance(primary, dict) else None
        rows.append(
            {
                "K": int(k),
                "S": int(s),
                "pool_size": int(len(pool)),
                "stats": (float(stats) if stats is not None else np.nan),
                "pval": (float(pval) if pval is not None else np.nan),
            }
        )

    if not rows:
        return pd.DataFrame(
            {
                "K": pd.Series(dtype="int64"),
                "S": pd.Series(dtype="int64"),
                "pool_size": pd.Series(dtype="int64"),
                "stats": pd.Series(dtype="float64"),
                "pval": pd.Series(dtype="float64"),
                "mean_jaccard": pd.Series(dtype="float64"),
                "min_jaccard": pd.Series(dtype="float64"),
                "coverage": pd.Series(dtype="float64"),
                "count_diff": pd.Series(dtype="int64"),
            }
        )

    anchor = (max(r["K"] for r in rows), max(r["S"] for r in rows))
    pool_anchor = pools.get(anchor, [])
    for row in rows:
        m = match_clusters_jaccard(
            pools[(int(row["K"]), int(row["S"]))], pool_anchor, coverage_floor=0.5
        )
        row["mean_jaccard"] = float(m.mean_jaccard)
        row["min_jaccard"] = float(m.min_jaccard)
        row["coverage"] = float(m.coverage)
        row["count_diff"] = int(m.count_diff)

    return pd.DataFrame(rows, columns=_GRID_COLUMNS).sort_values(["K", "S"]).reset_index(drop=True)


def screen_stable_significant(
    *,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    K_grid: Sequence[int],
    S_grid: Sequence[int],
    alpha: float = 0.05,
    simcount: int = 999,
    thresholds: StabilityThresholds | None = None,
    **kwargs: Any,
) -> Any:
    """Filter grid_search_stable output to stable and significant settings."""
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if not 0.0 < alpha_f < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    df = grid_search_stable(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        population=population,
        K_grid=K_grid,
        S_grid=S_grid,
        simcount=simcount,
        **kwargs,
    )
    if df.empty:
        return df

    thr = thresholds or StabilityThresholds()
    thr.__post_init__()
    stable_mask = (
        (df["mean_jaccard"] >= float(thr.mean_jaccard))
        & (df["min_jaccard"] >= float(thr.min_jaccard))
        & (df["coverage"] >= float(thr.coverage))
        & (df["count_diff"] <= int(thr.count_tol))
    )
    sig_mask = df["pval"].notna() & (df["pval"] <= alpha_f)
    out = df.loc[stable_mask & sig_mask].copy()
    return out.sort_values(["K", "S"]).reset_index(drop=True)


def find_stable_params_all_series(
    *,
    observed: np.ndarray,
    expected: np.ndarray | None = None,
    population: np.ndarray | None = None,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run find_stable_params_all independently per time slice and aggregate."""
    obs = np.asarray(observed)
    model_u = str(kwargs.get("model", "POISSON")).upper()
    raw_denom = population if model_u == "BINOMIAL" else expected
    if raw_denom is None:
        denom_name = "population" if model_u == "BINOMIAL" else "expected"
        raise ValueError(f"{denom_name} is required for model='{model_u}'")
    denom = np.asarray(raw_denom)
    if obs.ndim != 2 or denom.ndim != 2:
        raise ValueError("observed and denominator must be T x N arrays")
    if obs.shape != denom.shape:
        raise ValueError("observed and denominator must have the same shape")

    T = int(obs.shape[0])
    n_areas = int(obs.shape[1])
    if T < 1:
        raise ValueError("observed must contain at least one time slice")
    obs = validate_observed_array(obs.reshape(-1)).reshape(obs.shape)
    if model_u == "BINOMIAL":
        denom = validate_binomial_population(denom.reshape(-1)).reshape(denom.shape)
        if np.any(obs > denom):
            raise ValueError("observed counts cannot exceed population for BINOMIAL model")
    else:
        denom = validate_expected_array(denom.reshape(-1), allow_zero=True).reshape(denom.shape)

    per_time: list[dict[str, Any]] = []
    k_vals: list[int] = []
    s_vals: list[int] = []

    for t in range(T):
        sel = find_stable_params_all(
            coords=coords,
            adjacency=adjacency,
            observed=obs[t].tolist(),
            expected=(
                np.asarray(denom[t], dtype=float).tolist() if model_u != "BINOMIAL" else None
            ),
            population=(
                np.asarray(denom[t], dtype=float).tolist() if model_u == "BINOMIAL" else None
            ),
            **kwargs,
        )
        best = sel["best"]
        k = int(best["K"])
        s = int(best["S"])
        k_vals.append(k)
        s_vals.append(s)
        per_time.append(
            {
                "time": int(t),
                "K": k,
                "S": s,
                "selection": sel,
            }
        )

    k_med = int(np.median(np.asarray(k_vals, dtype=float))) if k_vals else 0
    s_med = int(np.median(np.asarray(s_vals, dtype=float))) if s_vals else 0
    if k_vals:
        max_valid_K = n_areas
        require_K_gt_S = bool(kwargs.get("require_K_gt_S", True))
        # Reuse shared bound-normalisation logic rather than re-implementing clipping.
        k_med, s_med, _, _ = _normalize_tuning_bounds(
            N=max_valid_K,
            K_start=k_med,
            S_start=s_med,
            K_min=kwargs.get("K_min"),
            S_min=kwargs.get("S_min"),
            require_K_gt_S=require_K_gt_S,
        )
        _validate_tuning_pair(K=k_med, S=s_med, N=max_valid_K, require_K_gt_S=require_K_gt_S)

    return {
        "best": {"K": k_med, "S": s_med},
        "K_values": k_vals,
        "S_values": s_vals,
        "per_time": per_time,
    }
