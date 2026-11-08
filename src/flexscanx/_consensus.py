"""Consensus sweep, trend analysis, and cross-K stability utilities.

Moved from validation.py (B1 split) — all external imports go through
validation.py which re-exports the public names.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ._core_loader import require_core as _require_core
from ._scan_internal import (
    _normalize_clustertype,
    _normalize_model_and_rantype,
    _normalize_scanmethod_and_stattype,
    _resolve_clustersize,
    _validate_denominator,
    _validate_scan_common,
)
from ._stability import match_clusters_jaccard
from ._utils import (
    _validate_coords_and_infer_N,
    _validate_csr_adjacency,
    build_validated_static_handle,
    validate_cluster_radius,
    validate_integer_parameter,
    validate_observed_array,
)


def consensus_sweep(
    *,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    clustersize: int,
    grid: dict[str, Sequence],
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "HOT",
    knn_mode: str = "K_FIXED",
    knn_refill_force: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a consensus primary cluster over a grid of parameters.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        Projected (x, y) coordinates for each area.
    adjacency : tuple of (row_ptr, col_idx)
        CSR adjacency for the scan graph.
    observed : sequence of int
        Observed counts, length N.
    expected : sequence of float, optional
        Expected counts, length N. Required for the Poisson model.
    population : sequence of int, optional
        Population sizes, length N. Required for the binomial model.
    clustersize : int
        Maximum cluster size (S) used in every scan run.
    grid : dict
        Parameter grid with optional keys ``'K'``, ``'ralpha'``, ``'radius'``.
        Each value is a sequence of values to sweep.  Missing keys default to
        ``[clustersize]``, ``[0.2]``, and ``[inf]`` respectively.
    model, scanmethod, stattype, clustertype : str
        Forwarded to the core scanner unchanged.
    knn_mode, knn_refill_force : str, bool
        KNN precompute options.
    threshold : float, default 0.5
        Minimum inclusion frequency for the consensus set.

    Returns
    -------
    dict
        Keys: ``'freq'`` (per-area frequencies), ``'consensus'`` (area IDs ≥ threshold),
        ``'runs'`` (total grid evaluations), ``'grid'`` (normalised grid dict).

    Notes
    -----
    Only the primary cluster area from each scan run contributes to the
    frequency tally.  Secondary clusters are not counted.  To include
    secondaries, iterate manually over scan results with ``secondary='ALL'``.
    """
    core = _require_core()

    model_u, _ = _normalize_model_and_rantype(model, None)
    scanmethod_u, stattype_u = _normalize_scanmethod_and_stattype(scanmethod, stattype)
    clustertype_u = _normalize_clustertype(clustertype, allow_both=False)

    coords, n_coords = _validate_coords_and_infer_N(coords)
    arr_obs_np = validate_observed_array(observed)
    if arr_obs_np.ndim != 1:
        raise ValueError("consensus_sweep: observed must be a 1D array")
    arr_obs = arr_obs_np.tolist()
    n_areas = len(arr_obs)
    try:
        arr_exp_np = _validate_denominator(model_u, expected, population, arr_obs_np, n_areas)
    except ValueError as exc:
        if str(exc).startswith("expected/population length mismatch"):
            denominator = population if model_u == "BINOMIAL" else expected
            denominator_size = len(denominator) if denominator is not None else 0
            raise ValueError(
                f"consensus_sweep: observed and expected/population must have the same length; "
                f"got observed={n_areas}, denominator={denominator_size}"
            ) from exc
        raise
    if arr_exp_np.ndim != 1:
        raise ValueError("consensus_sweep: expected/population must be a 1D array")
    arr_exp = arr_exp_np.tolist()
    if len(arr_exp) != n_areas:
        raise ValueError(
            f"consensus_sweep: observed and expected/population must have the same length; "
            f"got observed={n_areas}, denominator={len(arr_exp)}"
        )
    if n_areas == 0:
        raise ValueError("consensus_sweep: observed must contain at least one area")
    if n_coords != n_areas:
        raise ValueError(
            f"consensus_sweep: coords must have one row per area; "
            f"got coords={n_coords}, observed={n_areas}"
        )
    clustersize_i = validate_integer_parameter(clustersize, "clustersize", minimum=1)
    ks_raw = list(grid.get("K", [clustersize_i]))
    alphas_raw = list(grid.get("ralpha", [0.2]))
    radii_raw = list(grid.get("radius", [float("inf")]))
    if not ks_raw:
        raise ValueError("K grid must contain at least one value")
    if not alphas_raw:
        raise ValueError("ralpha grid must contain at least one value")
    if not radii_raw:
        raise ValueError("radius grid must contain at least one value")

    ks = [validate_integer_parameter(value, "K", minimum=1, maximum=n_areas) for value in ks_raw]
    alphas: list[float] = []
    for value in alphas_raw:
        alpha = float(value)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"ralpha grid values must be finite and in [0,1], got {value!r}")
        alphas.append(alpha)
    radii = [validate_cluster_radius(value) for value in radii_raw]

    threshold_f = float(threshold)
    if not np.isfinite(threshold_f):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    if not 0.0 <= threshold_f <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")

    row_ptr, col_idx = _validate_csr_adjacency(adjacency, n_areas)

    counts = np.zeros(n_areas, dtype=float)
    runs = 0
    settings_base: dict[str, Any] = {
        "model": model_u,
        "scanmethod": scanmethod_u,
        "stattype": stattype_u,
        "clustertype": clustertype_u,
        "simcount": 0,
    }

    for k_value in ks:
        for radius in radii:
            clustersize_eff = _resolve_clustersize(clustersize_i, k_value)
            handle = build_validated_static_handle(
                coords,
                (row_ptr, col_idx),
                k_value,
                radius,
                knn_mode=str(knn_mode),
                knn_refill_force=bool(knn_refill_force),
            )
            for alpha in alphas:
                settings = dict(settings_base)
                settings["clustersize"] = clustersize_eff
                settings["ralpha"] = alpha
                _validate_scan_common(alpha, clustersize_eff, radius)
                result = core.ScanSlice(handle, arr_obs, arr_exp, settings)
                if isinstance(result, dict) and "area" in result:
                    for area_id in result["area"]:
                        if 1 <= int(area_id) <= n_areas:
                            counts[int(area_id) - 1] += 1.0
                runs += 1

    freq = (counts / max(1, runs)).tolist()
    consensus = [idx + 1 for idx, value in enumerate(freq) if value >= threshold_f]
    grid_norm = {"K": ks, "ralpha": alphas, "radius": radii}
    return {"freq": freq, "consensus": consensus, "runs": int(runs), "grid": grid_norm}


def trend_level_slope(
    per_time: list[dict[str, Any]],
    window: int = 3,
    use: str = "S",
    jaccard_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compute trend level and slope from time-series scan results.

    Parameters
    ----------
    per_time : list of dict
        The ``'per_time'`` list from a ``scan_series()`` result.
    window : int, default 3
        Trailing window over which to compute slope.
    use : str, default "S"
        Score to track: ``'S'`` uses -log10(p-value); ``'LLR'`` uses the raw
        log-likelihood ratio statistic.
    jaccard_threshold : float, default 0.3
        Minimum Jaccard similarity between consecutive time-point cluster areas
        to treat them as the same persistent cluster segment.

    Returns
    -------
    dict
        Keys: ``'times'``, ``'level'``, ``'slope'``, ``'level_last'``.
    """
    import math as _math

    window_i = validate_integer_parameter(window, "window", minimum=1)
    use_u = str(use).strip().upper()
    if use_u not in {"S", "LLR"}:
        raise ValueError(f"use must be one of ['LLR', 'S'], got {use!r}")
    jaccard_threshold_f = float(jaccard_threshold)
    if not np.isfinite(jaccard_threshold_f):
        raise ValueError(f"jaccard_threshold must be finite, got {jaccard_threshold!r}")
    if not 0.0 <= jaccard_threshold_f <= 1.0:
        raise ValueError(f"jaccard_threshold must be in [0,1], got {jaccard_threshold}")

    def _score(pval: Any) -> float:
        if pval is None:
            return 0.0
        try:
            pval_f = float(pval)
        except Exception:
            return 0.0
        if not _math.isfinite(pval_f):
            return 0.0
        if pval_f <= 0:
            pval_f = 1e-16
        return -_math.log10(pval_f)

    series: list[tuple[int, list[int], float]] = []
    for t_idx, result in enumerate(per_time):
        if not isinstance(result, dict):
            continue
        if result.get("type", "").upper() == "BOTH":
            result = result.get("primary", {})
        area = [int(value) for value in result.get("area", [])]
        level = float(result.get("stats", 0.0)) if use_u == "LLR" else _score(result.get("pval"))
        series.append((t_idx, area, level))

    segments: list[list[tuple[int, list[int], float]]] = []
    current: list[tuple[int, list[int], float]] = []
    last_area: set[int] | None = None
    for t_idx, area, level in series:
        current_area = set(area)
        if not current:
            current = [(t_idx, area, level)]
            last_area = current_area
            continue
        union = current_area | (last_area or set())
        jaccard = len(current_area & (last_area or set())) / len(union) if union else 1.0
        if jaccard >= jaccard_threshold_f:
            current.append((t_idx, area, level))
            last_area = current_area
            continue
        segments.append(current)
        current = [(t_idx, area, level)]
        last_area = current_area

    if current:
        segments.append(current)
    if not segments:
        return {"times": [], "level": [], "slope": float("nan"), "level_last": 0.0}

    segment = segments[-1]
    times = [entry[0] for entry in segment]
    levels = [entry[2] for entry in segment]
    tail_times = times[-window_i:]
    tail_levels = levels[-window_i:]
    if len(tail_times) < 2:
        slope = 0.0
    else:
        t_arr = np.asarray(tail_times, dtype=float)
        y_arr = np.asarray(tail_levels, dtype=float)
        t_mean = float(np.mean(t_arr))
        y_mean = float(np.mean(y_arr))
        numerator = float(np.sum((t_arr - t_mean) * (y_arr - y_mean)))
        denominator = float(np.sum((t_arr - t_mean) ** 2))
        slope = numerator / denominator if denominator > 0 else 0.0
    return {
        "times": times,
        "level": levels,
        "slope": float(slope),
        "level_last": float(levels[-1] if levels else 0.0),
    }


def consensus_to_series(freq: Iterable[float], ids: Iterable[int] | None = None) -> Any:
    """Convert consensus frequencies to a pandas Series.

    Parameters
    ----------
    freq : iterable of float
        Per-area inclusion frequencies (as returned by ``consensus_sweep``).
    ids : iterable of int, optional
        Area IDs to use as the Series index.  Defaults to 1-based integers.

    Returns
    -------
    pd.Series
        Index = area IDs, values = frequencies.
    """
    import pandas as _pd  # type: ignore[import-untyped]

    series = _pd.Series(list(freq))
    if ids is not None:
        series.index = [int(value) for value in ids]
    else:
        series.index = list(range(1, len(series) + 1))
    return series


def stability_across_K(
    *,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    K_values: Sequence[int],
    clustersize: int,
    clusterradius: float = float("inf"),
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "HOT",
    knn_mode: str = "K_FIXED",
    knn_refill_force: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate cluster stability across adjacent K values.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        Projected (x, y) coordinates.
    adjacency : tuple of (row_ptr, col_idx)
        CSR adjacency.
    observed : sequence of int
        Observed counts.
    expected : sequence of float, optional
        Expected counts. Required for the Poisson model.
    population : sequence of int, optional
        Population sizes. Required for the binomial model.
    K_values : sequence of int
        KNN horizon values to compare (must contain at least one value).
    clustersize : int
        Maximum cluster size (S) used in every scan.
    clusterradius : float, default inf
        Maximum cluster radius.
    model, scanmethod, stattype, clustertype : str
        Forwarded to ``scan()``.
    knn_mode, knn_refill_force : str, bool
        KNN precompute options.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        Keys: ``'K_values'``, ``'runs'`` (per-K stats), ``'pairwise'`` (Jaccard metrics
        between adjacent K values).
    """
    from .api import scan as _scan
    from .helpers import cluster_pool as _cluster_pool

    ks_raw = list(K_values)
    if not ks_raw:
        raise ValueError("K_values must contain at least one K")
    n_areas = int(np.asarray(coords).shape[0])
    ks = [
        validate_integer_parameter(k_value, f"K_values[{idx}]", minimum=1, maximum=n_areas)
        for idx, k_value in enumerate(ks_raw)
    ]
    clustersize_i = validate_integer_parameter(clustersize, "clustersize", minimum=1)

    pools_by_k: dict[int, list[frozenset[int]]] = {}
    runs: list[dict[str, Any]] = []

    for k_value in ks:
        result = _scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            population=population,
            model=model,
            scanmethod=scanmethod,
            stattype=stattype,
            clustertype=clustertype,
            clustersize=clustersize_i,
            clusterradius=clusterradius,
            K=k_value,
            knn_method="graph",
            knn_mode=str(knn_mode),
            knn_refill_force=bool(knn_refill_force),
            simcount=0,
            secondary="ALL",
            seed=seed,
        )
        pools_by_k[k_value] = _cluster_pool(result)

        primary = result.get("primary", result) if isinstance(result, dict) else {}
        stats = None
        pval = None
        if isinstance(primary, dict):
            stats = primary.get("stats")
            pval = primary.get("pval")

        runs.append(
            {
                "K": k_value,
                "pool_size": int(len(pools_by_k[k_value])),
                "stats": (float(stats) if stats is not None else None),
                "pval": (float(pval) if pval is not None else None),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for idx in range(len(ks) - 1):
        k1 = int(ks[idx])
        k2 = int(ks[idx + 1])
        _m = match_clusters_jaccard(pools_by_k[k1], pools_by_k[k2])
        pairwise.append(
            {
                "mean_jaccard": float(_m.mean_jaccard),
                "min_jaccard": float(_m.min_jaccard),
                "coverage": float(_m.coverage),
                "count_diff": int(_m.count_diff),
                "nA": int(_m.nA),
                "nB": int(_m.nB),
                "K1": k1,
                "K2": k2,
            }
        )

    return {"K_values": ks, "runs": runs, "pairwise": pairwise}
