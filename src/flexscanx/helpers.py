from __future__ import annotations

import math
from typing import Any, Sequence, cast

from ._utils import validate_integer_parameter, warn_at_caller
from .summary import flatten_clusters as _flatten_clusters

__all__ = [
    "ensure_projected",
    "cluster_pool",
    "filter_significant_clusters",
    "merge_overlapping_clusters",
    "pool_membership_series",
    "pool_to_geodataframe",
]


def ensure_projected(gdf, target_crs: str | None = None, warn_only: bool = True):
    """Ensure a GeoDataFrame is in a projected CRS (Cartesian) for distance-based ops.

    - If gdf.crs is geographic (degrees), emit a warning by default.
    - If target_crs is provided and warn_only is False, return gdf.to_crs(target_crs).
    - Otherwise return the original gdf.
    """
    try:
        crs = getattr(gdf, "crs", None)
        if crs is None:
            return gdf
        # GeoPandas/pyproj CRS provides is_geographic in recent versions
        is_geo = False
        try:
            is_geo = bool(getattr(crs, "is_geographic", False))
        except (AttributeError, TypeError):
            # Fallback heuristic: EPSG:4326 or units in degrees
            is_geo = str(crs).upper().find("4326") >= 0
        if is_geo:
            warn_at_caller(
                "GeoDataFrame appears to be in a geographic CRS (degrees). "
                "For spatial scan statistics, preproject to a projected CRS (meters)."
            )
            if (target_crs is not None) and (warn_only is False):
                return gdf.to_crs(target_crs)
        return gdf
    except (AttributeError, TypeError):
        return gdf


def cluster_pool(result: Any, exact_dedup: bool = True) -> list[frozenset[int]]:
    """Extract a pool of clusters (as frozensets of integer area IDs).

    - Pools HOT and COLD; ignores primary/secondary distinction.
    - Removes empty clusters and exact duplicate sets when exact_dedup=True.
    - For overlap-based deduplication, apply merge_overlapping_clusters(...) explicitly.
    """
    # Tabular/significance surfaces deliberately flatten a BOTH envelope to its
    # two-sided primary.  This helper has a different, long-standing contract:
    # it is a diagnostic pool of every HOT and COLD branch cluster.
    if (
        isinstance(result, dict)
        and str(result.get("type", "")).upper() == "BOTH"
        and (isinstance(result.get("hot"), dict) or isinstance(result.get("cold"), dict))
    ):
        rows = []
        for direction in ("hot", "cold"):
            branch = result.get(direction)
            if isinstance(branch, dict):
                rows.extend(_flatten_clusters(branch, include_time=False))
    else:
        rows = _flatten_clusters(result, include_time=False)
    out: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    for r in rows:
        area = r.get("area") or []
        s = frozenset(int(v) for v in area if v is not None)
        if not s:
            continue
        if exact_dedup:
            if s in seen:
                continue
            seen.add(s)
        out.append(s)
    return out


def _passes_significance(d: dict, *, alpha: float | None, tau: float | None) -> bool:
    if alpha is not None:
        if not _passes_pvalue_threshold(d.get("pval"), alpha):
            return False
    if tau is not None:
        try:
            st = d.get("stats", None)
            if st is None:
                return False
            stats_f = float(st)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"cluster stats must be numeric or None, got {st!r}") from exc
        if not math.isfinite(stats_f):
            raise ValueError(f"cluster stats must be finite, got {st!r}")
        return stats_f >= tau
    return True


def _passes_pvalue_threshold(pval: object, alpha: float | None) -> bool:
    """Return whether a tested p-value passes an optional significance threshold.

    With a threshold, ``None`` denotes an untested cluster (the standard
    ``simcount=0`` result) and is excluded. Invalid numeric values are input
    errors rather than silently treated as a significance decision.
    """
    if alpha is None:
        return True
    if pval is None:
        return False
    try:
        p_f = float(cast(Any, pval))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cluster pval must be numeric or None, got {pval!r}") from exc
    if not math.isfinite(p_f) or not 0.0 <= p_f <= 1.0:
        raise ValueError(f"cluster pval must be finite and in [0,1], got {pval!r}")
    return p_f <= alpha


def filter_significant_clusters(
    result_or_pool: dict | list[frozenset[int]],
    *,
    alpha: float | None = 0.05,
    tau: float | None = None,
    min_size: int | None = None,
    max_radius: float | None = None,
) -> list[frozenset[int]]:
    """Return cluster sets filtered by significance/size/radius.

    - If a result dict is provided, flattens clusters and filters by p<=alpha and stats>=tau,
      then enforces max_radius using each cluster's max_dist metadata.
    - If a pool (list of frozensets) is provided, only min_size can be applied because
      radius metadata is not available on bare frozensets.
    """
    max_radius_f: float | None = None
    alpha_f: float | None = None
    tau_f: float | None = None
    min_size_i: int | None = None
    if alpha is not None:
        alpha_f = float(alpha)
        if not math.isfinite(alpha_f):
            raise ValueError(f"alpha must be finite, got {alpha!r}")
        if not 0.0 <= alpha_f <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if tau is not None:
        tau_f = float(tau)
        if not math.isfinite(tau_f):
            raise ValueError(f"tau must be finite, got {tau!r}")
    if min_size is not None:
        min_size_i = validate_integer_parameter(min_size, "min_size", minimum=1)
    if max_radius is not None:
        max_radius_f = float(max_radius)
        if not math.isfinite(max_radius_f):
            raise ValueError(f"max_radius must be finite, got {max_radius!r}")
        if max_radius_f < 0.0:
            raise ValueError("max_radius must be >= 0 when provided")

    if isinstance(result_or_pool, list):
        if max_radius_f is not None:
            raise ValueError(
                "max_radius requires a scan result dict with per-cluster max_dist metadata; "
                "a bare pool of frozensets does not carry radius information."
            )
        pool = [frozenset(s) for s in result_or_pool]
    else:
        rows = _flatten_clusters(result_or_pool, include_time=False)
        pool = []
        for d in rows:
            if _passes_significance(d, alpha=alpha_f, tau=tau_f):
                if max_radius_f is not None:
                    max_dist = d.get("max_dist", None)
                    if max_dist is None:
                        raise ValueError(
                            "max_radius filtering requires cluster max_dist metadata in the "
                            "scan result."
                        )
                    max_dist_f = float(max_dist)
                    if not math.isfinite(max_dist_f) or max_dist_f < 0.0:
                        raise ValueError(
                            f"cluster max_dist must be finite and non-negative, got {max_dist!r}"
                        )
                    if max_dist_f > max_radius_f:
                        continue
                s = frozenset(int(v) for v in d.get("area", []) if v is not None)
                if s:
                    pool.append(s)
    if min_size_i is not None:
        pool = [s for s in pool if len(s) >= min_size_i]
    return pool


def merge_overlapping_clusters(
    pool: Sequence[frozenset[int]], *, jaccard_threshold: float = 0.8
) -> list[frozenset[int]]:
    """Greedy merge/deduplicate clusters with high Jaccard overlap.

    Keeps a representative from groups of near-duplicates.
    """
    try:
        threshold = float(jaccard_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("jaccard_threshold must be finite and in [0,1]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"jaccard_threshold must be finite and in [0,1], got {jaccard_threshold!r}"
        )

    out: list[frozenset[int]] = []
    taken: list[bool] = [False] * len(pool)
    for i, A in enumerate(pool):
        if taken[i]:
            continue
        out.append(A)
        for j in range(i + 1, len(pool)):
            if taken[j]:
                continue
            B = pool[j]
            inter = len(A & B)
            union = len(A | B)
            jac = (inter / union) if union > 0 else 1.0
            if jac >= threshold:
                taken[j] = True
    return out


def pool_membership_series(pool: Sequence[frozenset[int]], ids: Sequence[int]) -> Any:
    """Return a pandas Series mapping id -> membership count across the pool.

    - ids: an iterable of integer area IDs (for example, `gdf[id_col]`) used as the index order.
    - pool: list of frozenset area IDs representing clusters.
    """
    try:
        import pandas as _pd  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("pandas is required for pool_membership_series") from e
    counts = {int(i): 0 for i in ids}
    for s in pool:
        for i in s:
            if int(i) in counts:
                counts[int(i)] += 1
    return _pd.Series(counts)


def _integer_area_ids(gdf: Any, id_col: str) -> list[int]:
    """Return the ``id_col`` values of *gdf* as integer area IDs, naming the column on failure.

    Integers and their decimal string form are accepted (string identifier columns
    that hold numbers are common in exported boundary files); anything else raises
    a ``ValueError`` that names the column and the offending value instead of
    Python's bare ``invalid literal for int()`` message (FXA-190).
    """
    if id_col not in getattr(gdf, "columns", []):
        raise ValueError(
            f"id_col '{id_col}' not found in geodataframe columns: {list(gdf.columns)}"
        )
    ids: list[int] = []
    for row, value in enumerate(gdf[id_col].tolist()):
        if isinstance(value, bool):
            raise ValueError(
                f"id_col '{id_col}' must contain integer area IDs matching the scan result; "
                f"row {row} has the boolean {value!r}"
            )
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"id_col '{id_col}' must contain integer area IDs (or their decimal string "
                f"form) matching the scan result; row {row} has {value!r}"
            ) from None
    return ids


def pool_to_geodataframe(
    gdf,
    pool: Sequence[frozenset[int]],
    id_col: str,
    *,
    member_col: str = "PoolMember",
    count_col: str = "PoolCount",
):
    """Attach pool membership columns to a GeoDataFrame.

    - id_col: column containing integer area IDs matching cluster area IDs.
    - member_col: name of 0/1 column indicating membership in at least one cluster.
    - count_col: name of column with the number of clusters each area belongs to.

    Raises
    ------
    ValueError
        If id_col is not found, holds a value that is not an integer area ID (or
        its decimal string form), or if no cluster IDs match GeoDataFrame IDs

    Warns
    -----
    UserWarning
        If some cluster area IDs are not present in the GeoDataFrame id space
    """
    ids = _integer_area_ids(gdf, id_col)
    gdf_id_set = set(ids)
    pool_id_set = {int(v) for s in pool for v in s}
    s = pool_membership_series(pool, ids)
    out = gdf.copy()
    out[count_col] = [int(s.get(area_id, 0)) for area_id in ids]
    out[member_col] = (out[count_col] > 0).astype(int)

    unmatched_pool_ids = sorted(pool_id_set - gdf_id_set)

    if unmatched_pool_ids and len(unmatched_pool_ids) == len(pool_id_set):
        # Complete mismatch - likely wrong id_col or IDs don't match
        raise ValueError(
            f"No cluster IDs matched GeoDataFrame {id_col} column. "
            f"Ensure GeoDataFrame {id_col} uses integer area IDs matching the scan result."
        )
    elif unmatched_pool_ids:
        # Partial mismatch - some cluster IDs do not exist in the GeoDataFrame
        warn_at_caller(
            f"{len(unmatched_pool_ids)} cluster area IDs were not found in GeoDataFrame "
            f"{id_col}: {unmatched_pool_ids[:10]}. "
            f"This may indicate an ID mismatch between the scan result and GeoDataFrame IDs."
        )

    return out
