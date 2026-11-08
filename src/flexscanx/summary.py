from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ._utils import extract_rr as _extract_rr
from .types import BothResultDict, ScanResultDict, SeriesResultBase

if TYPE_CHECKING:
    pass

__all__ = [
    "flatten_clusters",
    "clusters_to_dataframe",
    "write_clusters_csv",
    "write_series_clusters_csv",
    "scan_settings_to_text",
    "summarize_result",
]


def _require_pandas():
    try:
        import pandas as _pd  # type: ignore[import-untyped]

        return _pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for DataFrame/CSV helpers. "
            "It is listed as a required dependency; install flexscanx normally: "
            "pip install flexscanx"
        ) from exc


def _series_time_labels(result: dict) -> list:
    per_time = result.get("per_time")
    if per_time is None:
        return []
    try:
        count = len(per_time)
    except TypeError as exc:
        raise ValueError("series result 'per_time' must support len()") from exc

    times = result.get("times")
    if times is None:
        return list(range(count))
    if isinstance(times, (str, bytes)):
        raise ValueError("series result 'times' must be an iterable of labels")

    labels = list(times)
    if len(labels) != count:
        raise ValueError("series result 'times' length must equal len(per_time)")
    return labels


def flatten_clusters(result: dict, include_time: bool = True) -> list[dict]:
    """Return a flat list of clusters from a scan result.

    Works for:
    - single-direction results (dict with 'area' and optional 'secondaries')
    - BOTH results (dict with 'hot' and 'cold' dictionaries)
    - time-series results ({'per_time': [result_t, ...]})

    Each item contains: time (optional), direction, rank, area, n_case, expected,
    max_dist, RR, stats, pval, and name (if present).  For a BOTH result, rows are
    the inference-bearing envelope primary and its secondaries: ``pval`` is the
    two-sided value, while ``pval_hot`` and ``pval_cold`` retain the directional
    one-sided values on the primary row.
    """

    def _rows_one(res: dict, time: int | None) -> list[dict]:
        out: list[dict] = []

        def has_area(d: object) -> bool:
            return (
                isinstance(d, dict) and isinstance(d.get("area"), (list, tuple)) and bool(d["area"])
            )

        def rowify(d: dict, rank: int, direction: str, primary_type: str | None = None) -> dict:
            pval = d.get("pval")
            row = {
                "time": time,
                "direction": direction,
                "rank": rank,
                "area": list(d.get("area", [])),
                "n_case": float(d.get("n_case", 0.0)),
                "expected": float(d.get("expected", 0.0)),
                "max_dist": (None if d.get("max_dist") is None else float(d.get("max_dist", 0.0))),
                "RR": _extract_rr(d),
                "stats": float(d.get("stats", 0.0)),
                "pval": None if pval is None else float(pval),
                "name": d.get("name"),
            }
            if primary_type is not None:
                row["primary_type"] = primary_type
                for key in ("pval_hot", "pval_cold"):
                    value = d.get(key)
                    row[key] = None if value is None else float(value)
            return row

        # BOTH has one inferential result: its primary is ranked against the
        # combined two-sided null.  The retained HOT/COLD sub-results are
        # directional diagnostics, not separate two-sided significance claims.
        if str(res.get("type", "")).upper() == "BOTH" and isinstance(res.get("primary"), dict):
            primary_type = str(res.get("primary_type", "UNKNOWN")).upper()
            primary = res["primary"]
            if has_area(primary):
                out.append(rowify(primary, 1, "BOTH", primary_type))
            for rank, secondary in enumerate(primary.get("secondaries", []), start=2):
                if has_area(secondary):
                    out.append(rowify(secondary, rank, "BOTH", primary_type))
            return out

        # Compatibility fallback for hand-assembled legacy BOTH-like dictionaries
        # that predate the required envelope primary.
        if "hot" in res or "cold" in res:
            if isinstance(res.get("hot"), dict):
                r = 1
                if has_area(res["hot"]):
                    out.append(rowify(res["hot"], r, "HOT"))
                r += 1
                for s in res["hot"].get("secondaries", []):
                    if has_area(s):
                        out.append(rowify(s, r, "HOT"))
                    r += 1
            if isinstance(res.get("cold"), dict):
                r = 1
                if has_area(res["cold"]):
                    out.append(rowify(res["cold"], r, "COLD"))
                r += 1
                for s in res["cold"].get("secondaries", []):
                    if has_area(s):
                        out.append(rowify(s, r, "COLD"))
                    r += 1
            return out

        # Single-direction
        direction = str(res.get("type", "UNKNOWN")).upper()
        if has_area(res):
            out.append(rowify(res, 1, direction))
        r = 2
        for s in res.get("secondaries", []):
            if has_area(s):
                out.append(rowify(s, r, direction))
            r += 1
        return out

    if "per_time" in result:
        flat: list[dict] = []
        for time_label, rt in zip(_series_time_labels(result), result["per_time"]):
            flat.extend(_rows_one(rt, time_label if include_time else None))
        return flat
    else:
        return _rows_one(result, None)


def clusters_to_dataframe(result: dict, include_time: bool = True, sort: bool = True) -> Any:
    """Return a pandas DataFrame from clusters.

    Parameters
    ----------
    result : dict
        Scan result from ``scan()`` or ``scan_series()``.
    include_time : bool, default True
        Include per-time index for time series; non-series rows have ``time=None``.
    sort : bool, default True
        Sort rows by time (if any), direction (HOT, COLD, UNKNOWN), and rank.

    Returns
    -------
    pd.DataFrame
        Columns include time, direction, rank, area, n_case, expected, max_dist,
        RR, stats, and pval. BOTH rows also expose pval_hot/pval_cold as
        directional diagnostics; pval remains the two-sided authority.

    Notes
    -----
    The 'name' field (area names) is intentionally omitted from the DataFrame columns.
    """
    pd = _require_pandas()
    rows = flatten_clusters(result, include_time=include_time)
    cols = [
        "time",
        "direction",
        "rank",
        "area",
        "n_case",
        "expected",
        "max_dist",
        "RR",
        "stats",
        "pval",
    ]
    both_only_cols = ["primary_type", "pval_hot", "pval_cold"]
    if not rows:
        return pd.DataFrame(columns=cols)
    if any(any(column in row for column in both_only_cols) for row in rows):
        cols.extend(both_only_cols)
    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = None
    if sort:
        dir_order = {"HOT": 0, "COLD": 1, "UNKNOWN": 2}
        df["_dir_order"] = df["direction"].map(dir_order).fillna(2)
        if df["time"].notna().any():
            df = df.sort_values(by=["time", "_dir_order", "rank"], kind="mergesort")
        else:
            df = df.sort_values(by=["_dir_order", "rank"], kind="mergesort")
        df = df.drop(columns=["_dir_order"]).reset_index(drop=True)
    return df[cols]


def write_clusters_csv(
    result: dict,
    path: str,
    *,
    include_time: bool = True,
    sort: bool = True,
    index: bool = False,
) -> None:
    """Write flattened clusters to a CSV file (UTF-8).

    Parameters
    ----------
    result : dict
        Scan result dict (single/BOTH or ``{'per_time': [...]}``)
    path : str
        Output CSV path.
    include_time : bool, default True
        Include the 'time' column for time-series results.
    sort : bool, default True
        Sort by time, direction, and rank (stable).
    index : bool, default False
        Include DataFrame index in the file.
    """
    df = clusters_to_dataframe(result, include_time=include_time, sort=sort)
    df.to_csv(path, index=index, encoding="utf-8")


def write_series_clusters_csv(series_result: dict, path: str, *, index: bool = False) -> None:
    """Write a tidy CSV of per-time clusters for scan_series results.

    Produces columns: time, direction, rank, area (list), n_case, expected,
    max_dist, RR, stats, and pval.
    """
    if "per_time" not in series_result:
        raise ValueError("series_result must contain 'per_time'")
    df = clusters_to_dataframe(series_result, include_time=True, sort=True)
    df.to_csv(path, index=index, encoding="utf-8")


def scan_settings_to_text(settings: dict) -> str:
    """Return a one-line human-readable summary of scan_settings.

    Known keys are displayed in a canonical order; any additional keys
    present in the dict are appended alphabetically.
    """
    if not isinstance(settings, dict):
        return "{}"
    preferred_order = [
        "model",
        "scanmethod",
        "stattype",
        "clustertype",
        "clustersize",
        "K",
        "clusterradius",
        "ralpha",
        "simcount",
        "seed",
        "knn_method",
        "knn_mode",
        "knn_refill_force",
        "T",
        "N",
    ]
    seen = set()
    parts = []
    for k in preferred_order:
        if k in settings and settings[k] is not None:
            parts.append(f"{k}={settings[k]}")
            seen.add(k)
    for k in sorted(settings.keys()):
        if k not in seen and settings[k] is not None:
            parts.append(f"{k}={settings[k]}")
    return ", ".join(parts)


def summarize_result(result: ScanResultDict | BothResultDict | SeriesResultBase) -> Any:
    """Summarize a ScanSlice result dict into a DataFrame.

    Parameters
    ----------
    result : dict
        Accepts either:

        - a single-slice result dict with optional 'secondaries'
        - a combined BOTH dict with keys 'hot' and 'cold' (and 'primary')
        - or ``{'per_time': [result_t, ...]}`` returned by ``scan_series``

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Type, NumArea, MaxDist, Case, Expected, RR, Stats, P, Sig.

    Notes
    -----
    - Adds a 'Sig' column with significance stars (\\*\\*\\* < 0.001, \\*\\* < 0.01, \\* < 0.05).
    - Ranks clusters by descending Stats with P ascending as a tiebreaker
      (per JSS summary tables), reassigning Rank = 1..n accordingly. This matches
      rflexscan's printed summary order.
    """
    pd = _require_pandas()

    def one(res: dict) -> Any:
        rows: list[dict] = []

        def has_area(d: object) -> bool:
            return (
                isinstance(d, dict) and isinstance(d.get("area"), (list, tuple)) and bool(d["area"])
            )

        def rowify(
            d: dict,
            type_override: str | None = None,
            directional_pvals: tuple[object, object] | None = None,
        ) -> dict:
            t = type_override if type_override is not None else d.get("type")
            row = {
                "Type": (str(t).upper() if t is not None else None),
                "NumArea": len(d.get("area", [])),
                "MaxDist": round(float(d.get("max_dist", 0.0)), 6),
                "Case": float(d.get("n_case", 0.0)),
                "Expected": float(d.get("expected", 0.0)),
                "RR": _extract_rr(d),
                "Stats": float(d.get("stats", 0.0)),
                "P": (
                    float(d.get("pval", float("nan")))
                    if d.get("pval") is not None
                    else float("nan")
                ),
            }
            if directional_pvals is not None:
                hot_pval, cold_pval = directional_pvals
                row["P_hot"] = float(cast(Any, hot_pval)) if hot_pval is not None else float("nan")
                row["P_cold"] = (
                    float(cast(Any, cold_pval)) if cold_pval is not None else float("nan")
                )
            return row

        if str(res.get("type", "")).upper() == "BOTH" and isinstance(res.get("primary"), dict):
            primary = res["primary"]
            directional_pvals = (primary.get("pval_hot"), primary.get("pval_cold"))
            if has_area(primary):
                rows.append(
                    rowify(primary, type_override="BOTH", directional_pvals=directional_pvals)
                )
            for secondary in primary.get("secondaries", []):
                if has_area(secondary):
                    rows.append(rowify(secondary, type_override="BOTH"))
            df = pd.DataFrame(rows)
        elif "hot" in res or "cold" in res:
            if isinstance(res.get("hot"), dict):
                if has_area(res["hot"]):
                    rows.append(rowify(res["hot"], type_override="HOT"))
                for s in res["hot"].get("secondaries", []):
                    if has_area(s):
                        rows.append(rowify(s, type_override="HOT"))
            if isinstance(res.get("cold"), dict):
                if has_area(res["cold"]):
                    rows.append(rowify(res["cold"], type_override="COLD"))
                for s in res["cold"].get("secondaries", []):
                    if has_area(s):
                        rows.append(rowify(s, type_override="COLD"))
            df = pd.DataFrame(rows)
        else:
            t_override = str(res.get("type", "")).upper() or None
            if has_area(res):
                rows.append(rowify(res, type_override=t_override))
            for s in res.get("secondaries", []):
                if has_area(s):
                    rows.append(rowify(s, type_override=t_override))
            df = pd.DataFrame(rows)

        def _stars(p: float) -> str:
            try:
                if pd.isna(p):
                    return ""
                if p < 0.001:
                    return "***"
                if p < 0.01:
                    return "**"
                if p < 0.05:
                    return "*"
                return ""
            except Exception:
                return ""

        if df.empty:
            return df

        df["Sig"] = df["P"].map(_stars)
        # Sort by Stats desc, P asc; NaN p-values (MC disabled) sort last.
        # Using inf rather than 1.1 so any legitimately large p-value is still
        # ranked correctly and the sentinel can never collide with a real value.
        df["_Psort"] = df["P"].fillna(float("inf"))
        df = df.sort_values(
            by=["Stats", "_Psort"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        df = df.drop(columns=["_Psort"])
        for _col in ("Expected", "RR", "Stats", "MaxDist"):
            if _col in df.columns:
                try:
                    df[_col] = df[_col].astype(float).round(3)
                except Exception:
                    pass
        return df

    result_dict = cast(dict[str, Any], result)
    if "per_time" in result_dict:
        frames = [one(cast(dict[str, Any], rt)) for rt in result_dict["per_time"]]
        out = pd.concat(frames, keys=_series_time_labels(result_dict), names=["time", "cluster"])
        return out
    else:
        return one(result_dict)
