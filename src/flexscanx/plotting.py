from __future__ import annotations

import math
from typing import Any, Iterable, Sequence, cast

from ._utils import extract_rr as _extract_rr
from ._utils import validate_integer_parameter
from .helpers import _integer_area_ids, _passes_pvalue_threshold

__all__ = [
    "plot_cluster_membership",
    "plot_both_membership",
    "plot_all_clusters",
    "plot_all_both_clusters",
    "choropleth",
    "values_by_area_from_obsexp",
    "plot_values_from_obsexp",
]


def _validate_probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number in [0,1]") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {result}")
    return result


def _validate_nonnegative_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _validate_optional_finite(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ensure_gdf(geo: str | Any) -> Any:
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "geopandas is required for plotting. Install it with: pip install flexscanx[geo]"
        ) from exc
    if isinstance(geo, str):
        return gpd.read_file(geo)
    # Ensure copy to avoid mutating caller frames unexpectedly
    try:
        return geo.copy()
    except Exception:
        return geo


def _require_pyplot() -> Any:
    """Import pyplot with the public optional-dependency diagnostic."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install flexscanx[plot]"
        ) from exc
    return plt


def _close_owned_figure(fig: Any, *, created: bool, return_fig: bool) -> None:
    """Release a pyplot-managed figure only when this call created it."""
    if created and not return_fig:
        _require_pyplot().close(fig)


def _has_mapclassify() -> bool:
    try:
        import mapclassify as _mc  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_cmaps(
    n: int, cmaps_in: Sequence[Any] | None = None, palette: str | None = None
) -> list:
    """Return a list of n cluster colormaps (transparent for 0, solid for 1).

    palette: preferred palette name
      - "glasbey": try colorcet.glasbey
      - "colorblind": seaborn colorblind
      - "okabe-ito": Okabe–Ito set (excluding black)
      - "tab20": matplotlib tab20
      - None: try glasbey -> colorblind -> okabe-ito -> tab20
    """
    _plt = _require_pyplot()
    from matplotlib.colors import ListedColormap, to_rgba

    if cmaps_in is not None:
        resolved_cmaps = list(cmaps_in)
        if n > 0 and not resolved_cmaps:
            raise ValueError("cmaps must not be empty when clusters are displayed")
        if len(resolved_cmaps) < n:
            resolved_cmaps = resolved_cmaps + [
                resolved_cmaps[i % len(resolved_cmaps)] for i in range(n - len(resolved_cmaps))
            ]
        return resolved_cmaps

    valid_palettes = {None, "glasbey", "colorblind", "okabe-ito", "tab20"}
    if palette not in valid_palettes:
        raise ValueError(
            "palette must be one of None, 'glasbey', 'colorblind', 'okabe-ito', or 'tab20'"
        )

    colors: list = []

    def _okabe_ito_list():
        return [
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
        ]

    # Choose based on palette preference first
    if palette == "glasbey":
        try:
            import colorcet as cc  # type: ignore

            # Use modulo to avoid IndexError if n exceeds palette length
            plen = len(cc.glasbey)
            colors = [cc.glasbey[i % plen] for i in range(max(0, n))] if plen > 0 else []
        except ImportError:
            colors = []
    elif palette == "colorblind":
        try:
            import seaborn as sns  # type: ignore

            colors = list(sns.color_palette("colorblind", n_colors=n))
        except ImportError:
            colors = []
    elif palette == "okabe-ito":
        oi = _okabe_ito_list()
        colors = [oi[i % len(oi)] for i in range(n)]
    elif palette == "tab20":
        base = _plt.get_cmap("tab20", max(1, n))
        colors = [base(i) for i in range(n)]

    # If still not enough / no palette set, try cascade
    if len(colors) < n and palette is None:
        try:
            import colorcet as cc  # type: ignore

            plen = len(cc.glasbey)
            colors = [cc.glasbey[i % plen] for i in range(max(0, n))] if plen > 0 else []
        except ImportError:
            colors = []
        if len(colors) < n:
            try:
                import seaborn as sns  # type: ignore

                needed = n - len(colors)
                colors.extend(sns.color_palette("colorblind", n_colors=needed))
            except ImportError:
                pass

    # Complete the documented default cascade with one Okabe-Ito pass, then
    # tab20 for any further colours. Explicit palette choices retain their
    # palette's repeat behaviour.
    if len(colors) < n:
        oi = _okabe_ito_list()
        if palette is None:
            colors.extend(oi[: n - len(colors)])
        else:
            while len(colors) < n:
                colors.append(oi[len(colors) % len(oi)])

    if len(colors) < n:
        base = _plt.get_cmap("tab20", max(1, n))
        colors.extend(base(i) for i in range(len(colors), n))

    # Build ListedColormap per cluster with transparent '0' and solid '1'.
    resolved_cmaps = []
    for c in colors[:n]:
        rgba = to_rgba(c)
        resolved_cmaps.append(ListedColormap([(0, 0, 0, 0), rgba]))
    return resolved_cmaps


def _cluster_area_ids(gdf: Any, id_col: str, clusters: Sequence[Sequence[int]]) -> list[int]:
    """Return ``gdf[id_col]`` as integer area IDs for a membership plot (FXA-195).

    Integers and their decimal string form convert; any other value raises a
    ``ValueError`` naming the column, row and value (``helpers._integer_area_ids``).
    When at least one cluster has areas and none of them appears in the column,
    the map would show nothing at all, so that is rejected too, mirroring
    ``pool_to_geodataframe``.  A cluster that is only partly present is drawn
    partly, as before.
    """
    if id_col not in gdf.columns:
        raise ValueError(f"id_col '{id_col}' not found in geodata columns: {list(gdf.columns)}")
    ids = _integer_area_ids(gdf, id_col)
    wanted = {int(v) for areas in clusters for v in areas if v is not None}
    if wanted and not (wanted & set(ids)):
        raise ValueError(
            f"No cluster IDs matched GeoDataFrame {id_col} column. Ensure GeoDataFrame {id_col} "
            "uses the 1-based integer area IDs of the scan result "
            f"(cluster IDs {sorted(wanted)[:5]}..., column holds {sorted(set(ids))[:5]}...)."
        )
    return ids


def plot_cluster_membership(
    geodata: str | Any,
    result: dict,
    id_col: str,
    *,
    which: str = "primary",  # one of {"primary","hot","cold"}
    ax: Any | None = None,
    cmap: str = "Reds",
    edgecolor: str = "white",
    linewidth: float = 0.5,
    missing_color: str = "#f0f0f0",
    legend: bool = False,
    title: str | None = None,
    alpha: float = 1.0,
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Any:
    """Plot a choropleth highlighting cluster membership.

    Parameters
    - geodata: path to a vector dataset (e.g., shapefile/GeoPackage/GeoJSON) or a GeoDataFrame
    - result: dict returned by flexscanx.api.scan (single slice) or a BOTH structure
    - id_col: name of the column in geodata whose integer values match result area IDs
    - which: choose from {"primary","hot","cold"} for BOTH results; ignored for single-direction
    - ax: optional Matplotlib Axes to plot on
    - cmap: colormap name for membership shading
    - edgecolor, linewidth: boundary styling
    - missing_color: facecolor for geometries with missing id
    - legend: whether to draw a legend for membership
    - title: optional title

    Returns
    - Matplotlib Axes by default; when return_fig=True, returns (Figure, Axes)
    """
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    alpha = _validate_probability(alpha, "alpha")
    linewidth = _validate_nonnegative_finite(linewidth, "linewidth")

    plt = _require_pyplot()

    gdf = _ensure_gdf(geodata)

    # Extract area list in the same ID space used by id_col
    clus: Sequence[int] | None = None
    if result.get("type") == "BOTH":
        pick = which.lower()
        if pick not in ("primary", "hot", "cold"):
            raise ValueError("which must be one of 'primary', 'hot', 'cold'")
        src = result.get("primary") if pick == "primary" else result.get(pick)
        if isinstance(src, dict):
            clus = src.get("area")
    else:
        clus = result.get("area")
    if not isinstance(clus, (list, tuple)):
        raise ValueError("result does not contain an 'area' list to visualize")

    sel = set(int(v) for v in clus)
    # membership is 1 if geometry's id is in cluster else 0; a non-integer identifier
    # column or a cluster entirely absent from the geodata is an error (FXA-195)
    gdf["_area_id"] = _cluster_area_ids(gdf, id_col, [clus])
    gdf["InCluster"] = [int(area_id in sel) for area_id in gdf["_area_id"]]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        assert ax is not None
        fig = ax.figure
    assert ax is not None
    gdf.plot(
        column="InCluster",
        ax=ax,
        cmap=cmap,
        legend=legend,
        edgecolor=edgecolor,
        linewidth=linewidth,
        missing_kwds={"color": missing_color},
        categorical=True,
        alpha=alpha,
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, ax
    _close_owned_figure(fig, created=created, return_fig=return_fig)
    return ax


def plot_both_membership(
    geodata: str | Any,
    result_both: dict,
    id_col: str,
    *,
    titles: tuple[str, str] | None = ("HOT", "COLD"),
    cmaps: tuple[str, str] | None = ("Reds", "Blues"),
    edgecolor: str = "white",
    linewidth: float = 0.5,
    alpha: float = 1.0,
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Any:
    """Plot side-by-side membership maps for BOTH result (HOT, COLD)."""
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    alpha = _validate_probability(alpha, "alpha")
    linewidth = _validate_nonnegative_finite(linewidth, "linewidth")

    plt = _require_pyplot()

    if not isinstance(result_both, dict) or result_both.get("type") != "BOTH":
        raise ValueError("result_both must be a BOTH result (type=='BOTH')")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t_hot, t_cold = titles or ("HOT", "COLD")
    c_hot, c_cold = cmaps or ("Reds", "Blues")

    plot_cluster_membership(
        geodata,
        result_both,
        id_col,
        which="hot",
        ax=axes[0],
        cmap=c_hot,
        edgecolor=edgecolor,
        linewidth=linewidth,
        title=t_hot,
        alpha=alpha,
    )
    plot_cluster_membership(
        geodata,
        result_both,
        id_col,
        which="cold",
        ax=axes[1],
        cmap=c_cold,
        edgecolor=edgecolor,
        linewidth=linewidth,
        title=t_cold,
        alpha=alpha,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, axes
    _close_owned_figure(fig, created=True, return_fig=return_fig)
    return axes


def plot_all_both_clusters(
    geodata: str | Any,
    result_both: dict,
    id_col: str,
    *,
    overlay: bool = True,
    hot_cmap: str = "Reds",
    cold_cmap: str = "Blues",
    alpha_hot: float = 0.6,
    alpha_cold: float = 0.6,
    titles: tuple[str, str] | None = ("HOT clusters", "COLD clusters"),
    show_legend: bool = True,
    legend_loc: str = "upper right",
    significance_alpha: float | None = 0.05,
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
):
    """Plot all HOT and COLD clusters.

    overlay=True: overlay union of HOT and union of COLD on a single axis (different colors).
    overlay=False: two subplots (HOT union on left, COLD union on right).

    significance_alpha: if not None, only tested clusters with p <= alpha are
    included in the union. Clusters without p-values are shown only when it is None.
    """
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    alpha_hot = _validate_probability(alpha_hot, "alpha_hot")
    alpha_cold = _validate_probability(alpha_cold, "alpha_cold")
    if significance_alpha is not None:
        significance_alpha = _validate_probability(significance_alpha, "significance_alpha")

    plt = _require_pyplot()

    if not isinstance(result_both, dict) or result_both.get("type") != "BOTH":
        raise ValueError("result_both must be a BOTH result (type=='BOTH')")

    def _union_area(d: dict) -> list:
        areas: set[int] = set()

        # Helper to decide if a cluster passes the significance filter
        def _pass(cl: dict) -> bool:
            return _passes_pvalue_threshold(cl.get("pval"), significance_alpha)

        if isinstance(d, dict) and "area" in d and _pass(d):
            areas.update(int(v) for v in d["area"])
        for s in d.get("secondaries", []):
            if isinstance(s, dict) and "area" in s and _pass(s):
                areas.update(int(v) for v in s["area"])
        return sorted(areas)

    hot = result_both.get("hot")
    cold = result_both.get("cold")
    if not isinstance(hot, dict) or not isinstance(cold, dict):
        raise ValueError("result_both must contain HOT and COLD result dictionaries")
    union_hot = {"area": _union_area(hot)}
    union_cold = {"area": _union_area(cold)}

    if overlay:
        gdf_all = _ensure_gdf(geodata)
        area_ids = _cluster_area_ids(gdf_all, id_col, [union_hot["area"], union_cold["area"]])
        fig, ax = plt.subplots(figsize=(8, 6))

        def _draw_union(union: dict, cmap: str, alpha: float) -> None:
            selected = set(int(value) for value in union["area"])
            if not selected:
                return
            subset = gdf_all[gdf_all["_area_id"].isin(selected)]
            if subset.empty:
                return
            subset.plot(
                ax=ax,
                color=plt.get_cmap(cmap)(0.8),
                edgecolor="white",
                linewidth=0.5,
                alpha=alpha,
            )

        gdf_all["_area_id"] = area_ids
        gdf_all.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=0.5)
        _draw_union(union_hot, hot_cmap, alpha_hot)
        _draw_union(union_cold, cold_cmap, alpha_cold)
        ax.set_axis_off()
        if titles:
            ax.set_title(f"{titles[0]} / {titles[1]}")
        else:
            ax.set_title("HOT/COLD clusters")
        if show_legend:
            import matplotlib.patches as mpatches

            patches = [
                mpatches.Patch(color=plt.get_cmap(hot_cmap)(0.8), label="HOT union"),
                mpatches.Patch(color=plt.get_cmap(cold_cmap)(0.8), label="COLD union"),
            ]
            ax.legend(handles=patches, loc=cast(Any, legend_loc))
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if return_fig:
            return fig, ax
        _close_owned_figure(fig, created=True, return_fig=return_fig)
        return ax

    # Two subplots
    gdf_check = _ensure_gdf(geodata)
    _cluster_area_ids(gdf_check, id_col, [union_hot["area"], union_cold["area"]])
    fig, (ax_hot, ax_cold) = plt.subplots(1, 2, figsize=(12, 5))
    plot_cluster_membership(
        gdf_check,
        union_hot,
        id_col,
        ax=ax_hot,
        cmap=hot_cmap,
        alpha=alpha_hot,
        title=titles[0] if titles else "HOT clusters",
        legend=False,
    )
    plot_cluster_membership(
        gdf_check,
        union_cold,
        id_col,
        ax=ax_cold,
        cmap=cold_cmap,
        alpha=alpha_cold,
        title=titles[1] if titles else "COLD clusters",
        legend=False,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, (ax_hot, ax_cold)
    _close_owned_figure(fig, created=True, return_fig=return_fig)
    return ax_hot, ax_cold


def choropleth(
    geodata: str | Any,
    values_by_id: Iterable[tuple[int, float]] | Any,
    id_col: str,
    *,
    ax: Any | None = None,
    cmap: str = "OrRd",
    edgecolor: str = "white",
    linewidth: float = 0.5,
    missing_color: str = "#f0f0f0",
    legend: bool = True,
    legend_kwds: dict | None = None,
    title: str | None = None,
    scheme: str | None = None,  # e.g., "quantiles", "equal_interval"
    k: int = 5,
    classification_kwds: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Any:
    """General choropleth by numeric values keyed by 1-based region id.

    Unkeyed sequences (for example, lists, numpy arrays, or a default-index pandas Series)
    are interpreted in 1-based area-id order, so the first value maps to area id 1.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series({1: 2.5, 2: 0.9, 3: 1.2})  # id -> value
    >>> choropleth(geodata, s, id_col="id")
    """
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    k = validate_integer_parameter(k, "k", minimum=1)
    linewidth = _validate_nonnegative_finite(linewidth, "linewidth")
    vmin = _validate_optional_finite(vmin, "vmin")
    vmax = _validate_optional_finite(vmax, "vmax")
    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError(f"vmin must be <= vmax, got vmin={vmin}, vmax={vmax}")

    plt = _require_pyplot()
    import pandas as pd  # type: ignore[import-untyped]  # lazy import

    gdf = _ensure_gdf(geodata)

    if id_col not in gdf.columns:
        raise ValueError(f"id_col '{id_col}' not found in geodata columns: {list(gdf.columns)}")

    if hasattr(values_by_id, "items") and not isinstance(values_by_id, pd.Series):
        # dict-like
        values = pd.Series(dict(values_by_id))
    else:
        values = pd.Series(values_by_id)  # type: ignore[arg-type]
        # Plain sequence-like input gets a default 0-based RangeIndex from pandas,
        # but scan area ids are 1-based throughout the public API.
        try:
            if isinstance(values.index, pd.RangeIndex):
                is_zero_based = (
                    values.index.start == 0
                    and values.index.step == 1
                    and values.index.stop == len(values)
                )
            else:
                is_zero_based = list(values.index) == list(range(len(values)))
        except Exception:
            is_zero_based = False
        if is_zero_based:
            values.index = pd.RangeIndex(start=1, stop=len(values) + 1, step=1)

    # keys are 1-based ids; a non-integer identifier column is reported by name (FXA-190)
    gdf["_area_id"] = _integer_area_ids(gdf, id_col)
    gdf = gdf.merge(
        values.rename("_val").to_frame(),
        left_on="_area_id",
        right_index=True,
        how="left",
    )

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        assert ax is not None
        fig = ax.figure
    assert ax is not None

    plot_kwargs = dict(
        column="_val",
        ax=ax,
        cmap=cmap,
        legend=legend,
        legend_kwds=legend_kwds or {},
        edgecolor=edgecolor,
        linewidth=linewidth,
        missing_kwds={"color": missing_color},
    )
    if vmin is not None:
        plot_kwargs["vmin"] = vmin
    if vmax is not None:
        plot_kwargs["vmax"] = vmax
    if scheme is not None:
        if not _has_mapclassify():
            raise ImportError("scheme= requires 'mapclassify' (pip install flexscanx[geo])")
        plot_kwargs.update(dict(scheme=scheme, k=k, classification_kwds=classification_kwds))

    gdf.plot(**plot_kwargs)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, ax
    _close_owned_figure(fig, created=created, return_fig=return_fig)
    return ax


def values_by_area_from_obsexp(
    observed: Sequence[int],
    expected: Sequence[float],
    *,
    metric: str = "oe_ratio",
) -> Any:
    """Compute per-area values keyed by 1-based area id.

    metric options:
    - 'oe_ratio': observed/expected (NaN when expected<=0)
    - 'excess': observed - expected
    - 'observed': observed as float
    - 'expected': expected
    """
    import math as _math

    import pandas as pd  # type: ignore[import-untyped]

    if len(observed) != len(expected):
        raise ValueError("observed and expected must have the same length")

    idx = range(1, len(observed) + 1)
    obs = [float(o) for o in observed]
    exp = [float(e) for e in expected]

    if metric == "oe_ratio":
        vals = [(_o / _e) if _e > 0.0 else _math.nan for _o, _e in zip(obs, exp)]
    elif metric == "excess":
        vals = [(_o - _e) for _o, _e in zip(obs, exp)]
    elif metric == "observed":
        vals = obs
    elif metric == "expected":
        vals = exp
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return pd.Series(vals, index=list(idx))


def plot_values_from_obsexp(
    geodata: str | Any,
    observed: Sequence[int],
    expected: Sequence[float],
    id_col: str,
    *,
    metric: str = "oe_ratio",
    scheme: str | None = None,
    k: int = 5,
    cmap: str = "OrRd",
    title: str | None = None,
    ax: Any | None = None,
    legend: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Any:
    """Convenience: compute values from observed/expected and plot as a choropleth."""
    vals = values_by_area_from_obsexp(observed, expected, metric=metric)
    return choropleth(
        geodata,
        vals,
        id_col=id_col,
        ax=ax,
        cmap=cmap,
        legend=legend,
        title=title or metric,
        scheme=scheme,
        k=k,
        vmin=vmin,
        vmax=vmax,
        save_path=save_path,
        dpi=dpi,
        return_fig=return_fig,
    )


def plot_all_clusters(
    geodata: str | Any,
    result: dict,
    id_col: str,
    *,
    overlay: bool = False,
    max_clusters: int | None = None,
    cmaps: Sequence[Any] | None = None,
    cols: int = 3,
    alpha: float = 0.6,
    title_prefix: str = "Cluster",
    label_stats: bool = True,
    show_legend: bool = False,
    significance_alpha: float | None = 0.05,
    palette: str | None = "okabe-ito",
    save_path: str | None = None,
    dpi: int = 300,
    edge_color: str = "#222222",
    edge_width: float = 0.25,
    base_edge_color: str = "#bbbbbb",
    base_edge_width: float = 0.15,
    only_non_significant: bool = False,
    return_fig: bool = False,
):
    """Plot primary + all secondaries.

    overlay=False: grid of subplots; overlay=True: overlay on a single axis.

    significance_alpha: if not None, only tested clusters with p <= alpha are
    included. Clusters without p-values (e.g., when simcount=0) are shown only
    when significance_alpha is None.

    Color consistency: if you pass a 'cmaps' sequence, the same list will be used
    for both overlay and grid modes, ensuring the same cluster i uses the same color
    in both representations. If not provided, colors are auto-generated using
    color-blind friendly palettes and applied consistently within this call.

    palette: preferred palette ("okabe-ito" | "colorblind" | "glasbey" | "tab20")
    save_path: if provided, saves the figure to this path (PNG/PDF) with given dpi.

    edge_color/edge_width: style for cluster polygon borders (thin, dark by default)
    base_edge_color/base_edge_width: style for base outlines under the fills.

    only_non_significant: if True and significance_alpha is set, include only clusters
      with p-value strictly greater than the threshold (exclude missing p-values).
    """
    cols = validate_integer_parameter(cols, "cols", minimum=1)
    if max_clusters is not None:
        max_clusters = validate_integer_parameter(max_clusters, "max_clusters", minimum=0)
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    alpha = _validate_probability(alpha, "alpha")
    if significance_alpha is not None:
        significance_alpha = _validate_probability(significance_alpha, "significance_alpha")

    plt = _require_pyplot()

    if not isinstance(result, dict):
        raise ValueError("plot_all_clusters requires a single scan result dictionary")
    if "per_time" in result:
        raise ValueError(
            "plot_all_clusters requires a single scan result, not a scan_series result"
        )
    if str(result.get("type", "")).upper() == "BOTH":
        raise ValueError(
            "plot_all_clusters requires a single scan result; use plot_all_both_clusters"
        )
    if not isinstance(result.get("area"), (list, tuple)):
        raise ValueError("plot_all_clusters requires a single scan result with an 'area' list")
    secondaries = result.get("secondaries", [])
    if not isinstance(secondaries, (list, tuple)) or any(
        not isinstance(cluster, dict) or not isinstance(cluster.get("area"), (list, tuple))
        for cluster in secondaries
    ):
        raise ValueError(
            "plot_all_clusters requires single-result secondary clusters with 'area' lists"
        )
    if cmaps is not None and not cmaps and (result["area"] or secondaries):
        raise ValueError("cmaps must not be empty when clusters are displayed")

    clusters = [result]
    clusters.extend(secondaries)
    if max_clusters is not None:
        clusters = clusters[:max_clusters]

    # Filter by significance threshold if requested
    if significance_alpha is not None:
        if only_non_significant:

            def _keep_ns(cl: dict) -> bool:
                pval = cl.get("pval")
                return pval is not None and not _passes_pvalue_threshold(pval, significance_alpha)

            clusters = [cl for cl in clusters if _keep_ns(cl)]
        else:

            def _keep_sig(cl: dict) -> bool:
                return _passes_pvalue_threshold(cl.get("pval"), significance_alpha)

            clusters = [cl for cl in clusters if _keep_sig(cl)]

    # Build a color list matching the number of clusters if not provided
    cmaps = _ensure_cmaps(len(clusters), cmaps, palette=palette)

    # Load geodata once and precompute integer area ids for efficient filtering; a
    # non-integer identifier column, or clusters entirely absent from the geodata,
    # are errors rather than an empty map (FXA-195)
    gdf_all = _ensure_gdf(geodata)
    gdf_all["_area_id"] = _cluster_area_ids(
        gdf_all, id_col, [cl.get("area", []) for cl in clusters]
    )

    if overlay:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Draw base outlines
        gdf_all.plot(ax=ax, facecolor="none", edgecolor=base_edge_color, linewidth=base_edge_width)
        for i, cl in enumerate(clusters):
            cm = cmaps[i]
            try:
                color = cm(1.0)
            except Exception:
                color = plt.get_cmap(cm)(0.8)
            sel = set(int(v) for v in cl.get("area", []) if v is not None)
            if len(sel) == 0:
                continue
            subset = gdf_all[gdf_all["_area_id"].isin(sel)]
            if len(subset) == 0:
                continue
            subset.plot(
                ax=ax, color=color, edgecolor=edge_color, linewidth=float(edge_width), alpha=alpha
            )
        ax.set_title(
            "No clusters to display" if not clusters else f"All clusters (n={len(clusters)})"
        )
        if show_legend:
            import matplotlib.patches as mpatches

            patches = []
            for i in range(len(clusters)):
                cm = cmaps[i]
                try:
                    color = cm(1.0)
                except Exception:
                    color = plt.get_cmap(cm)(0.8)
                label = f"{title_prefix} {i + 1}"
                patches.append(mpatches.Patch(color=color, label=label))
            ax.legend(handles=patches, frameon=True, loc="upper right")
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if return_fig:
            return fig, ax
        _close_owned_figure(fig, created=True, return_fig=return_fig)
        return ax

    # grid mode
    n = len(clusters)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Draw base outlines to give geographical context even when no clusters match
        gdf_all.plot(ax=ax, facecolor="none", edgecolor=base_edge_color, linewidth=base_edge_width)
        ax.set_title("No clusters to display")
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if return_fig:
            return fig, ax
        _close_owned_figure(fig, created=True, return_fig=return_fig)
        return ax
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for i, cl in enumerate(clusters):
        ttl = f"{title_prefix} {i + 1}"
        if label_stats and isinstance(cl, dict):
            rr = _extract_rr(cl)
            pv = cl.get("pval")
            st = cl.get("stats")
            parts: list[str] = []
            try:
                if rr is not None:
                    parts.append(f"RR={float(rr):.2f}")
            except Exception:
                pass
            try:
                if pv is not None:
                    parts.append(f"p={float(pv):.3g}")
            except Exception:
                pass
            try:
                if st is not None:
                    parts.append(f"stat={float(st):.2f}")
            except Exception:
                pass
            if parts:
                ttl += " (" + ", ".join(parts) + ")"
        ax_i = axes_list[i]
        # base outlines
        gdf_all.plot(
            ax=ax_i, facecolor="none", edgecolor=base_edge_color, linewidth=base_edge_width
        )
        cm = cmaps[i]
        try:
            color = cm(1.0)
        except Exception:
            color = plt.get_cmap(cm)(0.8)
        sel = set(int(v) for v in cl.get("area", []) if v is not None)
        subset = gdf_all[gdf_all["_area_id"].isin(sel)]
        if len(subset) > 0:
            subset.plot(
                ax=ax_i, color=color, edgecolor=edge_color, linewidth=float(edge_width), alpha=alpha
            )
        ax_i.set_title(ttl)
    # hide extra in cluster grid
    for j in range(i + 1, rows * cols):
        axes_list[j].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, axes
    _close_owned_figure(fig, created=True, return_fig=return_fig)
    return axes
