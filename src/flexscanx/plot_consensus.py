from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import validate_integer_parameter
from .helpers import _integer_area_ids


def plot_consensus_frequency(
    geodata: str | Any,
    freq: list[float],
    id_col: str,
    *,
    ax: Any | None = None,
    cmap: str = "YlOrRd",
    legend: bool = True,
    title: str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    discrete_bins: list[float] | None = None,
    edgecolor: str = "white",
    linewidth: float = 0.5,
    missing_color: str = "#f0f0f0",
    save_path: str | None = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Any:
    """Plot consensus frequency on a choropleth map.

    Parameters
    ----------
    geodata : str or GeoDataFrame
        Path to a shapefile/GeoPackage or a GeoDataFrame with an integer id column
        matching result area IDs.
    freq : list of float
        Per-area inclusion frequencies in 1..N order (as returned by consensus_sweep).
    id_col : str
        Name of the column containing the 1-based integer area IDs (integers or
        their decimal string form).
    ax : matplotlib Axes, optional
        Existing axis to draw on.  A new figure is created when None.
    cmap : str, default "YlOrRd"
        Matplotlib colormap name.
    legend : bool, default True
        Whether to draw a colorbar.
    title : str, optional
        Axis title.
    vmin, vmax : float
        Colormap limits (default 0.0 and 1.0).
    discrete_bins : list of float, optional
        Bin edges for discrete colour classes, e.g. ``[0, 0.25, 0.5, 0.75, 1.0]``.
    edgecolor : str, default "white"
        Polygon edge colour.
    linewidth : float, default 0.5
        Polygon edge width.
    missing_color : str, default "#f0f0f0"
        Fill colour for areas absent from *freq*.
    save_path : str, optional
        If provided the figure is saved to this path.
    dpi : int, default 300
        Resolution for saved figures.
    return_fig : bool, default False
        When True return ``(fig, ax)``; otherwise return ``ax`` only.

    Returns
    -------
    ax or (fig, ax)
        Matplotlib axis (or figure + axis tuple when *return_fig* is True).

    Raises
    ------
    ImportError
        If *matplotlib* is not installed (``pip install matplotlib`` or
        ``pip install flexscanx[plot]``).
    ImportError
        If *geopandas* is not installed (``pip install flexscanx[geo]``).
    ValueError
        If *id_col* is missing, holds a value that is not an integer area ID
        (the column, row and value are named), or matches no area ID in
        ``1..N``; if *freq* is not one finite value in ``[0, 1]`` per row.
    """
    dpi = validate_integer_parameter(dpi, "dpi", minimum=1)
    try:
        linewidth_f = float(linewidth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("linewidth must be a finite non-negative number") from exc
    if not np.isfinite(linewidth_f):
        raise ValueError("linewidth must be finite")
    if linewidth_f < 0.0:
        raise ValueError(f"linewidth must be non-negative, got {linewidth_f}")
    linewidth = linewidth_f
    limits: dict[str, float | None] = {}
    for name, value in (("vmin", vmin), ("vmax", vmax)):
        if value is None:
            limits[name] = None
            continue
        try:
            limit = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(limit):
            raise ValueError(f"{name} must be finite")
        limits[name] = limit
    vmin_validated = limits["vmin"]
    vmax_validated = limits["vmax"]
    if (
        vmin_validated is not None
        and vmax_validated is not None
        and vmin_validated > vmax_validated
    ):
        raise ValueError(f"vmin must be <= vmax, got vmin={vmin_validated}, vmax={vmax_validated}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_consensus_frequency. "
            "Install it with: pip install matplotlib  or  pip install flexscanx[plot]"
        ) from exc

    try:
        import geopandas as gpd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for plot_consensus_frequency. "
            "Install it with: pip install flexscanx[geo]"
        ) from exc

    if isinstance(geodata, str):
        gdf = gpd.read_file(geodata)
    else:
        gdf = geodata.copy()

    if id_col not in gdf.columns:
        raise ValueError(f"id_col '{id_col}' not found in geodata columns: {list(gdf.columns)}")

    from .validation import consensus_to_series

    freq_values = np.asarray(freq, dtype=float)
    if freq_values.ndim != 1:
        raise ValueError("freq must be a 1D sequence")
    if len(freq_values) != len(gdf):
        raise ValueError(
            "freq length must equal the number of geodata rows: "
            f"got {len(freq_values)}, expected {len(gdf)}"
        )
    if not np.all(np.isfinite(freq_values)):
        raise ValueError("freq must contain only finite values")
    if np.any((freq_values < 0.0) | (freq_values > 1.0)):
        raise ValueError("freq values must be in [0,1]")

    # consensus_sweep frequencies are ordered by the public 1-based scan area
    # ID, independently of GeoDataFrame row order.  The identifier column is
    # converted like the other geospatial helpers (integers or their decimal
    # string form; anything else is reported by column, row and value), and a
    # column that matches no area ID is an error rather than an all-"missing"
    # map (FXA-194).  Rows whose ID is outside 1..N keep the missing colour.
    freq_series = consensus_to_series(freq_values)
    area_ids = _integer_area_ids(gdf, id_col)
    if not any(area_id in freq_series.index for area_id in area_ids):
        raise ValueError(
            f"No area IDs in geodata column '{id_col}' matched the consensus frequencies "
            f"(expected integers in 1..{len(freq_values)}; the column holds "
            f"{sorted(set(area_ids))[:5]}...). Ensure the column uses the 1-based scan area IDs."
        )
    gdf["_cons_freq"] = [freq_series.get(area_id) for area_id in area_ids]

    created = ax is None
    if created:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        assert ax is not None
        fig = ax.figure
    assert ax is not None

    if discrete_bins:
        norm = BoundaryNorm(discrete_bins, 256)
        gdf.plot(
            column="_cons_freq",
            ax=ax,
            cmap=cmap,
            norm=norm,
            legend=legend,
            vmin=vmin_validated,
            vmax=vmax_validated,
            edgecolor=edgecolor,
            linewidth=linewidth,
            missing_kwds={"color": missing_color},
        )
    else:
        gdf.plot(
            column="_cons_freq",
            ax=ax,
            cmap=cmap,
            legend=legend,
            vmin=vmin_validated,
            vmax=vmax_validated,
            edgecolor=edgecolor,
            linewidth=linewidth,
            missing_kwds={"color": missing_color},
        )

    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if return_fig:
        return fig, ax
    if created:
        plt.close(fig)
    return ax
