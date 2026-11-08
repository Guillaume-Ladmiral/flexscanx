from __future__ import annotations

import html
import math
import os
from typing import Any, cast

from ._utils import extract_rr as _extract_rr
from ._utils import validate_integer_parameter
from .types import BothResultDict, ScanResultDict

__all__ = [
    "export_cluster_report",
]


def _rank_kind(r: int) -> str:
    return "Primary" if int(r) == 1 else "Secondary"


def export_cluster_report(
    geodata: str | Any,
    result: dict,
    id_col: str,
    path: str,
    *,
    cols: int = 3,
    max_clusters: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    hot_cmap: str = "Reds",
    cold_cmap: str = "Blues",
    table_max_rows: int = 15,
    # Optional consensus overlay (single-direction reports only)
    consensus_areas: list[int] | None = None,
    consensus_title: str | None = None,
    consensus_cmap: str = "Greens",
) -> None:
    """Export a one-page report with cluster maps and a summary table.

    Supported outputs (by file extension):
    - .html: HTML file embedding the figure as a base64 PNG image
    - .pdf: vector PDF
    - .png: raster PNG

    Parameters
    - geodata: path or GeoDataFrame with an integer id column matching result area IDs
    - result: dict from flexscanx.api.scan (single or BOTH structure)
    - id_col: name of column containing integer area IDs
    - path: output path (.html, .pdf, or .png)
    - cols: map grid columns
    - max_clusters: cap number of clusters shown per direction (primary + secondaries)
    - title: optional report title
    - figsize: optional (width, height) in inches
    - hot_cmap/cold_cmap: colormaps for HOT/COLD or single-direction
    - table_max_rows: limit summary table rows
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a single scan result dictionary or a BOTH result")
    if "per_time" in result:
        raise ValueError(
            "export_cluster_report does not support scan_series results; "
            "select one time slice first"
        )

    cols = validate_integer_parameter(cols, "cols", minimum=1)
    if max_clusters is not None:
        max_clusters = validate_integer_parameter(max_clusters, "max_clusters", minimum=0)
    table_max_rows = validate_integer_parameter(table_max_rows, "table_max_rows", minimum=0)
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".html", ".pdf", ".png"}:
        raise ValueError("path extension must be one of '.html', '.pdf', or '.png'")
    if figsize is not None:
        try:
            width, height = figsize
        except (TypeError, ValueError) as exc:
            raise ValueError("figsize must contain exactly two positive finite numbers") from exc
        try:
            width_f = float(width)
            height_f = float(height)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("figsize must contain exactly two positive finite numbers") from exc
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not math.isfinite(width_f)
            or not math.isfinite(height_f)
            or width_f <= 0.0
            or height_f <= 0.0
        ):
            raise ValueError("figsize must contain exactly two positive finite numbers")
        figsize = (width_f, height_f)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "matplotlib is required for export_cluster_report. "
            "Install it with: pip install flexscanx[plot]"
        ) from exc

    try:
        import geopandas  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "geopandas is required for export_cluster_report. "
            "Install it with: pip install flexscanx[geo]"
        ) from exc

    from .plotting import plot_cluster_membership
    from .summary import summarize_result

    is_both = str(result.get("type", "")).upper() == "BOTH"

    def _clusters(res: dict) -> list[dict]:
        out: list[dict] = []
        if isinstance(res, dict):
            out.append(res)
            out.extend(res.get("secondaries", []))
        return out

    if is_both:
        hot = result.get("hot", {}) or {}
        cold = result.get("cold", {}) or {}
        hot_list = _clusters(hot)
        cold_list = _clusters(cold)
        if max_clusters is not None:
            hot_list = hot_list[:max_clusters]
            cold_list = cold_list[:max_clusters]
        n_hot = len(hot_list)
        n_cold = len(cold_list)
    else:
        cl_list = _clusters(result)
        if max_clusters is not None:
            cl_list = cl_list[:max_clusters]
        n_cl = len(cl_list)

    # Size heuristics
    if figsize is None:
        map_rows = (max(n_hot, n_cold) + cols - 1) // cols if is_both else (n_cl + cols - 1) // cols
        h = 4.0 * (map_rows if map_rows > 0 else 1) + 2.8  # + table
        w = 4.0 * cols if cols > 0 else 8.0
        figsize = (w, h)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if title is None:
        title = "Clusters Report" if not is_both else "Clusters Report (BOTH)"
    fig.suptitle(title, fontsize=14)

    gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1])
    maps_gs = gs[0].subgridspec(1, 2) if is_both else gs[0].subgridspec(1, 1)

    def _format_title(idx: int, cl: dict) -> str:
        ttl = f"Cluster {idx}"
        if isinstance(cl, dict):
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
        return ttl

    def _draw_grid(
        parent_gs,
        clusters: list[dict],
        cmap: str,
        label: str,
        consensus_areas: list[int] | None = None,
        consensus_title: str | None = None,
        consensus_cmap: str = "Greens",
    ):
        if not clusters and not consensus_areas:
            ax = fig.add_subplot(parent_gs)
            ax.set_axis_off()
            ax.set_title(f"No {label} clusters")
            return
        rows = (len(clusters) + cols - 1) // cols if clusters else 0
        add_consensus = bool(consensus_areas)
        rows_total = rows + (1 if add_consensus else 0)
        rows_total = max(rows_total, 1)
        subgs = parent_gs.subgridspec(rows_total, cols)
        # draw clusters grid
        for i, cl in enumerate(clusters):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(subgs[r, c])
            plot_cluster_membership(
                geodata,
                cl,
                id_col,
                ax=ax,
                cmap=cmap,
                legend=False,
                title=_format_title(i + 1, cl),
            )
        # hide extra in cluster grid
        for j in range(len(clusters), rows * cols):
            r, c = divmod(j, cols)
            ax = fig.add_subplot(subgs[r, c])
            ax.set_axis_off()
        # draw consensus
        if add_consensus:
            axc = fig.add_subplot(subgs[rows_total - 1, 0])
            cl = {"area": list(consensus_areas or [])}
            plot_cluster_membership(
                geodata,
                cl,
                id_col,
                ax=axc,
                cmap=consensus_cmap,
                legend=False,
                title=(consensus_title or "Consensus"),
            )

    if is_both:
        _draw_grid(maps_gs[0], hot_list, hot_cmap, "HOT")
        _draw_grid(maps_gs[1], cold_list, cold_cmap, "COLD")
    else:
        _draw_grid(
            maps_gs[0],
            cl_list,
            hot_cmap,
            "",
            consensus_areas=consensus_areas,
            consensus_title=consensus_title,
            consensus_cmap=consensus_cmap,
        )

    def _draw_table(ax: Any, df: Any, columns: list[str], table_title: str | None = None) -> None:
        ax.set_axis_off()
        if table_title is not None:
            ax.set_title(table_title, fontsize=10)
        if table_max_rows == 0:
            ax.text(0.5, 0.5, "Table suppressed", ha="center", va="center")
            return
        if df.empty:
            ax.text(0.5, 0.5, "No clusters", ha="center", va="center")
            return
        cols_use = [column for column in columns if column in df.columns]
        df_disp = df.iloc[:table_max_rows, :][cols_use].copy()
        if "Rank" in df_disp.columns and "Kind" not in df_disp.columns:
            df_disp.insert(
                df_disp.columns.get_loc("Rank") + 1, "Kind", df_disp["Rank"].map(_rank_kind)
            )
        for col in ("Case", "Expected", "RR", "Stats", "P", "P_hot", "P_cold"):
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].map(
                    lambda x: f"{x:.3g}" if isinstance(x, (int, float)) else x
                )
        table = ax.table(cellText=df_disp.values, colLabels=df_disp.columns, loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

    # The BOTH primary table is the inference authority.  Retain complete
    # one-sided branch tables alongside it: the non-winning branch is diagnostic
    # information, not a secondary of the winning two-sided primary.
    if is_both:
        tables_gs = gs[1].subgridspec(1, 3, wspace=0.25)
        _draw_table(
            fig.add_subplot(tables_gs[0]),
            summarize_result(cast(BothResultDict, result)),
            [
                "Rank",
                "Kind",
                "Type",
                "NumArea",
                "Case",
                "Expected",
                "RR",
                "Stats",
                "P",
                "P_hot",
                "P_cold",
            ],
            "BOTH primary (two-sided)",
        )
        _draw_table(
            fig.add_subplot(tables_gs[1]),
            summarize_result(cast(ScanResultDict, hot)),
            ["Rank", "Kind", "NumArea", "Case", "Expected", "RR", "Stats", "P"],
            "HOT diagnostic (one-sided)",
        )
        _draw_table(
            fig.add_subplot(tables_gs[2]),
            summarize_result(cast(ScanResultDict, cold)),
            ["Rank", "Kind", "NumArea", "Case", "Expected", "RR", "Stats", "P"],
            "COLD diagnostic (one-sided)",
        )
    else:
        _draw_table(
            fig.add_subplot(gs[1]),
            summarize_result(cast(ScanResultDict, result)),
            ["Rank", "Kind", "Type", "NumArea", "Case", "Expected", "RR", "Stats", "P"],
        )

    if ext == ".html":
        # Render to PNG in-memory and embed into an HTML document
        import base64
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        safe_title = html.escape(str(title or "Clusters Report"), quote=True)
        html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{safe_title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue',
                   Arial, sans-serif;
      margin: 1.5rem;
    }}
    .figure {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #ddd;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }}
    h1 {{
      font-size: 1.25rem;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <div class=\"figure\">
    <img alt=\"clusters report\" src=\"data:image/png;base64,{img_b64}\" />
  </div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        plt.close(fig)
        return
    if ext == ".pdf":
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
