from __future__ import annotations

from typing import Any

from ._core_loader import require_core as _require_core
from ._meta import __version__
from .api import build_static, describe_static, scan, scan_from_geodataframe, scan_series
from .graph import (
    build_contiguity,
    csr_from_dense,
    csr_from_pysal,
    csr_knn_from_coords,
    queen_adjacency,
    rook_adjacency,
)
from .helpers import (
    cluster_pool,
    ensure_projected,
    filter_significant_clusters,
    merge_overlapping_clusters,
    pool_membership_series,
    pool_to_geodataframe,
)
from .plot_consensus import plot_consensus_frequency
from .report import export_cluster_report
from .summary import (
    clusters_to_dataframe,
    flatten_clusters,
    scan_settings_to_text,
    summarize_result,
    write_clusters_csv,
    write_series_clusters_csv,
)
from .tuning import (
    find_stable_params_all,
    find_stable_params_all_series,
    grid_search_stable,
    screen_stable_significant,
)
from .validation import (
    boundary_contrast,
    calibrate_llr_threshold,
    conductance,
    consensus_sweep,
    consensus_to_series,
    cv_glm_gain,
    dunn_graph,
    join_counts,
    morans_I_within,
    silhouette_graph,
    stability_across_K,
    trend_level_slope,
)


def core_version() -> str:
    """Return the compiled core version string when available."""
    try:
        return _require_core().version()
    except (ImportError, AttributeError):
        return "unavailable"


def __getattr__(name: str) -> Any:
    if name == "_core":
        return _require_core()
    raise AttributeError(f"module 'flexscanx' has no attribute {name!r}")


__all__ = [
    "__version__",
    "core_version",
    "export_cluster_report",
    "plot_consensus_frequency",
    "scan",
    "scan_series",
    "csr_from_pysal",
    "csr_from_dense",
    "csr_knn_from_coords",
    "summarize_result",
    "flatten_clusters",
    "clusters_to_dataframe",
    "write_clusters_csv",
    "write_series_clusters_csv",
    "scan_settings_to_text",
    "build_static",
    "describe_static",
    "scan_from_geodataframe",
    "queen_adjacency",
    "rook_adjacency",
    "build_contiguity",
    "cluster_pool",
    "filter_significant_clusters",
    "merge_overlapping_clusters",
    "ensure_projected",
    "pool_membership_series",
    "pool_to_geodataframe",
    "silhouette_graph",
    "dunn_graph",
    "morans_I_within",
    "join_counts",
    "cv_glm_gain",
    "trend_level_slope",
    "consensus_sweep",
    "consensus_to_series",
    "conductance",
    "boundary_contrast",
    "stability_across_K",
    "calibrate_llr_threshold",
    "find_stable_params_all",
    "find_stable_params_all_series",
    "grid_search_stable",
    "screen_stable_significant",
]
