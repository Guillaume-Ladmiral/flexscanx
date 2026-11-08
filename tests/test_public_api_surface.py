from __future__ import annotations

import inspect

import flexscanx as flx
import flexscanx.api as api
import flexscanx.graph as graph
import flexscanx.helpers as helpers
import flexscanx.plot_consensus as plot_consensus
import flexscanx.report as report
import flexscanx.summary as summary
import flexscanx.tuning as tuning
import flexscanx.validation as validation

EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "boundary_contrast",
    "build_contiguity",
    "build_static",
    "calibrate_llr_threshold",
    "cluster_pool",
    "clusters_to_dataframe",
    "conductance",
    "consensus_sweep",
    "consensus_to_series",
    "core_version",
    "csr_from_dense",
    "csr_from_pysal",
    "csr_knn_from_coords",
    "cv_glm_gain",
    "describe_static",
    "dunn_graph",
    "ensure_projected",
    "export_cluster_report",
    "filter_significant_clusters",
    "find_stable_params_all",
    "find_stable_params_all_series",
    "flatten_clusters",
    "grid_search_stable",
    "join_counts",
    "merge_overlapping_clusters",
    "morans_I_within",
    "pool_membership_series",
    "plot_consensus_frequency",
    "pool_to_geodataframe",
    "queen_adjacency",
    "rook_adjacency",
    "scan",
    "scan_from_geodataframe",
    "scan_series",
    "scan_settings_to_text",
    "screen_stable_significant",
    "silhouette_graph",
    "stability_across_K",
    "summarize_result",
    "trend_level_slope",
    "write_clusters_csv",
    "write_series_clusters_csv",
}

ROOT_EXPORT_SOURCES = {
    "boundary_contrast": validation,
    "build_contiguity": graph,
    "build_static": api,
    "calibrate_llr_threshold": validation,
    "cluster_pool": helpers,
    "clusters_to_dataframe": summary,
    "conductance": validation,
    "consensus_sweep": validation,
    "consensus_to_series": validation,
    "csr_from_dense": graph,
    "csr_from_pysal": graph,
    "csr_knn_from_coords": graph,
    "cv_glm_gain": validation,
    "describe_static": api,
    "dunn_graph": validation,
    "ensure_projected": helpers,
    "export_cluster_report": report,
    "filter_significant_clusters": helpers,
    "find_stable_params_all": tuning,
    "find_stable_params_all_series": tuning,
    "flatten_clusters": summary,
    "grid_search_stable": tuning,
    "join_counts": validation,
    "merge_overlapping_clusters": helpers,
    "morans_I_within": validation,
    "plot_consensus_frequency": plot_consensus,
    "pool_membership_series": helpers,
    "pool_to_geodataframe": helpers,
    "queen_adjacency": graph,
    "rook_adjacency": graph,
    "scan": api,
    "scan_from_geodataframe": api,
    "scan_series": api,
    "scan_settings_to_text": summary,
    "screen_stable_significant": tuning,
    "silhouette_graph": validation,
    "stability_across_K": validation,
    "summarize_result": summary,
    "trend_level_slope": validation,
    "write_clusters_csv": summary,
    "write_series_clusters_csv": summary,
}

SCAN_SERIES_ONLY_PARAMS = {"crn", "n_jobs_time", "times", "warm_start"}


def test_root_public_api_surface_is_explicit_and_stable():
    assert set(flx.__all__) == EXPECTED_ROOT_EXPORTS
    for name in flx.__all__:
        assert hasattr(flx, name), name


def test_removed_root_wrappers_stay_absent():
    assert not hasattr(flx, "scan_all")
    assert not hasattr(flx, "select_scan_and_plot")


def test_root_exports_preserve_signatures_and_docstrings():
    assert inspect.signature(flx.scan) == inspect.signature(api.scan)
    assert inspect.signature(flx.scan_series) == inspect.signature(api.scan_series)
    assert inspect.signature(flx.scan_from_geodataframe) == inspect.signature(
        api.scan_from_geodataframe
    )
    assert flx.scan.__doc__ == api.scan.__doc__
    assert flx.scan_series.__doc__ == api.scan_series.__doc__
    assert flx.scan_from_geodataframe.__doc__ == api.scan_from_geodataframe.__doc__


def test_root_callables_are_direct_reexports_from_source_modules():
    for name, module in ROOT_EXPORT_SOURCES.items():
        assert getattr(flx, name) is getattr(module, name), name


def test_scan_series_retains_scan_parameter_parity_with_only_series_specific_extras():
    scan_params = set(inspect.signature(api.scan).parameters)
    series_params = set(inspect.signature(api.scan_series).parameters)

    assert scan_params <= series_params
    assert series_params - scan_params == SCAN_SERIES_ONLY_PARAMS


EXPECTED_SUBMODULE_EXPORTS = {
    "plotting": {
        "plot_cluster_membership",
        "plot_both_membership",
        "plot_all_clusters",
        "plot_all_both_clusters",
        "choropleth",
        "values_by_area_from_obsexp",
        "plot_values_from_obsexp",
    },
    "report": {"export_cluster_report"},
    "summary": {
        "flatten_clusters",
        "clusters_to_dataframe",
        "write_clusters_csv",
        "write_series_clusters_csv",
        "scan_settings_to_text",
        "summarize_result",
    },
    "graph": {
        "csr_from_pysal",
        "csr_from_dense",
        "csr_knn_from_coords",
        "queen_adjacency",
        "rook_adjacency",
        "build_contiguity",
        "csr_from_geodataframe_contiguity",
    },
    "helpers": {
        "ensure_projected",
        "cluster_pool",
        "filter_significant_clusters",
        "merge_overlapping_clusters",
        "pool_membership_series",
        "pool_to_geodataframe",
    },
    "types": {
        "RatioValue",
        "RatioCI",
        "ScanClusterBase",
        "ScanClusterDict",
        "SeriesEntryResultBase",
        "SeriesEntryResultDict",
        "ScanResultBase",
        "ScanResultDict",
        "SeriesEntryBothResultBase",
        "SeriesEntryBothResultDict",
        "BothResultBase",
        "BothResultDict",
        "SeriesResultBase",
    },
}


def test_documented_submodules_pin_their_public_surface():
    """FXA-178: every documented submodule declares exactly its public names in __all__."""
    import importlib

    for module_name, expected in EXPECTED_SUBMODULE_EXPORTS.items():
        module = importlib.import_module(f"flexscanx.{module_name}")
        assert set(module.__all__) == expected, module_name
        assert len(module.__all__) == len(set(module.__all__)), module_name
