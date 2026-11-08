"""Regression coverage for reporting and plotting contracts remediated in 2026-08-16."""

from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from flexscanx import report
from flexscanx.helpers import filter_significant_clusters
from flexscanx.plot_consensus import plot_consensus_frequency
from flexscanx.plotting import (
    _ensure_cmaps,
    choropleth,
    plot_all_both_clusters,
    plot_all_clusters,
    plot_both_membership,
    plot_cluster_membership,
    plot_values_from_obsexp,
)
from flexscanx.summary import clusters_to_dataframe, flatten_clusters, summarize_result
from tests.test_plotting import _make_grid


def _cluster(area: list[int], *, pval: float | None = 0.01) -> dict:
    return {
        "type": "HOT",
        "area": area,
        "n_case": 5.0,
        "expected": 2.0,
        "stats": 4.0,
        "pval": pval,
        "secondaries": [],
    }


def _both_result() -> dict:
    return {
        "type": "BOTH",
        "primary_type": "HOT",
        "primary": _cluster([1]),
        "hot": _cluster([1]),
        "cold": {**_cluster([2]), "type": "COLD"},
    }


def test_report_accepts_zero_table_rows_by_rendering_no_table(tmp_path) -> None:
    output = tmp_path / "report.png"

    report.export_cluster_report(
        _make_grid(2), _cluster([1]), id_col="id", path=str(output), table_max_rows=0
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_report_rejects_series_results_at_its_public_boundary(tmp_path) -> None:
    series_result = {"per_time": [_cluster([1])]}

    with pytest.raises(ValueError, match="scan_series"):
        report.export_cluster_report(
            _make_grid(2), series_result, id_col="id", path=str(tmp_path / "report.png")
        )


def test_report_dependency_errors_name_the_missing_extra(monkeypatch, tmp_path) -> None:
    gdf = _make_grid(2)
    with monkeypatch.context() as blocked_matplotlib:
        blocked_matplotlib.setitem(sys.modules, "matplotlib", None)
        blocked_matplotlib.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match=r"matplotlib.*flexscanx\[plot\]"):
            report.export_cluster_report(gdf, _cluster([1]), "id", str(tmp_path / "matplotlib.png"))

    with monkeypatch.context() as blocked_geopandas:
        blocked_geopandas.setitem(sys.modules, "geopandas", None)
        with pytest.raises(ImportError, match=r"geopandas.*flexscanx\[geo\]"):
            report.export_cluster_report(gdf, _cluster([1]), "id", str(tmp_path / "geopandas.png"))


def test_empty_cluster_slices_do_not_become_phantom_summary_rows() -> None:
    series_result = {"per_time": [_cluster([], pval=None), _cluster([2])]}

    flattened = flatten_clusters(series_result)
    dataframe = clusters_to_dataframe(series_result)
    summary = summarize_result(series_result)

    assert [(row["time"], row["area"]) for row in flattened] == [(1, [2])]
    assert dataframe[["time", "area"]].values.tolist() == [[1, [2]]]
    assert summary["NumArea"].tolist() == [1]
    assert summary.index.get_level_values("time").tolist() == [1]


def test_consensus_frequency_requires_one_value_per_map_area() -> None:
    with pytest.raises(ValueError, match="length"):
        plot_consensus_frequency(_make_grid(2), [0.5], id_col="id", legend=False)


def test_both_overlay_draws_only_member_subsets_in_their_own_colours() -> None:
    fig, ax = plot_all_both_clusters(
        _make_grid(2),
        _both_result(),
        id_col="id",
        overlay=True,
        significance_alpha=None,
        return_fig=True,
    )
    try:
        patch_counts = [len(collection.get_paths()) for collection in ax.collections]
        face_colours = [collection.get_facecolors()[0] for collection in ax.collections[1:]]

        assert patch_counts == [4, 1, 1]
        assert face_colours[0][0] > face_colours[0][2]
        assert face_colours[1][2] > face_colours[1][0]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "result",
    [
        {"per_time": [_cluster([1])]},
        _both_result(),
        {"type": "HOT", "secondaries": []},
        {"hello": "world"},
        {"area": "not-a-list"},
    ],
)
def test_plot_all_clusters_rejects_non_single_result_shapes(result) -> None:
    with pytest.raises(ValueError, match="single"):
        plot_all_clusters(_make_grid(2), result, id_col="id", overlay=True)


def test_both_membership_rejects_non_mapping_results_before_creating_a_figure() -> None:
    with pytest.raises(ValueError, match="BOTH result"):
        plot_both_membership(_make_grid(2), [], id_col="id")


def test_missing_pvalues_are_excluded_consistently_by_filters_and_plots() -> None:
    result = _cluster([1], pval=None)

    assert filter_significant_clusters(result, alpha=0.05) == []
    assert filter_significant_clusters(result, alpha=None) == [frozenset({1})]
    fig, ax = plot_all_clusters(_make_grid(2), result, id_col="id", overlay=True, return_fig=True)
    try:
        assert ax.get_title() == "No clusters to display"
    finally:
        plt.close(fig)

    fig, ax = plot_all_clusters(
        _make_grid(2), result, id_col="id", overlay=True, significance_alpha=None, return_fig=True
    )
    try:
        assert ax.get_title() == "All clusters (n=1)"
    finally:
        plt.close(fig)


def test_plotters_close_only_figures_they_create_when_not_returning_them() -> None:
    gdf = _make_grid(2)
    single = _cluster([1])
    both = _both_result()
    plt.close("all")
    try:
        outputs = [
            plot_cluster_membership(gdf, single, "id"),
            plot_both_membership(gdf, both, "id"),
            plot_all_both_clusters(gdf, both, "id", overlay=True, significance_alpha=None),
            plot_all_both_clusters(gdf, both, "id", overlay=False, significance_alpha=None),
            plot_all_clusters(gdf, single, "id", overlay=True),
            plot_all_clusters(gdf, single, "id", overlay=False),
            choropleth(gdf, [0.1, 0.2, 0.3, 0.4], "id", legend=False),
            plot_values_from_obsexp(gdf, [1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], "id"),
            plot_consensus_frequency(gdf, [0.1, 0.2, 0.3, 0.4], "id", legend=False),
        ]

        def _first_axis(output):
            if isinstance(output, np.ndarray):
                return output.flat[0]
            if isinstance(output, tuple):
                return output[0]
            return output

        assert all(not plt.fignum_exists(_first_axis(output).figure.number) for output in outputs)

        external_figure, external_axis = plt.subplots()
        plot_cluster_membership(gdf, single, "id", ax=external_axis)
        assert plt.fignum_exists(external_figure.number)
    finally:
        plt.close("all")


def test_default_colormap_cascade_reaches_tab20_after_okabe_ito(monkeypatch) -> None:
    original_import = builtins.__import__

    def _block_optional_colours(name, *args, **kwargs):
        if name.split(".")[0] in {"colorcet", "seaborn"}:
            raise ImportError(f"blocked optional dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_optional_colours)
    cmaps = _ensure_cmaps(8, palette=None)

    assert tuple(cmaps[0](1.0)) != tuple(cmaps[7](1.0))
