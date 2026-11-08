from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flexscanx as flx

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line_graph(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr = [0]
    col_idx: list[int] = []
    for node in range(n):
        if node:
            col_idx.append(node - 1)
        if node + 1 < n:
            col_idx.append(node + 1)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_series_tuning_aggregate_respects_explicit_minimum_bounds() -> None:
    coords, adjacency = _line_graph(20)
    observed = np.full((1, 20), 10, dtype=int)
    observed[0, 0] = 20
    expected = np.full((1, 20), 10.0)

    result = flx.find_stable_params_all_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=adjacency,
        K_start=2,
        S_start=1,
        K_min=2,
        S_min=1,
        model="POISSON",
        clustertype="HOT",
    )

    assert result["best"] == {"K": 2, "S": 1}
    assert result["best"] == {
        "K": result["per_time"][0]["K"],
        "S": result["per_time"][0]["S"],
    }


@pytest.mark.parametrize("reuse_handle", [False, True])
def test_scan_series_rejects_zero_time_slices(reuse_handle: bool) -> None:
    coords, adjacency = _line_graph(3)
    kwargs = {
        "observed": np.empty((0, 3), dtype=int),
        "expected": np.empty((0, 3), dtype=float),
        "simcount": 0,
    }
    if reuse_handle:
        kwargs["handle"] = flx.build_static(coords, adjacency, K=3)
    else:
        kwargs.update(coords=coords, adjacency=adjacency, K=3)

    with pytest.raises(ValueError, match="at least one time slice"):
        flx.scan_series(**kwargs)


def test_plot_consensus_frequency_rejects_misaligned_frequency_vector() -> None:
    pytest.importorskip("geopandas")

    from tests.test_plotting import _make_grid

    gdf = _make_grid(2).iloc[[0, 2]].copy()
    with pytest.raises(ValueError, match="freq length"):
        flx.plot_consensus_frequency(gdf, [0.5], id_col="id", legend=False)


def test_stability_across_k_accepts_binomial_population() -> None:
    coords, adjacency = _line_graph(4)

    result = flx.stability_across_K(
        coords=coords,
        adjacency=adjacency,
        observed=[1, 4, 1, 1],
        population=[10, 10, 10, 10],
        K_values=[2, 3],
        clustersize=2,
        model="BINOMIAL",
    )

    assert result["K_values"] == [2, 3]


def test_consensus_sweep_accepts_binomial_population() -> None:
    coords, adjacency = _line_graph(4)

    result = flx.consensus_sweep(
        coords=coords,
        adjacency=adjacency,
        observed=[1, 4, 1, 1],
        population=[10, 10, 10, 10],
        clustersize=2,
        grid={"K": [2], "ralpha": [0.2]},
        model="BINOMIAL",
    )

    assert result["runs"] == 1


def test_calibration_cache_distinguishes_spatial_topology(tmp_path, monkeypatch) -> None:
    from flexscanx import validation

    coords, line_adjacency = _line_graph(6)
    triangle_adjacency = (
        [0, 2, 4, 6, 8, 10, 12],
        [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4],
    )
    cache_file = tmp_path / "calibration.json"
    monkeypatch.setattr(validation, "_CACHE_FILE", str(cache_file))
    common = {
        "coords": coords,
        "observed": [8, 7, 5, 4, 3, 2],
        "expected": [5.0] * 6,
        "K": 6,
        "clustersize": 4,
        "scanmethod": "FLEXIBLE",
        "stattype": "ORIGINAL",
        "clustertype": "HOT",
        "knn_mode": "K_FIXED",
        "n_sims": 40,
        "alpha": 0.1,
        "seed": 42,
    }

    flx.calibrate_llr_threshold(adjacency=line_adjacency, **common)
    flx.calibrate_llr_threshold(adjacency=triangle_adjacency, **common)

    payload = validation._cache_load_impl(
        str(cache_file),
        cache_file=str(cache_file),
        cache_schema_version=validation._CACHE_SCHEMA_VERSION,
        cache_format=validation._CACHE_FORMAT,
    )
    assert len(payload) == 2


@pytest.mark.parametrize("area_ids", [[1, 1], [1, True], [0, 2], [1.5, 2]])
def test_extract_sequences_rejects_invalid_or_duplicate_area_ids(area_ids) -> None:
    from flexscanx.sequence import extract_sequences

    result = {"per_time": [{"type": "HOT", "area": [1]}]}

    with pytest.raises(ValueError, match="area_ids"):
        extract_sequences(result, area_ids=area_ids)


@pytest.mark.parametrize("base", [np.nan, np.inf, -np.inf])
def test_seq_entropy_rejects_nonfinite_base(base) -> None:
    from flexscanx.sequence import seq_entropy

    with pytest.raises(ValueError, match="base must be finite"):
        seq_entropy(np.array([[0, 1], [1, 0]]), base=base)


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.0, np.nan], [np.nan, 0.0]],
        [[0.0, -1.0], [-1.0, 0.0]],
        [[0.0, 1.0], [2.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
    ],
)
def test_seq_dist_matrix_info_rejects_invalid_distance_matrices(matrix) -> None:
    from flexscanx.sequence import seq_dist_matrix_info

    with pytest.raises(ValueError):
        seq_dist_matrix_info(np.asarray(matrix))


@pytest.mark.parametrize("top_n", [1.5, np.nan, True])
def test_seq_freq_rejects_noninteger_top_n(top_n) -> None:
    from flexscanx.sequence import seq_freq

    with pytest.raises(ValueError, match="top_n must be an integer or None"):
        seq_freq(np.array([[0], [1]]), top_n=top_n)


@pytest.mark.parametrize(
    ("sequences", "expected"),
    [
        (np.array([["B"], ["A"]], dtype=object), ("A",)),
        (np.array([[2], [10]]), (10,)),
    ],
)
def test_seq_freq_uses_traminex_lexical_order_at_tied_cutoff(sequences, expected) -> None:
    from flexscanx.sequence import seq_freq

    result = seq_freq(sequences, top_n=1)

    assert result.loc[0, "sequence"] == expected


def test_sequence_plots_support_mixed_and_missing_categorical_states() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    import pandas as pd
    from matplotlib.colors import to_rgb

    from flexscanx.sequence import seq_dplot, seq_fplot, seq_iplot

    mixed = np.array([[0, "A"], ["0", "B"]], dtype=object)
    fig_i_mixed, _ = seq_iplot(mixed)
    fig_d_mixed, _ = seq_dplot(mixed)

    missing = np.array([[np.nan, 0], [np.nan, 1]], dtype=object)
    fig_i, ax_i = seq_iplot(missing)
    fig_d, ax_d = seq_dplot(missing)
    fig_f, _ = seq_fplot(np.array([[np.nan, 0], [pd.NA, 1]], dtype=object))

    image = np.asarray(ax_i.images[0].get_array())
    assert np.allclose(image[:, 0, :], to_rgb("#808080"))
    for time_index in range(missing.shape[1]):
        stacked_height = sum(
            patch.get_height()
            for patch in ax_d.patches
            if np.isclose(patch.get_x() + patch.get_width() / 2, time_index)
        )
        assert stacked_height == pytest.approx(1.0)

    for fig in (fig_i_mixed, fig_d_mixed, fig_i, fig_d, fig_f):
        plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_jaccard": np.nan},
        {"min_jaccard": -0.1},
        {"coverage": 1.1},
        {"coverage_floor": np.inf},
        {"count_tol": 1.5},
        {"count_tol": True},
        {"count_tol": -1},
    ],
)
def test_stability_thresholds_reject_invalid_values(kwargs) -> None:
    from flexscanx.tuning import StabilityThresholds

    with pytest.raises(ValueError):
        StabilityThresholds(**kwargs)


@pytest.mark.parametrize("coverage_floor", [np.nan, -0.1, 1.1])
def test_match_clusters_jaccard_rejects_invalid_coverage_floor(coverage_floor) -> None:
    from flexscanx.tuning import match_clusters_jaccard

    with pytest.raises(ValueError, match="coverage_floor"):
        match_clusters_jaccard([{1}], [{1}], coverage_floor=coverage_floor)


@pytest.mark.parametrize("threshold", [np.nan, -0.1, 1.1, "bad"])
@pytest.mark.parametrize("pool", [[], [frozenset({1})]])
def test_merge_overlapping_clusters_rejects_invalid_threshold(pool, threshold) -> None:
    with pytest.raises(ValueError, match="jaccard_threshold"):
        flx.merge_overlapping_clusters(pool, jaccard_threshold=threshold)


@pytest.mark.parametrize(
    ("function_name", "kwargs", "message"),
    [
        ("plot_cluster_membership", {"dpi": 1.5}, "dpi must be an integer"),
        ("plot_cluster_membership", {"alpha": np.nan}, "alpha must be finite"),
        ("plot_both_membership", {"linewidth": -1.0}, "linewidth"),
        ("choropleth", {"k": True}, "k must be an integer"),
        ("choropleth", {"vmin": np.nan}, "vmin must be finite"),
        ("plot_consensus_frequency", {"dpi": True}, "dpi must be an integer"),
        ("plot_consensus_frequency", {"linewidth": np.inf}, "linewidth must be finite"),
        ("plot_consensus_frequency", {"vmin": object()}, "vmin must be finite"),
    ],
)
def test_public_plotters_reject_invalid_numeric_controls(function_name, kwargs, message) -> None:
    from flexscanx import plot_consensus_frequency
    from flexscanx.plotting import choropleth, plot_both_membership, plot_cluster_membership
    from tests.test_plotting import _make_grid

    gdf = _make_grid(2)
    if function_name == "plot_cluster_membership":
        args = (gdf, {"area": [1]}, "id")
    elif function_name == "plot_both_membership":
        both = {
            "type": "BOTH",
            "hot": {"area": [1]},
            "cold": {"area": [2]},
        }
        args = (gdf, both, "id")
    elif function_name == "choropleth":
        args = (gdf, [0.1, 0.2, 0.3, 0.4], "id")
    else:
        args = (gdf, [0.1, 0.2, 0.3, 0.4], "id")

    functions = {
        "plot_cluster_membership": plot_cluster_membership,
        "plot_both_membership": plot_both_membership,
        "choropleth": choropleth,
        "plot_consensus_frequency": plot_consensus_frequency,
    }
    with pytest.raises(ValueError, match=message):
        functions[function_name](*args, **kwargs)


def test_ci_installs_plot_dependencies_for_geo_and_vignette_jobs() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m pip install -e .[geo,plot]" in workflow
    assert "python -m pip install -e .[geo,plot] pyogrio" in workflow


def test_cibuildwheel_runs_repository_tests_from_isolated_directory() -> None:
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")

    assert 'CIBW_TEST_COMMAND: "pytest -q {project}/tests"' in workflow


def test_bench_reports_primary_cluster_for_both_results(tmp_path) -> None:
    output = tmp_path / "both.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bench.py"),
            "--clustertype",
            "BOTH",
            "--grid",
            "4",
            "--K",
            "4",
            "--clustersize",
            "2",
            "--simcounts",
            "0",
            "--repeat",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    row = json.loads(output.read_text(encoding="utf-8"))[0]
    assert row["stats"] is not None
    assert row["area"] > 0


@pytest.mark.parametrize("script", ["bench.py", "perf_checks.py"])
def test_benchmark_scripts_reject_unknown_output_extensions(script, tmp_path) -> None:
    output = tmp_path / "results.txt"
    command = [sys.executable, str(ROOT / "scripts" / script)]
    if script == "bench.py":
        command += [
            "--grid",
            "2",
            "--K",
            "2",
            "--clustersize",
            "1",
            "--simcounts",
            "0",
            "--repeat",
            "1",
        ]
    else:
        command += [
            "--grid",
            "4",
            "--K",
            "4",
            "--clustersize",
            "4",
            "--quick",
        ]
    command += ["--output", str(output)]

    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    assert proc.returncode != 0
    assert "expected .csv or .json" in (proc.stderr + proc.stdout)
    assert not output.exists()


def test_docs_signature_checker_is_independent_of_caller_cwd(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_docs_signatures.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_ci_gated_time_slice_benchmark_has_smoke_coverage(tmp_path) -> None:
    output = tmp_path / "timeslices.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bench_timeslices.py"),
            "--grid",
            "4",
            "--K",
            "4",
            "--T",
            "2",
            "--workers",
            "1",
            "--simcount",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert output.exists()
    assert len(output.read_text(encoding="utf-8").splitlines()) >= 2


def test_otani_workflow_preserves_valid_area_ids_and_requires_known_crs() -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    module = _load_script_module("otani_takahashi_2021_audit", "scripts/otani_takahashi_2021.py")
    projected = gpd.GeoDataFrame(
        {"area_id": [101, 205, 999]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:3857",
    )
    missing_crs = projected.set_crs(None, allow_override=True)

    assert module._valid_area_id_column(projected, "area_id")
    assert module._ensure_projected(projected) is projected
    with pytest.raises(ValueError, match="declare a CRS"):
        module._ensure_projected(missing_crs)

    projected["area_id"] = [101, 101, 999]
    assert not module._valid_area_id_column(projected, "area_id")


def test_otani_workflow_uses_id_preserving_geodataframe_scan_wrapper() -> None:
    source = (ROOT / "scripts" / "otani_takahashi_2021.py").read_text(encoding="utf-8")

    assert "api.scan_from_geodataframe" in source
    assert "api.scan(" not in source


@pytest.mark.parametrize(
    "workflow",
    [
        "restricted_llr_guidance.qmd",
        "stable_significant_workflow.qmd",
        "synthetic_grid_stability.qmd",
    ],
)
def test_workflow_python_blocks_are_executable_quarto_cells(workflow: str) -> None:
    source = (ROOT / "vignettes" / "workflows" / workflow).read_text(encoding="utf-8")

    assert "```{python}" in source
    assert "```python\n" not in source
