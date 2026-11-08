import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flexscanx
import flexscanx.sequence as sequence
import flexscanx.tuning as tuning
from flexscanx.sequence import seq_freq

REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_graph(n_areas: int):
    coords = np.column_stack([np.arange(n_areas, dtype=float), np.zeros(n_areas)])
    row_ptr = [0]
    col_idx = []
    for area in range(n_areas):
        if area > 0:
            col_idx.append(area - 1)
        if area + 1 < n_areas:
            col_idx.append(area + 1)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_seq_freq_top_n_proportions_use_all_sequences_as_denominator():
    sequences = np.array(
        [
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ]
    )

    result = seq_freq(sequences, top_n=2, normalize=True)

    np.testing.assert_allclose(result["proportion"], [2 / 5, 1 / 5])
    np.testing.assert_allclose(result["cumulative"], [2 / 5, 3 / 5])


def test_sequence_package_docstring_does_not_advertise_removed_gower_distance():
    assert "Gower" not in (sequence.__doc__ or "")


def test_core_developer_guide_describes_cyclic_revisitation_not_visit_once():
    developer_guide = (REPO_ROOT / "src" / "flexscanx_core" / "DEVELOPER.md").read_text(
        encoding="utf-8"
    )

    assert "generated exactly once" not in developer_guide
    assert "cyclic" in developer_guide.lower()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"observed": [1.5, 2, 1, 2]}, "integer values"),
        ({"grid": {"K": [2.5]}}, "K must be an integer"),
        ({"grid": {"K": []}}, "K grid must contain at least one value"),
        ({"threshold": np.nan}, "threshold must be finite"),
        ({"clustertype": "BOTH"}, "clustertype must be one of"),
    ],
)
def test_consensus_sweep_rejects_invalid_contract_inputs(overrides, message):
    coords, adjacency = _line_graph(4)
    kwargs = {
        "coords": coords,
        "adjacency": adjacency,
        "observed": [1, 2, 1, 2],
        "expected": [1.0, 1.0, 1.0, 1.0],
        "clustersize": 2,
        "grid": {"K": [2]},
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        flexscanx.consensus_sweep(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"window": 0}, "window must be >= 1"),
        ({"window": 1.5}, "window must be an integer"),
        ({"use": "PVALUE"}, "use must be one of"),
        ({"jaccard_threshold": np.nan}, "jaccard_threshold must be finite"),
    ],
)
def test_trend_level_slope_rejects_invalid_controls(overrides, message):
    kwargs = {
        "per_time": [{"area": [1], "pval": 0.05}, {"area": [1], "pval": 0.01}],
        "window": 2,
        "use": "S",
        "jaccard_threshold": 0.3,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        flexscanx.trend_level_slope(**kwargs)


def test_trend_level_slope_treats_consecutive_empty_clusters_as_one_segment():
    result = flexscanx.trend_level_slope(
        [{"area": [], "pval": 0.5}, {"area": [], "pval": 0.25}],
        window=2,
    )

    assert result["times"] == [0, 1]


def test_stability_across_k_rejects_fractional_k():
    coords, adjacency = _line_graph(4)

    with pytest.raises(ValueError, match="K_values\\[0\\] must be an integer"):
        flexscanx.stability_across_K(
            coords=coords,
            adjacency=adjacency,
            observed=[2, 3, 1, 1],
            expected=[1.0] * 4,
            K_values=[2.5],
            clustersize=2,
        )


def test_grid_search_stable_accepts_explicit_monte_carlo_count():
    coords, adjacency = _line_graph(4)

    result = flexscanx.grid_search_stable(
        coords=coords,
        adjacency=adjacency,
        observed=[4, 3, 1, 1],
        expected=[1.0] * 4,
        K_grid=[3],
        S_grid=[2],
        simcount=9,
        seed=7,
    )

    assert np.isfinite(result.loc[0, "pval"])


def test_grid_search_stable_rejects_fractional_grid_values():
    coords, adjacency = _line_graph(4)

    with pytest.raises(ValueError, match="K_grid\\[0\\] must be an integer"):
        flexscanx.grid_search_stable(
            coords=coords,
            adjacency=adjacency,
            observed=[4, 3, 1, 1],
            expected=[1.0] * 4,
            K_grid=[2.5],
            S_grid=[1],
        )


def test_find_stable_params_all_rejects_fractional_search_bounds():
    coords, adjacency = _line_graph(4)

    with pytest.raises(ValueError, match="K_start must be an integer"):
        flexscanx.find_stable_params_all(
            coords=coords,
            adjacency=adjacency,
            observed=[4, 3, 1, 1],
            expected=[1.0] * 4,
            K_start=2.5,
            S_start=1,
            K_min=2,
            S_min=1,
        )


def test_find_stable_params_all_rejects_non_decreasing_step():
    coords, adjacency = _line_graph(4)

    with pytest.raises(ValueError, match="stepK must be finite and in"):
        flexscanx.find_stable_params_all(
            coords=coords,
            adjacency=adjacency,
            observed=[4, 3, 1, 1],
            expected=[1.0] * 4,
            K_start=3,
            S_start=1,
            K_min=2,
            S_min=1,
            stepK=1.1,
        )


def test_find_stable_params_all_series_rejects_fractional_observed_counts():
    coords, adjacency = _line_graph(4)

    with pytest.raises(ValueError, match="integer values"):
        flexscanx.find_stable_params_all_series(
            coords=coords,
            adjacency=adjacency,
            observed=np.array([[4.5, 3, 1, 1]]),
            expected=np.ones((1, 4)),
            K_start=3,
            S_start=1,
            K_min=2,
            S_min=1,
        )


def test_screen_stable_significant_does_not_treat_missing_pvalues_as_significant(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            {
                "K": 3,
                "S": 2,
                "mean_jaccard": 1.0,
                "min_jaccard": 1.0,
                "coverage": 1.0,
                "count_diff": 0,
                "pval": np.nan,
            }
        ]
    )
    monkeypatch.setattr(tuning, "grid_search_stable", lambda **_kwargs: frame)

    result = tuning.screen_stable_significant(
        coords=np.zeros((3, 2)),
        adjacency=([0, 0, 0, 0], []),
        observed=[0, 0, 0],
        expected=[1.0, 1.0, 1.0],
        K_grid=[2],
        S_grid=[1],
    )

    assert result.empty


def test_filter_significant_clusters_excludes_clusters_without_pvalues():
    result = {
        "type": "HOT",
        "area": [1, 2],
        "pval": None,
        "stats": 5.0,
        "secondaries": [],
    }

    assert flexscanx.filter_significant_clusters(result, alpha=0.05) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": np.nan}, "alpha must be finite"),
        ({"tau": np.inf}, "tau must be finite"),
        ({"min_size": 1.5}, "min_size must be an integer"),
        ({"max_radius": np.nan}, "max_radius must be finite"),
    ],
)
def test_filter_significant_clusters_rejects_invalid_controls(kwargs, message):
    result = {
        "type": "HOT",
        "area": [1, 2],
        "pval": 0.01,
        "stats": 5.0,
        "max_dist": 0.0,
        "secondaries": [],
    }

    with pytest.raises(ValueError, match=message):
        flexscanx.filter_significant_clusters(result, **kwargs)


def test_plot_consensus_frequency_maps_plain_frequencies_by_one_based_area_id(
    monkeypatch,
):
    pytest.importorskip("geopandas")
    from geopandas import plotting as gpd_plotting

    from tests.test_plotting import _make_grid

    gdf = _make_grid(2).iloc[[2, 0, 3, 1]].copy()
    captured = {}
    original_plot_dataframe = gpd_plotting.plot_dataframe

    def _spy_plot_dataframe(dataframe, *args, **kwargs):
        captured["frame"] = dataframe.copy()
        return original_plot_dataframe(dataframe, *args, **kwargs)

    monkeypatch.setattr(gpd_plotting, "plot_dataframe", _spy_plot_dataframe)

    flexscanx.plot_consensus_frequency(
        gdf,
        [0.1, 0.2, 0.3, 0.4],
        id_col="id",
        legend=False,
    )

    plotted = captured["frame"]
    assert list(plotted["_cons_freq"]) == [0.3, 0.1, 0.4, 0.2]


def test_multinomial_monte_carlo_rejects_total_above_native_integer_range():
    coords, adjacency = _line_graph(2)

    with pytest.raises(ValueError, match="multinomial total exceeds"):
        flexscanx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1_500_000_000, 1_500_000_000],
            expected=[1.0, 1.0],
            K=2,
            clustersize=1,
            simcount=1,
            rantype="MULTINOMIAL",
        )


def test_scan_rejects_nonfinite_secondary_statistic_threshold():
    coords, adjacency = _line_graph(2)

    with pytest.raises(ValueError, match="secondary_stats_stop must be finite"):
        flexscanx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[5, 1],
            expected=[2.0, 2.0],
            K=2,
            clustersize=1,
            simcount=0,
            secondary_stats_stop=np.nan,
        )


def test_native_scan_rejects_negative_seed():
    handle = flexscanx.build_static(
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        adjacency=([0, 1, 2], [1, 0]),
        K=2,
    )

    with pytest.raises(ValueError, match="seed must be in"):
        flexscanx._core.ScanSlice(
            handle,
            [5, 1],
            [2.0, 2.0],
            {
                "model": "POISSON",
                "scanmethod": "FLEXIBLE",
                "stattype": "RESTRICTED",
                "clustertype": "HOT",
                "clustersize": 1,
                "simcount": 1,
                "seed": -1,
            },
        )


def test_base_import_does_not_warn_when_optional_numba_is_absent():
    code = r"""
import builtins
import warnings

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "numba":
        raise ImportError("blocked for test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import flexscanx
messages = [str(item.message) for item in caught]
assert not any("numba is not installed" in message for message in messages), messages
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_graph_metric_warns_once_when_optional_numba_is_absent():
    code = r"""
import builtins
import warnings

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "numba":
        raise ImportError("blocked for test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from flexscanx import silhouette_graph
    args = ([0, 1, 3, 5, 6], [1, 0, 2, 1, 3, 2], [1, 1, 0, 0])
    silhouette_graph(*args)
    silhouette_graph(*args)
messages = [str(item.message) for item in caught if "numba is not installed" in str(item.message)]
assert len(messages) == 1, messages
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_binomial_result_area_order_is_stable_across_parallelism():
    coords, adjacency = _line_graph(6)
    kwargs = {
        "coords": coords,
        "adjacency": adjacency,
        "observed": [12, 10, 6, 2, 1, 1],
        "population": [20.0] * 6,
        "model": "BINOMIAL",
        "rantype": "BINOMIAL",
        "scanmethod": "FLEXIBLE",
        "clustertype": "HOT",
        "stattype": "ORIGINAL",
        "K": 4,
        "clustersize": 3,
        "simcount": 7,
        "seed": 1729,
    }

    serial = flexscanx.scan(**kwargs, n_jobs_centers=1)
    parallel_results = [flexscanx.scan(**kwargs, n_jobs_centers=2) for _ in range(100)]

    assert serial["area"] == [1, 2, 3]
    for parallel in parallel_results:
        assert parallel["area"] == serial["area"]
        assert parallel.get("from_area") == serial.get("from_area")
        assert parallel.get("to_area") == serial.get("to_area")
