"""Regression tests for the public-contract repairs of the 2026-09-07 audit-and-fix run.

Each test names the finding it pins. The fixtures are tiny deterministic graphs so
every test runs in well under a second and never touches the calibration cache.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import flexscanx as flx


def _path_graph(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """A collinear path graph 0-1-2-...-(n-1) with unit spacing."""
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = [0], []
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        col_idx.extend(neighbours)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _irregular_path(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """A path graph with strictly distinct pairwise distances (no KNN ties)."""
    spacing = 1.0 + 0.37 * np.arange(n - 1)
    xs = np.concatenate([[0.0], np.cumsum(spacing)])
    coords = np.column_stack([xs, np.zeros(n)])
    distances = sorted(abs(xs[i] - xs[j]) for i in range(n) for j in range(i + 1, n))
    assert len(set(distances)) == len(distances), "fixture must be tie-free"
    row_ptr, col_idx = [0], []
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        col_idx.extend(neighbours)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


# ---------------------------------------------------------------------------
# FXA-179: a reused handle must not silently discard a caller-supplied graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argument", ["coords", "adjacency"])
def test_reused_handle_rejects_caller_supplied_graph_in_scan(argument: str) -> None:
    coords, adjacency = _path_graph(6)
    handle = flx.build_static(coords, adjacency, K=4)
    extra = {argument: coords if argument == "coords" else adjacency}
    with pytest.raises(ValueError, match=rf"handle.*{argument}"):
        flx.scan(
            handle=handle, observed=[1, 1, 9, 9, 1, 1], expected=[2.0] * 6, simcount=0, **extra
        )


@pytest.mark.parametrize("argument", ["coords", "adjacency"])
def test_reused_handle_rejects_caller_supplied_graph_in_scan_series(argument: str) -> None:
    coords, adjacency = _path_graph(6)
    handle = flx.build_static(coords, adjacency, K=4)
    extra = {argument: coords if argument == "coords" else adjacency}
    with pytest.raises(ValueError, match=rf"handle.*{argument}"):
        flx.scan_series(
            handle=handle,
            observed=[[1, 1, 9, 9, 1, 1]],
            expected=[[2.0] * 6],
            simcount=0,
            **extra,
        )


def test_reused_handle_alone_still_scans() -> None:
    coords, adjacency = _path_graph(6)
    handle = flx.build_static(coords, adjacency, K=4)
    result = flx.scan(handle=handle, observed=[1, 1, 9, 9, 1, 1], expected=[2.0] * 6, simcount=0)
    assert result["area"] == [3, 4]
    assert result["scan_settings"]["handle_reused"] is True


# ---------------------------------------------------------------------------
# FXA-180: float-valued controls name themselves when conversion fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["clusterradius", "ralpha", "mc_alpha"])
@pytest.mark.parametrize("bad", ["abc", [1, 2], {"a": 1}])
def test_float_parameter_conversion_errors_name_the_parameter(name: str, bad: object) -> None:
    coords, adjacency = _path_graph(5)
    with pytest.raises((ValueError, TypeError)) as excinfo:
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1] * 5,
            expected=[1.0] * 5,
            simcount=1,
            **{name: bad},
        )
    assert name in str(excinfo.value)
    # A string is a ValueError (as before); a container stays a TypeError, so callers
    # catching TypeError specifically keep seeing it.
    expected_type = ValueError if isinstance(bad, str) else TypeError
    assert isinstance(excinfo.value, expected_type)


@pytest.mark.parametrize("name", ["secondary_p_stop", "secondary_stats_stop"])
def test_secondary_stop_conversion_errors_name_the_parameter(name: str) -> None:
    coords, adjacency = _path_graph(5)
    with pytest.raises(ValueError, match=name):
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1] * 5,
            expected=[1.0] * 5,
            simcount=0,
            secondary=1,
            **{name: "abc"},
        )


def test_calibration_alpha_conversion_error_names_the_parameter() -> None:
    coords, adjacency = _path_graph(5)
    with pytest.raises(ValueError, match="alpha"):
        flx.calibrate_llr_threshold(
            coords=coords,
            adjacency=adjacency,
            observed=[1] * 5,
            expected=[1.0] * 5,
            K=3,
            clustersize=2,
            n_sims=1,
            alpha="abc",
        )


def test_float_parameter_still_accepts_numeric_types() -> None:
    coords, adjacency = _path_graph(5)
    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[1] * 5,
        expected=[1.0] * 5,
        simcount=0,
        clusterradius=np.float32(2.5),
        ralpha=np.float64(0.25),
    )
    assert result["scan_settings"]["ralpha"] == pytest.approx(0.25)
    assert result["scan_settings"]["clusterradius"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# FXA-171: an invalid handle is a public ValueError that names ``handle``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["not-a-handle", 3, object()])
def test_bad_handle_is_reported_as_a_public_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="handle") as excinfo:
        flx.describe_static(bad)  # type: ignore[arg-type]
    assert "DescribeStatic" not in str(excinfo.value)
    with pytest.raises(ValueError, match="handle") as excinfo2:
        flx.scan(handle=bad, observed=[1] * 4, expected=[1.0] * 4, simcount=0)  # type: ignore[arg-type]
    assert "DescribeStatic" not in str(excinfo2.value)


def test_describe_static_none_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="handle"):
        flx.describe_static(None)


# ---------------------------------------------------------------------------
# FXA-173: library warnings are attributed to the caller's frame
# ---------------------------------------------------------------------------


def test_clustersize_cap_warning_blames_the_caller() -> None:
    coords, adjacency = _path_graph(4)
    with pytest.warns(UserWarning, match="capping clustersize") as record:
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1] * 4,
            expected=[1.0] * 4,
            K=2,
            clustersize=20,
            simcount=0,
        )
    caps = [w for w in record if "capping clustersize" in str(w.message)]
    assert caps and caps[0].filename == __file__


def test_scan_series_cap_warning_blames_the_caller() -> None:
    coords, adjacency = _path_graph(4)
    with pytest.warns(UserWarning, match="capping clustersize") as record:
        flx.scan_series(
            coords=coords,
            adjacency=adjacency,
            observed=[[1] * 4],
            expected=[[1.0] * 4],
            K=2,
            clustersize=20,
            simcount=0,
        )
    caps = [w for w in record if "capping clustersize" in str(w.message)]
    assert caps and caps[0].filename == __file__


# ---------------------------------------------------------------------------
# FXA-172: scan_series stores JSON-safe time labels without losing them
# ---------------------------------------------------------------------------


def _two_slice_series(times):
    coords, adjacency = _path_graph(4)
    return flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=[[1, 2, 1, 1], [1, 1, 3, 1]],
        expected=[[1.0] * 4] * 2,
        simcount=0,
        times=times,
    )


def test_series_numpy_integer_times_become_python_ints() -> None:
    result = _two_slice_series(np.arange(2))
    json.dumps(result, allow_nan=False)
    assert result["times"] == [0, 1]
    assert all(type(t) is int for t in result["times"])


def test_series_pandas_timestamps_become_iso_strings() -> None:
    labels = pd.date_range("2026-01-01", periods=2, freq="MS")
    result = _two_slice_series(labels)
    json.dumps(result, allow_nan=False)
    assert result["times"] == [t.isoformat() for t in labels]


def test_series_numpy_datetime64_become_iso_strings() -> None:
    labels = np.array(["2026-01-01", "2026-02-01"], dtype="datetime64[D]")
    result = _two_slice_series(labels)
    json.dumps(result, allow_nan=False)
    assert result["times"] == ["2026-01-01", "2026-02-01"]


def test_series_string_and_int_times_are_unchanged() -> None:
    assert _two_slice_series(["a", "b"])["times"] == ["a", "b"]
    assert _two_slice_series([7, 9])["times"] == [7, 9]


# ---------------------------------------------------------------------------
# FXA-181: a non-finite ratio never reaches the public envelope
# ---------------------------------------------------------------------------


def test_denormal_expected_keeps_the_envelope_strict_json() -> None:
    coords, adjacency = _path_graph(8)
    expected = [1.0] * 8
    expected[7] = 1e-310  # legal: finite and strictly positive
    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[1] * 7 + [5],
        expected=expected,
        simcount=0,
        K=3,
    )
    json.dumps(result, allow_nan=False)
    assert result["area"] == [8]
    assert result["Rate_Ratio"] is None
    assert result["Rate_Ratio_CI"] is None


def test_denormal_expected_sanitises_secondaries_and_series_too() -> None:
    coords, adjacency = _path_graph(8)
    expected = [1.0] * 8
    expected[7] = 1e-310
    expected[0] = 1e-310
    observed = [5] + [1] * 6 + [5]
    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        simcount=0,
        K=3,
        secondary="ALL",
    )
    json.dumps(result, allow_nan=False)
    series = flx.scan_series(
        coords=coords,
        adjacency=adjacency,
        observed=[observed],
        expected=[expected],
        simcount=0,
        K=3,
    )
    json.dumps(series, allow_nan=False)


# ---------------------------------------------------------------------------
# FXA-166 / FXA-167: the per-cluster completed-replicate count is declared and truthful
# ---------------------------------------------------------------------------


def _grid(side: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    n = side * side
    coords = np.array([[i % side, i // side] for i in range(n)], dtype=float)
    row_ptr, col_idx = [0], []
    for i in range(n):
        r, c = divmod(i, side)
        neighbours = []
        if r > 0:
            neighbours.append(i - side)
        if r < side - 1:
            neighbours.append(i + side)
        if c > 0:
            neighbours.append(i - 1)
        if c < side - 1:
            neighbours.append(i + 1)
        col_idx.extend(sorted(neighbours))
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_secondary_simcount_actual_is_declared_in_the_cluster_type() -> None:
    from flexscanx.types import ScanClusterDict

    assert "simcount_actual" in ScanClusterDict.__annotations__


def test_secondary_simcount_actual_is_the_denominator_behind_its_pval() -> None:
    coords, adjacency = _grid(6)
    rng = np.random.default_rng(4)
    observed = rng.poisson(2.0, 36)
    observed[0] += 12
    result = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=np.full(36, 2.0),
        simcount=999,
        seed=4,
        secondary="ALL",
        secondary_null="conditional",
        mc_early_stop=True,
        K=6,
    )
    assert "simcount_actual" not in result  # the primary's count lives in scan_settings
    assert result["secondaries"], "fixture must produce secondaries"
    for secondary in result["secondaries"]:
        count = secondary["simcount_actual"]
        assert isinstance(count, int) and count >= 1
        assert secondary["pval"] == pytest.approx(secondary["rank"] / (count + 1))
    # The conditional-null early stop is what makes the per-cluster count differ.
    assert any(
        s["simcount_actual"] != result["scan_settings"]["simcount_actual"]
        for s in result["secondaries"]
    )


# ---------------------------------------------------------------------------
# D5: enabling an opt-in pruning heuristic warns that the maximum may be lost
# ---------------------------------------------------------------------------

_PRUNE_MATCH = "may return a suboptimal cluster"


def _pruning_scan_kwargs():
    coords, adjacency = _grid(4)
    rng = np.random.default_rng(7)
    return dict(
        coords=coords,
        adjacency=adjacency,
        observed=rng.poisson(3.0, 16),
        expected=np.full(16, 3.0),
        simcount=0,
        K=6,
        clustersize=4,
    )


@pytest.mark.parametrize("flag", ["enable_frame_ub_prune", "enable_center_ub_gating"])
def test_enabling_a_pruning_heuristic_warns_once_and_blames_the_caller(flag: str) -> None:
    with pytest.warns(UserWarning, match=_PRUNE_MATCH) as record:
        flx.scan(**_pruning_scan_kwargs(), **{flag: True})
    prune = [w for w in record if _PRUNE_MATCH in str(w.message)]
    assert len(prune) == 1
    assert flag in str(prune[0].message)
    assert prune[0].filename == __file__


def test_scan_series_warns_once_for_a_pruning_heuristic() -> None:
    kwargs = _pruning_scan_kwargs()
    with pytest.warns(UserWarning, match=_PRUNE_MATCH) as record:
        flx.scan_series(
            coords=kwargs["coords"],
            adjacency=kwargs["adjacency"],
            observed=np.stack([kwargs["observed"]] * 3),
            expected=np.stack([kwargs["expected"]] * 3),
            simcount=0,
            K=6,
            clustersize=4,
            enable_frame_ub_prune=True,
        )
    assert sum(_PRUNE_MATCH in str(w.message) for w in record) == 1


def _tie_free_scan_kwargs():
    coords, adjacency = _irregular_path(8)
    rng = np.random.default_rng(7)
    return dict(
        coords=coords,
        adjacency=adjacency,
        observed=rng.poisson(3.0, 8),
        expected=np.full(8, 3.0),
        simcount=0,
        K=4,
        clustersize=3,
    )


def test_default_and_explicitly_disabled_heuristics_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flx.scan(**_tie_free_scan_kwargs())
        flx.scan(
            **_tie_free_scan_kwargs(), enable_frame_ub_prune=False, enable_center_ub_gating=False
        )


def test_circular_scans_ignore_the_flexible_only_heuristics_silently() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flx.scan(**_tie_free_scan_kwargs(), scanmethod="CIRCULAR", enable_frame_ub_prune=True)


# ---------------------------------------------------------------------------
# FXA-190: a non-integer identifier column is reported by name, not as a bare int() error
# ---------------------------------------------------------------------------


def test_pool_to_geodataframe_names_the_identifier_column_for_non_integer_ids() -> None:
    frame = pd.DataFrame({"region": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="region") as excinfo:
        flx.pool_to_geodataframe(frame, [frozenset({1, 2})], "region")
    assert "'a'" in str(excinfo.value)
    assert "invalid literal" not in str(excinfo.value)


def test_pool_to_geodataframe_keeps_accepting_numeric_strings() -> None:
    frame = pd.DataFrame({"region": ["1", "2", "3"]})
    out = flx.pool_to_geodataframe(frame, [frozenset({1, 2})], "region")
    assert out["PoolMember"].tolist() == [1, 1, 0]


def test_choropleth_names_the_identifier_column_for_non_integer_ids() -> None:
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("matplotlib")
    from shapely.geometry import box

    from flexscanx import plotting

    frame = gpd.GeoDataFrame(
        {"region": ["a", "b"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857"
    )
    with pytest.raises(ValueError, match="region") as excinfo:
        plotting.choropleth(frame, {1: 0.5, 2: 0.7}, "region")
    assert "invalid literal" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Graph-metric degenerate-case conventions (documented 2026-09-07; pins of existing behaviour)
# ---------------------------------------------------------------------------


def test_dunn_graph_degenerate_conventions_match_the_docstring() -> None:
    from flexscanx._graph_metrics import dunn_graph

    edge_free = ([0, 0, 0, 0, 0], [])
    assert math.isnan(dunn_graph(*edge_free, [1, 1, 0, 0]))  # nothing reachable
    assert dunn_graph(*edge_free, [1, 0, 0, 0]) == math.inf  # single node: zero diameter
    path = ([0, 1, 3, 5, 6], [1, 0, 2, 1, 3, 2])
    assert dunn_graph(*path, [1, 0, 0, 1]) == pytest.approx(1 / 3)  # members joined via the rest
    assert (
        dunn_graph([0, 1, 3, 4, 4], [1, 0, 2, 1], [1, 1, 1, 0]) == math.inf
    )  # complement unreachable
    assert dunn_graph([0, 1, 2, 2], [1, 0], [1, 0, 1]) == 0.0  # members mutually unreachable


def test_conductance_zero_volume_side_is_inf() -> None:
    from flexscanx._graph_metrics import conductance

    assert conductance([0, 0, 0, 0, 0], [], [1, 1, 0, 0]) == math.inf
    assert conductance([0, 1, 3, 4, 4], [1, 0, 2, 1], [1, 1, 1, 0]) == math.inf
    assert conductance([0, 1, 3, 5, 6], [1, 0, 2, 1, 3, 2], [1, 1, 0, 0]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# FXA-194: plot_consensus_frequency identifier handling matches its siblings
# ---------------------------------------------------------------------------


def _square_gdf(ids: list):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"region": ids}, geometry=[box(i, 0, i + 1, 1) for i in range(len(ids))]
    )


def _plotted_values(ax) -> list[float]:
    arrays = [c.get_array() for c in ax.collections if c.get_array() is not None]
    return sorted(float(v) for arr in arrays for v in np.asarray(arr).ravel())


def test_plot_consensus_frequency_accepts_numeric_string_ids() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plot_consensus import plot_consensus_frequency

    freq = [0.2, 0.9, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax_int = plot_consensus_frequency(_square_gdf([1, 2, 3]), freq, "region")
        ax_str = plot_consensus_frequency(_square_gdf(["1", "2", "3"]), freq, "region")
    assert _plotted_values(ax_int) == sorted(freq)
    assert _plotted_values(ax_str) == sorted(freq), "numeric strings must colour the map"


def test_plot_consensus_frequency_names_the_identifier_column_for_non_integer_ids() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plot_consensus import plot_consensus_frequency

    with pytest.raises(
        ValueError, match=r"id_col 'region' must contain integer area IDs.*row 0 has 'a'"
    ):
        plot_consensus_frequency(_square_gdf(["a", "b", "c"]), [0.2, 0.9, 0.5], "region")


def test_plot_consensus_frequency_rejects_a_column_matching_no_area_id() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plot_consensus import plot_consensus_frequency

    with pytest.raises(ValueError, match=r"No area IDs .*region.* matched"):
        plot_consensus_frequency(_square_gdf([10, 11, 12]), [0.2, 0.9, 0.5], "region")


# ---------------------------------------------------------------------------
# FXA-195: the cluster plotters name a non-integer identifier column and refuse an all-empty map
# ---------------------------------------------------------------------------

_HOT_RESULT = {
    "type": "HOT",
    "area": [2, 3],
    "stats": 4.0,
    "pval": 0.01,
    "secondaries": [{"area": [1], "stats": 1.0, "pval": 0.02}],
    "scan_settings": {},
}


def _both_result() -> dict:
    hot = {"type": "HOT", "area": [2, 3], "stats": 4.0, "pval": 0.01, "secondaries": []}
    cold = {"type": "COLD", "area": [], "stats": 0.0, "pval": None, "secondaries": []}
    return {
        "type": "BOTH",
        "primary_type": "HOT",
        "primary": dict(hot),
        "hot": hot,
        "cold": cold,
        "scan_settings": {},
    }


def _in_cluster_flags(ax) -> list[int]:
    collections = [c for c in ax.collections if len(c.get_paths())]
    arrays = [c.get_array() for c in collections]
    if all(arr is not None for arr in arrays):
        return sorted(int(v) for arr in arrays for v in np.asarray(arr).ravel())
    # geopandas 1.2 draws a categorical column as one collection per category, in category
    # order, with no array. InCluster's categories are 0 and 1, and these tests draw both.
    return sorted(code for code, c in enumerate(collections) for _ in c.get_paths())


def test_plot_cluster_membership_accepts_numeric_strings_and_names_letters() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plotting import plot_cluster_membership

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax_int = plot_cluster_membership(_square_gdf([1, 2, 3]), _HOT_RESULT, "region")
        ax_str = plot_cluster_membership(_square_gdf(["1", "2", "3"]), _HOT_RESULT, "region")
    assert _in_cluster_flags(ax_int) == [0, 1, 1]
    assert _in_cluster_flags(ax_str) == [0, 1, 1]
    with pytest.raises(
        ValueError, match=r"id_col 'region' must contain integer area IDs.*row 0 has 'a'"
    ):
        plot_cluster_membership(_square_gdf(["a", "b", "c"]), _HOT_RESULT, "region")


def test_plot_cluster_membership_rejects_a_cluster_absent_from_the_geodata() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plotting import plot_cluster_membership

    with pytest.raises(ValueError, match=r"No cluster IDs matched .*region"):
        plot_cluster_membership(_square_gdf([10, 11, 12]), _HOT_RESULT, "region")
    # an empty cluster is legitimate (COLD side of a BOTH result with nothing detected)
    empty = {"type": "COLD", "area": [], "stats": 0.0, "pval": None, "scan_settings": {}}
    plot_cluster_membership(_square_gdf([10, 11, 12]), empty, "region")


def test_plot_all_clusters_names_letters_and_rejects_an_all_empty_map() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plotting import plot_all_clusters

    for overlay in (False, True):
        with pytest.raises(
            ValueError, match=r"id_col 'region' must contain integer area IDs.*row 0 has 'a'"
        ):
            plot_all_clusters(_square_gdf(["a", "b", "c"]), _HOT_RESULT, "region", overlay=overlay)
        with pytest.raises(ValueError, match=r"No cluster IDs matched .*region"):
            plot_all_clusters(_square_gdf([10, 11, 12]), _HOT_RESULT, "region", overlay=overlay)
        plot_all_clusters(_square_gdf(["1", "2", "3"]), _HOT_RESULT, "region", overlay=overlay)


def test_plot_all_both_clusters_names_letters_and_rejects_an_all_empty_map() -> None:
    pytest.importorskip("matplotlib")
    from flexscanx.plotting import plot_all_both_clusters

    for overlay in (False, True):
        with pytest.raises(
            ValueError, match=r"id_col 'region' must contain integer area IDs.*row 0 has 'a'"
        ):
            plot_all_both_clusters(
                _square_gdf(["a", "b", "c"]), _both_result(), "region", overlay=overlay
            )
        with pytest.raises(ValueError, match=r"No cluster IDs matched .*region"):
            plot_all_both_clusters(
                _square_gdf([10, 11, 12]), _both_result(), "region", overlay=overlay
            )
        plot_all_both_clusters(
            _square_gdf(["1", "2", "3"]), _both_result(), "region", overlay=overlay
        )
