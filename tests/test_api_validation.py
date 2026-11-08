import warnings

import numpy as np
import pytest

import flexscanx as p
import flexscanx._core as core


def _small_scan_kwargs():
    return {
        "coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "adjacency": ([0, 1, 2], [1, 0]),
        "observed": [1, 2],
        "expected": [1.0, 2.0],
        "K": 2,
        "clustersize": 2,
        "simcount": 0,
    }


def test_scan_validation_errors():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]
    observed = [1, 2]
    expected = [1.0, 2.0]

    with pytest.raises(ValueError):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            clustersize=0,
        )
    with pytest.raises(ValueError):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            ralpha=-0.1,
        )
    with pytest.raises(ValueError):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            clusterradius=-1.0,
        )


def test_scan_series_validation_errors():
    obs = np.array([[1, 2], [3, 4]])
    exp = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]

    with pytest.raises(ValueError):
        p.scan_series(
            observed=obs,
            expected=exp,
            coords=x,
            adjacency=(row_ptr, col_idx),
            clustersize=0,
        )
    with pytest.raises(ValueError):
        p.scan_series(
            observed=obs,
            expected=exp,
            coords=x,
            adjacency=(row_ptr, col_idx),
            ralpha=2.0,
        )
    with pytest.raises(ValueError):
        p.scan_series(
            observed=obs,
            expected=exp,
            coords=x,
            adjacency=(row_ptr, col_idx),
            clusterradius=-0.01,
        )


def test_scan_rejects_invalid_ratio_smoothing():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]

    with pytest.raises(ValueError, match="ratio_smoothing must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            ratio_smoothing="WEB",
            simcount=0,
        )


def test_scan_series_rejects_invalid_ratio_smoothing():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = np.array([[1, 2, 1], [1, 1, 1]])
    expected = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="ratio_smoothing must be one of"):
        p.scan_series(
            observed=observed,
            expected=expected,
            coords=x,
            adjacency=(row_ptr, col_idx),
            ratio_smoothing="HALDANE_X",
            simcount=0,
        )


def test_scan_rejects_invalid_modes_before_core_dispatch():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]

    with pytest.raises(ValueError, match="knn_method must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            knn_method="mesh",
            simcount=0,
        )

    with pytest.raises(ValueError, match="knn_mode must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            knn_mode="REFILL",
            simcount=0,
        )

    with pytest.raises(ValueError, match="scheduler_backend must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            scheduler_backend="work-steal",
            simcount=0,
        )

    with pytest.raises(ValueError, match="scanmethod must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            scanmethod="SPIRAL",
            simcount=0,
        )

    with pytest.raises(ValueError, match="stattype must be one of"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            scanmethod="FLEXIBLE",
            stattype="CLASSIC",
            simcount=0,
        )


def test_scan_and_series_reject_invalid_cluster_direction():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    adjacency = ([0, 1, 2], [1, 0])

    with pytest.raises(ValueError, match="clustertype must be one of"):
        p.scan(
            coords=x,
            adjacency=adjacency,
            observed=[1, 2],
            expected=[1.0, 2.0],
            clustertype="SIDEWAYS",
            simcount=0,
        )

    with pytest.raises(ValueError, match="clustertype must be one of"):
        p.scan_series(
            coords=x,
            adjacency=adjacency,
            observed=np.array([[1, 2]]),
            expected=np.array([[1.0, 2.0]]),
            clustertype="SIDEWAYS",
            simcount=0,
        )


def test_core_scan_slice_rejects_invalid_cluster_direction():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    handle = core.BuildStatic(x, [0, 1, 2], [1, 0], 2, float("inf"))

    with pytest.raises(ValueError, match="clustertype must be one of"):
        core.ScanSlice(
            handle,
            [1, 2],
            [1.0, 2.0],
            {"clustertype": "SIDEWAYS", "simcount": 0},
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"model": "QUASIPOISSON"}, "model must be one of"),
        ({"model": "POISSON", "rantype": "BINOMIAL"}, "rantype"),
        ({"clustersize": 0}, "clustersize must be >= 1"),
        ({"ralpha": np.nan}, "ralpha must be finite"),
        ({"n_jobs_centers": 0}, "n_jobs_centers must be >= 1"),
        ({"simcount": -1}, "simcount must be >= 0"),
        ({"mc_alpha": 0.0}, "mc_alpha must be in"),
        ({"mc_progress_every": 0}, "mc_progress_every must be >= 1"),
    ],
)
def test_core_scan_slice_rejects_invalid_settings(settings, message):
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    handle = core.BuildStatic(x, [0, 1, 2], [1, 0], 2, float("inf"))

    with pytest.raises(ValueError, match=message):
        core.ScanSlice(handle, [1, 2], [1.0, 2.0], settings)


@pytest.mark.parametrize(
    ("observed", "expected", "settings", "message"),
    [
        ([-1, 2], [1.0, 2.0], {"model": "POISSON"}, "observed must be non-negative"),
        ([1, 2], [1.0, np.nan], {"model": "POISSON"}, "expected must be finite"),
        ([1, 2], [1.5, 3.0], {"model": "BINOMIAL"}, "population must contain integers"),
        (
            [2, 2],
            [1.0, 3.0],
            {"model": "BINOMIAL"},
            "observed cannot exceed population",
        ),
    ],
)
def test_core_scan_slice_rejects_invalid_statistical_arrays(observed, expected, settings, message):
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    handle = core.BuildStatic(x, [0, 1, 2], [1, 0], 2, float("inf"))

    with pytest.raises(ValueError, match=message):
        core.ScanSlice(handle, observed, expected, settings)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"clustersize": 1.5}, "clustersize must be an integer"),
        ({"K": 1.5}, "K must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"simcount": -1}, "simcount must be >= 0"),
        ({"simcount": 1.5}, "simcount must be an integer"),
        ({"n_jobs_centers": 0}, "n_jobs_centers must be >= 1"),
        ({"mc_alpha": np.nan}, "mc_alpha must be finite"),
        ({"mc_alpha": 0.0}, "mc_alpha must be in"),
        ({"mc_progress_every": 0}, "mc_progress_every must be >= 1"),
        ({"secondary": -1}, "secondary must be >= 0"),
        ({"secondary": 1.5}, "secondary must be an integer"),
        ({"secondary_p_stop": 1.5}, "secondary_p_stop must be in"),
        ({"clusterradius": np.nan}, "clusterradius must be finite"),
    ],
)
def test_scan_rejects_invalid_numeric_control_contracts(overrides, message):
    kwargs = _small_scan_kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        p.scan(**kwargs)


def test_scan_requires_one_dimensional_observed_counts():
    kwargs = _small_scan_kwargs()
    kwargs["observed"] = np.array([[1, 2]])

    with pytest.raises(ValueError, match="observed must be a 1D array"):
        p.scan(**kwargs)


def test_scan_series_validates_worker_count_and_time_labels():
    common = _small_scan_kwargs()
    common["observed"] = np.array([[1, 2], [2, 1]])
    common["expected"] = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="n_jobs_time must be >= 1"):
        p.scan_series(**common, n_jobs_time=0)

    with pytest.raises(ValueError, match="times must have length T"):
        p.scan_series(**common, times=["only-one"])


@pytest.mark.parametrize(
    "coords",
    [
        np.array([[0.0, 0.0], [np.nan, 0.0]]),
        np.array([[0.0, 0.0], [np.inf, 0.0]]),
    ],
)
def test_static_builder_rejects_nonfinite_coordinates(coords):
    with pytest.raises(ValueError, match="coords must contain only finite"):
        p.build_static(coords=coords, adjacency=([0, 1, 2], [1, 0]), K=2)


def test_static_builder_rejects_fractional_csr_indices_and_invalid_radius():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="integer indices"):
        p.build_static(coords=coords, adjacency=([0, 1.5, 2], [1, 0]), K=2)

    with pytest.raises(ValueError, match="clusterradius must be non-negative"):
        p.build_static(
            coords=coords,
            adjacency=([0, 1, 2], [1, 0]),
            K=2,
            clusterradius=-1.0,
        )


def test_scan_rejects_reused_handle_with_wrong_n():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]
    handle = p.build_static(coords=x, adjacency=(row_ptr, col_idx), K=1)

    with pytest.raises(ValueError, match="static handle N mismatch"):
        p.scan(
            handle=handle,
            observed=[1, 2, 3],
            expected=[1.0, 2.0, 3.0],
            simcount=0,
        )


def test_reused_handle_reports_authoritative_static_provenance():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    adjacency = ([0, 1, 3, 4], [1, 0, 2, 1])
    handle = p.build_static(
        coords=coords,
        adjacency=adjacency,
        K=2,
        clusterradius=1.25,
        knn_mode="K_FIXED",
    )
    static = p.describe_static(handle)

    single = p.scan(
        handle=handle,
        observed=[4, 2, 1],
        expected=[2.0, 2.0, 2.0],
        clustersize=2,
        simcount=0,
    )
    series = p.scan_series(
        handle=handle,
        observed=np.array([[4, 2, 1]]),
        expected=np.array([[2.0, 2.0, 2.0]]),
        clustersize=2,
        simcount=0,
    )

    for settings in (single["scan_settings"], series["scan_settings"]):
        assert settings["K"] == static["K"] == 2
        assert settings["clusterradius"] == static["radius"] == 1.25
        assert settings["knn_mode"] == static["knn_mode"] == "K_FIXED"
        assert settings["knn_method"] is None
        assert settings["knn_refill_force"] is None
        assert settings["static"] == static


def test_core_scan_slice_rejects_invalid_ratio_smoothing():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    handle = core.BuildStatic(x, row_ptr, col_idx, 2, float("inf"))

    with pytest.raises(ValueError, match="ratio_smoothing must be one of"):
        core.ScanSlice(
            handle,
            [1, 2, 1],
            [1.0, 2.0, 1.0],
            {"ratio_smoothing": "WEB", "simcount": 0},
        )


def test_scan_rejects_names_length_mismatch():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]

    with pytest.raises(ValueError, match="names must have length N"):
        p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            names=["A"],
            simcount=0,
        )


def test_scan_series_rejects_names_length_mismatch():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = np.array([[1, 2, 1], [1, 1, 1]])
    expected = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="names must have length N"):
        p.scan_series(
            observed=observed,
            expected=expected,
            coords=x,
            adjacency=(row_ptr, col_idx),
            names=["A"],
            simcount=0,
        )


def test_clustersize_exceeding_k_is_capped_scan():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [1, 2, 1]
    expected = [1.0, 2.0, 1.0]

    with pytest.warns(UserWarning, match="capping clustersize to K"):
        res = p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            expected=expected,
            clustersize=3,
            K=2,
            simcount=0,
        )
    assert res["scan_settings"]["clustersize"] == 2


def test_clustersize_exceeding_k_is_capped_scan_series():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = np.array([[1, 2, 1], [1, 1, 1]])
    expected = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    with pytest.warns(UserWarning, match="capping clustersize to K"):
        res = p.scan_series(
            observed=observed,
            expected=expected,
            coords=x,
            adjacency=(row_ptr, col_idx),
            clustersize=3,
            K=2,
            simcount=0,
        )
    assert res["scan_settings"]["clustersize"] == 2


def test_default_clustersize_capping_does_not_warn_when_only_k_is_small_scan():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.scan(
            coords=x,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2, 1],
            expected=[1.0, 2.0, 1.0],
            K=2,
            simcount=0,
        )
    assert res["scan_settings"]["clustersize"] == 2
    assert not any("capping clustersize to K" in str(w.message) for w in caught)


def test_default_clustersize_capping_does_not_warn_when_only_k_is_small_scan_series():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = np.array([[1, 2, 1], [1, 1, 1]])
    expected = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.scan_series(
            observed=observed,
            expected=expected,
            coords=x,
            adjacency=(row_ptr, col_idx),
            K=2,
            simcount=0,
        )
    assert res["scan_settings"]["clustersize"] == 2
    assert not any("capping clustersize to K" in str(w.message) for w in caught)


def test_graph_csr_validation_rejects_bad_row_ptr_length():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    observed = [1, 2]
    expected = [1.0, 2.0]

    with pytest.raises(ValueError, match="row_ptr must have length N\\+1"):
        p.scan(
            coords=x,
            adjacency=([0, 1], [1]),  # invalid row_ptr length
            observed=observed,
            expected=expected,
            simcount=0,
        )


def test_graph_csr_validation_rejects_self_loops():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    observed = [1, 2]
    expected = [1.0, 2.0]

    with pytest.raises(ValueError, match="self-loop"):
        p.scan(
            coords=x,
            adjacency=([0, 1, 2], [0, 0]),  # row 0 has self-loop 0->0
            observed=observed,
            expected=expected,
            simcount=0,
        )


def test_graph_csr_validation_rejects_asymmetry():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    observed = [1, 2]
    expected = [1.0, 2.0]

    with pytest.raises(ValueError, match="must be symmetric"):
        p.scan(
            coords=x,
            adjacency=([0, 1, 1], [1]),  # edge 0->1 exists, 1->0 missing
            observed=observed,
            expected=expected,
            simcount=0,
        )


def test_graph_csr_validation_rejects_duplicate_neighbors():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    observed = [1, 2, 1]
    expected = [1.0, 2.0, 1.0]

    with pytest.raises(ValueError, match="duplicate neighbors"):
        p.scan(
            coords=x,
            adjacency=([0, 2, 4, 5], [1, 1, 0, 2, 1]),  # row 0 duplicates neighbor 1
            observed=observed,
            expected=expected,
            simcount=0,
        )


def test_build_static_rejects_bad_row_ptr_length():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="row_ptr must have length N\\+1"):
        p.build_static(
            coords=x,
            adjacency=([0, 1], [1]),
            K=2,
        )


def test_build_static_rejects_self_loops():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="self-loop"):
        p.build_static(
            coords=x,
            adjacency=([0, 1, 2], [0, 0]),
            K=2,
        )


def test_build_static_rejects_asymmetry():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="must be symmetric"):
        p.build_static(
            coords=x,
            adjacency=([0, 1, 1], [1]),
            K=2,
        )


def test_build_static_rejects_duplicate_neighbors():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    with pytest.raises(ValueError, match="duplicate neighbors"):
        p.build_static(
            coords=x,
            adjacency=([0, 2, 4, 5], [1, 1, 0, 2, 1]),
            K=3,
        )


def test_core_buildstatic_rejects_self_loops():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(RuntimeError, match="self-loop"):
        core.BuildStatic(x, [0, 1, 2], [0, 0], 2, float("inf"))


def test_core_buildstatic_rejects_out_of_range_indices():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(RuntimeError, match="out-of-range"):
        core.BuildStatic(x, [0, 1, 2], [1, 2], 2, float("inf"))


def test_both_result_primary_type_is_consistent():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [5, 2, 1]
    expected = [2.0, 2.0, 2.0]

    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        clustertype="BOTH",
        K=3,
        clustersize=2,
        simcount=0,
    )
    assert res["type"] == "BOTH"
    assert res["primary_type"] in ("HOT", "COLD")
    assert res["primary"]["type"] == res["primary_type"]
    assert "primary_direction" not in res
