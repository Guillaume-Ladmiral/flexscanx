from __future__ import annotations

from concurrent.futures import Future

import numpy as np

import flexscanx.api as api


def _fake_ensure_static_handle(
    handle,
    coords,
    adjacency,
    K,
    N_from_obs,
    clusterradius,
    knn_method,
    knn_mode,
    knn_refill_force,
):
    if N_from_obs is not None:
        n = int(N_from_obs)
    elif coords is not None:
        n = int(np.asarray(coords).shape[0])
    elif adjacency is not None and len(adjacency[0]) >= 1:
        n = int(len(adjacency[0]) - 1)
    else:  # pragma: no cover - defensive
        raise AssertionError("Unable to infer N in fake static handle helper")
    eff_k = int(K) if K is not None else n
    return object(), n, eff_k, {"K": eff_k}


def test_scan_settings_normalized_by_shared_builder(monkeypatch):
    captured: list[dict] = []

    def _fake_scan_slice(_handle, _obs, _exp, settings):
        captured.append(dict(settings))
        return {
            "area": [1],
            "n_case": 1,
            "expected": 1.0,
            "stats": 1.0,
            "rank": 1,
            "pval": 1.0,
            "secondaries": [],
        }

    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)
    monkeypatch.setenv("FLEXSCANX_MC_ALIAS", "1")

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    api.scan(
        coords=coords,
        observed=[4, 2, 1],
        expected=[2.0, 2.0, 2.0],
        K=3,
        simcount=0,
        secondary="ALL",
        ratio_smoothing="haldane",
    )
    settings = captured[-1]
    assert settings["secondary"] == 3
    assert settings["secondary_p_stop"] == 1.0
    assert settings["mc_alias"] is True
    assert settings["ratio_smoothing"] == "HALDANE"

    api.scan(
        coords=coords,
        observed=[4, 2, 1],
        expected=[2.0, 2.0, 2.0],
        K=3,
        simcount=0,
    )
    settings_no_secondary = captured[-1]
    assert "secondary" not in settings_no_secondary
    assert "secondary_p_stop" not in settings_no_secondary


def test_scan_series_settings_normalized_by_shared_builder(monkeypatch):
    captured: list[dict] = []

    def _fake_scan_slice(_handle, _obs, _exp, settings):
        captured.append(dict(settings))
        return {
            "area": [1],
            "n_case": 1,
            "expected": 1.0,
            "stats": 1.0,
            "rank": 1,
            "pval": 1.0,
            "secondaries": [],
        }

    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    observed = np.asarray([[4, 2, 1], [5, 2, 1]], dtype=int)
    expected = np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=float)

    api.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        K=3,
        simcount=0,
        secondary="ALL",
        ratio_smoothing="eb",
        seed=123,
        crn=True,
    )
    first_run = captured[:]
    assert len(first_run) == observed.shape[0]
    for t, settings in enumerate(first_run):
        assert settings["secondary"] == 3
        assert settings["secondary_p_stop"] == 1.0
        assert settings["ratio_smoothing"] == "EB"
        assert settings["seed"] == 123
        assert settings["crn"] is True
        assert settings["time_index"] == t

    captured.clear()
    api.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        K=3,
        simcount=0,
    )
    assert len(captured) == observed.shape[0]
    assert all("secondary" not in settings for settings in captured)
    assert all("secondary_p_stop" not in settings for settings in captured)


def test_scan_series_forwards_secondary_stop_controls_and_both_parallel(monkeypatch):
    captured: list[dict] = []
    threadpool_calls: list[int] = []

    class _ImmediateExecutor:
        def __init__(self, *, max_workers: int):
            threadpool_calls.append(int(max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut: Future = Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    def _fake_scan_slice(_handle, _obs, _exp, settings):
        captured.append(dict(settings))
        return {
            "area": [1],
            "n_case": 1,
            "expected": 1.0,
            "stats": 1.0,
            "rank": 1,
            "pval": 0.01,
            "secondaries": [],
        }

    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _ImmediateExecutor)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    observed = np.asarray([[4, 2, 1]], dtype=int)
    expected = np.asarray([[2.0, 2.0, 2.0]], dtype=float)

    api.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        K=3,
        simcount=0,
        clustertype="BOTH",
        both_parallel=True,
        secondary="ALL",
        secondary_p_stop=0.25,
        secondary_stats_stop=3.5,
    )

    assert threadpool_calls == [2]
    assert len(captured) == 2
    assert {settings["clustertype"] for settings in captured} == {"HOT", "COLD"}
    for settings in captured:
        assert settings["secondary"] == 3
        assert settings["secondary_p_stop"] == 0.25
        assert settings["secondary_stats_stop"] == 3.5
