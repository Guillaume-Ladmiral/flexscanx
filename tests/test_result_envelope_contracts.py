from __future__ import annotations

import numpy as np

import flexscanx.api as api
from flexscanx._contracts import (
    BOTH_REQUIRED_KEYS,
    DIRECTION_REQUIRED_KEYS,
    SERIES_REQUIRED_KEYS,
    validate_scan_result_contract,
    validate_series_result_contract,
)


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


def _fake_scan_slice(_handle, arr_obs, arr_exp, settings):
    ct = str(settings.get("clustertype", "HOT")).upper()
    stats = 10.0 if ct == "HOT" else 5.0
    return {
        "area": [1, 2],
        "n_case": int(np.asarray(arr_obs).sum()),
        "expected": float(np.asarray(arr_exp).sum()),
        "stats": float(stats),
        "rank": 1,
        "secondaries": [],
    }


def test_scan_hot_result_envelope_contract(monkeypatch):
    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = api.scan(
        coords=coords,
        observed=[5, 2, 1],
        expected=[2.0, 2.0, 2.0],
        K=3,
        simcount=0,
    )
    assert DIRECTION_REQUIRED_KEYS <= set(out)
    validate_scan_result_contract(out, require_scan_settings=True)
    assert out["type"] == "HOT"
    assert out["scan_settings"]["clustertype"] == "HOT"


def test_scan_both_result_envelope_contract(monkeypatch):
    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = api.scan(
        coords=coords,
        observed=[5, 2, 1],
        expected=[2.0, 2.0, 2.0],
        clustertype="BOTH",
        K=3,
        simcount=0,
    )
    assert BOTH_REQUIRED_KEYS <= set(out)
    validate_scan_result_contract(out, require_scan_settings=True)
    assert out["type"] == "BOTH"
    assert out["primary_type"] == "HOT"
    assert out["primary"]["type"] == "HOT"
    assert "primary_direction" not in out


def test_scan_series_result_envelope_contract_default_times(monkeypatch):
    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    observed = np.asarray([[5, 2, 1], [6, 1, 1]], dtype=int)
    expected = np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=float)
    out = api.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        K=3,
        simcount=0,
    )

    assert SERIES_REQUIRED_KEYS <= set(out)
    validate_series_result_contract(out)
    assert out["times"] == [0, 1]
    assert len(out["per_time"]) == 2
    assert all("type" in r for r in out["per_time"])
    assert out["scan_settings"]["T"] == 2
    assert out["scan_settings"]["N"] == 3


def test_scan_series_result_envelope_contract_custom_times_and_both(monkeypatch):
    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    observed = np.asarray([[5, 2, 1], [6, 1, 1]], dtype=int)
    expected = np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=float)
    labels = ["2026-01", "2026-02"]
    out = api.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        clustertype="BOTH",
        K=3,
        simcount=0,
        times=labels,
    )

    validate_series_result_contract(out)
    assert out["times"] == labels
    assert len(out["per_time"]) == 2
    first = out["per_time"][0]
    assert BOTH_REQUIRED_KEYS <= set(first)
    assert first["type"] == "BOTH"
    assert "primary_direction" not in first
