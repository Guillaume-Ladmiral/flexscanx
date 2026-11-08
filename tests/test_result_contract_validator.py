from __future__ import annotations

import numpy as np
import pytest

import flexscanx.api as api
from flexscanx._contracts import (
    validate_scan_result_contract,
    validate_series_entry_contract,
    validate_series_result_contract,
)


def test_validate_scan_result_contract_rejects_missing_primary_type_for_both():
    bad = {
        "type": "BOTH",
        "primary": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "HOT",
        },
        "hot": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "HOT",
        },
        "cold": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "COLD",
        },
    }
    with pytest.raises(ValueError, match="missing required keys"):
        validate_scan_result_contract(bad)


def test_validate_scan_result_contract_rejects_primary_type_mismatch():
    bad = {
        "type": "BOTH",
        "primary_type": "COLD",
        "primary": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "HOT",
        },
        "hot": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "HOT",
        },
        "cold": {
            "area": [],
            "n_case": 0,
            "expected": 0.0,
            "stats": 0.0,
            "rank": 1,
            "type": "COLD",
        },
    }
    with pytest.raises(ValueError, match="must match"):
        validate_scan_result_contract(bad)


def test_validate_series_result_contract_rejects_times_per_time_length_mismatch():
    bad = {
        "per_time": [
            {"area": [], "n_case": 0, "expected": 0.0, "stats": 0.0, "rank": 1, "type": "HOT"}
        ],
        "times": [],
        "scan_settings": {},
    }
    with pytest.raises(ValueError, match="must equal"):
        validate_series_result_contract(bad)


def test_validate_series_entry_contract_accepts_per_time_payload_without_scan_settings():
    entry = {"area": [], "n_case": 0, "expected": 0.0, "stats": 0.0, "rank": 1, "type": "HOT"}
    out = validate_series_entry_contract(entry)
    assert out["type"] == "HOT"


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
    else:
        n = int(np.asarray(coords).shape[0])
    eff_k = int(K) if K is not None else n
    return object(), n, eff_k, {"K": eff_k}


def test_scan_runtime_contract_enforced(monkeypatch):
    def _bad_scan_slice(_handle, _obs, _exp, settings):
        # Missing n_case/expected/stats/rank required by contract.
        return {"area": [1], "secondaries": []}

    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _bad_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="missing required keys"):
        api.scan(
            coords=coords,
            observed=[1, 2],
            expected=[1.0, 2.0],
            K=2,
            simcount=0,
        )


def test_scan_series_runtime_contract_enforced(monkeypatch):
    def _bad_scan_slice(_handle, _obs, _exp, settings):
        return {"area": [1], "secondaries": []}

    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _bad_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    observed = np.array([[1, 2], [2, 1]], dtype=int)
    expected = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="missing required keys"):
        api.scan_series(
            observed=observed,
            expected=expected,
            coords=coords,
            K=2,
            simcount=0,
        )
