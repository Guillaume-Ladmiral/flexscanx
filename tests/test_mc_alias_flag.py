import numpy as np

import flexscanx as p
import flexscanx.api as api


def test_scan_mc_alias_flag_smoke(monkeypatch):
    monkeypatch.setenv("FLEXSCANX_MC_ALIAS", "1")
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 2, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=3,
        ralpha=0.2,
        K=3,
        simcount=99,
        seed=123,
    )
    assert isinstance(res, dict)
    assert 1 <= int(res.get("rank", 1)) <= 100


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


def test_scan_series_mc_alias_flag_sets_core_option(monkeypatch):
    captured: list[dict] = []

    def _fake_scan_slice(_handle, _obs, _exp, settings):
        captured.append(dict(settings))
        return {
            "area": [1],
            "n_case": 1,
            "expected": 1.0,
            "stats": 1.0,
            "rank": 1,
            "secondaries": [],
        }

    monkeypatch.setenv("FLEXSCANX_MC_ALIAS", "1")
    monkeypatch.setattr(api, "ensure_static_handle", _fake_ensure_static_handle)
    monkeypatch.setattr(api.core, "ScanSlice", _fake_scan_slice)

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    observed = np.array([[1, 2], [2, 1]], dtype=int)
    expected = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    out = p.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        K=2,
        simcount=0,
    )
    assert isinstance(out, dict)
    assert len(captured) == observed.shape[0]
    assert all(s.get("mc_alias") is True for s in captured)
