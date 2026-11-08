from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import flexscanx.validation as v
from flexscanx import _core as core_mod
from flexscanx.validation import _get_cache_dir, _validate_cache_entry


def _line_graph_csr(n: int) -> tuple[list[int], list[int]]:
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        nbrs: list[int] = []
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def test_calibration_cache_hits_and_misses_on_expected_fingerprint(monkeypatch, tmp_path):
    cache_file = tmp_path / "calibration_cache.json"
    monkeypatch.setattr(v, "_CACHE_FILE", str(cache_file), raising=False)

    # Avoid depending on the real static-handle builder in this cache-focused test.
    def _fake_ensure_static_handle(*args, **kwargs):
        k = kwargs.get("K")
        if k is None and len(args) >= 4:
            k = args[3]
        return object(), 0, (None if k is None else int(k)), None

    monkeypatch.setattr(v, "_ensure_static_handle", _fake_ensure_static_handle)

    calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def _fake_scan_slice(handle, observed, expected, settings):
        calls.append((tuple(int(x) for x in observed), tuple(float(x) for x in expected)))
        return {"stats": float(sum(observed))}

    monkeypatch.setattr(core_mod, "ScanSlice", _fake_scan_slice)

    n = 4
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    row_ptr, col_idx = _line_graph_csr(n)
    observed = [10, 8, 7, 5]
    expected_a = [1.0, 2.0, 3.0, 4.0]  # sum = 10
    expected_b = [4.0, 1.0, 3.0, 2.0]  # sum = 10 (same total, different distribution)

    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        n_sims=5,
        alpha=0.1,
        seed=123,
    )

    out1 = v.calibrate_llr_threshold(expected=expected_a, **common)
    assert len(calls) == 5

    # Same inputs: must hit cache (no additional ScanSlice calls).
    out2 = v.calibrate_llr_threshold(expected=expected_a, **common)
    assert len(calls) == 5
    assert out2 == out1

    # Different expected distribution (same sum): must miss cache.
    out3 = v.calibrate_llr_threshold(expected=expected_b, **common)
    assert len(calls) == 10
    assert set(out3.keys()) == {"tau", "mu", "sigma", "alpha", "n_sims"}

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["__meta__"]["schema_version"] == v._CACHE_SCHEMA_VERSION
    assert payload["__meta__"]["format"] == v._CACHE_FORMAT
    assert len(payload["entries"]) == 2


def test_calibration_multinomial_uses_observed_total_and_misses_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "calibration_cache.json"
    monkeypatch.setattr(v, "_CACHE_FILE", str(cache_file), raising=False)
    monkeypatch.delenv("FLEXSCANX_MC_ALIAS", raising=False)

    def _fake_ensure_static_handle(*args, **kwargs):
        k = kwargs.get("K")
        if k is None and len(args) >= 4:
            k = args[3]
        return object(), 0, (None if k is None else int(k)), None

    monkeypatch.setattr(v, "_ensure_static_handle", _fake_ensure_static_handle)

    simulated_totals: list[int] = []

    def _fake_scan_slice(handle, observed, expected, settings):
        total = int(sum(int(x) for x in observed))
        simulated_totals.append(total)
        return {"stats": float(total)}

    monkeypatch.setattr(core_mod, "ScanSlice", _fake_scan_slice)

    n = 4
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    row_ptr, col_idx = _line_graph_csr(n)
    expected = [1.0, 2.0, 3.0, 4.0]

    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        n_sims=5,
        alpha=0.1,
        seed=123,
    )

    out_a = v.calibrate_llr_threshold(observed=[1, 2, 3, 4], **common)
    assert out_a == {"tau": 10.0, "mu": 10.0, "sigma": 0.0, "alpha": 0.1, "n_sims": 5}
    assert simulated_totals == [10] * 5

    out_b = v.calibrate_llr_threshold(observed=[2, 4, 6, 8], **common)
    assert out_b == {"tau": 20.0, "mu": 20.0, "sigma": 0.0, "alpha": 0.1, "n_sims": 5}
    assert simulated_totals == ([10] * 5) + ([20] * 5)

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["__meta__"]["schema_version"] == v._CACHE_SCHEMA_VERSION
    assert len(payload["entries"]) == 2


def test_cache_load_invalidates_schema_mismatch_and_legacy_payload(tmp_path):
    cache_file = tmp_path / "calibration_cache.json"

    stale_payload = {
        "__meta__": {
            "format": v._CACHE_FORMAT,
            "schema_version": v._CACHE_SCHEMA_VERSION - 1,
        },
        "entries": {"k1": {"tau": 1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 99}},
    }
    cache_file.write_text(json.dumps(stale_payload), encoding="utf-8")
    assert v._cache_load(str(cache_file)) == {}

    # Legacy flat payload (pre-versioned format) should be invalidated.
    legacy_payload = {"k1": {"tau": 1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 99}}
    cache_file.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert v._cache_load(str(cache_file)) == {}


def test_cache_entry_validation_rejects_nonfinite_and_boolean_numbers():
    valid = {"tau": 1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 99}
    assert _validate_cache_entry(valid)

    for field, invalid in (("tau", np.nan), ("sigma", np.inf), ("n_sims", True)):
        candidate = {**valid, field: invalid}
        assert not _validate_cache_entry(candidate), (field, invalid)


def test_cache_save_merges_stale_writer_snapshot(tmp_path):
    """A delayed writer must not erase entries committed after its load."""
    cache_file = tmp_path / "calibration_cache.json"
    entry_a = {"tau": 1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 99}
    entry_b = {"tau": 2.0, "mu": 1.0, "sigma": 2.0, "alpha": 0.10, "n_sims": 49}

    stale_snapshot = v._cache_load(str(cache_file))
    v._cache_save({"newer": entry_a}, str(cache_file))
    stale_snapshot["delayed"] = entry_b
    v._cache_save(stale_snapshot, str(cache_file))

    assert v._cache_load(str(cache_file)) == {"newer": entry_a, "delayed": entry_b}


def test_cache_save_preserves_all_concurrent_writer_entries(tmp_path):
    cache_file = tmp_path / "calibration_cache.json"
    entry = {"tau": 1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 9}
    writer_count = 16
    barrier = Barrier(writer_count)

    def save_one(index: int) -> None:
        barrier.wait()
        v._cache_save({str(index): entry}, str(cache_file))

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(save_one, range(writer_count)))

    assert set(v._cache_load(str(cache_file))) == {str(index) for index in range(writer_count)}


def test_calibrate_cache_rantype_none_and_multinomial_share_key(monkeypatch, tmp_path):
    """rantype=None and rantype='MULTINOMIAL' must resolve to the same cache key.

    Regression test for the A1 audit finding: prior to the fix, these two calls
    produced different SHA-256 keys ('', 'MULTINOMIAL') causing a permanent cache
    miss for any caller mixing the two forms.
    """
    cache_file = tmp_path / "calibration_cache.json"
    monkeypatch.setattr(v, "_CACHE_FILE", str(cache_file), raising=False)

    def _fake_ensure_static_handle(*args, **kwargs):
        k = kwargs.get("K")
        if k is None and len(args) >= 4:
            k = args[3]
        return object(), 0, (None if k is None else int(k)), None

    monkeypatch.setattr(v, "_ensure_static_handle", _fake_ensure_static_handle)

    scan_calls: list[int] = []

    def _fake_scan_slice(handle, observed, expected, settings):
        scan_calls.append(1)
        return {"stats": float(sum(observed))}

    monkeypatch.setattr(core_mod, "ScanSlice", _fake_scan_slice)

    n = 4
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = _line_graph_csr(n)
    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=[10, 8, 7, 5],
        expected=[1.0, 2.0, 3.0, 4.0],
        model="POISSON",
        K=3,
        clustersize=3,
        n_sims=4,
        alpha=0.1,
        seed=42,
    )

    # First call: rantype=None (default)
    out_none = v.calibrate_llr_threshold(rantype=None, **common)
    assert len(scan_calls) == 4

    # Second call: same computation, explicit rantype="MULTINOMIAL" — must hit cache
    out_mult = v.calibrate_llr_threshold(rantype="MULTINOMIAL", **common)
    assert len(scan_calls) == 4, (
        "Expected cache hit for rantype='MULTINOMIAL' after rantype=None, "
        "but ScanSlice was called again (cache key mismatch)."
    )
    assert out_none == out_mult


def test_calibrate_binomial_requires_population_not_expected(monkeypatch, tmp_path):
    """calibrate_llr_threshold must reject expected= for BINOMIAL and require population=."""
    import pytest

    cache_file = tmp_path / "calib.json"
    monkeypatch.setattr(v, "_CACHE_FILE", str(cache_file), raising=False)

    n = 4
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = _line_graph_csr(n)
    base = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=[2, 3, 1, 1],
        model="BINOMIAL",
        K=3,
        clustersize=3,
        n_sims=2,
        seed=0,
    )

    with pytest.raises(ValueError, match="population"):
        v.calibrate_llr_threshold(expected=[10.0, 10.0, 10.0, 10.0], **base)

    with pytest.raises(ValueError, match="population"):
        v.calibrate_llr_threshold(**base)  # neither provided


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "NEGATIVE_BINOMIAL"}, "Unknown model"),
        ({"rantype": "BOOTSTRAP"}, "Invalid rantype"),
        ({"clustertype": "BOTH"}, "clustertype must be one of"),
        ({"scanmethod": "SPIRAL"}, "scanmethod must be one of"),
        ({"stattype": "CLASSIC"}, "stattype must be one of"),
        ({"n_sims": 1.5}, "n_sims must be an integer"),
        ({"alpha": np.nan}, "alpha must be finite"),
        ({"alpha": 0.0}, "alpha must be in"),
        ({"observed": [1.5, 2]}, "integer values"),
        ({"expected": [1.0, np.nan]}, "NaN values"),
    ],
)
def test_calibration_rejects_invalid_contract_inputs(monkeypatch, tmp_path, overrides, message):
    cache_file = tmp_path / "calibration_cache.json"
    monkeypatch.setattr(v, "_CACHE_FILE", str(cache_file), raising=False)
    monkeypatch.setattr(v, "_ensure_static_handle", lambda **kwargs: (object(), 0, 2, None))
    monkeypatch.setattr(core_mod, "ScanSlice", lambda *args, **kwargs: {"stats": 1.0})

    kwargs = {
        "coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "adjacency": ([0, 1, 2], [1, 0]),
        "observed": [1, 2],
        "expected": [1.0, 2.0],
        "model": "POISSON",
        "scanmethod": "FLEXIBLE",
        "stattype": "RESTRICTED",
        "clustertype": "HOT",
        "K": 2,
        "clustersize": 2,
        "n_sims": 2,
        "alpha": 0.05,
        "seed": 1,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        v.calibrate_llr_threshold(**kwargs)


@pytest.mark.parametrize(
    ("population", "observed", "message"),
    [
        ([2.5, 3], [1, 2], "integer values"),
        ([1, 3], [2, 2], "observed counts cannot exceed population"),
    ],
)
def test_calibration_validates_binomial_population(
    monkeypatch, tmp_path, population, observed, message
):
    monkeypatch.setattr(v, "_CACHE_FILE", str(tmp_path / "cache.json"), raising=False)
    monkeypatch.setattr(v, "_ensure_static_handle", lambda **kwargs: (object(), 0, 2, None))
    monkeypatch.setattr(core_mod, "ScanSlice", lambda *args, **kwargs: {"stats": 1.0})

    with pytest.raises(ValueError, match=message):
        v.calibrate_llr_threshold(
            coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
            adjacency=([0, 1, 2], [1, 0]),
            observed=observed,
            population=population,
            model="BINOMIAL",
            K=2,
            clustersize=2,
            n_sims=2,
            seed=1,
        )


def test_get_cache_dir_without_platformdirs(monkeypatch):
    """_get_cache_dir() must not raise NameError when platformdirs is absent.

    Regression test for missing ``import os`` in _validation_calibration.py.
    """

    # Simulate platformdirs not being installed
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _no_platformdirs(name, *args, **kwargs):
        if name == "platformdirs":
            raise ImportError("mocked: platformdirs not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys.modules["builtins"], "__import__", _no_platformdirs)

    # Ensure XDG_CACHE_HOME and LOCALAPPDATA are not set so we hit the fallback
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    result = _get_cache_dir()
    assert isinstance(result, pathlib.Path)
    assert "flexscanx" in str(result)
