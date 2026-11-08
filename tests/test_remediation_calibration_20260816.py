"""Regression tests for the 2026-08-16 calibration-cache remediation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import flexscanx._calibration as calibration_module
import flexscanx.validation as validation
from flexscanx import _core as core_module

VALID_ENTRY = {
    "tau": 2.5,
    "mu": 1.0,
    "sigma": 0.5,
    "alpha": 0.05,
    "n_sims": 20,
}


def test_private_calibration_module_has_no_shadow_public_facade() -> None:
    assert not hasattr(calibration_module, "calibrate_llr_threshold")


def test_non_utf8_cache_is_ignored_and_repaired(tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    cache_file.write_bytes(b"\xff\xfe\x00\x01garbage")

    assert validation._cache_load(str(cache_file)) == {}
    validation._cache_save({"key": VALID_ENTRY}, str(cache_file))

    assert validation._cache_load(str(cache_file)) == {"key": VALID_ENTRY}
    assert json.loads(cache_file.read_text(encoding="utf-8"))["entries"]


def test_cache_entry_integrity_rejects_tampering(tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    validation._cache_save({"key": VALID_ENTRY}, str(cache_file))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["entries"]["key"]["value"]["tau"] = 0.0
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    assert validation._cache_load(str(cache_file)) == {}


def test_truncated_integrity_key_is_repaired_on_next_save(tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    validation._cache_save({"first": VALID_ENTRY}, str(cache_file))
    key_file = cache_file.with_suffix(".json.key")
    key_file.write_bytes(b"truncated")

    assert validation._cache_load(str(cache_file)) == {}
    validation._cache_save({"second": VALID_ENTRY}, str(cache_file))

    assert key_file.stat().st_size == 32
    assert validation._cache_load(str(cache_file)) == {"second": VALID_ENTRY}


def test_cache_save_failure_warns_and_preserves_symlink_target(tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    cache_file.with_suffix(".json.lock").symlink_to(victim)

    with pytest.warns(UserWarning, match="could not save calibration cache"):
        validation._cache_save({"key": VALID_ENTRY}, str(cache_file))

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert not cache_file.exists()


def test_unseeded_calibration_bypasses_persistent_cache(monkeypatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    monkeypatch.setattr(validation, "_CACHE_FILE", str(cache_file), raising=False)
    monkeypatch.setattr(
        validation,
        "_ensure_static_handle",
        lambda **kwargs: (object(), 0, int(kwargs["K"]), None),
    )
    scan_calls: list[int] = []

    def fake_scan_slice(handle, observed, expected, settings):
        scan_calls.append(1)
        return {"stats": float(len(scan_calls))}

    monkeypatch.setattr(core_module, "ScanSlice", fake_scan_slice)
    kwargs = {
        "coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "adjacency": ([0, 1, 2], [1, 0]),
        "observed": [1, 1],
        "expected": [1.0, 1.0],
        "model": "POISSON",
        "K": 2,
        "clustersize": 2,
        "n_sims": 2,
        "seed": None,
    }

    first = validation.calibrate_llr_threshold(**kwargs)
    second = validation.calibrate_llr_threshold(**kwargs)

    assert len(scan_calls) == 4
    assert first != second
    assert not cache_file.exists()


def test_seeded_calibration_still_uses_integrity_checked_cache(monkeypatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "calibration.json"
    monkeypatch.setattr(validation, "_CACHE_FILE", str(cache_file), raising=False)
    monkeypatch.setattr(
        validation,
        "_ensure_static_handle",
        lambda **kwargs: (object(), 0, int(kwargs["K"]), None),
    )
    scan_calls: list[int] = []

    def fake_scan_slice(handle, observed, expected, settings):
        scan_calls.append(1)
        return {"stats": float(len(scan_calls))}

    monkeypatch.setattr(core_module, "ScanSlice", fake_scan_slice)
    kwargs = {
        "coords": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "adjacency": ([0, 1, 2], [1, 0]),
        "observed": [1, 1],
        "expected": [1.0, 1.0],
        "model": "POISSON",
        "K": 2,
        "clustersize": 2,
        "n_sims": 2,
        "seed": 42,
    }

    first = validation.calibrate_llr_threshold(**kwargs)
    second = validation.calibrate_llr_threshold(**kwargs)

    assert len(scan_calls) == 2
    assert first == second
    assert validation._cache_load(str(cache_file))
