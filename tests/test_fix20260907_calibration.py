"""Regression tests for the calibration repairs of the 2026-09-07 audit-and-fix run.

Every test redirects the on-disk cache into ``tmp_path`` through the public
``flexscanx.validation`` facade so the user's real cache is never read or written.
"""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx
from flexscanx import validation


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


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "_CACHE_FILE", str(tmp_path / "cache" / "calibration_cache.json")
    )
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# FXA-165: the documented handle= parameter must be usable
# ---------------------------------------------------------------------------


def test_calibrate_llr_threshold_accepts_a_reused_handle(isolated_cache) -> None:
    coords, adjacency = _grid(6)
    observed = np.ones(36, dtype=int)
    expected = np.ones(36)
    handle = flx.build_static(coords, adjacency, K=8)
    via_handle = flx.calibrate_llr_threshold(
        handle=handle, observed=observed, expected=expected, K=8, clustersize=4, n_sims=6, seed=11
    )
    via_graph = flx.calibrate_llr_threshold(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        K=8,
        clustersize=4,
        n_sims=6,
        seed=11,
    )
    assert via_handle["tau"] == pytest.approx(via_graph["tau"])
    assert via_handle["n_sims"] == 6


def test_calibrate_llr_threshold_handle_accepts_matching_knn_mode(isolated_cache) -> None:
    coords, adjacency = _grid(5)
    handle = flx.build_static(coords, adjacency, K=6)
    out = flx.calibrate_llr_threshold(
        handle=handle,
        observed=np.ones(25, dtype=int),
        expected=np.ones(25),
        K=6,
        clustersize=3,
        n_sims=3,
        knn_mode="K_REFILL",
    )
    assert set(out) == {"tau", "mu", "sigma", "alpha", "n_sims"}


def test_calibrate_llr_threshold_handle_rejects_explicit_refill_force_like_scan(
    isolated_cache,
) -> None:
    """An explicit build-time option beside a handle is rejected, exactly as scan() does."""
    coords, adjacency = _grid(5)
    handle = flx.build_static(coords, adjacency, K=6)
    with pytest.raises(ValueError, match="knn_refill_force"):
        flx.calibrate_llr_threshold(
            handle=handle,
            observed=np.ones(25, dtype=int),
            expected=np.ones(25),
            K=6,
            clustersize=3,
            n_sims=3,
            knn_refill_force=False,
        )
    with pytest.raises(ValueError, match="knn_refill_force"):
        flx.scan(
            handle=handle,
            observed=[1] * 25,
            expected=[1.0] * 25,
            simcount=0,
            knn_refill_force=False,
        )


def test_calibrate_llr_threshold_handle_rejects_conflicting_K(isolated_cache) -> None:
    coords, adjacency = _grid(5)
    handle = flx.build_static(coords, adjacency, K=6)
    with pytest.raises(ValueError, match="K"):
        flx.calibrate_llr_threshold(
            handle=handle,
            observed=np.ones(25, dtype=int),
            expected=np.ones(25),
            K=4,
            clustersize=3,
            n_sims=3,
        )


# ---------------------------------------------------------------------------
# FXA-182: the cache directory is private, repaired when permissive, refused when foreign
# ---------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402
import warnings  # noqa: E402

from flexscanx._calibration import _cache_entry_digest  # noqa: E402


def _calibrate(seed: int = 5) -> dict:
    coords, adjacency = _grid(5)
    return flx.calibrate_llr_threshold(
        coords=coords,
        adjacency=adjacency,
        observed=np.ones(25, dtype=int),
        expected=np.ones(25),
        K=6,
        clustersize=3,
        n_sims=8,
        seed=seed,
    )


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def permissive_umask():
    old = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_cache_directory_is_created_private(isolated_cache, permissive_umask) -> None:
    _calibrate()
    assert isolated_cache.is_dir()
    assert _mode(isolated_cache) == 0o700
    for name in ("calibration_cache.json", "calibration_cache.json.key"):
        assert _mode(isolated_cache / name) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_permissive_existing_directory_is_tightened_and_its_entries_discarded(
    isolated_cache, permissive_umask
) -> None:
    isolated_cache.mkdir(parents=True)
    os.chmod(isolated_cache, 0o775)
    with pytest.warns(UserWarning, match="group- or world-writable"):
        first = _calibrate()
    assert _mode(isolated_cache) == 0o700
    assert first["n_sims"] == 8
    # A second call now hits a trusted, private cache and returns the same value.
    assert _calibrate()["tau"] == pytest.approx(first["tau"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_key_replacement_in_a_permissive_directory_does_not_poison_the_threshold(
    isolated_cache, permissive_umask
) -> None:
    honest = _calibrate()
    cache_file = isolated_cache / "calibration_cache.json"
    key_path = isolated_cache / "calibration_cache.json.key"
    # Threat model B: an attacker with directory write (possible on a 0775 directory)
    # replaces the key and re-signs a chosen entry.
    os.chmod(isolated_cache, 0o775)
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    (key,) = payload["entries"].keys()
    attacker_secret = os.urandom(32)
    forged = {"tau": -1.0, "mu": 0.0, "sigma": 1.0, "alpha": 0.05, "n_sims": 8}
    payload["entries"][key] = {
        "value": forged,
        "hmac": _cache_entry_digest(key, forged, attacker_secret),
    }
    key_path.unlink()
    key_path.write_bytes(attacker_secret)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="group- or world-writable"):
        recovered = _calibrate()
    assert recovered["tau"] == pytest.approx(honest["tau"])
    assert recovered["tau"] != -1.0
    assert _mode(isolated_cache) == 0o700
    # The attacker's key was rotated away and the forged entry did not survive.
    assert key_path.read_bytes() != attacker_secret
    assert not any(
        entry["value"]["tau"] == -1.0
        for entry in json.loads(cache_file.read_text(encoding="utf-8"))["entries"].values()
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
def test_directory_owned_by_another_user_is_refused_not_used(isolated_cache, monkeypatch) -> None:
    isolated_cache.mkdir(parents=True, mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr("flexscanx._calibration.os.getuid", lambda: real_uid + 1)
    with pytest.warns(UserWarning, match="not owned by the current user") as record:
        result = _calibrate()
    assert result["n_sims"] == 8
    assert not (isolated_cache / "calibration_cache.json").exists()
    assert not (isolated_cache / "calibration_cache.json.key").exists()
    # Warned once per directory per process, not on every call.
    _calibrate()
    assert sum("not owned by the current user" in str(w.message) for w in record) == 1


# ---------------------------------------------------------------------------
# FXA-191: a corrupt cache file is announced once, not silently rebuilt
# ---------------------------------------------------------------------------


def test_unreadable_cache_file_warns_once_and_is_rebuilt(isolated_cache) -> None:
    first = _calibrate()
    cache_file = isolated_cache / "calibration_cache.json"
    assert cache_file.exists()
    cache_file.write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning, match="calibration cache .* could not be read") as record:
        again = _calibrate()
    assert again["tau"] == pytest.approx(first["tau"])
    assert sum("could not be read" in str(w.message) for w in record) == 1
    # Rebuilt: the file parses again and the next call is silent.
    json.loads(cache_file.read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*equidistant.*")
        _calibrate()


def test_cold_cache_is_silent(isolated_cache) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*equidistant.*")
        _calibrate()


def test_wrong_shaped_cache_payload_warns(isolated_cache) -> None:
    _calibrate()
    (isolated_cache / "calibration_cache.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.warns(UserWarning, match="could not be read"):
        _calibrate()
