"""LLR threshold calibration and on-disk cache.

Moved from validation.py (B1 split) — all external imports go through
validation.py which re-exports the public names.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import pathlib
import stat
import tempfile
from hashlib import sha256
from typing import Any, Callable, Sequence, TypeGuard

import numpy as np

from ._core_loader import require_core as _require_core
from ._scan_internal import (
    _normalize_clustertype,
    _normalize_model_and_rantype,
    _normalize_scanmethod_and_stattype,
    _validate_denominator,
    _validate_scan_common,
    _validate_seed,
    warn_exact_flexible_large_k,
)
from ._utils import (
    mc_alias_env_enabled,
    validate_float_parameter,
    validate_observed_array,
    warn_at_caller,
)  # noqa: F401

# ---------------------------------------------------------------------------
# Calibration cache configuration
# ---------------------------------------------------------------------------


def _get_cache_dir() -> pathlib.Path:
    """Return a platform-appropriate user cache directory for flexscanx."""
    try:
        import platformdirs  # type: ignore

        return pathlib.Path(platformdirs.user_cache_dir("flexscanx"))
    except ImportError:
        pass
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    if xdg:
        return pathlib.Path(xdg) / "flexscanx"
    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            return pathlib.Path(local_app) / "flexscanx"
    return pathlib.Path.home() / ".cache" / "flexscanx"


_CACHE_DIR: pathlib.Path = _get_cache_dir()
_CACHE_FILE: str = str(_CACHE_DIR / "calibration_cache.json")
# Cache schema version history:
#   v1 — initial development schema
#   v2 — added knn_mode / knn_refill_force to cache key
#   v3 — added expected_fingerprint (content hash) so same-total, different-distribution
#         arrays produce distinct keys instead of sharing a cached result
#   v4 — added seed to the key (audit AUD-6: different seeds previously shared
#         one cached estimate); cache writes made atomic (temp + os.replace)
#   v5 — added a spatial-input fingerprint; opaque reused handles bypass the
#         persistent cache because their topology cannot be reconstructed here
#   v6 — integrity-protected entries with a per-cache secret; invalidates
#         unauthenticated legacy entries rather than trusting user-writable JSON
#   (2026-09-07, no schema change) — the cache directory is created 0o700, a
#         group- or world-writable directory is tightened and its key rotated,
#         and a directory owned by another user disables the cache (FXA-182)
_CACHE_SCHEMA_VERSION: int = 6
_CACHE_FORMAT: str = "flexscanx.calibration_cache"

# ---------------------------------------------------------------------------
# Calibration cache internals
# ---------------------------------------------------------------------------


def _safe_package_version() -> str:
    try:
        from ._meta import __version__ as _pkg_version

        return str(_pkg_version)
    except Exception:
        return "unknown"


def _safe_core_version(core_module: Any = None) -> str:
    if core_module is None:
        try:
            core_module = _require_core()
        except ImportError:
            return "unknown"
    try:
        return str(core_module.version())
    except Exception:
        return "unknown"


def _expected_fingerprint(expected: Sequence[float] | np.ndarray) -> str:
    arr = np.asarray(expected, dtype=np.float64)
    arr_le = np.ascontiguousarray(arr.astype("<f8", copy=False))
    hasher = sha256()
    hasher.update(str(arr_le.shape).encode("utf-8"))
    hasher.update(arr_le.tobytes(order="C"))
    return hasher.hexdigest()


def _spatial_fingerprint(
    coords: np.ndarray | Sequence[Sequence[float]],
    adjacency: tuple[Sequence[int], Sequence[int]],
) -> str:
    """Return a stable content fingerprint for the effective spatial inputs."""
    coords_arr = np.ascontiguousarray(np.asarray(coords, dtype="<f8"))
    row_ptr = np.ascontiguousarray(np.asarray(adjacency[0], dtype="<i8"))
    col_idx = np.ascontiguousarray(np.asarray(adjacency[1], dtype="<i8"))
    hasher = sha256()
    for name, array in (
        ("coords", coords_arr),
        ("row_ptr", row_ptr),
        ("col_idx", col_idx),
    ):
        hasher.update(name.encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _validate_cache_entry(entry: object) -> TypeGuard[dict[str, Any]]:
    if not isinstance(entry, dict):
        return False
    required = {"tau", "mu", "sigma", "alpha", "n_sims"}
    if not required.issubset(entry.keys()):
        return False
    numeric_fields = ("tau", "mu", "sigma", "alpha")
    if any(
        isinstance(entry[key], (bool, np.bool_))
        or not isinstance(entry[key], (int, float))
        or not math.isfinite(float(entry[key]))
        for key in numeric_fields
    ):
        return False
    n_sims = entry["n_sims"]
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, int) or n_sims < 1:
        return False
    return float(entry["sigma"]) >= 0.0 and 0.0 < float(entry["alpha"]) < 1.0


def _cache_secret_path(target: pathlib.Path) -> pathlib.Path:
    return target.with_suffix(target.suffix + ".key")


_PERMISSIVE_DIR_BITS = stat.S_IWGRP | stat.S_IWOTH
_UNTRUSTED_DIRS_WARNED: set[str] = set()


def _warn_cache_dir_once(directory: pathlib.Path, message: str) -> None:
    key = str(directory)
    if key in _UNTRUSTED_DIRS_WARNED:
        return
    _UNTRUSTED_DIRS_WARNED.add(key)
    warn_at_caller(message)


def _prepare_cache_dir(target: pathlib.Path, *, create: bool) -> str:
    """Make the cache directory private and say whether its contents can be trusted.

    Returns one of ``"private"`` (owned by this user and writable by nobody else),
    ``"repaired"`` (it was group- or world-writable; it has just been tightened to
    ``0o700`` and the integrity key rotated, so anything already inside is
    discarded), ``"missing"`` (absent and *create* is False) or ``"untrusted"``
    (owned by another user, or its permissions cannot be fixed).

    The HMAC key lives beside the cache it authenticates, so the integrity scheme
    only holds while nobody else can write the directory (FXA-182).  The files
    were always created ``0o600``; the directory used to inherit the process
    umask, which on the development platform gives ``0o775``.  On non-POSIX
    systems the mode bits are not meaningful and the directory is trusted as is.
    """
    directory = target.parent
    try:
        info = os.stat(directory)
    except FileNotFoundError:
        if not create:
            return "missing"
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = os.stat(directory)
        except OSError as exc:
            _warn_cache_dir_once(
                directory, f"calibration cache disabled: cannot create {directory}: {exc}"
            )
            return "untrusted"
    if not stat.S_ISDIR(info.st_mode):
        _warn_cache_dir_once(
            directory, f"calibration cache disabled: {directory} is not a directory"
        )
        return "untrusted"
    if os.name != "posix":
        return "private"
    if info.st_uid != os.getuid():
        _warn_cache_dir_once(
            directory,
            f"calibration cache disabled: {directory} is not owned by the current user, "
            "so its integrity key cannot be trusted",
        )
        return "untrusted"
    if not (info.st_mode & _PERMISSIVE_DIR_BITS):
        return "private"
    try:
        os.chmod(directory, 0o700)
        after = os.stat(directory)
    except OSError as exc:
        _warn_cache_dir_once(
            directory,
            f"calibration cache disabled: {directory} is group- or world-writable and "
            f"could not be made private: {exc}",
        )
        return "untrusted"
    if after.st_mode & _PERMISSIVE_DIR_BITS:
        _warn_cache_dir_once(
            directory,
            f"calibration cache disabled: {directory} stays group- or world-writable",
        )
        return "untrusted"
    # Anything written while the directory was permissive may be an attacker's,
    # including the key; rotate it so existing entries fail authentication.
    try:
        _cache_secret_path(target).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        _warn_cache_dir_once(
            directory,
            f"calibration cache disabled: could not rotate the integrity key in {directory}: {exc}",
        )
        return "untrusted"
    warn_at_caller(
        f"calibration cache directory {directory} was group- or world-writable; its "
        "permissions were tightened to 0700 and the entries it held were discarded"
    )
    return "repaired"


def _read_cache_secret(secret_path: pathlib.Path) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(secret_path, flags)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as stream:
        secret = stream.read(33)
    return secret if len(secret) == 32 else None


def _load_or_create_cache_secret(target: pathlib.Path, *, create: bool) -> bytes | None:
    secret_path = _cache_secret_path(target)
    secret = _read_cache_secret(secret_path)
    if secret is not None or not create:
        return secret

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    generated = os.urandom(32)
    try:
        fd = os.open(secret_path, flags, 0o600)
    except FileExistsError:
        secret = _read_cache_secret(secret_path)
        if secret is not None:
            return secret
        # The cache and key are disposable.  Repair a truncated key while the
        # caller holds the cache lock; old entries then fail authentication and
        # are replaced from the caller's freshly computed snapshot.
        try:
            secret_path.unlink()
            fd = os.open(secret_path, flags, 0o600)
        except OSError:
            return None
    with os.fdopen(fd, "wb") as stream:
        stream.write(generated)
        stream.flush()
        os.fsync(stream.fileno())
    return generated


def _cache_entry_digest(cache_key: str, entry: dict, secret: bytes) -> str:
    canonical = json.dumps(
        entry,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    message = cache_key.encode("utf-8") + b"\0" + canonical
    return hmac.new(secret, message, sha256).hexdigest()


def _cache_load_impl(
    path: str | None = None,
    *,
    cache_file: str,
    cache_schema_version: int,
    cache_format: str,
) -> dict:
    target = pathlib.Path(path or cache_file)
    if _prepare_cache_dir(target, create=False) != "private":
        return {}
    try:
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}  # a cold cache is the normal first-run state
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        # An existing file that cannot be read is not a cold cache: say so once,
        # so a corrupt or tampered file is distinguishable from first use
        # (FXA-191).  The file is rebuilt from the next computed result.
        _warn_cache_dir_once(
            target,
            f"calibration cache {target} could not be read ({type(exc).__name__}); "
            "it is ignored and will be rebuilt",
        )
        return {}

    meta = payload.get("__meta__") if isinstance(payload, dict) else None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(meta, dict) or not isinstance(entries, dict):
        _warn_cache_dir_once(
            target,
            f"calibration cache {target} could not be read (unexpected layout); "
            "it is ignored and will be rebuilt",
        )
        return {}
    try:
        schema_ok = int(meta.get("schema_version", -1)) == int(cache_schema_version)
    except (TypeError, ValueError):
        schema_ok = False
    format_ok = str(meta.get("format", "")) == str(cache_format)
    if not (schema_ok and format_ok):
        return {}

    secret = _load_or_create_cache_secret(target, create=False)
    if secret is None:
        return {}
    verified: dict[str, dict] = {}
    for cache_key, envelope in entries.items():
        if not isinstance(cache_key, str) or not isinstance(envelope, dict):
            continue
        value = envelope.get("value")
        digest = envelope.get("hmac")
        if not _validate_cache_entry(value) or not isinstance(digest, str):
            continue
        expected_digest = _cache_entry_digest(cache_key, value, secret)
        if hmac.compare_digest(digest, expected_digest):
            verified[cache_key] = value
    return verified


def _cache_save_impl(
    data: dict,
    *,
    path: str | None = None,
    cache_file: str,
    cache_schema_version: int,
    cache_format: str,
) -> None:
    target = pathlib.Path(path or cache_file)
    tmp_path: pathlib.Path | None = None
    lock_fh: Any = None
    if _prepare_cache_dir(target, create=True) not in ("private", "repaired"):
        return
    try:

        # Serialize the read/merge/write transaction on the active Linux
        # platform. O_NOFOLLOW prevents a pre-created lock-file symlink from
        # redirecting the open outside the cache directory.
        lock_path = target.with_suffix(target.suffix + ".lock")
        lock_flags = os.O_CREAT | os.O_RDWR
        lock_flags |= getattr(os, "O_CLOEXEC", 0)
        lock_flags |= getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(lock_path, lock_flags, 0o600)
        lock_fh = os.fdopen(lock_fd, "a+b")
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        secret = _load_or_create_cache_secret(target, create=True)
        if secret is None:
            raise OSError("could not create or read the calibration-cache integrity key")

        # Callers save a snapshot loaded before a potentially long simulation.
        # Merge under the lock so a delayed writer cannot erase keys committed
        # by another process while that simulation was running.
        merged = _cache_load_impl(
            str(target),
            cache_file=cache_file,
            cache_schema_version=cache_schema_version,
            cache_format=cache_format,
        )
        if isinstance(data, dict):
            merged.update(data)

        stored_entries = {
            cache_key: {
                "value": entry,
                "hmac": _cache_entry_digest(cache_key, entry, secret),
            }
            for cache_key, entry in merged.items()
            if isinstance(cache_key, str) and _validate_cache_entry(entry)
        }
        payload = {
            "__meta__": {
                "format": str(cache_format),
                "schema_version": int(cache_schema_version),
                "package_version": _safe_package_version(),
                "core_version": _safe_core_version(),
                "integrity": "hmac-sha256",
            },
            "entries": stored_entries,
        }

        # A unique same-directory temporary file avoids collisions between
        # writers and keeps os.replace atomic on one filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = pathlib.Path(fh.name)
            json.dump(payload, fh, allow_nan=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
        os.chmod(target, 0o600)

        # Persist the directory entry as well as the file contents.
        if os.name == "posix":
            dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            dir_fd = os.open(target.parent, dir_flags)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except (OSError, TypeError) as exc:
        warn_at_caller(f"could not save calibration cache at {target}: {exc}")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if lock_fh is not None:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fh.close()
        # Deliberately retain the lock inode. Unlinking a pathname-based advisory
        # lock after release lets a third process create and lock a new inode
        # while an existing waiter still holds the old one, breaking exclusion.


def _calib_key_impl(**kw: Any) -> str:
    items = [
        ("cache_schema_version", int(kw.get("cache_schema_version", 0))),
        ("cache_format", str(kw.get("cache_format", ""))),
        ("package_version", str(kw.get("package_version", ""))),
        ("core_version", str(kw.get("core_version", ""))),
        ("model", str(kw.get("model", "POISSON")).upper()),
        ("scanmethod", str(kw.get("scanmethod", "FLEXIBLE")).upper()),
        ("stattype", str(kw.get("stattype", "RESTRICTED")).upper()),
        ("clustertype", str(kw.get("clustertype", "HOT")).upper()),
        ("K", int(kw.get("K", 0))),
        ("clustersize", int(kw.get("clustersize", 0))),
        ("radius", float(kw.get("clusterradius", float("inf")))),
        ("N", int(kw.get("N", 0))),
        ("alpha", float(kw.get("alpha", 0.05))),
        ("ralpha", float(kw.get("ralpha", 0.2))),
        ("n_sims", int(kw.get("n_sims", 0))),
        ("rantype", str(kw.get("rantype", "")).upper()),
        ("knn_mode", str(kw.get("knn_mode", "K_REFILL")).upper()),
        ("knn_refill_force", bool(kw.get("knn_refill_force", False))),
        ("observed_total", int(kw.get("observed_total", 0))),
        ("expected_total", float(kw.get("expected_total", 0.0))),
        ("expected_fingerprint", str(kw.get("expected_fingerprint", ""))),
        ("spatial_fingerprint", str(kw.get("spatial_fingerprint", ""))),
        # v4: the RNG seed is part of the key — without it, a call with a
        # different seed silently returned another seed's cached estimate.
        # seed=None (unseeded) keys as "None" and is shared by unseeded calls.
        ("seed", str(kw.get("seed", None))),
    ]
    return sha256(json.dumps(items).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Alias sampling helpers (Python-side; used only by calibrate_llr_threshold_impl)
#
# Note: a parallel C++ alias implementation exists in monte_carlo.hpp
# (build_alias_from_weights / multinomial_counts_alias_crn) used by ScanSlice.
# Both implement the same Vose alias method.  Any algorithmic fix must be
# applied in both places until the Python calibration path is migrated to
# delegate to the C++ binding.
# ---------------------------------------------------------------------------


def _alias_build(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    n_probs = int(p.size)
    if n_probs == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    q = p * float(n_probs)
    prob = np.zeros(n_probs, dtype=float)
    alias = np.zeros(n_probs, dtype=int)
    small: list[int] = []
    large: list[int] = []
    for idx in range(n_probs):
        (small if q[idx] < 1.0 else large).append(idx)
    while small and large:
        s_idx = small.pop()
        l_idx = large.pop()
        prob[s_idx] = q[s_idx]
        alias[s_idx] = l_idx
        q[l_idx] = q[l_idx] - (1.0 - q[s_idx])
        if q[l_idx] < 1.0:
            small.append(l_idx)
        else:
            large.append(l_idx)
    while large:
        l_idx = large.pop()
        prob[l_idx] = 1.0
        alias[l_idx] = l_idx
    while small:
        s_idx = small.pop()
        prob[s_idx] = 1.0
        alias[s_idx] = s_idx
    return prob, alias


def _alias_multinomial_counts(
    prob: np.ndarray, alias: np.ndarray, n: int, rng: np.random.Generator
) -> list[int]:
    n_probs = int(prob.size)
    if n <= 0 or n_probs == 0:
        return [0] * n_probs
    counts = np.zeros(n_probs, dtype=int)
    idx = rng.integers(0, n_probs, size=int(n))
    choose_alias = rng.random(size=int(n)) >= prob[idx]
    chosen = idx.copy()
    chosen[choose_alias] = alias[idx[choose_alias]]
    binc = np.bincount(chosen, minlength=n_probs)
    counts[:n_probs] = binc[:n_probs]
    return counts.astype(int).tolist()


def _simulate_poisson_multinomial(
    expected: Sequence[float],
    total: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    if rng is None:
        rng = np.random.default_rng()
    exp = np.asarray(expected, dtype=float)
    weights = np.clip(exp, 0.0, np.inf)
    mass = float(weights.sum())
    if mass <= 0.0:
        return [0] * len(expected)
    probs = (weights / mass).astype(float)
    n_draws = int(total if total is not None else round(mass))
    if n_draws <= 0:
        return [0] * len(expected)
    if mc_alias_env_enabled():
        prob, alias = _alias_build(probs)
        return _alias_multinomial_counts(prob, alias, n_draws, rng)
    return rng.multinomial(n_draws, probs, size=1)[0].astype(int).tolist()


# ---------------------------------------------------------------------------
# Public facade wrappers for cache I/O
# ---------------------------------------------------------------------------


def _cache_load(path: str | None = None) -> dict:
    return _cache_load_impl(
        path,
        cache_file=_CACHE_FILE,
        cache_schema_version=_CACHE_SCHEMA_VERSION,
        cache_format=_CACHE_FORMAT,
    )


def _cache_save(data: dict, path: str | None = None) -> None:
    _cache_save_impl(
        data,
        path=path,
        cache_file=_CACHE_FILE,
        cache_schema_version=_CACHE_SCHEMA_VERSION,
        cache_format=_CACHE_FORMAT,
    )


def _calib_key(**kw: Any) -> str:
    kw.setdefault("cache_schema_version", _CACHE_SCHEMA_VERSION)
    kw.setdefault("cache_format", _CACHE_FORMAT)
    return _calib_key_impl(**kw)


# ---------------------------------------------------------------------------
# calibrate_llr_threshold implementation
# ---------------------------------------------------------------------------


def calibrate_llr_threshold_impl(
    *,
    coords: np.ndarray | None = None,
    adjacency: tuple[Sequence[int], Sequence[int]] | None = None,
    handle: Any | None = None,
    observed: Sequence[int],
    expected: Sequence[float] | None = None,
    population: Sequence[float] | None = None,
    model: str = "POISSON",
    scanmethod: str = "FLEXIBLE",
    stattype: str = "RESTRICTED",
    clustertype: str = "HOT",
    K: int,
    clustersize: int,
    clusterradius: float = float("inf"),
    n_sims: int = 999,
    alpha: float = 0.05,
    seed: int | None = None,
    knn_mode: str = "K_REFILL",
    knn_refill_force: bool | None = None,
    rantype: str | None = None,
    ralpha: float = 0.2,
    core_module: Any = None,
    ensure_static_handle: Callable[..., Any],
    cache_load: Callable[[], dict],
    cache_save: Callable[[dict], None],
    cache_schema_version: int,
    cache_format: str,
) -> dict[str, float]:
    """Calibrate an LLR threshold and z-normalization moments under the null."""
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, (int, np.integer)):
        raise ValueError(f"n_sims must be an integer, got {n_sims!r}")
    n_sims_i = int(n_sims)
    if n_sims_i < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    alpha_f = validate_float_parameter(alpha, "alpha")
    if not math.isfinite(alpha_f):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if not 0.0 < alpha_f < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise ValueError(f"K must be an integer, got {K!r}")
    if isinstance(clustersize, (bool, np.bool_)) or not isinstance(clustersize, (int, np.integer)):
        raise ValueError(f"clustersize must be an integer, got {clustersize!r}")

    _validate_scan_common(ralpha, int(clustersize), clusterradius)
    seed_i = _validate_seed(seed)
    model_u, rantype_eff = _normalize_model_and_rantype(model, rantype)
    scanmethod_u, stattype_u = _normalize_scanmethod_and_stattype(scanmethod, stattype)
    clustertype_u = _normalize_clustertype(clustertype, allow_both=False)

    observed_arr = validate_observed_array(observed, name="observed")
    if observed_arr.ndim != 1:
        raise ValueError(f"observed must be a 1D array, got shape {observed_arr.shape}")
    n_areas = len(observed_arr)
    denominator = _validate_denominator(model_u, expected, population, observed_arr, n_areas)
    if denominator.ndim != 1:
        denominator_name = "population" if model_u == "BINOMIAL" else "expected"
        raise ValueError(f"{denominator_name} must be a 1D array, got shape {denominator.shape}")

    if core_module is None:
        core_module = _require_core()

    cacheable = handle is None and seed_i is not None
    # ``None`` means "not requested".  Passing a concrete bool here made every
    # handle= call fail, because the reuse path treats any explicit value as a
    # build-time request it cannot honour (FXA-165).
    knn_refill_force_eff = False if knn_refill_force is None else bool(knn_refill_force)
    static_handle, _, effective_K, _ = ensure_static_handle(
        handle=handle,
        coords=coords,
        adjacency=adjacency,
        K=int(K),
        N=None if handle is None else n_areas,
        clusterradius=float(clusterradius),
        knn_method="graph",
        knn_mode=str(knn_mode),
        knn_refill_force=knn_refill_force,
    )
    settings: dict[str, Any] = {
        "ralpha": float(ralpha),
        "clustersize": int(clustersize),
        "scanmethod": scanmethod_u,
        "stattype": stattype_u,
        "model": model_u,
        "simcount": 0,
        "clustertype": clustertype_u,
    }

    expected_arr = denominator
    expected_total = float(expected_arr.sum())
    observed_total = int(observed_arr.sum(dtype=np.int64))
    expected_fp = _expected_fingerprint(expected_arr)
    spatial_fp = ""
    if handle is None:
        # Defensive: ensure_static_handle already rejects these missing inputs.
        if coords is None or adjacency is None:
            raise ValueError("coords and adjacency are required when handle is not provided")
        spatial_fp = _spatial_fingerprint(coords, adjacency)
    pkg_version = _safe_package_version()
    core_version = _safe_core_version(core_module)

    key = _calib_key_impl(
        cache_schema_version=cache_schema_version,
        cache_format=cache_format,
        package_version=pkg_version,
        core_version=core_version,
        model=model_u,
        scanmethod=scanmethod_u,
        stattype=stattype_u,
        clustertype=clustertype_u,
        K=K,
        clustersize=clustersize,
        clusterradius=clusterradius,
        N=n_areas,
        alpha=alpha_f,
        ralpha=ralpha,
        n_sims=n_sims,
        rantype=rantype_eff,
        knn_mode=knn_mode,
        knn_refill_force=knn_refill_force_eff,
        observed_total=observed_total,
        expected_total=expected_total,
        expected_fingerprint=expected_fp,
        spatial_fingerprint=spatial_fp,
        seed=seed_i,
    )
    cache = cache_load() if cacheable else {}
    if cacheable and key in cache:
        entry = cache[key]
        if _validate_cache_entry(entry):
            return entry

    warn_exact_flexible_large_k(
        scanmethod_u,
        effective_K if effective_K is not None else int(K),
        int(clustersize),
        None,
        None,
        stattype=stattype_u,
        repetitions=n_sims_i,
    )

    rng = np.random.default_rng(seed_i)
    total = observed_total
    use_alias = mc_alias_env_enabled()
    alias_prob: np.ndarray | None = None
    alias_idx: np.ndarray | None = None
    if use_alias and rantype_eff not in ("POISSON", "BINOMIAL"):
        weights = np.clip(expected_arr, 0.0, np.inf)
        mass = float(weights.sum())
        if mass > 0.0:
            alias_prob, alias_idx = _alias_build(weights / mass)

    def _simulate_max_once() -> float:
        if rantype_eff == "POISSON":
            sim_obs = rng.poisson(lam=expected_arr).astype(int).tolist()
        elif rantype_eff == "BINOMIAL":
            pop_arr = expected_arr
            pooled = float(observed_total) / float(np.sum(pop_arr))
            sim_obs = rng.binomial(pop_arr.astype(int), pooled).astype(int).tolist()
        elif use_alias and alias_prob is not None and alias_idx is not None:
            sim_obs = _alias_multinomial_counts(alias_prob, alias_idx, total, rng)
        else:
            sim_obs = _simulate_poisson_multinomial(expected_arr.tolist(), total=total, rng=rng)

        result = core_module.ScanSlice(static_handle, sim_obs, expected_arr.tolist(), settings)
        try:
            return float(result.get("stats", float("nan")))
        except Exception:
            return float("nan")

    stats = [_simulate_max_once() for _ in range(n_sims_i)]
    stats = [
        float(value) for value in stats if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not stats:
        return {
            "tau": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "alpha": alpha_f,
            "n_sims": n_sims_i,
        }

    mu = float(np.mean(stats))
    sigma = float(np.std(stats, ddof=1)) if len(stats) >= 2 else 0.0
    tau = float(np.quantile(np.asarray(stats, dtype=float), 1.0 - alpha_f))

    out = {"tau": tau, "mu": mu, "sigma": sigma, "alpha": alpha_f, "n_sims": n_sims_i}
    if cacheable:
        cache[key] = out
        cache_save(cache)
    return out
