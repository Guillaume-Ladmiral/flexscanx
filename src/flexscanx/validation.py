"""Public validation API — thin facade over the focused sub-modules.

This module re-exports every public symbol from the four focused sub-modules
so that all existing imports (``from flexscanx.validation import X``,
``import flexscanx.validation as v; v._CACHE_FILE``, etc.) continue to work
without change.

Sub-module responsibilities:
  _calibration.py  — LLR threshold calibration, on-disk cache, alias sampling
  _graph_metrics.py — BFS-based graph metrics (silhouette, Dunn, Moran's I, …)
  _predictive.py   — Cross-validation log-likelihood gain (cv_glm_gain)
  _consensus.py    — Consensus sweep, trend analysis, stability across K

Monkeypatch compatibility note
-------------------------------
Several tests in ``test_validation_calibration_cache.py`` monkeypatch
``validation._ensure_static_handle`` to avoid building a real precompute handle.
Because ``calibrate_llr_threshold`` is defined here (in this facade module)
rather than in ``_calibration``, the patch correctly intercepts the call:
Python resolves ``_ensure_static_handle`` from *this* module's ``__dict__``
at call time, so ``monkeypatch.setattr(v, '_ensure_static_handle', fake)``
works without any test changes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Calibration internals — all re-exported for test monkeypatching
# ---------------------------------------------------------------------------
from ._calibration import (  # noqa: F401  (facade: re-exported for back-compat / test monkeypatching)
    _CACHE_FILE,
    _CACHE_FORMAT,
    _CACHE_SCHEMA_VERSION,
    _alias_build,
    _alias_multinomial_counts,
    _cache_load,
    _cache_load_impl,
    _cache_save,
    _cache_save_impl,
    _calib_key,
    _calib_key_impl,
    _expected_fingerprint,
    _get_cache_dir,
    _safe_core_version,
    _safe_package_version,
    _simulate_poisson_multinomial,
    _validate_cache_entry,
    calibrate_llr_threshold_impl,
)
from ._core_loader import require_core as _require_core

# _ensure_static_handle re-exported for monkeypatching in tests.
# Keep as a module-level name so setattr(v, '_ensure_static_handle', fake) works.
from ._utils import ensure_static_handle as _ensure_static_handle  # noqa: F401


def calibrate_llr_threshold(
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
) -> dict[str, float]:
    """Calibrate an LLR threshold tau_K and z-score moments under the null.

    Parameters
    ----------
    observed : sequence of int
        Length-N integer count vector (same as passed to ``scan()``).
    expected : sequence of float, optional
        Length-N float expected-count vector.  Required for POISSON.
        Not accepted for BINOMIAL (pass *population* instead).
    population : sequence of float, optional
        Length-N integer trial-count vector.  Required for BINOMIAL.
        Not accepted for POISSON (pass *expected* instead).
    model : str, default "POISSON"
        Statistical model: ``'POISSON'`` or ``'BINOMIAL'``.
    K : int
        KNN horizon to calibrate for.
    clustersize : int
        Maximum cluster size to calibrate for.
    n_sims : int, default 999
        Number of null simulations (must be >= 1).
    alpha : float, default 0.05
        Significance level for the tau quantile.
    seed : int, optional
        RNG seed for reproducibility. Seeded, reconstructable calibrations may
        use the persistent cache. ``None`` draws fresh entropy and deliberately
        bypasses persistent caching.
    handle : ScanHandle, optional
        Precomputed static handle from ``build_static()``. When provided,
        *coords* and *adjacency* must be omitted and *K*, *clusterradius* and
        *knn_mode* must agree with the handle; results computed through an
        opaque handle are not persisted in the cache.
    knn_mode : str, default "K_REFILL"
        KNN precompute mode for a newly built handle.
    knn_refill_force : bool, optional
        Force K_REFILL on large N for a newly built handle. ``None`` (default)
        leaves the build-time guardrail in place and is the only value accepted
        beside a reused *handle*, whose refill behaviour is already fixed.
    rantype : str, optional
        Null distribution override.  Default ``None`` resolves to
        ``'MULTINOMIAL'`` for POISSON and ``'BINOMIAL'`` for BINOMIAL.

    Returns
    -------
    dict
        Keys: ``'tau'``, ``'mu'``, ``'sigma'``, ``'alpha'``, ``'n_sims'``.
        Seeded results are cached on disk with integrity protection and keyed by
        all calibration parameters. Unseeded and opaque-handle results are not
        persisted.
    """
    core = _require_core()
    # Read everything that tests monkeypatch from THIS module's namespace.
    # Tests do: monkeypatch.setattr(v, "_CACHE_FILE", str(tmp_path/...), ...)
    # and:      monkeypatch.setattr(v, "_ensure_static_handle", fake)
    # Both patches land on validation.__dict__, not _calibration.__dict__.
    # Reading via sys.modules[__name__] ensures the patched values are used.
    import sys

    _mod = sys.modules[__name__]
    _esh = _mod._ensure_static_handle  # type: ignore[attr-defined]
    _cfile = _mod._CACHE_FILE  # type: ignore[attr-defined]
    _cver = _mod._CACHE_SCHEMA_VERSION  # type: ignore[attr-defined]
    _cfmt = _mod._CACHE_FORMAT  # type: ignore[attr-defined]

    def _cl(path: str | None = None) -> dict:
        return _cache_load_impl(
            path,
            cache_file=_cfile,
            cache_schema_version=_cver,
            cache_format=_cfmt,
        )

    def _cs(data: dict, path: str | None = None) -> None:
        _cache_save_impl(
            data,
            path=path,
            cache_file=_cfile,
            cache_schema_version=_cver,
            cache_format=_cfmt,
        )

    return calibrate_llr_threshold_impl(
        coords=coords,
        adjacency=adjacency,
        handle=handle,
        observed=observed,
        expected=expected,
        population=population,
        model=model,
        scanmethod=scanmethod,
        stattype=stattype,
        clustertype=clustertype,
        K=K,
        clustersize=clustersize,
        clusterradius=clusterradius,
        n_sims=n_sims,
        alpha=alpha,
        seed=seed,
        knn_mode=knn_mode,
        knn_refill_force=knn_refill_force,
        rantype=rantype,
        ralpha=ralpha,
        core_module=core,
        ensure_static_handle=_esh,
        cache_load=_cl,
        cache_save=_cs,
        cache_schema_version=_cver,
        cache_format=_cfmt,
    )


# ---------------------------------------------------------------------------
# Graph-based validation metrics
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Consensus and trend utilities
# ---------------------------------------------------------------------------
from ._consensus import (  # noqa: E402  (facade: late re-export after shim defs)
    consensus_sweep,
    consensus_to_series,
    stability_across_K,
    trend_level_slope,
)
from ._graph_metrics import (  # noqa: E402  (facade: late re-export after shim defs)
    boundary_contrast,
    conductance,
    dunn_graph,
    join_counts,
    morans_I_within,
    silhouette_graph,
)

# ---------------------------------------------------------------------------
# Predictive validation
# ---------------------------------------------------------------------------
from ._predictive import cv_glm_gain  # noqa: E402  (facade: late re-export after shim defs)

__all__ = [
    # calibration
    "calibrate_llr_threshold",
    "calibrate_llr_threshold_impl",
    # graph metrics
    "silhouette_graph",
    "dunn_graph",
    "join_counts",
    "boundary_contrast",
    "morans_I_within",
    "conductance",
    # predictive
    "cv_glm_gain",
    # consensus / trend / stability
    "consensus_sweep",
    "consensus_to_series",
    "trend_level_slope",
    "stability_across_K",
]
