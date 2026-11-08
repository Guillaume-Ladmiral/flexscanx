"""Type stubs for the flexscanx compiled C++ core extension.

Provides BuildStatic, DescribeStatic, ScanSlice, and middle-p probability
functions.  The compiled extension must be built before these are available
at runtime (pip install -e . or pip install flexscanx).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, TypedDict

import numpy as np

class ScanHandle(Protocol):
    """Protocol for the opaque precompute handle returned by BuildStatic.

    The handle is an opaque C++ object (PyCapsule).  It carries the precomputed
    KNN order, distance arrays, and graph structure so that ``ScanSlice`` can be
    called repeatedly without rebuilding.

    The protocol declares ``__hash__`` so that handles can be used as dict keys
    (e.g. for handle-keyed pool caches in tuning.py) and compared by identity.
    """

    def __hash__(self) -> int:
        """Handles are identity-hashed (same object ↔ same precompute)."""
        ...

class CoreScanClusterBase(TypedDict):
    """Required raw single-direction fields emitted by ScanSlice."""

    area: list[int]
    n_case: float
    expected: float
    stats: float
    rank: int
    simcount_actual: int

class CoreScanClusterDict(CoreScanClusterBase, total=False):
    """Raw single-direction cluster/result payload emitted by ScanSlice.

    Note: The Python layer adds 'type', 'name', and 'scan_settings' via
    _finalize_direction_result after ScanSlice returns.
    """

    Rate_Ratio: float | None
    Rate_Ratio_CI: tuple[float, float] | None
    Risk_Ratio: float | None
    Risk_Ratio_CI: tuple[float, float] | None
    ratio_smoothing_applied: Literal["NONE", "HALDANE", "EB"]
    max_dist: float
    from_area: int
    to_area: int
    pval: float | None
    secondaries: list["CoreScanClusterDict"]

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def version() -> str:
    """Return the core version string (e.g. '0.0.1-core')."""
    ...

def source_digest() -> str:
    """Return the build-time SHA-256 digest of the native sources (FXA-168).

    Computed by CMake over the sorted ``"<relative path>\n<sha256>\n"`` records of
    every ``src/flexscanx_core/*.cpp|*.hpp|*.in`` file plus ``CMakeLists.txt``.
    """
    ...

# ---------------------------------------------------------------------------
# Static precompute
# ---------------------------------------------------------------------------

def BuildStatic(
    coords: np.ndarray,
    row_ptr: list[int],
    col_idx: list[int],
    K: int,
    radius: float = ...,
    knn_mode: str = "K_REFILL",
    knn_refill_force: bool = False,
) -> ScanHandle:
    """Build a static precompute handle (opaque PyCapsule).

    Parameters
    - coords: Nx2 float64 array of (x, y) coordinates.
    - row_ptr: CSR row-pointer array (length N+1).
    - col_idx: CSR column-index array.
    - K: KNN horizon (number of nearest neighbours per centre).
    - radius: maximum cluster radius (default inf).
    - knn_mode: 'K_REFILL' (default) or 'K_FIXED'.
    - knn_refill_force: force-enable refill even on large N.

    Returns an opaque handle suitable for DescribeStatic / ScanSlice.
    """
    ...

def DescribeUniverse(
    handle: ScanHandle, center: int, excluded: Sequence[int] | None = None
) -> dict[str, Any]:
    """Return the effective per-centre universe the scan cores see (debug/verification).

    Keys: 'center', 'flexible' (sorted 0-based indices admitted by build_universe),
    'circular_order' (distance-ordered indices used by the circular scanner) and
    'circular_dist2' (their squared distances).  ``excluded`` masks areas already
    detected (secondary search).  Not part of the public inference surface (U-6).
    """
    ...

def DescribeStatic(handle: ScanHandle) -> dict[str, Any]:
    """Return a diagnostic dictionary for a precompute handle.

    Keys: 'N', 'K', 'radius', 'knn_mode', 'refill_enabled', 'has_refill_data',
    'knn_tie_boundary_centers' (centres whose K-th and (K+1)-th nearest
    neighbours are exactly equidistant, FXA-187).
    """
    ...

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def ScanSlice(
    handle: ScanHandle,
    observed: Sequence[int] | np.ndarray,
    expected: Sequence[float] | np.ndarray,
    settings: dict[str, Any],
) -> CoreScanClusterDict:
    """Run a single scan slice using the given precompute handle.

    Parameters
    - handle: opaque precompute handle from BuildStatic.
    - observed: length-N integer count vector.
    - expected: length-N float denominator vector.
    - settings: dict of scan settings (model, scanmethod, stattype,
      clustertype, clustersize, ralpha, simcount, secondary, …).

    Returns a raw single-direction result dict containing keys such as:
    - 'area', 'n_case', 'expected', 'stats', 'rank'
    - model-specific 'Rate_Ratio' / 'Risk_Ratio' (float or None on degenerate/empty outputs)
    - reporting CI keys 'Rate_Ratio_CI' / 'Risk_Ratio_CI' (tuple or None)
    - optional 'pval', 'max_dist', 'from_area', 'to_area', 'secondaries'
    - 'rank' is the Monte Carlo rank among simulated maxima; it is 1 when simcount == 0

    Note: ScanSlice returns raw core payloads only. Higher-level Python helpers
    attach keys such as 'type', 'name', and 'scan_settings'.
    """
    ...

# ---------------------------------------------------------------------------
# Middle-p (exact) probability functions
# ---------------------------------------------------------------------------

def poisson_middle_p_upper(ax: int, mean: float) -> float:
    """Poisson upper-tail middle-p: P(X >= ax | mean) - 0.5*P(X == ax | mean)."""
    ...

def poisson_middle_p_lower(ax: int, mean: float) -> float:
    """Poisson lower-tail middle-p: P(X <= ax | mean) - 0.5*P(X == ax | mean)."""
    ...

def binomial_middle_p_upper(ax: int, n: int, p: float) -> float:
    """Binomial upper-tail middle-p: P(X >= ax | n, p) - 0.5*P(X == ax | n, p)."""
    ...

def binomial_middle_p_lower(ax: int, n: int, p: float) -> float:
    """Binomial lower-tail middle-p: P(X <= ax | n, p) - 0.5*P(X == ax | n, p)."""
    ...
