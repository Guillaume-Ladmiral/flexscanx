"""Internal utility functions for flexscanx.

This module contains shared utilities used across the package.
Not part of the public API.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from types import FrameType
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ._core_loader import require_core as _require_core

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def mc_alias_env_enabled() -> bool:
    """Return True if the FLEXSCANX_MC_ALIAS env var is set to a truthy value."""
    return os.environ.get("FLEXSCANX_MC_ALIAS", "0").strip() in ("1", "true", "True")


def warn_at_caller(message: str, category: type[Warning] = UserWarning) -> None:
    """Emit *message* attributed to the nearest stack frame outside this package.

    Library warnings are consumed by ``warnings`` filters keyed on the caller's
    module and by readers of the printed source line, so a fixed ``stacklevel``
    that blames ``api.py`` is a defect whenever the call depth varies (FXA-173).
    Walking the stack to the first frame outside ``flexscanx`` gives every entry
    point the right attribution without per-site arithmetic.
    """
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_PACKAGE_DIR + os.sep):
            break
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level + 1)


def format_validation_error(
    param_name: str,
    issue: str,
    hint: str | None = None,
    indices: Sequence[int] | None = None,
    max_indices: int = 5,
) -> str:
    """Format a validation error with consistent structure."""
    lines = [f'Parameter "{param_name}" validation failed.']
    if indices is not None and len(indices) > 0:
        idx_list = list(indices)[:max_indices]
        more = " and more" if len(indices) > max_indices else ""
        lines.append(f"Issue: {issue} at indices {idx_list}{more}")
    else:
        lines.append(f"Issue: {issue}")
    if hint:
        lines.append(f"Hint: {hint}")
    return "\n".join(lines)


def extract_rr(d: dict) -> float | None:
    """Extract rate/risk ratio from cluster dict.

    C++ bindings set 'Rate_Ratio' (POISSON model) or 'Risk_Ratio' (BINOMIAL model).
    Returns None if neither key is present.

    Note: This is v0.0.1 - no backward compatibility needed.
    """
    if "Rate_Ratio" in d:
        return d["Rate_Ratio"]
    if "Risk_Ratio" in d:
        return d["Risk_Ratio"]
    return None


def _validate_numeric_array_common(
    arr: np.ndarray, name: str, allow_zero: bool = True, allow_negative: bool = False
) -> None:
    """Common validation for numeric arrays: NaN, Inf, and range checks.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate (must be float64)
    name : str
        Name for error messages
    allow_zero : bool, default True
        Whether zero values are allowed
    allow_negative : bool, default False
        Whether negative values are allowed

    Raises
    ------
    ValueError
        If array contains NaN, Inf, or values outside allowed range
    """
    # Check for NaN
    if np.any(np.isnan(arr)):
        nan_idx = np.where(np.isnan(arr))[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "contains NaN values",
                hint="Ensure all values are valid numbers (not NaN)",
                indices=nan_idx,
            )
        )

    # Check for Inf
    if np.any(np.isinf(arr)):
        inf_idx = np.where(np.isinf(arr))[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "contains infinite values",
                hint="Ensure all values are finite numbers (not infinity)",
                indices=inf_idx,
            )
        )

    # Check range
    if not allow_negative:
        if np.any(arr < 0):
            neg_idx = np.where(arr < 0)[0].tolist()
            raise ValueError(
                format_validation_error(
                    name,
                    "must be non-negative",
                    hint="Ensure all values are >= 0",
                    indices=neg_idx,
                )
            )

    if not allow_zero:
        if np.any(arr <= 0):
            zero_idx = np.where(arr <= 0)[0].tolist()
            raise ValueError(
                format_validation_error(
                    name,
                    "must be strictly positive",
                    hint=f"For model calculations, {name}=0 would cause division by zero",
                    indices=zero_idx,
                )
            )


_OBSERVED_INTEGER_TOLERANCE = 1e-9


def validate_integer_parameter(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate a scalar integer control without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}")
    return result


def validate_float_parameter(value: object, name: str) -> float:
    """Convert a scalar real-valued control, naming the parameter on failure.

    Strings that are not numbers keep raising ``ValueError`` and non-numeric
    objects (lists, mappings, ``None``) keep raising ``TypeError``, exactly as
    ``float()`` does, so callers that catch one type specifically are unaffected;
    only the message changes, and it now names the parameter (FXA-180).
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)  # type: ignore[arg-type]
    except ValueError:
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    except TypeError:
        raise TypeError(f"{name} must be a real number, got {value!r}") from None


def validate_cluster_radius(value: Any) -> float:
    """Validate a non-negative finite radius or the +infinity sentinel."""
    radius = validate_float_parameter(value, "clusterradius")
    if math.isnan(radius) or radius == -math.inf:
        raise ValueError(f"clusterradius must be finite or +inf, got {value!r}")
    if radius < 0.0:
        raise ValueError(f"clusterradius must be non-negative or +inf, got {value}")
    return radius


def validate_observed_array(observed: npt.ArrayLike, name: str = "observed") -> np.ndarray:
    """Validate observed counts array.

    Parameters
    ----------
    observed : Sequence[int]
        Array of observed counts
    name : str, default "observed"
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated array as int32

    Raises
    ------
    ValueError
        If array contains NaN, Inf, or negative values
    """
    arr = np.asarray(observed, dtype=np.float64)  # Use float to detect NaN/Inf first
    _validate_numeric_array_common(arr, name, allow_zero=True, allow_negative=False)

    # Count data must be integer-valued; reject silent truncation.
    frac_mask = np.abs(arr - np.rint(arr)) > _OBSERVED_INTEGER_TOLERANCE
    if np.any(frac_mask):
        frac_idx = np.where(frac_mask)[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "must contain integer values",
                hint="Observed counts are discrete events; use integer counts only",
                indices=frac_idx,
            )
        )

    int32_max = np.iinfo(np.int32).max
    if np.any(arr > float(int32_max)):
        big_idx = np.where(arr > float(int32_max))[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "contains values that exceed int32 range",
                hint=f"Maximum supported value is {int32_max}",
                indices=big_idx,
            )
        )

    return np.rint(arr).astype(np.int32)


def validate_expected_array(
    expected: npt.ArrayLike,
    name: str = "expected",
    allow_zero: bool = False,
) -> np.ndarray:
    """Validate expected/population array.

    Parameters
    ----------
    expected : Sequence[float]
        Array of expected counts or population sizes
    name : str, default "expected"
        Name for error messages
    allow_zero : bool, default False
        Whether to allow zero values

    Returns
    -------
    np.ndarray
        Validated array as float64

    Raises
    ------
    ValueError
        If array contains NaN, Inf, or negative values (zero allowed when allow_zero=True)
    """
    arr = np.asarray(expected, dtype=np.float64)
    _validate_numeric_array_common(arr, name, allow_zero=allow_zero, allow_negative=False)
    return arr


def validate_binomial_population(population: npt.ArrayLike, name: str = "population") -> np.ndarray:
    """Validate BINOMIAL model population array (must be positive integers).

    Parameters
    ----------
    population : Sequence[float]
        Array of population sizes
    name : str, default "population"
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated array as float64 (with integer values)

    Raises
    ------
    ValueError
        If array contains non-integers or non-positive values
    """
    arr = validate_expected_array(population, name=name, allow_zero=False)

    # Check for non-integer values
    frac_mask = np.abs(arr - np.rint(arr)) > _OBSERVED_INTEGER_TOLERANCE
    if np.any(frac_mask):
        frac_idx = np.where(frac_mask)[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "must contain integer values for BINOMIAL model",
                hint="BINOMIAL model requires integer population sizes (e.g., number of trials)",
                indices=frac_idx,
            )
        )

    int32_max = np.iinfo(np.int32).max
    if np.any(arr > float(int32_max)):
        big_idx = np.where(arr > float(int32_max))[0].tolist()
        raise ValueError(
            format_validation_error(
                name,
                "contains values that exceed int32 range",
                hint=f"Maximum supported population is {int32_max}",
                indices=big_idx,
            )
        )

    return arr


# Parallelism configuration constants
_PARALLELISM_MIN_N = 5000  # Threshold below which single-threading is preferred
_PARALLELISM_SCALE = 2000  # Scale factor: ~4 threads for N=8000


def compute_default_n_jobs(
    N: int, *, min_N: int = _PARALLELISM_MIN_N, scale: int = _PARALLELISM_SCALE
) -> int:
    """Compute smart default for parallelism based on problem size.

    Strategy:
    - N <= min_N: use 1 thread (overhead not worth it)
    - N > min_N: use min(cpu_count, max(2, N // scale))

    Rationale: Scale factor of 2000 means ~4 threads for N=8000,
    which balances overhead vs. speedup on typical 4-8 core machines.

    Parameters
    ----------
    N : int
        Problem size (number of areas)
    min_N : int, default 5000
        Threshold below which single-threading is preferred
    scale : int, default 2000
        Scale factor for gradual parallelism ramp-up

    Returns
    -------
    int
        Number of threads (1 or more)
    """
    if N <= min_N:
        return 1
    cpu_count = max(1, os.cpu_count() or 1)
    return min(cpu_count, max(2, N // scale))


# Default K value for spatial scan
_DEFAULT_K = 15
_VALID_KNN_METHODS = frozenset({"graph", "geometry"})
_VALID_KNN_MODES = frozenset({"K_FIXED", "K_REFILL"})


def _validate_coords_and_infer_N(coords: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Validate coordinates array and infer N.

    Returns
    -------
    coords : np.ndarray
        Validated Nx2 coordinates
    N : int
        Number of areas
    """
    if coords is None:
        raise ValueError("coords are required when handle is not provided")
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be Nx2 array, got shape {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite numeric values")
    return coords, coords.shape[0]


def _validate_and_default_K(K: int | None, N: int) -> int:
    """Validate K parameter and apply default if needed.

    Returns
    -------
    K : int
        Validated K value in range [1, N]
    """
    if K is None:
        return min(_DEFAULT_K, int(N))

    return validate_integer_parameter(K, "K", minimum=1, maximum=int(N))


def _normalize_knn_method(knn_method: str) -> str:
    """Validate and normalize a KNN adjacency construction method."""
    method = str(knn_method).strip().lower()
    if method not in _VALID_KNN_METHODS:
        raise ValueError(
            f"knn_method must be one of {sorted(_VALID_KNN_METHODS)}, got {knn_method!r}"
        )
    return method


def _normalize_knn_mode(knn_mode: str) -> str:
    """Validate and normalize the static precompute KNN mode."""
    mode = str(knn_mode).strip().upper()
    if mode not in _VALID_KNN_MODES:
        raise ValueError(f"knn_mode must be one of {sorted(_VALID_KNN_MODES)}, got {knn_mode!r}")
    return mode


def _build_adjacency_from_method(
    knn_method: str,
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]] | None,
    K: int,
    clusterradius: float,
) -> tuple[Sequence[int], Sequence[int]]:
    """Build or validate adjacency based on knn_method.

    Returns
    -------
    row_ptr : Sequence[int]
        CSR row pointers
    col_idx : Sequence[int]
        CSR column indices
    """
    method = _normalize_knn_method(knn_method)
    if method == "geometry":
        from .graph import csr_knn_from_coords

        return csr_knn_from_coords(
            coords, int(K), radius=float(clusterradius), include_self=False, backend="auto"
        )
    if adjacency is None:
        raise ValueError("adjacency is required when knn_method='graph' and handle is not provided")
    return _validate_csr_adjacency(adjacency, N=int(coords.shape[0]))


def _validate_csr_adjacency(
    adjacency: tuple[Sequence[int], Sequence[int]], N: int
) -> tuple[list[int], list[int]]:
    """Validate CSR adjacency for an undirected graph.

    Validates shape, bounds, duplicates, self-loops, and symmetry.
    Returns normalized Python int lists suitable for pybind calls.
    """
    if not isinstance(adjacency, tuple) or len(adjacency) != 2:
        raise ValueError("adjacency must be a (row_ptr, col_idx) tuple")

    row_ptr_raw, col_idx_raw = adjacency
    row_ptr_untyped = np.asarray(row_ptr_raw)
    col_idx_untyped = np.asarray(col_idx_raw)
    if row_ptr_untyped.dtype.kind not in "iu" or (
        col_idx_untyped.size > 0 and col_idx_untyped.dtype.kind not in "iu"
    ):
        raise ValueError("adjacency row_ptr and col_idx must contain integer indices")
    row_ptr = row_ptr_untyped.astype(np.int64, copy=False)
    # ``np.asarray([])`` defaults to float64 even when a public graph builder
    # produced the empty list.  An edge-free CSR has no values whose integer
    # nature could be violated, so normalize that valid closure case directly.
    col_idx = col_idx_untyped.astype(np.int64, copy=False)

    if row_ptr.ndim != 1:
        raise ValueError(f"adjacency row_ptr must be 1D, got shape {row_ptr.shape}")
    if col_idx.ndim != 1:
        raise ValueError(f"adjacency col_idx must be 1D, got shape {col_idx.shape}")
    if row_ptr.size != N + 1:
        raise ValueError(f"adjacency row_ptr must have length N+1 ({N + 1}), got {row_ptr.size}")
    if row_ptr[0] != 0:
        raise ValueError("adjacency row_ptr must start at 0")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("adjacency row_ptr must be non-decreasing")
    if int(row_ptr[-1]) != int(col_idx.size):
        raise ValueError("adjacency CSR is invalid: row_ptr[N] must equal len(col_idx)")

    if np.any(col_idx < 0) or np.any(col_idx >= N):
        raise ValueError("adjacency col_idx contains out-of-range vertex indices")

    row_sets: list[set[int]] = []
    for i in range(N):
        start = int(row_ptr[i])
        end = int(row_ptr[i + 1])
        nbrs = col_idx[start:end]

        if np.any(nbrs == i):
            raise ValueError(f"adjacency contains self-loop at node {i}")
        if nbrs.size > 1 and np.unique(nbrs).size != nbrs.size:
            raise ValueError(f"adjacency row {i} contains duplicate neighbors")

        row_sets.append(set(int(v) for v in nbrs.tolist()))

    for i, row_neighbors in enumerate(row_sets):
        for j in row_neighbors:
            if i not in row_sets[j]:
                raise ValueError(
                    "adjacency must be symmetric for an undirected graph: "
                    f"missing reverse edge {j}->{i} for edge {i}->{j}"
                )

    return row_ptr.astype(int).tolist(), col_idx.astype(int).tolist()


def _build_core_static_handle(
    coords: np.ndarray,
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    K: int,
    clusterradius: float,
    *,
    knn_mode: str,
    knn_refill_force: bool,
) -> Any:
    """Call the core static-builder with normalized Python values."""
    core = _require_core()
    mode = _normalize_knn_mode(knn_mode)

    handle = core.BuildStatic(
        np.asarray(coords, dtype=float),
        list(row_ptr),
        list(col_idx),
        int(K),
        float(clusterradius),
        mode,
        bool(knn_refill_force),
    )
    _warn_on_tied_universe_boundary(core, handle, int(K))
    return handle


def _warn_on_tied_universe_boundary(core: Any, handle: Any, K: int) -> None:
    """Warn when a centre's K-limited universe boundary falls inside a distance tie.

    A K-nearest-neighbour universe has no labelling-independent definition when
    the K-th and (K+1)-th nearest neighbours are exactly equidistant; flexscanx
    resolves the tie by area index, so on a regular lattice the reported cluster,
    statistic and p-value can change when the input rows are relabelled
    (FXA-187).  The compiled core counts such centres at build time; making the
    dependence discoverable is decision D4 of the 2026-09-07 remediation.
    """
    try:
        description = core.DescribeStatic(handle)
    except (TypeError, RuntimeError):
        return
    tied = (
        int(description.get("knn_tie_boundary_centers", 0)) if isinstance(description, dict) else 0
    )
    if tied <= 0:
        return
    total = int(description.get("N", 0))
    warn_at_caller(
        f"{tied} of {total} centres have exactly equidistant neighbours at the "
        f"K={K} universe boundary (regular lattice or coarsely rounded coordinates). "
        "Equidistant neighbours are resolved by area index, so the reported cluster, "
        "statistic and p-value can depend on the row order of the input. Choose a K "
        "that does not split a tied distance group, or break the ties (for example "
        "by adding a negligible coordinate jitter); see API_REFERENCE.md, parameter K."
    )


def build_validated_static_handle(
    coords: np.ndarray,
    adjacency: tuple[Sequence[int], Sequence[int]],
    K: int | None,
    clusterradius: float = float("inf"),
    *,
    knn_mode: str = "K_REFILL",
    knn_refill_force: bool = False,
) -> Any:
    """Validate static-builder inputs and construct a reusable handle."""
    coords_arr, N = _validate_coords_and_infer_N(coords)
    K_eff = _validate_and_default_K(K, N)
    radius = validate_cluster_radius(clusterradius)
    mode = _normalize_knn_mode(knn_mode)
    row_ptr, col_idx = _validate_csr_adjacency(adjacency, N=N)
    return _build_core_static_handle(
        coords_arr,
        row_ptr,
        col_idx,
        K_eff,
        radius,
        knn_mode=mode,
        knn_refill_force=bool(knn_refill_force),
    )


def describe_static_handle(handle: Any) -> dict:
    """Return the core's description of *handle*, or raise a public ``ValueError``.

    The compiled ``DescribeStatic`` binding rejects anything that is not the
    capsule ``build_static()`` returns with a pybind11 ``TypeError`` naming the
    internal symbol.  That symbol is not part of the public contract, so the
    failure is translated into a ``ValueError`` that names the ``handle``
    parameter instead (FXA-171).
    """
    core = _require_core()
    if handle is None:
        raise ValueError("handle must be a static precompute handle returned by build_static()")
    try:
        description = core.DescribeStatic(handle)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "handle must be a static precompute handle returned by build_static(); "
            f"got an object of type {type(handle).__name__}"
        ) from exc
    return dict(description) if isinstance(description, dict) else {}


def _get_static_description(handle: Any) -> dict | None:
    """Get the static handle description, raising ``ValueError`` for a bad handle.

    Returns
    -------
    static_desc : dict or None
        Description dict, or None if the core offers no description
    """
    description = describe_static_handle(handle)
    return description if description else None


def ensure_static_handle(
    handle: Any | None,
    coords: np.ndarray | None,
    adjacency: tuple[Sequence[int], Sequence[int]] | None,
    K: int | None,
    N: int | None,
    clusterradius: float | None,
    knn_method: str | None,
    knn_mode: str | None,
    knn_refill_force: bool | None,
) -> tuple[Any, int, int | None, dict | None]:
    """Ensure a valid static precompute handle exists, building if necessary.

    This is a unified utility to avoid code duplication between api.py and validation.py.

    Parameters
    ----------
    handle : Any, optional
        Existing handle or None
    coords : np.ndarray, optional
        Nx2 coordinates (required if handle is None)
    adjacency : tuple, optional
        (row_ptr, col_idx) CSR adjacency (required if handle is None and knn_method='graph')
    K : int, optional
        KNN horizon (defaults to min(15, N) if handle is None)
    N : int, optional
        Number of areas (inferred from coords or must be provided if handle exists)
    clusterradius : float, optional
        Maximum cluster radius. ``None`` means unlimited for a new handle and
        means "not specified" when reusing an existing handle.
    knn_method : str, optional
        'graph' or 'geometry'. ``None`` resolves to 'graph' for a new handle.
    knn_mode : str, optional
        'K_FIXED' or 'K_REFILL'. ``None`` resolves to 'K_REFILL' for a new handle.
    knn_refill_force : bool, optional
        Force refill on large N. ``None`` resolves to False for a new handle.

    Returns
    -------
    handle : Any
        Valid precompute handle
    N : int
        Number of areas
    K : int or None
        Effective K value when available (from requested/default K or handle metadata)
    static_desc : dict or None
        Static handle description

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
    """
    radius_was_requested = clusterradius is not None
    method_was_requested = knn_method is not None
    mode_was_requested = knn_mode is not None
    refill_force_was_requested = knn_refill_force is not None

    mode = _normalize_knn_mode("K_REFILL" if knn_mode is None else knn_mode)
    method = _normalize_knn_method("graph" if knn_method is None else knn_method)
    radius = validate_cluster_radius(float("inf") if clusterradius is None else clusterradius)
    refill_force = False if knn_refill_force is None else bool(knn_refill_force)
    if handle is None:
        # Build new handle: validate inputs and construct
        coords, N = _validate_coords_and_infer_N(coords)
        K = _validate_and_default_K(K, N)
        row_ptr, col_idx = _build_adjacency_from_method(method, coords, adjacency, K, radius)

        # Build static precompute handle
        handle = _build_core_static_handle(
            coords,
            row_ptr,
            col_idx,
            K,
            radius,
            knn_mode=mode,
            knn_refill_force=refill_force,
        )

        static_desc = _get_static_description(handle)
        return handle, int(N), int(K), static_desc
    else:
        # Reuse existing handle: validate N is provided
        if N is None:
            raise ValueError("N must be provided when reusing a handle")
        static_desc = _get_static_description(handle)
        if isinstance(static_desc, dict) and "N" in static_desc:
            try:
                handle_N = int(static_desc["N"])
            except (TypeError, ValueError):
                handle_N = None
            if handle_N is not None and handle_N != int(N):
                raise ValueError(
                    f"static handle N mismatch: handle has N={handle_N}, "
                    f"observed length is N={int(N)}"
                )
        # The handle owns the geometry and the graph.  A caller-supplied graph can
        # no longer take effect, so it is rejected rather than silently discarded
        # (FXA-179), exactly as K, clusterradius and knn_mode already are.
        if coords is not None:
            raise ValueError(
                "handle reuse is incompatible with a caller-supplied coords array; "
                "pass either handle= or coords=/adjacency=, or rebuild the handle "
                "with build_static()"
            )
        if adjacency is not None:
            raise ValueError(
                "handle reuse is incompatible with a caller-supplied adjacency; "
                "pass either handle= or coords=/adjacency=, or rebuild the handle "
                "with build_static()"
            )
        if method_was_requested and method != "graph":
            raise ValueError(
                "handle reuse is incompatible with knn_method other than 'graph'; "
                "the handle's precomputed graph cannot be rebuilt during scan"
            )
        if isinstance(static_desc, dict):
            if K is not None and "K" in static_desc and int(K) != int(static_desc["K"]):
                raise ValueError(
                    f"handle was built with K={int(static_desc['K'])}; "
                    f"incompatible requested K={int(K)}"
                )
            if radius_was_requested and "radius" in static_desc:
                handle_radius = float(static_desc["radius"])
                if not (
                    (math.isinf(radius) and math.isinf(handle_radius))
                    or math.isclose(radius, handle_radius, rel_tol=0.0, abs_tol=0.0)
                ):
                    raise ValueError(
                        f"handle was built with clusterradius={handle_radius}; "
                        f"incompatible requested clusterradius={radius}"
                    )
            if mode_was_requested and "knn_mode" in static_desc:
                handle_mode = str(static_desc["knn_mode"]).upper()
                if mode != handle_mode:
                    raise ValueError(
                        f"handle was built with knn_mode={handle_mode!r}; "
                        f"incompatible requested knn_mode={mode!r}"
                    )
        if refill_force_was_requested:
            raise ValueError(
                "handle reuse is incompatible with an explicit knn_refill_force; "
                "that build-time option cannot alter an existing handle"
            )
        eff_K: int | None = None
        if isinstance(static_desc, dict) and "K" in static_desc:
            try:
                eff_K = int(static_desc["K"])
            except (TypeError, ValueError):
                eff_K = None
        return handle, int(N), eff_K, static_desc
