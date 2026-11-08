"""Graph-based cluster validation metrics.

Moved from validation.py (B1 split) — all external imports go through
validation.py which re-exports the public names.

The silhouette and Dunn metrics use BFS shortest-path distances on the scan
adjacency graph, while boundary contrast has a separate accelerated inner loop.
When numba is installed (``pip install flexscanx[jit]``), those paths are
JIT-compiled. Otherwise their pure-Python fallbacks are used; importing this
module remains silent, and the first fallback-backed metric call warns once per
process.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Sequence

import numpy as np

from ._utils import (
    _validate_csr_adjacency,
    validate_binomial_population,
    validate_expected_array,
    validate_observed_array,
    warn_at_caller,
)

# ``FLEXSCANX_DISABLE_JIT=1`` forces the pure-Python paths even when numba is
# installed.  CI sets it on its non-``jit`` matrix cells; until 2026-09 nothing
# read it (FXA-161).
_JIT_DISABLED_BY_ENV = os.environ.get("FLEXSCANX_DISABLE_JIT", "0").strip() in (
    "1",
    "true",
    "True",
)

_nb: Any = None
_HAS_NUMBA = False
if not _JIT_DISABLED_BY_ENV:
    try:
        import numba as _numba  # type: ignore[import-untyped]
    except ImportError:
        pass
    else:
        _nb = _numba
        _HAS_NUMBA = True

# ---------------------------------------------------------------------------
# BFS distance helpers
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @_nb.njit(cache=True)
    def _bfs_distances_numba(row_ptr: np.ndarray, col_idx: np.ndarray, src: int) -> np.ndarray:
        n_nodes = len(row_ptr) - 1
        dist = np.full(n_nodes, np.inf, dtype=np.float64)
        if not (0 <= src < n_nodes):
            return dist
        dist[src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            start, end = row_ptr[node], row_ptr[node + 1]
            for pos in range(start, end):
                neighbor = col_idx[pos]
                if np.isinf(dist[neighbor]):
                    dist[neighbor] = dist[node] + 1.0
                    queue.append(neighbor)
        return dist


def _bfs_distances_py(row_ptr: Sequence[int], col_idx: Sequence[int], src: int) -> np.ndarray:
    n_nodes = int(len(row_ptr) - 1)
    dist = np.full(n_nodes, np.inf, dtype=float)
    src = int(src)
    if not (0 <= src < n_nodes):
        return dist
    dist[src] = 0.0
    queue = deque([src])
    while queue:
        node = queue.popleft()
        start, end = row_ptr[node], row_ptr[node + 1]
        for pos in range(start, end):
            neighbor = int(col_idx[pos])
            if dist[neighbor] == np.inf:
                dist[neighbor] = dist[node] + 1.0
                queue.append(neighbor)
    return dist


if _HAS_NUMBA:

    def _bfs_distances(row_ptr: Sequence[int], col_idx: Sequence[int], src: int) -> np.ndarray:
        return _bfs_distances_numba(
            np.asarray(row_ptr, dtype=np.int64),
            np.asarray(col_idx, dtype=np.int64),
            int(src),
        )

else:
    _bfs_distances = _bfs_distances_py  # type: ignore


_NUMBA_FALLBACK_WARNED = False


def _warn_numba_fallback() -> None:
    """Warn once, and only when a graph metric actually uses the fallback."""
    global _NUMBA_FALLBACK_WARNED
    if _HAS_NUMBA or _NUMBA_FALLBACK_WARNED or _JIT_DISABLED_BY_ENV:
        return
    warn_at_caller(
        "numba is not installed: this graph metric is using a pure-Python fallback "
        "that can be significantly slower on large graphs. Install numba for "
        "JIT-accelerated performance: pip install flexscanx[jit]"
    )
    _NUMBA_FALLBACK_WARNED = True


# ---------------------------------------------------------------------------
# Numba-accelerated boundary contrast inner loop
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @_nb.njit(cache=True)
    def _bc_core(obs_a, exp_a, row_ptr_a, col_idx_a, mem_a, model_flag, p_gl):
        n_edges = 0
        max_edges = col_idx_a.shape[0] // 2
        deltas = np.empty(max_edges, dtype=np.float64)
        zvals = np.empty(max_edges, dtype=np.float64)
        n_nodes = row_ptr_a.shape[0] - 1
        for i in range(n_nodes):
            start = row_ptr_a[i]
            end = row_ptr_a[i + 1]
            for pos in range(start, end):
                j = col_idx_a[pos]
                if i < j and mem_a[i] != mem_a[j]:
                    ru = obs_a[i] / exp_a[i] if exp_a[i] > 0.0 else 0.0
                    rv = obs_a[j] / exp_a[j] if exp_a[j] > 0.0 else 0.0
                    delta = ru - rv
                    vi = 1.0 / exp_a[i] if exp_a[i] > 0.0 else 0.0
                    vj = 1.0 / exp_a[j] if exp_a[j] > 0.0 else 0.0
                    if model_flag == 1:
                        var = p_gl * (1.0 - p_gl) * (vi + vj)
                    else:
                        var = vi + vj
                    z_val = delta / np.sqrt(var) if var > 0.0 else 0.0
                    deltas[n_edges] = delta
                    zvals[n_edges] = z_val
                    n_edges += 1
        return deltas[:n_edges], zvals[:n_edges]

else:
    _bc_core = None  # type: ignore

# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _validate_graph_membership(
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    membership: Sequence[int],
    *,
    name: str = "membership",
) -> tuple[list[int], list[int], np.ndarray]:
    row_ptr_raw = np.asarray(row_ptr)
    if row_ptr_raw.ndim != 1 or row_ptr_raw.size < 1:
        raise ValueError("row_ptr must be a 1D CSR row pointer array")
    n_nodes = int(row_ptr_raw.size - 1)
    row_ptr_valid, col_idx_valid = _validate_csr_adjacency((row_ptr, col_idx), n_nodes)

    mem_raw = np.asarray(membership)
    if mem_raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if mem_raw.size != n_nodes:
        raise ValueError(f"{name} length must equal graph node count {n_nodes}, got {mem_raw.size}")
    if mem_raw.dtype.kind not in "biu" or np.any((mem_raw != 0) & (mem_raw != 1)):
        raise ValueError(f"{name} must contain only 0 or 1")
    return row_ptr_valid, col_idx_valid, mem_raw.astype(np.int8, copy=False)


def _validate_observed_expected_lengths(
    observed: Sequence[int], expected: Sequence[float], n_nodes: int, model: str
) -> tuple[np.ndarray, np.ndarray, str]:
    model_u = str(model).strip().upper()
    if model_u not in {"POISSON", "BINOMIAL"}:
        raise ValueError(f"model must be one of ['BINOMIAL', 'POISSON'], got {model!r}")

    obs = validate_observed_array(observed).astype(np.float64)
    if model_u == "BINOMIAL":
        exp = validate_binomial_population(expected)
    else:
        exp = validate_expected_array(expected, allow_zero=True)
    if obs.ndim != 1 or exp.ndim != 1:
        raise ValueError("observed and expected must be 1D arrays")
    if obs.size != n_nodes or exp.size != n_nodes:
        raise ValueError(
            f"observed and expected must have length {n_nodes}, got {obs.size} and {exp.size}"
        )
    if model_u == "BINOMIAL" and np.any(obs > exp):
        raise ValueError("observed counts cannot exceed population for BINOMIAL model")
    return obs, exp, model_u


# ---------------------------------------------------------------------------
# Public graph metrics
# ---------------------------------------------------------------------------


def silhouette_graph(
    row_ptr: Sequence[int], col_idx: Sequence[int], membership: Sequence[int]
) -> dict[str, Any]:
    """Silhouette score using graph shortest-path distances.

    Parameters
    ----------
    row_ptr : sequence of int
        CSR row-pointer array (length N+1).
    col_idx : sequence of int
        CSR column-index array.
    membership : sequence of int
        Binary membership vector (1 = in-cluster, 0 = out-of-cluster).

    Returns
    -------
    dict
        Keys ``'mean'`` (float) and ``'values'`` (list of per-node silhouette scores).

    Notes
    -----
    Intended for diagnostic use on modest graphs; this runs one BFS per
    in-cluster node.  Install ``flexscanx[jit]`` for JIT-accelerated BFS.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    idx_in = np.where(mem == 1)[0].tolist()
    idx_out = np.where(mem == 0)[0].tolist()
    if not idx_in or not idx_out:
        return {"mean": float("nan"), "values": []}

    if len(idx_in) > 1:
        _warn_numba_fallback()

    N_graph = len(row_ptr) - 1
    if len(idx_in) * N_graph > 250_000:
        warn_at_caller(
            f"silhouette_graph: {len(idx_in)} in-cluster nodes × {N_graph} graph nodes "
            f"= {len(idx_in) * N_graph:,} BFS steps. This may be slow on large graphs."
        )

    values: list[float] = []
    for node in idx_in:
        if len(idx_in) == 1:
            # Rousseeuw (1987): s(i) = 0 for singleton clusters.
            values.append(0.0)
            continue
        dist = _bfs_distances(row_ptr, col_idx, node)
        # a(i) averages over the OTHER |C|-1 members; dist[node] == 0, so the
        # sum is unchanged but the divisor must exclude self (audit AUD-4:
        # including the self-distance inflated every silhouette value).
        intra = float(np.sum(dist[idx_in])) / float(len(idx_in) - 1)
        extra = float(np.mean(dist[idx_out]))
        if math.isinf(intra):
            # A split cluster has unbounded within-cluster distance: retain a
            # finite diagnostic convention rather than propagating NaN.
            values.append(-1.0)
        elif math.isinf(extra):
            # As b(i) tends to infinity with finite a(i), Rousseeuw's formula
            # tends to one (perfect separation from an islanded complement).
            values.append(1.0)
        else:
            denom = max(intra, extra)
            values.append((extra - intra) / denom if denom > 0 else 0.0)

    return {"mean": float(np.mean(values)) if values else float("nan"), "values": values}


def dunn_graph(row_ptr: Sequence[int], col_idx: Sequence[int], membership: Sequence[int]) -> float:
    """Dunn-style separation/compactness ratio on graph shortest-path distances.

    Distances are shortest paths in the whole graph, so two members may be
    joined through the complement.  Degenerate cases are returned rather than
    raised so callers can filter them: ``nan`` when no member reaches another
    member or the complement (for example a graph with no edges); ``inf`` when
    the cluster is a single node (zero diameter) or the complement is unreachable
    from every member; ``0.0`` when two members are mutually unreachable
    (infinite diameter) while the complement is reachable.

    Note: this is a cluster-vs-rest *variant* of the Dunn index — the
    denominator is the diameter of the cluster only (the complement's diameter
    is not meaningful for a binary cluster-vs-rest split), whereas the
    classical Dunn index uses the maximum diameter over all clusters.

    Parameters
    ----------
    row_ptr, col_idx : sequence of int
        CSR adjacency arrays.
    membership : sequence of int
        Binary membership vector (1 = in-cluster, 0 = out-of-cluster).

    Returns
    -------
    float
        Min separation / cluster diameter, or ``nan``/``inf`` when undefined.

    Notes
    -----
    Intended for diagnostic use on modest graphs; this runs one BFS per
    in-cluster node.  Install ``flexscanx[jit]`` for JIT-accelerated BFS.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    idx_in = np.where(mem == 1)[0].tolist()
    idx_out = np.where(mem == 0)[0].tolist()
    if not idx_in or not idx_out:
        return float("nan")

    _warn_numba_fallback()

    N_graph = len(row_ptr) - 1
    if len(idx_in) * N_graph > 250_000:
        warn_at_caller(
            f"dunn_graph: {len(idx_in)} in-cluster nodes × {N_graph} graph nodes "
            f"= {len(idx_in) * N_graph:,} BFS steps. This may be slow on large graphs."
        )

    max_diameter = 0.0
    min_separation = float("inf")
    for node in idx_in:
        dist = _bfs_distances(row_ptr, col_idx, node)
        if len(idx_in) > 1:
            max_diameter = max(max_diameter, float(np.max(dist[idx_in])))
        min_separation = min(min_separation, float(np.min(dist[idx_out])))

    if max_diameter <= 0:
        return float("inf")
    return min_separation / max_diameter


def join_counts(
    row_ptr: Sequence[int], col_idx: Sequence[int], membership: Sequence[int]
) -> dict[str, int]:
    """Compute binary join counts for an undirected graph partition.

    Parameters
    ----------
    row_ptr, col_idx : sequence of int
        CSR adjacency arrays.
    membership : sequence of int
        Binary membership vector (1 = in-cluster, 0 = out-of-cluster).

    Returns
    -------
    dict
        Keys ``'J11'``, ``'J10'``, ``'J00'`` — edge counts within/across clusters.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    j11 = j10 = j00 = 0
    for i in range(len(row_ptr) - 1):
        start, end = row_ptr[i], row_ptr[i + 1]
        for pos in range(start, end):
            j = col_idx[pos]
            if i < j:
                if mem[i] == 1 and mem[j] == 1:
                    j11 += 1
                elif mem[i] == 0 and mem[j] == 0:
                    j00 += 1
                else:
                    j10 += 1
    return {"J11": j11, "J10": j10, "J00": j00}


def boundary_contrast(
    observed: Sequence[int],
    expected: Sequence[float],
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    membership: Sequence[int],
    model: str = "POISSON",
) -> dict[str, float]:
    """Compute descriptive boundary contrast metrics across the cut.

    Parameters
    ----------
    observed : sequence of int
        Observed counts, length N.
    expected : sequence of float
        Expected counts or population sizes, length N.
    row_ptr, col_idx : sequence of int
        CSR adjacency arrays.
    membership : sequence of int
        Binary membership vector.
    model : str, default "POISSON"
        ``'POISSON'`` or ``'BINOMIAL'`` (affects variance formula).

    Returns
    -------
    dict
        Keys: ``'n_edges'``, ``'mean_delta'``, ``'std_delta'``, ``'mean_z'``, ``'std_z'``.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    obs, exp, model_u = _validate_observed_expected_lengths(observed, expected, len(mem), model)
    p_glob = float(obs.sum()) / float(exp.sum()) if float(exp.sum()) > 0 else 0.0
    p_glob = max(0.0, min(1.0, p_glob))

    if _bc_core is not None:
        d_arr, z_arr = _bc_core(
            obs,
            exp,
            np.asarray(row_ptr, dtype=np.int64),
            np.asarray(col_idx, dtype=np.int64),
            np.asarray(mem, dtype=np.int64),
            1 if model_u == "BINOMIAL" else 0,
            float(p_glob),
        )
        n_edges = len(d_arr)
        if n_edges == 0:
            return {"n_edges": 0, "mean_delta": 0.0, "std_delta": 0.0, "mean_z": 0.0, "std_z": 0.0}
        return {
            "n_edges": int(n_edges),
            "mean_delta": float(d_arr.mean()),
            "std_delta": float(d_arr.std(ddof=1)) if n_edges > 1 else 0.0,
            "mean_z": float(z_arr.mean()),
            "std_z": float(z_arr.std(ddof=1)) if n_edges > 1 else 0.0,
        }

    _warn_numba_fallback()

    deltas: list[float] = []
    zvals: list[float] = []
    for i in range(len(row_ptr) - 1):
        start, end = row_ptr[i], row_ptr[i + 1]
        for pos in range(start, end):
            j = col_idx[pos]
            if i < j and mem[i] != mem[j]:
                ratio_i = obs[i] / exp[i] if exp[i] > 0.0 else 0.0
                ratio_j = obs[j] / exp[j] if exp[j] > 0.0 else 0.0
                delta = ratio_i - ratio_j
                deltas.append(delta)
                vi = 1.0 / exp[i] if exp[i] > 0.0 else 0.0
                vj = 1.0 / exp[j] if exp[j] > 0.0 else 0.0
                var = p_glob * (1.0 - p_glob) * (vi + vj) if model_u == "BINOMIAL" else vi + vj
                zvals.append(delta / math.sqrt(var) if var > 0.0 else 0.0)

    if not deltas:
        return {"n_edges": 0, "mean_delta": 0.0, "std_delta": 0.0, "mean_z": 0.0, "std_z": 0.0}

    d_arr = np.asarray(deltas, dtype=float)
    z_arr = np.asarray(zvals, dtype=float)
    return {
        "n_edges": int(len(d_arr)),
        "mean_delta": float(d_arr.mean()),
        "std_delta": float(d_arr.std(ddof=1)) if len(d_arr) > 1 else 0.0,
        "mean_z": float(z_arr.mean()),
        "std_z": float(z_arr.std(ddof=1)) if len(z_arr) > 1 else 0.0,
    }


def morans_I_within(
    observed: Sequence[int],
    expected: Sequence[float],
    row_ptr: Sequence[int],
    col_idx: Sequence[int],
    membership: Sequence[int],
    model: str = "POISSON",
) -> float:
    """Compute Moran's I within the cluster on standardized residuals.

    Parameters
    ----------
    observed : sequence of int
        Observed counts, length N.
    expected : sequence of float
        Expected counts or population sizes, length N.
    row_ptr, col_idx : sequence of int
        CSR adjacency arrays.
    membership : sequence of int
        Binary membership vector.
    model : str, default "POISSON"
        ``'POISSON'`` or ``'BINOMIAL'``.

    Returns
    -------
    float
        Moran's I statistic, or ``nan`` when undefined.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    idx_in = np.where(mem == 1)[0]
    if idx_in.size == 0:
        return float("nan")

    obs, exp, model_u = _validate_observed_expected_lengths(observed, expected, len(mem), model)
    resid = np.zeros_like(obs, dtype=float)
    if model_u == "BINOMIAL":
        p_glob = float(obs.sum()) / float(exp.sum()) if float(exp.sum()) > 0 else 0.0
        p_glob = max(0.0, min(1.0, p_glob))
        mu = exp * p_glob
        var = exp * p_glob * (1.0 - p_glob)
        mask = var > 0
        resid[mask] = (obs[mask] - mu[mask]) / np.sqrt(var[mask])
    else:
        mask = exp > 0
        resid[mask] = (obs[mask] - exp[mask]) / np.sqrt(exp[mask])

    resid_in = resid[idx_in]
    xbar = float(np.mean(resid_in)) if resid_in.size > 0 else 0.0
    numerator = 0.0
    weights = 0.0
    for u in idx_in:
        start, end = row_ptr[u], row_ptr[u + 1]
        for pos in range(start, end):
            v = col_idx[pos]
            if mem[v] != 1:
                continue
            numerator += (resid[u] - xbar) * (resid[v] - xbar)
            weights += 1.0

    denominator = float(np.sum((resid_in - xbar) ** 2))
    n_nodes = float(idx_in.size)
    if denominator <= 0 or weights <= 0 or n_nodes <= 1:
        return float("nan")
    return (n_nodes / weights) * (numerator / denominator)


def conductance(row_ptr: Sequence[int], col_idx: Sequence[int], membership: Sequence[int]) -> float:
    """Compute graph conductance for a binary partition.

    Parameters
    ----------
    row_ptr, col_idx : sequence of int
        CSR adjacency arrays.
    membership : sequence of int
        Binary membership vector.

    Returns
    -------
    float
        Conductance (cut / min(vol_cluster, vol_complement)), or ``inf`` when undefined:
        an edge-free graph, or a side of the partition with no incident edges, has zero
        volume and the ratio has no value. ``inf`` is returned rather than raised so a
        caller ranking many partitions can filter it.
    """
    row_ptr, col_idx, mem = _validate_graph_membership(row_ptr, col_idx, membership)
    deg = np.diff(np.asarray(row_ptr, dtype=int))
    vol_c = int(deg[mem == 1].sum())
    vol_o = int(deg[mem == 0].sum())
    cut = 0
    for i in range(len(row_ptr) - 1):
        start, end = row_ptr[i], row_ptr[i + 1]
        for pos in range(start, end):
            j = col_idx[pos]
            if i < j and mem[i] != mem[j]:
                cut += 1
    denom = min(vol_c, vol_o)
    if denom <= 0:
        return float("inf")
    return float(cut) / float(denom)
