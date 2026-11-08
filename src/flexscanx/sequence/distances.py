"""Sequence distance metrics.

Copyright (c) 2025 flexscanx contributors
Portions adapted from traminex (MIT License)
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .._utils import warn_at_caller


def seq_dist_hamming(
    sequences: np.ndarray,
    weights: np.ndarray | None = None,
    norm: bool = True,
    condensed: bool = False,
) -> np.ndarray:
    """Compute pairwise Hamming distance between sequences.

    The Hamming distance counts the number of positions at which two sequences differ.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix where each row is a sequence and each column is a time point.
        Values typically encode states: 0 (none), 1 (HOT), -1 (COLD), etc.
    weights : ndarray of shape (T,), optional
        Per-position weights for distance computation.  If None, all positions are
        weighted equally.  Must be non-negative and sum to a positive value.
    norm : bool, default True
        If True, normalize distance by sequence length (or sum of weights).
        Normalised distance is in [0, 1] where 0 = identical, 1 = differ everywhere.
    condensed : bool, default False
        If True, return a 1-D condensed distance vector (upper triangle, row-major)
        compatible with ``scipy.spatial.distance.squareform``.  This halves memory
        usage compared to the full N×N matrix and is preferable for large N.

    Returns
    -------
    distances : ndarray of shape (N, N) or shape (N*(N-1)//2,)
        Symmetric distance matrix (or condensed upper-triangle vector when
        ``condensed=True``).  Diagonal entries are zero.

    Warns
    -----
    UserWarning
        Emitted when the full N×N matrix would exceed 500 MB. Pass
        ``condensed=True`` to use half the memory.

    Examples
    --------
    >>> seqs = np.array([
    ...     [0, 1, 1, 0],   # seq 0
    ...     [0, 1, 0, 0],   # seq 1: differs from seq 0 at position 2
    ...     [1, 1, 1, 1]    # seq 2: differs from seq 0 at positions 0, 3
    ... ])
    >>> dist = seq_dist_hamming(seqs, norm=True)
    >>> dist[0, 1]   # 1 mismatch / 4 positions
    0.25
    >>> dist[0, 2]   # 2 mismatches / 4 positions
    0.5
    >>> # Memory-efficient condensed form
    >>> cdist = seq_dist_hamming(seqs, condensed=True)
    >>> cdist.shape
    (3,)

    Notes
    -----
    - Time complexity: O(N² × T).
    - Full-matrix memory: N² × 8 bytes.  For N=10 000 this is ~800 MB; use
      ``condensed=True`` (≈400 MB) or subsample *sequences* first.

    Ported from traminex.distances.python_kernels.ham_distance()
    """

    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    N, T = sequences.shape

    if T == 0:
        raise ValueError("sequences must have at least one time position (T > 0)")

    # Factorization preserves arbitrary hashable categorical labels instead of
    # silently narrowing them to int32.  pandas uses one sentinel code for all
    # missing values, so missing-at-the-same-position compares equal and the
    # distance diagonal remains zero.
    try:
        codes, _ = pd.factorize(sequences.ravel(), sort=False)
    except TypeError as exc:
        raise ValueError("sequence states must be hashable categorical values") from exc
    encoded = codes.reshape(sequences.shape)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be a finite 1D array")
        if weights.shape[0] != T:
            raise ValueError(f"weights must have length {T}, got {weights.shape[0]}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if weights.sum() <= 0:
            raise ValueError("sum of weights must be positive")
        w = weights
    else:
        w = np.ones(T, dtype=np.float64)

    w_sum = float(w.sum()) if norm else 1.0

    # Warn when the N×N output matrix would exceed 500 MB.
    full_mb = N * N * 8 / 1_000_000
    if full_mb > 500 and not condensed:
        warn_at_caller(
            f"seq_dist_hamming: N={N} requires a {full_mb:.0f} MB output matrix "
            f"({N}×{N}×8 bytes). Pass condensed=True to use ~{full_mb/2:.0f} MB "
            "instead, or subsample the sequence matrix first."
        )

    # Chunk over rows to avoid the N×N×T intermediate boolean array.
    CHUNK = 64
    if condensed:
        pairs = N * (N - 1) // 2
        out = np.empty(pairs, dtype=np.float64)
        ptr = 0
        for i in range(N - 1):
            # Compare row i against rows i+1..N-1 (upper triangle)
            mismatch = encoded[i, None, :] != encoded[i + 1 :, :]  # (N-1-i, T)
            d = np.tensordot(mismatch.astype(np.float64), w, axes=([1], [0]))
            if norm:
                d /= w_sum
            n_pairs = N - 1 - i
            out[ptr : ptr + n_pairs] = d
            ptr += n_pairs
        return out

    D = np.zeros((N, N), dtype=np.float64)
    for i_start in range(0, N, CHUNK):
        i_end = min(i_start + CHUNK, N)
        mismatch = encoded[i_start:i_end, None, :] != encoded[None, :, :]
        dist = np.tensordot(mismatch.astype(np.float64), w, axes=([2], [0]))
        if norm:
            dist /= w_sum
        D[i_start:i_end, :] = dist
    return D


def seq_dist_matrix_info(D: np.ndarray) -> dict:
    """Compute summary statistics for a distance matrix.

    Parameters
    ----------
    D : ndarray of shape (N, N)
        Square distance matrix (typically from seq_dist_hamming).

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'n': number of sequences
        - 'mean': mean pairwise distance
        - 'std': standard deviation of pairwise distances
        - 'min': minimum non-zero distance
        - 'max': maximum distance
        - 'median': median pairwise distance

    Examples
    --------
    >>> seqs = np.random.randint(0, 3, size=(50, 12))
    >>> D = seq_dist_hamming(seqs)
    >>> info = seq_dist_matrix_info(D)
    >>> print(f"Mean distance: {info['mean']:.3f}")
    >>> print(f"Range: [{info['min']:.3f}, {info['max']:.3f}]")
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")
    if not np.all(np.isfinite(D)):
        raise ValueError("D must contain only finite distances")
    if np.any(D < 0.0):
        raise ValueError("D must contain only non-negative distances")
    if not np.allclose(D, D.T, rtol=0.0, atol=1e-12):
        raise ValueError("D must be symmetric")
    if not np.allclose(np.diag(D), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("D must have a zero diagonal")

    N = D.shape[0]

    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(N, k=1)
    distances = D[triu_idx]

    if len(distances) == 0:
        return {"n": N, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    return {
        "n": N,
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances[distances > 0])) if np.any(distances > 0) else 0.0,
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
    }
