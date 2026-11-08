"""Sequence analysis statistics.

Copyright (c) 2025 flexscanx contributors
Portions adapted from traminex (MIT License)
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_MISSING_STATE = object()


def _state_key(value: object) -> object:
    """Return a hash key that coalesces scalar missing-state representations."""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return _MISSING_STATE
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("sequence states must be hashable scalar values") from exc
    return value


def _row_counts(sequences: np.ndarray) -> tuple[list[tuple[object, ...]], np.ndarray]:
    """Count rows, using NumPy's ordering unless object/missing labels require fallback."""
    has_missing = bool(np.asarray(pd.isna(sequences), dtype=bool).any())
    if sequences.dtype != object and not has_missing:
        unique_rows, counts = np.unique(sequences, axis=0, return_counts=True)
        return [tuple(row.tolist()) for row in unique_rows], counts.astype(float)

    counts_by_key: dict[tuple[object, ...], int] = {}
    representatives: dict[tuple[object, ...], tuple[object, ...]] = {}
    for row in sequences:
        representative = tuple(row.tolist())
        key = tuple(_state_key(value) for value in representative)
        if key not in counts_by_key:
            representatives[key] = representative
            counts_by_key[key] = 0
        counts_by_key[key] += 1

    rows = [representatives[key] for key in counts_by_key]
    counts = np.asarray(list(counts_by_key.values()), dtype=float)
    return rows, counts


def _value_counts(values: np.ndarray) -> np.ndarray:
    """Count categorical values, coalescing mixed-type missing representations."""
    has_missing = bool(np.asarray(pd.isna(values), dtype=bool).any())
    if values.dtype != object and not has_missing:
        return np.unique(values, return_counts=True)[1]

    counts_by_key: dict[object, int] = {}
    for value in values:
        key = _state_key(value)
        counts_by_key[key] = counts_by_key.get(key, 0) + 1
    return np.asarray(list(counts_by_key.values()), dtype=np.intp)


def _unique_state_count(sequences: np.ndarray) -> int:
    """Count distinct categorical states without ordering incomparable labels."""
    return len(_value_counts(sequences.ravel()))


def seq_freq(sequences: np.ndarray, top_n: int | None = 10, normalize: bool = True) -> pd.DataFrame:
    """Count frequency of unique sequences.

    Identifies distinct trajectory patterns and their occurrence frequencies.
    Useful for summarizing the diversity of cluster membership patterns.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix where each row is a sequence.
    top_n : int, default=10
        Number of most frequent sequences to return. If None or <= 0,
        returns all unique sequences.
    normalize : bool, default=True
        If True, include 'proportion' and 'cumulative' columns with
        proportions and cumulative proportions.

    Returns
    -------
    freq_table : pd.DataFrame
        DataFrame with columns:
        - 'sequence': tuple representation of each unique sequence
        - 'count': number of occurrences
        - 'proportion': fraction of total sequences (if normalize=True)
        - 'cumulative': cumulative proportion (if normalize=True)
        Sorted by descending count.

    Examples
    --------
    >>> seqs = np.array([
    ...     [0, 1, 1],
    ...     [0, 1, 0],
    ...     [0, 1, 1],  # duplicate of row 0
    ...     [1, 0, 0],
    ...     [0, 1, 0]   # duplicate of row 1
    ... ])
    >>> freq = seq_freq(seqs, top_n=3)
    >>> print(freq)
       sequence  count  proportion  cumulative
    0  (0,1,1)      2        0.4         0.4
    1  (0,1,0)      2        0.4         0.8
    2  (1,0,0)      1        0.2         1.0

    >>> # Without normalization
    >>> freq = seq_freq(seqs, normalize=False)
    >>> print(freq.columns)
    Index(['sequence', 'count'], dtype='object')

    Notes
    -----
    - Sequences are compared element-wise; two sequences are considered identical
      if all corresponding positions have the same values.
    - For large datasets with many unique sequences, use top_n to limit output size.
    - The 'sequence' column contains tuples for easy inspection and comparison.

    Adapted from traminex.stats.seqtab()
    """
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")
    if top_n is not None and (
        isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer))
    ):
        raise ValueError(f"top_n must be an integer or None, got {top_n!r}")

    unique_seqs, counts = _row_counts(sequences)

    # traminex stringifies sequence labels before extracting unique rows, so
    # tied frequencies are ordered lexically by those labels rather than by
    # first occurrence or native numeric order.
    def _lexical_key(row: tuple[object, ...]) -> str:
        labels = ("<NA>" if _state_key(value) is _MISSING_STATE else str(value) for value in row)
        return "|".join(labels)

    sort_idx = sorted(
        range(len(unique_seqs)),
        key=lambda idx: (-float(counts[idx]), _lexical_key(unique_seqs[idx])),
    )
    unique_seqs = [unique_seqs[i] for i in sort_idx]
    counts = counts[sort_idx]

    # Normalize against the complete input before top_n truncation, matching
    # traminex.seqtab(...).head(top_n) and the documented "fraction of total
    # sequences" contract.
    total = counts.sum()

    # Limit to top_n if specified
    if top_n is not None and int(top_n) > 0:
        unique_seqs = unique_seqs[: int(top_n)]
        counts = counts[: int(top_n)]

    # Build DataFrame
    df_dict = {"sequence": unique_seqs, "count": counts}

    if normalize:
        proportions = counts / total if total > 0 else np.zeros_like(counts)
        cumulative = np.cumsum(proportions)
        df_dict["proportion"] = proportions
        df_dict["cumulative"] = cumulative

    return pd.DataFrame(df_dict)


def seq_trans(sequences: np.ndarray, lag: int = 1, normalize: bool = True) -> pd.DataFrame:
    """Compute state transition matrix.

    Aggregates state-to-state transitions across all sequences to estimate
    transition probabilities or counts.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix where each row is a sequence. States are integer-coded.
    lag : int, default=1
        Time lag for transitions. lag=1 means immediate next-position transitions.
        lag=2 means transitions from position t to position t+2, etc.
    normalize : bool, default=True
        If True, rows are normalized to probabilities (sum to 1).
        If False, returns raw transition counts.

    Returns
    -------
    trans_matrix : pd.DataFrame
        Transition matrix as DataFrame where:
        - index: 'from' states
        - columns: 'to' states
        - values: transition probabilities (if normalize=True) or counts
        Rows and columns labeled with state values.

    Examples
    --------
    >>> # Simple 3-state system: 0 (none), 1 (HOT), -1 (COLD)
    >>> seqs = np.array([
    ...     [0, 1, 1, 0],   # none→HOT→HOT→none
    ...     [0, 1, 0, 0],   # none→HOT→none→none
    ...     [1, 0, -1, -1]  # HOT→none→COLD→COLD
    ... ])
    >>> trans = seq_trans(seqs, lag=1, normalize=True)
    >>> print(trans)
    to        -1    0    1
    from
    -1       1.0  0.0  0.0
     0       0.0  0.5  0.5
     1       0.0  0.5  0.5

    >>> # Raw counts
    >>> trans_counts = seq_trans(seqs, lag=1, normalize=False)
    >>> print(trans_counts.loc[0, 1])  # 0→1 transitions: 2 occurrences
    2.0

    >>> # Lag-2 transitions (skip one position)
    >>> trans_lag2 = seq_trans(seqs, lag=2)
    >>> # Example: state at t=0 to state at t=2

    Notes
    -----
    - Transition probabilities help identify common patterns:
      - High P(HOT|HOT): clusters persist
      - High P(none|HOT): clusters are transient
      - Low P(COLD|HOT): rare direct HOT→COLD transitions
    - For lag > 1, transitions skip intermediate states.
    - Missing transitions (never observed) have probability/count of 0.

    Adapted from traminex transition utilities. For ``lag=1``, matrix rates and
    counts are parity-tested against ``traminex.seqtrate(count=False)`` and
    ``traminex.seqtrate(count=True)`` respectively. ``traminex.seqtransn`` is a
    separate per-sequence statistic with keyword ``norm``.
    """
    raw_sequences = np.asarray(sequences)
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
        raise ValueError(f"lag must be an integer, got {lag!r}")

    if np.issubdtype(raw_sequences.dtype, np.integer):
        if np.issubdtype(raw_sequences.dtype, np.unsignedinteger):
            max_value = raw_sequences.max(initial=0)
            if max_value > np.iinfo(np.int64).max:
                raise ValueError("integer-coded states must fit in signed 64-bit range")
        sequences = raw_sequences.astype(np.int64, copy=False)
    elif np.issubdtype(raw_sequences.dtype, np.floating):
        if not np.all(np.isfinite(raw_sequences)) or not np.all(
            raw_sequences == np.rint(raw_sequences)
        ):
            raise ValueError("sequences must contain finite integer-coded states")
        int64_info = np.iinfo(np.int64)
        if np.any(raw_sequences < int64_info.min) or np.any(raw_sequences > int64_info.max):
            raise ValueError("integer-coded states must fit in signed 64-bit range")
        sequences = raw_sequences.astype(np.int64)
    else:
        raise ValueError("sequences must contain finite integer-coded states")

    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    _, T = sequences.shape

    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    if T < lag + 1:
        # Not enough time points for any transitions with this lag
        states = np.unique(sequences)
        K = len(states)
        empty = pd.DataFrame(0.0, index=states, columns=states)
        empty.index.name = "from"
        empty.columns.name = "to"
        return empty

    # Extract all transitions with given lag
    from_states = sequences[:, : T - lag].ravel()  # states at time t
    to_states = sequences[:, lag:].ravel()  # states at time t+lag

    # Get all unique states
    all_states = np.unique(sequences)
    K = len(all_states)

    # Count transitions. Search in the sorted unique-state array so memory is
    # proportional to the number of states, not the numeric span of their labels.
    trans_matrix = np.zeros((K, K), dtype=np.float64)
    idx_from = np.searchsorted(all_states, from_states)
    idx_to = np.searchsorted(all_states, to_states)
    np.add.at(trans_matrix, (idx_from, idx_to), 1.0)

    # Normalize if requested
    if normalize:
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        trans_matrix = np.divide(
            trans_matrix, row_sums, out=np.zeros_like(trans_matrix), where=row_sums > 0
        )

    # Convert to DataFrame with labeled axes
    df = pd.DataFrame(trans_matrix, index=all_states, columns=all_states)
    df.index.name = "from"
    df.columns.name = "to"

    return df


def seq_entropy(
    sequences: np.ndarray, by_time: bool = False, base: float = 2.0, normalize: bool = True
) -> float | np.ndarray:
    """Compute Shannon entropy of sequence distribution.

    Entropy measures the diversity or unpredictability of cluster membership
    patterns. High entropy indicates many different patterns; low entropy
    indicates dominant patterns.

    Parameters
    ----------
    normalize : bool, default=True
        Only affects the cross-sectional (``by_time=True``) entropy. When True
        (the default), the per-position entropy is divided by ``log(K)``, where
        ``K`` is the number of distinct states across the whole matrix (the
        alphabet size). This yields a value in [0, 1] that is independent of
        ``base`` and matches ``traminex.seqstatd(include_missing=True)``'s
        normalized ``Entropy`` row (the source of truth for this statistic).
        Missing categorical values are therefore counted as an alphabet state.
        When False, the unnormalized entropy in ``base`` units is returned. The
        overall (``by_time=False``) diversity entropy is a flexscanx-specific
        measure with no traminex counterpart and is unaffected by this flag.
    sequences : ndarray of shape (N, T)
        Sequence matrix where each row is a sequence.
    by_time : bool, default=False
        If False, return overall sequence diversity (scalar).
        If True, return per-position state diversity (array of length T).
    base : float, default=2.0
        Logarithm base for entropy computation:
        - 2.0: entropy in bits
        - np.e: entropy in nats
        - 10.0: entropy in dits

    Returns
    -------
    entropy : float or ndarray
        If by_time=False:
            Scalar entropy of the entire sequence distribution.
            Range: [0, log_base(N)] where N is number of sequences.
            - 0 = all sequences identical
            - log_base(N) = all sequences unique (maximum diversity)
        If by_time=True:
            Array of shape (T,) with entropy at each time position.
            Measures state diversity at each time point across sequences.
            Range: [0, 1] when normalize=True (matches traminex.seqstatd's
            Entropy row); [0, log_base(K)] when normalize=False, where K is
            the alphabet size (distinct states across the whole matrix).

    Examples
    --------
    >>> # Overall sequence diversity
    >>> seqs = np.array([
    ...     [0, 1, 1],
    ...     [0, 1, 0],
    ...     [0, 1, 1],  # duplicate
    ...     [1, 0, 0]
    ... ])
    >>> ent = seq_entropy(seqs, by_time=False)
    >>> # 3 unique patterns out of 4 sequences → moderate entropy

    >>> # Per-position state diversity
    >>> ent_time = seq_entropy(seqs, by_time=True)
    >>> print(ent_time)  # length 3 (T=3)
    >>> # Position 0: mix of 0 and 1 → higher entropy
    >>> # Position 1: mostly 1 → lower entropy
    >>> # Position 2: mix of 0 and 1 → higher entropy

    >>> # Extreme cases
    >>> identical = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    >>> seq_entropy(identical)  # all identical → entropy ≈ 0
    0.0

    >>> unique = np.array([[0, 0], [1, 1], [2, 2]])
    >>> seq_entropy(unique)  # all unique → maximum entropy
    1.5849625007211563  # log2(3)

    Notes
    -----
    - Normalized entropy = H / log(K) scales entropy to [0, 1] range,
      useful for comparing across different numbers of states.
    - For cluster membership analysis:
      - High overall entropy: diverse membership trajectories
      - High position entropy: many states represented at that time
      - Low position entropy: dominant state at that time (e.g., most areas not in clusters)
    - Entropy is computed with small epsilon to avoid log(0).

    Deliberate in-sync copy of
    traminex.seqstatd(include_missing=True)'s normalized Entropy row for the
    cross-sectional (by_time=True) case; verified by
    tests/test_traminex_parity_b3_entropy.py.
    """
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    N, T = sequences.shape

    try:
        base_f = float(base)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"base must be finite, got {base!r}") from exc
    if not np.isfinite(base_f):
        raise ValueError(f"base must be finite, got {base!r}")
    if base_f <= 0 or base_f == 1:
        raise ValueError(f"base must be > 0 and != 1, got {base}")
    if N == 0:
        raise ValueError("sequences must contain at least one sequence")

    if by_time:
        # Per-position (cross-sectional) state entropy. With normalize=True this
        # matches traminex.seqstatd's Entropy row: the natural-log Shannon
        # entropy divided by log(K), where K is the number of distinct states
        # across the whole matrix (the alphabet size). The normalized value lies
        # in [0, 1] and is independent of `base`. With normalize=False the
        # unnormalized entropy in `base` units is returned.
        entropy_per_pos = np.zeros(T, dtype=np.float64)
        log_base = np.log(base_f)
        K_full = _unique_state_count(sequences)

        for t in range(T):
            counts = _value_counts(sequences[:, t])
            probs = counts.astype(np.float64) / N
            H_nat = -np.sum(probs * np.log(probs))  # natural-log entropy (nats)
            if normalize:
                entropy_per_pos[t] = H_nat / np.log(K_full) if K_full > 1 else 0.0
            else:
                entropy_per_pos[t] = H_nat / log_base

        return entropy_per_pos

    else:
        # Overall sequence diversity entropy
        _, counts = _row_counts(sequences)
        probs = counts / N

        # Shannon entropy; probs are strictly >0 from unique counts.
        H = -np.sum(probs * np.log(probs) / np.log(base_f))
        return float(H)
