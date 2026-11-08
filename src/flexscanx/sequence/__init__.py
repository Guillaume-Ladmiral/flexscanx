"""Sequence analysis module for temporal cluster patterns.

This module provides tools for analyzing temporal evolution of spatial clusters
detected by scan_series(). Key capabilities:

- **Sequence extraction**: Convert scan_series results to membership sequences
- **Distance metrics**: Hamming distances for trajectory comparison
- **Analysis**: Frequency tables, transition matrices, entropy measures
- **Visualization**: Index plots, distribution plots, frequency plots

Typical workflow:
    >>> import flexscanx as flx
    >>> from flexscanx.sequence import extract_sequences, seq_dist_hamming, seq_iplot
    >>>
    >>> # 1. Run time-series scan
    >>> result = flx.scan_series(observed=obs_TN, expected=exp_TN, ...)
    >>>
    >>> # 2. Extract cluster membership sequences
    >>> sequences, area_ids = extract_sequences(result)
    >>> # sequences shape: (N_areas, T_timepoints)
    >>> # sequences[i,t] = 1 (HOT), -1 (COLD), or 0 (none)
    >>>
    >>> # 3. Compute pairwise trajectory distances
    >>> distances = seq_dist_hamming(sequences, norm=True)
    >>> # distances[i,j] = proportion of time points where areas i,j differ
    >>>
    >>> # 4. Visualize patterns (requires matplotlib: pip install matplotlib)
    >>> fig, ax = seq_iplot(sequences[:30], area_ids=area_ids[:30])
    >>> plt.savefig('trajectories.png')

Copyright (c) 2025 flexscanx contributors
Portions adapted from traminex (MIT License)
"""

from __future__ import annotations

# Sequence extraction
from ._utils import extract_sequences

# Analysis functions
from .analysis import (
    seq_entropy,
    seq_freq,
    seq_trans,
)

# Distance metrics
from .distances import (
    seq_dist_hamming,
    seq_dist_matrix_info,
)

# seq_dist_gower was removed in v0.0.1: it was an unexported thin wrapper that
# delegated 100 % to seq_dist_hamming without implementing true Gower semantics
# for mixed data types.  Use seq_dist_hamming directly.
# Visualization
from .plotting import (
    seq_dplot,
    seq_fplot,
    seq_iplot,
)

__all__ = [
    # Extraction
    "extract_sequences",
    # Distances
    "seq_dist_hamming",
    "seq_dist_matrix_info",
    # Analysis
    "seq_freq",
    "seq_trans",
    "seq_entropy",
    # Plotting
    "seq_iplot",
    "seq_dplot",
    "seq_fplot",
]
