"""Sequence visualization functions.

Copyright (c) 2025 flexscanx contributors
Portions adapted from traminex (MIT License)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flexscanx._utils import validate_integer_parameter

from .analysis import seq_freq

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_MISSING_PLOT_STATE = object()


def _categorical_state_codes(sequences: np.ndarray) -> tuple[np.ndarray, list[object]]:
    """Factor categorical states without sorting incomparable labels."""
    state_to_code: dict[object, int] = {}
    states: list[object] = []
    codes = np.empty(sequences.size, dtype=np.intp)
    for idx, value in enumerate(sequences.ravel()):
        missing = pd.isna(value)
        key: object
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            key = _MISSING_PLOT_STATE
        else:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError("sequence states must be hashable scalar values") from exc
            key = value
        if key not in state_to_code:
            state_to_code[key] = len(states)
            states.append(key)
        codes[idx] = state_to_code[key]
    order = sorted(
        range(len(states)),
        key=lambda state_idx: (
            "<NA>" if states[state_idx] is _MISSING_PLOT_STATE else str(states[state_idx])
        ),
    )
    old_to_new = np.empty(len(states), dtype=np.intp)
    for new_code, old_code in enumerate(order):
        old_to_new[old_code] = new_code
    sorted_states = [states[old_code] for old_code in order]
    return old_to_new[codes].reshape(sequences.shape), sorted_states


def _state_label(state: object) -> str:
    if state is _MISSING_PLOT_STATE:
        return "Missing"
    if state == -1:
        return "COLD"
    if state == 0:
        return "None"
    if state == 1:
        return "HOT"
    return f"State {state}"


def seq_iplot(
    sequences: np.ndarray,
    area_ids: list[int] | None = None,
    time_labels: list[str] | None = None,
    colors: dict[object, str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    title: str = "Cluster Membership Trajectories",
    show_grid: bool = True,
    max_display: int = 50,
) -> tuple[Figure, Axes]:
    """Plot individual sequences as horizontal colored bars (index plot).

    Each row represents one area/sequence, each column represents a time point,
    and colors encode cluster membership states.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix where rows are areas and columns are time points.
        Values encode states: 0 (none), 1 (HOT), -1 (COLD), etc.
    area_ids : list of int, optional
        Area IDs corresponding to rows of sequences. Used for y-axis labels.
        If None, uses row indices.
    time_labels : list of str, optional
        Labels for x-axis (time points). If None, uses numeric indices.
    colors : dict of int -> str, optional
        Custom color mapping for states. Keys are state values, values are
        matplotlib color specifications. Default:
        {-1: 'blue', 0: 'white', 1: 'red'}
    figsize : tuple of (width, height), default=(12, 8)
        Figure size in inches.
    title : str, default='Cluster Membership Trajectories'
        Plot title.
    show_grid : bool, default=True
        Whether to show grid lines between cells.
    max_display : int, default=50
        Maximum number of sequences to display. If N > max_display,
        only first max_display sequences are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Examples
    --------
    >>> import flexscanx as flx
    >>> from flexscanx.sequence import extract_sequences, seq_iplot
    >>>
    >>> # After running scan_series
    >>> result = flx.scan_series(...)
    >>> seqs, ids = extract_sequences(result)
    >>>
    >>> # Plot first 30 areas
    >>> fig, ax = seq_iplot(seqs[:30], area_ids=ids[:30],
    ...                      time_labels=['Jan', 'Feb', 'Mar', ..., 'Dec'])
    >>> plt.tight_layout()
    >>> plt.savefig('trajectories.png', dpi=150)

    >>> # Custom colors
    >>> colors = {-1: '#0000FF', 0: '#EEEEEE', 1: '#FF0000'}
    >>> fig, ax = seq_iplot(seqs, colors=colors)

    Notes
    -----
    - Useful for visualizing temporal patterns and identifying areas with
      similar trajectories.
    - For large N, consider filtering to specific areas of interest or using
      max_display to limit output.
    - Horizontal lines separate individual sequences for clarity.

    Adapted from traminex.visualization.plots.sequence_index_plot()
    """
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    N, T = sequences.shape
    if N == 0:
        raise ValueError("sequences must contain at least one sequence")
    if T == 0:
        raise ValueError("sequences must contain at least one time point")
    max_display = validate_integer_parameter(max_display, "max_display", minimum=1)

    # Limit display if too many sequences
    if N > max_display:
        sequences = sequences[:max_display]
        if area_ids is not None:
            area_ids = area_ids[:max_display]
        N = max_display

    if time_labels is not None and len(time_labels) != T:
        raise ValueError(f"time_labels length ({len(time_labels)}) must match T ({T})")
    if area_ids is not None and len(area_ids) < N:
        raise ValueError(f"area_ids length ({len(area_ids)}) must be >= N ({N})")

    # Default colors for 3-state system (COLD, none, HOT)
    if colors is None:
        colors = {
            -1: "#4575b4",  # blue for COLD
            0: "#f7f7f7",  # light gray for none
            1: "#d73027",  # red for HOT
        }

    state_codes, unique_states = _categorical_state_codes(sequences)

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    # Create RGB array for imshow
    rgb_array = np.zeros((N, T, 3))
    for state_code, state in enumerate(unique_states):
        mask = state_codes == state_code
        color_str = colors.get(state, "#808080")  # gray as fallback
        # Convert hex or name to RGB
        rgb = to_rgb(color_str)
        rgb_array[mask] = rgb

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Display as image
    ax.imshow(rgb_array, aspect="auto", interpolation="nearest")

    # Configure axes
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Sequence / Area", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # X-axis: time labels
    if time_labels is not None:
        ax.set_xticks(range(T))
        ax.set_xticklabels(time_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(range(T))
        ax.set_xticklabels([str(t) for t in range(T)])

    # Y-axis: area IDs or indices
    if area_ids is not None:
        # Show subset of labels to avoid crowding
        step = max(1, N // 20)  # show ~20 labels
        ytick_pos = list(range(0, N, step))
        ytick_labels = [str(area_ids[i]) for i in ytick_pos]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels)
    else:
        step = max(1, N // 20)
        ytick_pos = list(range(0, N, step))
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([str(i) for i in ytick_pos])

    # Grid
    if show_grid:
        ax.set_xticks(np.arange(T) - 0.5, minor=True)
        ax.set_yticks(np.arange(N) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.5)

    # Add legend
    legend_elements = []
    for state in unique_states:
        color = colors.get(state, "#808080")
        legend_elements.append(Patch(facecolor=color, label=_state_label(state)))

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    return fig, ax


def seq_dplot(
    sequences: np.ndarray,
    time_labels: list[str] | None = None,
    colors: dict[object, str] | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Cluster Membership Distribution Over Time",
    show_entropy: bool = False,
) -> tuple[Figure, Axes]:
    """Plot state distribution over time (distribution plot).

    Shows proportion of areas in each state at each time point as stacked
    area chart or bar chart.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix.
    time_labels : list of str, optional
        Labels for x-axis (time points).
    colors : dict of int -> str, optional
        Custom color mapping for states.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str
        Plot title.
    show_entropy : bool, default=False
        If True, overlay entropy line on secondary y-axis.

    Returns
    -------
    fig : Figure
    ax : Axes

    Examples
    --------
    >>> fig, ax = seq_dplot(seqs, time_labels=months)
    >>> # Shows how proportion of areas in HOT/COLD/none changes over time

    Notes
    -----
    - Useful for identifying time periods with high cluster activity.
    - Stacked bars sum to 1.0 at each time point.
    - Missing categorical values are counted as a state, matching
      ``traminex.seqstatd(include_missing=True)``. flexscanx sorts the displayed
      missing state with the ``"<NA>"`` label rather than using traminex's
      missing-state column order.

    Adapted from traminex.visualization.plots.state_distribution_plot()
    """
    import matplotlib.pyplot as plt

    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2D array, got shape {sequences.shape}")

    N, T = sequences.shape

    # Default colors
    if colors is None:
        colors = {-1: "#4575b4", 0: "#f7f7f7", 1: "#d73027"}

    # Compute state proportions at each time point. Factorization coalesces all
    # scalar missing values and supports mixed, incomparable categorical labels.
    state_codes, unique_states = _categorical_state_codes(sequences)
    K = len(unique_states)

    freq = np.zeros((T, K))
    for t in range(T):
        counts = np.bincount(state_codes[:, t], minlength=K)
        freq[t, :] = counts / N

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    xs = np.arange(T)
    bottom = np.zeros(T)

    for i, state in enumerate(unique_states):
        color = colors.get(state, "#808080")

        ax.bar(
            xs,
            freq[:, i],
            bottom=bottom,
            color=color,
            label=_state_label(state),
            width=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += freq[:, i]

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Proportion of Areas", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)

    if time_labels is not None:
        if len(time_labels) != T:
            raise ValueError("time_labels length must match T")
        ax.set_xticks(xs)
        ax.set_xticklabels(time_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(xs)

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Optional entropy overlay
    if show_entropy:
        from .analysis import seq_entropy

        ent = seq_entropy(sequences, by_time=True)
        ax2 = ax.twinx()
        ax2.plot(xs, ent, "k-", linewidth=2, label="Entropy", marker="o")
        ax2.set_ylabel("Normalized entropy", fontsize=12)
        ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    return fig, ax


def seq_fplot(
    sequences: np.ndarray,
    top_n: int = 10,
    figsize: tuple[int, int] = (12, 6),
    title: str = "Most Frequent Sequence Patterns",
) -> tuple[Figure, Axes]:
    """Plot most frequent sequences as horizontal bars (frequency plot).

    Displays the top N most common trajectory patterns and their frequencies.

    Parameters
    ----------
    sequences : ndarray of shape (N, T)
        Sequence matrix.
    top_n : int, default=10
        Number of most frequent patterns to display.
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    fig : Figure
    ax : Axes

    Examples
    --------
    >>> fig, ax = seq_fplot(seqs, top_n=15)
    >>> # Shows bar chart of 15 most common membership patterns

    Notes
    -----
    - Useful for identifying dominant trajectory types.
    - Sequences displayed as string representations.
    """
    import matplotlib.pyplot as plt

    freq_table = seq_freq(sequences, top_n=top_n, normalize=True)

    if freq_table.empty:
        raise ValueError("No sequences to plot")

    fig, ax = plt.subplots(figsize=figsize)

    # Convert sequence tuples to readable strings
    labels = [_format_sequence(seq) for seq in freq_table["sequence"]]
    counts = freq_table["count"].values
    proportions = freq_table.get("proportion", counts / counts.sum()).values

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, proportions, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # most frequent at top
    ax.set_xlabel("Proportion", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Annotate with counts
    for i, (prop, count) in enumerate(zip(proportions, counts)):
        ax.text(prop + 0.01, i, f"n={int(count)}", va="center", fontsize=8)

    plt.tight_layout()
    return fig, ax


def _format_sequence(seq_tuple: tuple, max_len: int = 20) -> str:
    """Format sequence tuple as readable string."""
    # Convert states to symbols for readability
    symbols = []
    for val in seq_tuple:
        missing = pd.isna(val)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            symbols.append("?")
        elif val == -1:
            symbols.append("C")  # COLD
        elif val == 0:
            symbols.append("·")  # none (middle dot)
        elif val == 1:
            symbols.append("H")  # HOT
        else:
            symbols.append(str(val))

    s = "".join(symbols)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
