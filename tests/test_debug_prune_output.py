from __future__ import annotations

import numpy as np

import flexscanx as flx


def _line_graph(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    coords = np.stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)], axis=1)
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        nbrs: list[int] = []
        if i - 1 >= 0:
            nbrs.append(i - 1)
        if i + 1 < n:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def test_debug_prune_emits_flexible_summary_to_stderr(capfd) -> None:
    coords, adjacency = _line_graph(5)
    flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[12, 10, 1, 1, 1],
        expected=[5.0] * 5,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        K=3,
        simcount=0,
        debug_prune=True,
    )

    captured = capfd.readouterr()
    assert "[flexscanx][debug_prune]" in captured.err
    assert "scanner=flexible" in captured.err
    assert "model=poisson" in captured.err
    assert "tail=hot" in captured.err
    assert "dfs_nodes_expanded=" in captured.err
    assert "bound_prunes=" in captured.err


def test_debug_prune_emits_circular_summary_to_stderr(capfd) -> None:
    coords, adjacency = _line_graph(5)
    flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[12, 10, 1, 1, 1],
        expected=[5.0] * 5,
        model="POISSON",
        scanmethod="CIRCULAR",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        K=3,
        simcount=0,
        debug_prune=True,
    )

    captured = capfd.readouterr()
    assert "[flexscanx][debug_prune]" in captured.err
    assert "scanner=circular" in captured.err
    assert "model=poisson" in captured.err
    assert "tail=hot" in captured.err
    assert "best_updates=" in captured.err
