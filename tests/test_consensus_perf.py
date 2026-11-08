import inspect

import numpy as np
import pytest

from flexscanx import consensus_sweep


def test_consensus_sweep_smoke():
    assert "seed" not in inspect.signature(consensus_sweep).parameters
    # Small 4x4 grid
    coords = np.column_stack([np.repeat(np.arange(4), 4), np.tile(np.arange(4), 4)])
    # N = 16  # not used
    # Build adjacency CSR (4-neighbor)
    row_ptr = [0]
    col_idx = []
    for r in range(4):
        for c in range(4):
            # i = r * 4 + c  # not used
            nbrs = []
            if r > 0:
                nbrs.append((r - 1) * 4 + c)
            if r < 3:
                nbrs.append((r + 1) * 4 + c)
            if c > 0:
                nbrs.append(r * 4 + c - 1)
            if c < 3:
                nbrs.append(r * 4 + c + 1)
            nbrs.sort()
            col_idx.extend(nbrs)
            row_ptr.append(len(col_idx))
    observed = [5, 5, 5, 5, 6, 12, 11, 6, 5, 5, 5, 5, 5, 5, 5, 5]
    expected = [5.0] * 16
    # Run consensus sweep
    res = consensus_sweep(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        clustersize=5,
        grid={"K": [5, 8], "ralpha": [0.2, 0.25]},
        model="POISSON",
        threshold=0.5,
    )
    assert "freq" in res and "consensus" in res
    assert len(res["freq"]) == 16
    assert res["runs"] == 4  # 2 K * 2 ralpha * 1 radius


def test_consensus_sweep_rejects_removed_seed_parameter():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    row_ptr = [0, 1, 2]
    col_idx = [1, 0]

    with pytest.raises(TypeError, match="seed"):
        consensus_sweep(
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=[1, 2],
            expected=[1.0, 1.0],
            clustersize=2,
            grid={"K": [2], "ralpha": [0.2]},
            seed=7,
        )
