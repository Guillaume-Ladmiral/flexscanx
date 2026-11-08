import numpy as np
import pytest

import flexscanx as p


def test_mc_progress_callback_invoked():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr = [0, 1, 3, 4]
    col_idx = [1, 0, 2, 1]
    observed = [10, 5, 5]
    expected = [5.0, 5.0, 5.0]

    seen = []

    def cb(status: dict):
        # minimal callback that records progress
        done = int(status.get("done", 0))
        simcount = int(status.get("simcount", 0))
        if done >= 0 and simcount >= 0:
            seen.append(done)

    res = p.scan(
        coords=x,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        clustersize=2,
        ralpha=0.2,
        K=3,
        simcount=50,
        mc_early_stop=False,
        mc_progress=cb,
        mc_progress_every=10,
        seed=123,
    )
    assert isinstance(res, dict)
    # We should have seen progress around every 10 sims (5 reports plus the final)
    assert len(seen) >= 4
    assert max(seen) <= 50


def test_mc_progress_callback_exception_propagates_to_caller():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])

    def fail_callback(_status):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        p.scan(
            coords=x,
            adjacency=([0, 1, 2], [1, 0]),
            observed=[5, 1],
            expected=[2.0, 2.0],
            K=2,
            clustersize=1,
            simcount=2,
            seed=1,
            mc_progress=fail_callback,
            mc_progress_every=1,
        )
