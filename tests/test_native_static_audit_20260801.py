import warnings

import numpy as np
import pytest

import flexscanx._core as core
from flexscanx import scan


def _complete_graph_csr(n: int) -> tuple[list[int], list[int]]:
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        col_idx.extend(j for j in range(n) if j != i)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


@pytest.mark.parametrize(
    ("coords", "row_ptr", "col_idx", "k", "radius", "message"),
    [
        (np.empty((0, 2)), [0], [], 1, np.inf, "at least one point"),
        (np.array([[0.0, 0.0]]), [0, 0], [], 0, np.inf, "K must be in"),
        (np.array([[0.0, 0.0]]), [0, 0], [], 2, np.inf, "K must be in"),
        (np.array([[np.nan, 0.0]]), [0, 0], [], 1, np.inf, "coordinates must be finite"),
        (np.array([[0.0, np.inf]]), [0, 0], [], 1, np.inf, "coordinates must be finite"),
        (np.array([[0.0, 0.0]]), [0, 0], [], 1, np.nan, "radius must be"),
        (np.array([[0.0, 0.0]]), [0, 0], [], 1, -1.0, "radius must be"),
        (np.array([[0.0, 0.0]]), [0, 0], [], 1, -np.inf, "radius must be"),
    ],
)
def test_direct_buildstatic_rejects_invalid_geometry_boundaries(
    coords, row_ptr, col_idx, k, radius, message
):
    with pytest.raises(ValueError, match=message):
        core.BuildStatic(coords, row_ptr, col_idx, k, radius)


def test_buildstatic_propagates_guardrail_warning_as_error():
    n = 5_001
    coords = np.zeros((n, 2), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="disabled due to guardrail"):
            core.BuildStatic(coords, [0] * (n + 1), [], 1, np.inf, "K_REFILL", False)


def test_large_finite_coordinates_respect_a_large_finite_radius():
    coords = np.array([[0.0, 0.0], [1.0e200, 0.0], [2.0e200, 0.0]])
    row_ptr, col_idx = _complete_graph_csr(len(coords))

    with pytest.warns(UserWarning, match="excludes nearly all neighbors"):
        result = scan(
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=[10, 10, 0],
            expected=[5.0, 5.0, 5.0],
            model="POISSON",
            scanmethod="CIRCULAR",
            stattype="ORIGINAL",
            clustertype="HOT",
            K=3,
            clustersize=2,
            clusterradius=1.0e199,
            simcount=0,
            knn_method="geometry",
            knn_mode="K_FIXED",
            n_jobs_centers=1,
        )

    assert result["area"] == [1]


@pytest.mark.parametrize(
    ("scanmethod", "observed"),
    [
        ("CIRCULAR", [0, 10, 0, 0, 0, 0]),
        ("FLEXIBLE", [2, 6, 0, 11, 5, 0]),
    ],
)
def test_k_refill_keeps_duplicate_coordinate_center_first(scanmethod, observed):
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    row_ptr, col_idx = _complete_graph_csr(len(coords))
    common = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=np.full(len(coords), 3.0),
        model="POISSON",
        scanmethod=scanmethod,
        stattype="ORIGINAL",
        clustertype="HOT",
        K=2,
        clustersize=2,
        simcount=0,
        knn_method="geometry",
        knn_refill_force=True,
        n_jobs_centers=1,
    )

    fixed = scan(knn_mode="K_FIXED", **common)
    refilled = scan(knn_mode="K_REFILL", **common)

    assert refilled["area"] == fixed["area"]
    assert refilled["stats"] == pytest.approx(fixed["stats"], rel=0.0, abs=1e-12)
