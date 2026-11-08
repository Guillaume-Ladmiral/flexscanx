"""Graph-construction regressions of the 2026-09-07 run (U-5 / D3, FXA-164 / D7)."""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx as flx

# ---------------------------------------------------------------------------
# U-5 / D3: csr_from_dense enforces the documented 0/1 contract
# ---------------------------------------------------------------------------

_BINARY = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
_BINARY_CSR = ([0, 1, 3, 4], [1, 0, 2, 1])


def test_binary_matrices_still_convert_and_ignore_the_diagonal() -> None:
    assert flx.csr_from_dense(_BINARY) == _BINARY_CSR
    with_diagonal = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
    assert flx.csr_from_dense(with_diagonal) == _BINARY_CSR
    assert flx.csr_from_dense(np.array(_BINARY, dtype=bool)) == _BINARY_CSR
    assert flx.csr_from_dense(np.array(_BINARY, dtype=np.float64)) == _BINARY_CSR
    assert flx.csr_from_dense(np.zeros((3, 3), dtype=int)) == ([0, 0, 0, 0], [])


@pytest.mark.parametrize(
    "value",
    [2, 0.5, -1, 7, float("nan"), float("inf"), 1e-300, "1", "0", "true"],
)
def test_entries_outside_zero_one_are_rejected_naming_the_position(value: object) -> None:
    matrix = [[0, 1, 0], [1, 0, value], [0, value, 0]]
    with pytest.raises(ValueError, match=r"csr_from_dense.*\(1, 2\)") as excinfo:
        flx.csr_from_dense(matrix)
    assert "0 or 1" in str(excinfo.value) or "{0, 1}" in str(excinfo.value)


def test_weighted_but_symmetric_matrix_is_rejected_not_binarised() -> None:
    weighted = [[0, 2, 0], [2, 0, 3], [0, 3, 0]]
    with pytest.raises(ValueError, match="csr_from_dense"):
        flx.csr_from_dense(weighted)


def test_asymmetric_weights_are_rejected_as_values_before_symmetry() -> None:
    # bool(2) == bool(3) used to pass the symmetry check; both values are now invalid.
    matrix = [[0, 2, 0], [3, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        flx.csr_from_dense(matrix)


def test_string_zero_is_not_an_edge() -> None:
    with pytest.raises(ValueError, match="csr_from_dense"):
        flx.csr_from_dense([["0", "1"], ["1", "0"]])


def test_symmetry_is_still_checked_on_a_valid_binary_matrix() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        flx.csr_from_dense([[0, 1, 0], [0, 0, 1], [0, 1, 0]])


def test_diagonal_entries_outside_zero_one_are_still_rejected() -> None:
    # The diagonal is ignored for edges, but the contract is a 0/1 matrix.
    with pytest.raises(ValueError, match=r"\(1, 1\)"):
        flx.csr_from_dense([[0, 1, 0], [1, 5, 1], [0, 1, 0]])


# ---------------------------------------------------------------------------
# FXA-164 / D7: queen and rook implement the documented geometric predicates on
# every backend, so the presence of libpysal cannot change a scan result
# ---------------------------------------------------------------------------

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

CONTIGUITY_FIXTURES = {
    # A tall left polygon abutted by two right-hand polygons meeting inside its edge.
    "t_junction": [box(0, 0, 1, 2), box(1, 0, 2, 1), box(1, 1, 2, 2), box(2, 0, 3, 2)],
    # A tall polygon beside three stacked boxes: the middle box shares no vertex with it.
    "stacked_column": [box(0, 0, 1, 3), box(1, 0, 2, 1), box(1, 1, 2, 2), box(1, 2, 2, 3)],
    # A vertex-congruent 2x2 grid: queen adds the two diagonal pairs.
    "grid_2x2": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
}


def _oracle_edges(geoms, method: str) -> list[tuple[int, int]]:
    """The two docstring sentences, implemented literally with shapely."""
    edges = []
    for i, a in enumerate(geoms):
        for j, b in enumerate(geoms):
            if i == j:
                continue
            touches = a.touches(b)
            if method == "queen":
                ok = touches
            else:
                ok = touches and a.boundary.intersection(b.boundary).length > 0
            if ok:
                edges.append((i, j))
    return sorted(edges)


def _edges(csr) -> list[tuple[int, int]]:
    row_ptr, col_idx = csr
    return sorted(
        (i, j) for i in range(len(row_ptr) - 1) for j in col_idx[row_ptr[i] : row_ptr[i + 1]]
    )


_BLOCKED_LIBPYSAL_SRC = """
import json, sys
sys.modules["libpysal"] = None
sys.modules["libpysal.weights"] = None
import geopandas as gpd
from shapely.geometry import box
from flexscanx.graph import csr_from_geodataframe_contiguity
boxes = json.loads(sys.argv[1])
gdf = gpd.GeoDataFrame(geometry=[box(*b) for b in boxes], crs="EPSG:3857")
out = {m: list(csr_from_geodataframe_contiguity(gdf, method=m)) for m in ("queen", "rook")}
print(json.dumps(out))
"""


def _contiguity_with_libpysal_blocked(geoms) -> dict:
    import json
    import subprocess
    import sys

    boxes = json.dumps([list(g.bounds) for g in geoms])
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_LIBPYSAL_SRC, boxes],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("fixture", sorted(CONTIGUITY_FIXTURES))
@pytest.mark.parametrize("method", ["queen", "rook"])
def test_contiguity_matches_the_documented_geometric_predicate(fixture: str, method: str) -> None:
    geoms = CONTIGUITY_FIXTURES[fixture]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3857")
    got = _edges(flx.graph.csr_from_geodataframe_contiguity(gdf, method=method))
    assert got == _oracle_edges(geoms, method)


@pytest.mark.parametrize("fixture", sorted(CONTIGUITY_FIXTURES))
def test_contiguity_is_identical_with_libpysal_blocked(fixture: str) -> None:
    geoms = CONTIGUITY_FIXTURES[fixture]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3857")
    blocked = _contiguity_with_libpysal_blocked(geoms)
    for method in ("queen", "rook"):
        present = flx.graph.csr_from_geodataframe_contiguity(gdf, method=method)
        assert list(present) == blocked[method], (fixture, method)
        assert _edges(present) == _oracle_edges(geoms, method)


def test_t_junction_rook_scan_does_not_depend_on_libpysal() -> None:
    geoms = CONTIGUITY_FIXTURES["t_junction"]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3857")
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    blocked = tuple(_contiguity_with_libpysal_blocked(geoms)["rook"])
    present = flx.rook_adjacency(gdf)
    results = []
    for csr in (present, blocked):
        res = flx.scan(
            coords=coords,
            adjacency=csr,
            observed=[24, 24, 1, 1],
            expected=[5.0] * 4,
            K=4,
            clustersize=3,
            simcount=99,
            seed=1,
        )
        results.append((res["area"], res["stats"], res["pval"]))
    assert results[0] == results[1]
    # Areas 1 and 2 share the edge x=1, y in [0, 1]; the documented rook predicate joins them.
    assert results[0][0] == [1, 2]


def test_queen_include_intersects_admits_overlapping_polygons() -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 6, 6)], crs="EPSG:3857"
    )
    strict = _edges(flx.queen_adjacency(gdf))
    loose = _edges(flx.queen_adjacency(gdf, include_intersects=True))
    assert strict == []  # overlapping interiors do not "touch"
    assert loose == [(0, 1), (1, 0)]
