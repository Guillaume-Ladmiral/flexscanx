import inspect
import warnings

import numpy as np
import pytest

import flexscanx as pfs
from flexscanx.graph import csr_knn_from_coords

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
Polygon = shapely_geometry.Polygon


def test_helpers_pool_membership_series():
    pool = [frozenset([1, 2, 3]), frozenset([2, 3, 4])]
    ids = [1, 2, 3, 4, 5]
    s = pfs.pool_membership_series(pool, ids)
    assert int(s[1]) == 1
    assert int(s[2]) == 2
    assert int(s[5]) == 0


def test_helpers_pool_to_geodataframe():
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]
    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": polys}, geometry="geometry", crs="EPSG:3857")
    pool = [frozenset([2])]
    with warnings.catch_warnings(record=True) as caught:
        out = pfs.pool_to_geodataframe(gdf, pool, id_col="id")
    assert "PoolMember" in out.columns and "PoolCount" in out.columns
    assert int(out.loc[out.id == 2, "PoolMember"].iloc[0]) == 1
    assert int(out.loc[out.id == 1, "PoolCount"].iloc[0]) == 0
    assert caught == []


def test_helpers_pool_to_geodataframe_warns_on_unmatched_cluster_ids():
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]
    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": polys}, geometry="geometry", crs="EPSG:3857")
    pool = [frozenset([2, 99])]
    with pytest.warns(UserWarning, match="cluster area IDs were not found"):
        out = pfs.pool_to_geodataframe(gdf, pool, id_col="id")
    assert int(out.loc[out.id == 2, "PoolMember"].iloc[0]) == 1


def test_filter_significant_clusters_enforces_max_radius():
    result = {
        "type": "HOT",
        "area": [1, 2],
        "pval": 0.01,
        "max_dist": 2.0,
        "secondaries": [
            {"area": [3, 4], "pval": 0.03, "max_dist": 4.5},
            {"area": [5], "pval": 0.20, "max_dist": 0.0},
        ],
    }

    pool = pfs.filter_significant_clusters(result, alpha=0.05, max_radius=3.0)
    assert pool == [frozenset([1, 2])]


def test_filter_significant_clusters_applies_alpha_and_tau():
    result = {
        "type": "HOT",
        "area": [1],
        "pval": 0.01,
        "stats": 9.0,
        "secondaries": [
            {"area": [2], "pval": 0.01, "stats": 3.0},
            {"area": [3], "pval": 0.20, "stats": 10.0},
            {"area": [4], "pval": 0.01, "stats": 8.0},
        ],
    }

    pool = pfs.filter_significant_clusters(result, alpha=0.05, tau=8.0)

    assert pool == [frozenset([1]), frozenset([4])]


def test_filter_significant_clusters_rejects_radius_on_bare_pool():
    pool = [frozenset([1, 2])]
    with pytest.raises(ValueError, match="max_radius requires a scan result dict"):
        pfs.filter_significant_clusters(pool, max_radius=2.0)


def test_filter_significant_clusters_requires_max_dist_metadata():
    result = {"type": "HOT", "area": [1, 2], "pval": 0.01, "secondaries": []}
    with pytest.raises(ValueError, match="max_dist metadata"):
        pfs.filter_significant_clusters(result, max_radius=2.0)


def test_csr_knn_from_coords_rejects_unknown_backend():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="backend must be one of"):
        csr_knn_from_coords(coords, k=1, backend="nearest")


def test_csr_knn_tie_breaking_matches_backends_by_distance_then_index():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    kdtree = csr_knn_from_coords(coords, k=2, backend="kdtree")
    balltree = csr_knn_from_coords(coords, k=2, backend="balltree")

    assert kdtree == balltree


@pytest.mark.parametrize(
    ("coords", "k", "radius", "message"),
    [
        (np.array([[0.0, 0.0], [1.0, 0.0]]), 1.5, np.inf, "k must be an integer"),
        (np.array([[0.0, 0.0], [1.0, 0.0]]), 0, np.inf, "k must be >= 1"),
        (np.empty((0, 2)), 1, np.inf, "at least one coordinate"),
        (
            np.array([[0.0, 0.0], [np.nan, 0.0]]),
            1,
            np.inf,
            "finite",
        ),
        (np.array([[0.0, 0.0], [1.0, 0.0]]), 1, -1.0, "radius must be non-negative"),
    ],
)
def test_csr_knn_rejects_invalid_numeric_inputs(coords, k, radius, message):
    with pytest.raises(ValueError, match=message):
        csr_knn_from_coords(coords, k=k, radius=radius)


def test_csr_haversine_interprets_coordinates_as_longitude_latitude():
    lon_lat = np.array([[0.0, 68.0], [135.0, 71.0], [-107.0, 81.0], [134.0, 70.0]])

    row_ptr, col_idx = csr_knn_from_coords(lon_lat, k=1, metric="haversine", backend="balltree")

    assert row_ptr == [0, 1, 3, 5, 6]
    assert col_idx == [2, 2, 3, 0, 1, 1]


def test_csr_haversine_rejects_incompatible_kdtree_backend():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="haversine.*BallTree"):
        csr_knn_from_coords(coords, k=1, metric="haversine", backend="kdtree")


def test_cluster_pool_exact_dedup_matches_behavior_and_old_keyword_is_gone():
    result = {
        "type": "HOT",
        "area": [1, 2, 3],
        "secondaries": [
            {"area": [1, 2, 3]},
            {"area": [1, 2, 4]},
        ],
    }

    assert "exact_dedup" in str(inspect.signature(pfs.cluster_pool))
    assert pfs.cluster_pool(result) == [frozenset([1, 2, 3]), frozenset([1, 2, 4])]
    assert pfs.cluster_pool(result, exact_dedup=False) == [
        frozenset([1, 2, 3]),
        frozenset([1, 2, 3]),
        frozenset([1, 2, 4]),
    ]
    with pytest.raises(TypeError):
        pfs.cluster_pool(result, jaccard_dedup=True)  # type: ignore[call-arg]


def test_graph_contiguity_builders():
    # 2x2 grid
    polys = []
    ids = []
    for r in range(2):
        for c in range(2):
            x0, y0 = c * 1.0, r * 1.0
            polys.append(Polygon([(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)]))
            ids.append(r * 2 + c + 1)
    gdf = gpd.GeoDataFrame({"id": ids, "geometry": polys}, geometry="geometry", crs="EPSG:3857")

    rp_q, ci_q = pfs.queen_adjacency(gdf)
    rp_r, ci_r = pfs.rook_adjacency(gdf)
    # basic invariants
    assert len(rp_q) == len(rp_r) == len(gdf) + 1
    # top-left (id=1) has at least 2 queen neighbors and 2 rook neighbors in this grid
    deg_q_1 = rp_q[1] - rp_q[0]
    deg_r_1 = rp_r[1] - rp_r[0]
    assert deg_q_1 >= 2 and deg_r_1 >= 2


def test_graph_contiguity_follows_geodataframe_row_order():
    polys = [
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # top-right
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # bottom-left
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # top-left
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # bottom-right
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [101, 202, 303, 404], "geometry": polys},
        geometry="geometry",
        crs="EPSG:3857",
        index=[20, 10, 40, 30],
    )

    row_ptr, col_idx = pfs.rook_adjacency(gdf)

    # Expected rook adjacency in current row order:
    # row 0 (top-right)  -> rows 2,3
    # row 1 (bottom-left)-> rows 2,3
    # row 2 (top-left)   -> rows 0,1
    # row 3 (bottom-right)-> rows 0,1
    assert row_ptr == [0, 2, 4, 6, 8]
    assert col_idx == [2, 3, 2, 3, 0, 1, 0, 1]


def test_graph_contiguity_passes_use_index_false(monkeypatch):
    import libpysal.weights as weights

    from flexscanx import graph as graph_mod

    class _DummyW:
        def __init__(self):
            self.id_order = [0, 1]
            self.neighbors = {0: [1], 1: [0]}

    class _DummyQueen:
        @staticmethod
        def from_dataframe(gdf, **kwargs):
            assert kwargs["use_index"] is False
            return _DummyW()

    class _DummyRook:
        @staticmethod
        def from_dataframe(gdf, **kwargs):
            assert kwargs["use_index"] is False
            return _DummyW()

    monkeypatch.setattr(weights, "Queen", _DummyQueen)
    monkeypatch.setattr(weights, "Rook", _DummyRook)

    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    assert graph_mod.csr_from_geodataframe_contiguity(gdf, method="queen") == ([0, 1, 2], [1, 0])
    assert graph_mod.csr_from_geodataframe_contiguity(gdf, method="rook") == ([0, 1, 2], [1, 0])


def test_contiguity_rejects_unknown_method():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ]
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="method must be one of"):
        pfs.build_contiguity(gdf, method="bishop")
