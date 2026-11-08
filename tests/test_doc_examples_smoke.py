from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")

import flexscanx as flx
from flexscanx.graph import csr_from_geodataframe_contiguity
from flexscanx.sequence import extract_sequences


def _example_grid(n: int = 3):
    import geopandas as gpd
    from shapely.geometry import box

    polys = []
    ids = []
    k = 0
    for i in range(n):
        for j in range(n):
            k += 1
            ids.append(k)
            polys.append(box(j * 1000.0, i * 1000.0, (j + 1) * 1000.0, (i + 1) * 1000.0))
    return gpd.GeoDataFrame({"id": ids, "geometry": polys}, crs="EPSG:3857")


def _coords_from_geodataframe(gdf) -> np.ndarray:
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])


def test_doc_example_geodataframe_contiguity_scan_smoke():
    gdf = _example_grid(3)
    coords = _coords_from_geodataframe(gdf)
    row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf, method="queen")

    observed = np.full(len(gdf), 5, dtype=int)
    observed[0] = 15
    expected = np.full(len(gdf), 5.0, dtype=float)

    result = flx.scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="HOT",
        K=4,
        clustersize=2,
        simcount=0,
    )

    assert isinstance(result, dict)
    assert "area" in result
    assert len(result["area"]) >= 1


def test_doc_example_timeseries_workflow_smoke():
    gdf = _example_grid(3)
    coords = _coords_from_geodataframe(gdf)
    row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf, method="queen")

    observed_tn = np.full((2, len(gdf)), 5, dtype=int)
    observed_tn[0, 0] = 15
    observed_tn[1, -1] = 15
    expected_tn = np.full((2, len(gdf)), 5.0, dtype=float)

    result_series = flx.scan_series(
        observed=observed_tn,
        expected=expected_tn,
        coords=coords,
        adjacency=(row_ptr, col_idx),
        model="POISSON",
        clustertype="BOTH",
        K=4,
        clustersize=2,
        simcount=0,
        crn=True,
        seed=42,
    )

    sequences, area_ids = extract_sequences(result_series, include_type="both")
    assert result_series["per_time"]
    assert sequences.shape[1] == observed_tn.shape[0]
    assert sequences.shape[0] == len(area_ids)


def test_doc_example_flatten_and_radius_filter_smoke():
    gdf = _example_grid(3)
    coords = _coords_from_geodataframe(gdf)
    row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf, method="queen")

    observed = np.full(len(gdf), 5, dtype=int)
    observed[0] = 12
    expected = np.full(len(gdf), 5.0, dtype=float)

    result = flx.scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        model="POISSON",
        clustersize=1,
        K=4,
        simcount=0,
    )

    clusters = flx.flatten_clusters(result)
    pooled = flx.cluster_pool(result)
    radius_filtered = flx.filter_significant_clusters(result, alpha=None, max_radius=0.1)

    assert "max_dist" in clusters[0]
    assert clusters[0]["max_dist"] == 0.0
    assert pooled == [frozenset(result["area"])]
    assert radius_filtered == [frozenset(result["area"])]
