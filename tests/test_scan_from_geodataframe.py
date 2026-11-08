import numpy as np
import pytest

import flexscanx as flx

gpd = pytest.importorskip("geopandas")
box = pytest.importorskip("shapely.geometry").box


def _grid_gdf(ids: list[int], *, crs: str):
    polys = [
        box(0.0, 0.0, 1000.0, 1000.0),
        box(1000.0, 0.0, 2000.0, 1000.0),
        box(0.0, 1000.0, 1000.0, 2000.0),
        box(1000.0, 1000.0, 2000.0, 2000.0),
    ]
    return gpd.GeoDataFrame({"region_id": ids, "geometry": polys}, crs=crs)


def test_scan_from_geodataframe_requires_projected_crs():
    gdf = _grid_gdf([10, 20, 30, 40], crs="EPSG:4326")
    observed = np.array([5, 5, 5, 15], dtype=int)
    expected = np.full(4, 5.0, dtype=float)

    with pytest.raises(ValueError, match="projected CRS"):
        flx.scan_from_geodataframe(
            gdf,
            id_col="region_id",
            observed=observed,
            expected=expected,
            clustersize=1,
            K=4,
            simcount=0,
        )


def test_scan_from_geodataframe_requires_id_col():
    gdf = _grid_gdf([10, 20, 30, 40], crs="EPSG:3857")
    observed = np.array([5, 5, 5, 15], dtype=int)
    expected = np.full(4, 5.0, dtype=float)

    with pytest.raises(TypeError, match="required keyword-only argument: 'id_col'"):
        flx.scan_from_geodataframe(
            gdf,
            observed=observed,
            expected=expected,
            clustersize=1,
            K=4,
            simcount=0,
        )


def test_scan_from_geodataframe_remaps_result_ids_to_id_col():
    gdf = _grid_gdf([10, 20, 30, 40], crs="EPSG:3857")
    observed = np.array([5, 5, 5, 15], dtype=int)
    expected = np.full(4, 5.0, dtype=float)

    result = flx.scan_from_geodataframe(
        gdf,
        id_col="region_id",
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="HOT",
        clustersize=1,
        K=4,
        simcount=0,
    )

    assert result["area"] == [40]
    assert result["name"] == ["40"]

    out = flx.pool_to_geodataframe(gdf, flx.cluster_pool(result), id_col="region_id")
    assert int(out.loc[out.region_id == 40, "PoolMember"].iloc[0]) == 1
    assert int(out.loc[out.region_id == 10, "PoolCount"].iloc[0]) == 0


def test_scan_from_geodataframe_deterministically_remaps_distance_endpoints():
    gdf = _grid_gdf([10, 20, 30, 40], crs="EPSG:3857")
    observed = np.array([5, 5, 15, 15], dtype=int)
    expected = np.full(4, 5.0, dtype=float)

    first = flx.scan_from_geodataframe(
        gdf,
        id_col="region_id",
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="HOT",
        clustersize=2,
        K=4,
        simcount=0,
    )
    second = flx.scan_from_geodataframe(
        gdf,
        id_col="region_id",
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="HOT",
        clustersize=2,
        K=4,
        simcount=0,
    )

    for result in (first, second):
        assert result["area"] == [30, 40]
        assert result["name"] == ["30", "40"]
        assert result["from_area"] == 30
        assert result["to_area"] == 40

    assert first["area"] == second["area"]
    assert first["from_area"] == second["from_area"]
    assert first["to_area"] == second["to_area"]


def test_scan_from_geodataframe_rejects_duplicate_id_col_values():
    gdf = _grid_gdf([10, 20, 20, 40], crs="EPSG:3857")
    observed = np.array([5, 5, 5, 15], dtype=int)
    expected = np.full(4, 5.0, dtype=float)

    with pytest.raises(ValueError, match="must contain unique area IDs"):
        flx.scan_from_geodataframe(
            gdf,
            id_col="region_id",
            observed=observed,
            expected=expected,
            clustersize=1,
            K=4,
            simcount=0,
        )


@pytest.mark.parametrize("bad_id", [True, "20"])
def test_scan_from_geodataframe_rejects_non_integer_typed_ids(bad_id):
    gdf = _grid_gdf([10, bad_id, 30, 40], crs="EPSG:3857")

    with pytest.raises(ValueError, match="positive integer area IDs"):
        flx.scan_from_geodataframe(
            gdf,
            id_col="region_id",
            observed=[5, 5, 5, 15],
            expected=[5.0] * 4,
            clustersize=1,
            K=4,
            simcount=0,
        )


@pytest.mark.parametrize("geometry_kind", ["empty", "invalid"])
def test_scan_from_geodataframe_rejects_unusable_geometries(geometry_kind):
    gdf = _grid_gdf([10, 20, 30, 40], crs="EPSG:3857")
    if geometry_kind == "empty":
        from shapely.geometry import Polygon

        gdf.at[1, "geometry"] = Polygon()
        message = "empty geometry"
    else:
        from shapely.geometry import Polygon

        gdf.at[1, "geometry"] = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        message = "invalid geometry"

    with pytest.raises(ValueError, match=message):
        flx.scan_from_geodataframe(
            gdf,
            id_col="region_id",
            observed=[5, 5, 5, 15],
            expected=[5.0] * 4,
            clustersize=1,
            K=4,
            simcount=0,
        )
