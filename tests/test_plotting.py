import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")


def _make_grid(n=3):
    import geopandas as gpd
    from shapely.geometry import box

    polys = []
    ids = []
    k = 0
    for i in range(n):
        for j in range(n):
            k += 1
            ids.append(k)  # 1-based
            polys.append(box(j, i, j + 1, i + 1))
    return gpd.GeoDataFrame({"id": ids, "geometry": polys}, crs="EPSG:4326")


def test_plot_cluster_membership_smoke():
    from flexscanx.plotting import plot_cluster_membership

    gdf = _make_grid(2)
    res = {"area": [1, 2, 3]}
    ax = plot_cluster_membership(gdf, res, id_col="id", legend=False)
    # basic axis
    assert hasattr(ax, "set_axis_off")


def test_plot_both_membership_smoke():
    from flexscanx.plotting import plot_both_membership

    gdf = _make_grid(2)
    res_both = {"type": "BOTH", "hot": {"area": [1, 2]}, "cold": {"area": [3, 4]}}
    axes = plot_both_membership(gdf, res_both, id_col="id")
    assert len(axes) == 2


def test_choropleth_smoke():
    import pandas as pd

    from flexscanx.plotting import choropleth

    gdf = _make_grid(2)
    values = pd.Series({1: 0.1, 2: 0.2, 3: 0.5, 4: 0.9})
    ax = choropleth(gdf, values, id_col="id", legend=False)
    assert hasattr(ax, "set_axis_off")


def test_choropleth_plain_sequence_uses_one_based_area_ids(monkeypatch):
    from geopandas import plotting as gpd_plotting

    from flexscanx.plotting import choropleth

    gdf = _make_grid(2)
    captured: dict[str, object] = {}
    original_plot_dataframe = gpd_plotting.plot_dataframe

    def _spy_plot_dataframe(dataframe, *args, **kwargs):
        captured["frame"] = dataframe.copy()
        return original_plot_dataframe(dataframe, *args, **kwargs)

    monkeypatch.setattr(gpd_plotting, "plot_dataframe", _spy_plot_dataframe)
    choropleth(gdf, [10.0, 20.0, 30.0, 40.0], id_col="id", legend=False)

    plotted = captured["frame"]
    assert list(plotted.sort_values("id")["_val"]) == [10.0, 20.0, 30.0, 40.0]


def test_choropleth_passes_returned_axis_to_geopandas(monkeypatch):
    from geopandas import plotting as gpd_plotting

    from flexscanx.plotting import choropleth

    gdf = _make_grid(2)
    captured: dict[str, object] = {}
    original_plot_dataframe = gpd_plotting.plot_dataframe

    def _spy_plot_dataframe(dataframe, *args, **kwargs):
        captured["ax"] = kwargs.get("ax")
        return original_plot_dataframe(dataframe, *args, **kwargs)

    monkeypatch.setattr(gpd_plotting, "plot_dataframe", _spy_plot_dataframe)
    ax = choropleth(gdf, [10.0, 20.0, 30.0, 40.0], id_col="id", legend=False)

    assert captured["ax"] is ax


def test_plot_values_from_obsexp_smoke():
    from flexscanx.plotting import plot_values_from_obsexp

    gdf = _make_grid(2)
    observed = [10, 20, 5, 15]
    expected = [8.0, 18.0, 10.0, 20.0]
    ax = plot_values_from_obsexp(gdf, observed, expected, id_col="id", metric="oe_ratio")
    assert hasattr(ax, "set_axis_off")


def test_choropleth_with_scheme_quantiles():
    pytest.importorskip("mapclassify")
    import pandas as pd

    from flexscanx.plotting import choropleth

    gdf = _make_grid(3)
    values = pd.Series({i: float(i) for i in range(1, 10)})
    ax = choropleth(gdf, values, id_col="id", legend=False, scheme="quantiles", k=3)
    assert hasattr(ax, "set_axis_off")
