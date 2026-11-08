import numpy as np
import pytest

from flexscanx.plotting import (
    plot_all_both_clusters,
    plot_all_clusters,
)
from tests.test_plotting import _make_grid

pytest.importorskip("geopandas")
pytest.importorskip("shapely")


def test_plot_all_clusters_grid_overlay_smoke():
    gdf = _make_grid(3)
    res = {"area": [1, 2, 3], "secondaries": [{"area": [4]}, {"area": [5, 6]}]}

    # grid mode
    plot_all_clusters(gdf, res, id_col="id", overlay=False, cols=2)
    # overlay mode
    ax = plot_all_clusters(gdf, res, id_col="id", overlay=True)
    assert hasattr(ax, "set_axis_off")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cmaps": []}, "cmaps must not be empty"),
        ({"palette": "not-a-palette"}, "palette must be one of"),
        ({"cols": 0}, "cols must be >= 1"),
        ({"cols": 1.5}, "cols must be an integer"),
        ({"max_clusters": 1.5}, "max_clusters must be an integer"),
        ({"significance_alpha": np.nan}, "significance_alpha must be finite"),
        ({"dpi": 72.5}, "dpi must be an integer"),
    ],
)
def test_plot_all_clusters_rejects_invalid_layout_and_palette_inputs(kwargs, message):
    gdf = _make_grid(2)
    result = {"area": [1], "secondaries": []}

    with pytest.raises(ValueError, match=message):
        plot_all_clusters(gdf, result, id_col="id", **kwargs)


def test_plot_all_clusters_optional_glasbey_palette_falls_back_when_unavailable():
    pytest.importorskip("geopandas")
    gdf = _make_grid(2)
    result = {"area": [1], "secondaries": []}

    axes = plot_all_clusters(gdf, result, id_col="id", palette="glasbey")

    assert axes is not None


def test_plot_all_both_clusters_overlay_smoke():
    gdf = _make_grid(3)
    res_both = {
        "type": "BOTH",
        "hot": {"area": [1, 2], "secondaries": [{"area": [3]}]},
        "cold": {"area": [4], "secondaries": [{"area": [5, 6]}]},
    }
    ax = plot_all_both_clusters(gdf, res_both, id_col="id", overlay=True)
    assert hasattr(ax, "set_axis_off")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"significance_alpha": np.nan}, "significance_alpha must be finite"),
        ({"dpi": 72.5}, "dpi must be an integer"),
    ],
)
def test_plot_all_both_clusters_rejects_invalid_controls(kwargs, message):
    gdf = _make_grid(2)
    result = {
        "type": "BOTH",
        "hot": {"area": [1], "secondaries": []},
        "cold": {"area": [2], "secondaries": []},
    }

    with pytest.raises(ValueError, match=message):
        plot_all_both_clusters(gdf, result, id_col="id", **kwargs)
