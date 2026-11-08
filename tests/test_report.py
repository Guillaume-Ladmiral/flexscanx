import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from flexscanx.report import export_cluster_report  # noqa: E402
from tests.test_plotting import _make_grid  # noqa: E402


def test_export_cluster_report_png(tmp_path):
    gdf = _make_grid(3)
    res = {"area": [1, 2, 3], "secondaries": [{"area": [4]}, {"area": [5, 6]}]}
    out = tmp_path / "report.png"
    export_cluster_report(gdf, res, id_col="id", path=str(out), cols=2, max_clusters=3)
    assert out.exists() and out.stat().st_size > 0


def test_export_cluster_report_both_pdf(tmp_path):
    gdf = _make_grid(3)
    res_both = {
        "type": "BOTH",
        "hot": {"area": [1, 2], "secondaries": [{"area": [3]}]},
        "cold": {"area": [4], "secondaries": [{"area": [5, 6]}]},
    }
    out = tmp_path / "report.pdf"
    export_cluster_report(gdf, res_both, id_col="id", path=str(out), cols=2, max_clusters=2)
    assert out.exists() and out.stat().st_size > 0


def test_export_cluster_report_html_escapes_title(tmp_path):
    gdf = _make_grid(3)
    res = {"area": [1, 2, 3], "secondaries": []}
    out = tmp_path / "report.html"

    export_cluster_report(
        gdf,
        res,
        id_col="id",
        path=str(out),
        title="<script>alert(1)</script>",
    )

    text = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_export_cluster_report_html_with_consensus_areas(tmp_path):
    gdf = _make_grid(3)
    res = {"area": [1, 2], "secondaries": []}
    out = tmp_path / "consensus.html"

    export_cluster_report(
        gdf,
        res,
        id_col="id",
        path=str(out),
        consensus_areas=[1, 2, 3],
        consensus_title="Consensus cluster",
    )

    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text


def test_export_cluster_report_both_html(tmp_path):
    gdf = _make_grid(3)
    res_both = {
        "type": "BOTH",
        "hot": {"area": [1, 2], "secondaries": []},
        "cold": {"area": [4], "secondaries": []},
    }
    out = tmp_path / "both_report.html"

    export_cluster_report(gdf, res_both, id_col="id", path=str(out))

    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text


@pytest.mark.parametrize(
    ("kwargs", "filename", "match"),
    [
        ({"cols": 0}, "report.png", "cols"),
        ({"cols": 1.5}, "report.png", "cols"),
        ({"max_clusters": 1.5}, "report.png", "max_clusters"),
        ({"table_max_rows": 1.5}, "report.png", "table_max_rows"),
        ({"figsize": (0.0, 4.0)}, "report.png", "figsize"),
        ({"figsize": ("wide", 4.0)}, "report.png", "figsize"),
        ({}, "report.svg", "extension"),
    ],
)
def test_export_cluster_report_rejects_invalid_controls(tmp_path, kwargs, filename, match):
    gdf = _make_grid(2)
    result = {"area": [1], "secondaries": []}

    with pytest.raises(ValueError, match=match):
        export_cluster_report(
            gdf,
            result,
            id_col="id",
            path=str(tmp_path / filename),
            **kwargs,
        )
