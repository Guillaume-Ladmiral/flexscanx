"""Regression coverage for the 2026-08-16 inference-contract remediation."""

from __future__ import annotations

import numpy as np
import pytest

import flexscanx.api as api
import flexscanx.summary as summary
from flexscanx import report
from flexscanx._result_format import _build_both_result
from flexscanx._stability import match_clusters_jaccard
from flexscanx.helpers import cluster_pool, filter_significant_clusters
from flexscanx.summary import clusters_to_dataframe, summarize_result, write_clusters_csv


def _line_graph() -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    return np.asarray([[0.0, 0.0], [1.0, 0.0]]), ([0, 1, 2], [1, 0])


def _both_result_with_straddling_pvalues() -> dict:
    hot = {
        "area": [1],
        "n_case": 8,
        "expected": 3.0,
        "stats": 10.0,
        "rank": 1,
        "pval": 0.04,
        "secondaries": [],
        "sim_max": [1.0, 2.0, 3.0, 4.0],
    }
    cold = {
        "area": [2],
        "n_case": 1,
        "expected": 3.0,
        "stats": 5.0,
        "rank": 1,
        "pval": 0.80,
        "secondaries": [
            {
                "area": [3],
                "n_case": 0,
                "expected": 2.0,
                "stats": 2.0,
                "rank": 2,
                "pval": 0.90,
            }
        ],
        "sim_max": [1.0, 2.0, 3.0, 4.0],
    }
    return _build_both_result(hot, cold, provenance={})


def test_poisson_zero_expected_is_rejected_by_scan_and_series() -> None:
    coords, adjacency = _line_graph()

    with pytest.raises(ValueError, match=r"must be strictly positive"):
        api.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1, 0],
            expected=[1.0, 0.0],
            K=2,
            clustersize=1,
            simcount=0,
        )

    with pytest.raises(ValueError, match=r"must be strictly positive"):
        api.scan_series(
            coords=coords,
            adjacency=adjacency,
            observed=np.asarray([[1, 0]], dtype=int),
            expected=np.asarray([[1.0, 0.0]]),
            K=2,
            clustersize=1,
            simcount=0,
        )


def test_both_outputs_use_the_two_sided_primary_pvalue(tmp_path, monkeypatch) -> None:
    result = _both_result_with_straddling_pvalues()

    dataframe = clusters_to_dataframe(result)
    assert dataframe["pval"].tolist() == [0.2]
    assert dataframe["pval_hot"].tolist() == [0.04]
    assert dataframe["pval_cold"].tolist() == [0.8]

    single_dataframe = clusters_to_dataframe(
        {
            "type": "HOT",
            "area": [1],
            "n_case": 8,
            "expected": 3.0,
            "stats": 10.0,
            "pval": 0.04,
        }
    )
    assert "pval_hot" not in single_dataframe
    assert "pval_cold" not in single_dataframe

    table_summary = summarize_result(result)
    assert table_summary["P"].tolist() == [0.2]
    assert table_summary["P_hot"].tolist() == [0.04]
    assert table_summary["P_cold"].tolist() == [0.8]
    assert table_summary["Sig"].tolist() == [""]

    csv_path = tmp_path / "clusters.csv"
    write_clusters_csv(result, str(csv_path))
    assert "0.2" in csv_path.read_text(encoding="utf-8")
    assert filter_significant_clusters(result, alpha=0.05) == []
    assert cluster_pool(result) == [frozenset({1}), frozenset({2}), frozenset({3})]

    pytest.importorskip("geopandas")
    from tests.test_plotting import _make_grid

    original_summarize = summary.summarize_result
    seen: list[dict] = []

    def _capture_summary(value: dict):
        seen.append(value)
        return original_summarize(value)

    monkeypatch.setattr(summary, "summarize_result", _capture_summary)
    report.export_cluster_report(
        _make_grid(2), result, id_col="id", path=str(tmp_path / "report.png")
    )
    assert seen == [result, result["hot"], result["cold"]]
    assert original_summarize(seen[2])["P"].tolist() == [0.8, 0.9]


def test_jaccard_unmatched_clusters_contribute_zero_scores() -> None:
    metrics = match_clusters_jaccard([{1, 2}, {3, 4}], [{1, 2}, {7, 8}])

    assert metrics.matched == [(0, 0, 1.0)]
    assert metrics.mean_jaccard == 0.5
    assert metrics.min_jaccard == 0.0
