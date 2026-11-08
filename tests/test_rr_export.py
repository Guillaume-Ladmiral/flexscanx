"""Regression test for RR key bug (Rate_Ratio/Risk_Ratio export).

This test ensures that Rate_Ratio (POISSON) and Risk_Ratio (BINOMIAL)
values are correctly extracted and exported to summaries, DataFrames, and CSV.
"""

import numpy as np
import pandas as pd

import flexscanx._core as core
from flexscanx.summary import (
    clusters_to_dataframe,
    flatten_clusters,
    summarize_result,
    write_clusters_csv,
)


def test_poisson_rate_ratio_export():
    """Verify Rate_Ratio from POISSON scan appears in summaries as RR."""
    # 4-node chain with hot spot
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]

    handle = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {"ralpha": 0.2, "clustersize": 3}
    res = core.ScanSlice(handle, observed, expected, settings)

    # Verify C++ bindings set Rate_Ratio
    assert "Rate_Ratio" in res
    assert "Rate_Ratio_CI" in res
    assert "RR" not in res
    rate_ratio = res["Rate_Ratio"]
    assert rate_ratio is not None and rate_ratio > 0.0
    assert res["Rate_Ratio_CI"] is None

    # Verify flatten_clusters extracts it as RR
    flat = flatten_clusters(res, include_time=False)
    assert len(flat) > 0
    assert flat[0]["RR"] == rate_ratio

    # Verify DataFrame conversion
    df = clusters_to_dataframe(res, include_time=False)
    assert "RR" in df.columns
    assert df.iloc[0]["RR"] == rate_ratio

    # Verify summarize_result
    summary_df = summarize_result(res)
    assert "RR" in summary_df.columns
    assert summary_df.iloc[0]["RR"] == rate_ratio


def test_binomial_risk_ratio_export():
    """Verify Risk_Ratio from BINOMIAL scan appears in summaries as RR."""
    # 4-node chain with hot spot in binomial setting
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [8, 4, 1, 1]
    population = [10.0, 10.0, 10.0, 10.0]

    handle = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {"model": "BINOMIAL", "ralpha": 0.2, "clustersize": 3}
    res = core.ScanSlice(handle, observed, population, settings)

    # Verify C++ bindings set Risk_Ratio
    assert "Risk_Ratio" in res
    assert "Risk_Ratio_CI" in res
    assert "RR" not in res
    risk_ratio = res["Risk_Ratio"]
    assert risk_ratio is not None and risk_ratio > 0.0
    assert res["Risk_Ratio_CI"] is None

    # Verify flatten_clusters extracts it as RR
    flat = flatten_clusters(res, include_time=False)
    assert len(flat) > 0
    assert flat[0]["RR"] == risk_ratio

    # Verify DataFrame conversion
    df = clusters_to_dataframe(res, include_time=False)
    assert "RR" in df.columns
    assert df.iloc[0]["RR"] == risk_ratio

    # Verify summarize_result
    summary_df = summarize_result(res)
    assert "RR" in summary_df.columns
    assert summary_df.iloc[0]["RR"] == risk_ratio


def test_rr_csv_export(tmp_path):
    """Verify RR values appear correctly in CSV export."""
    # POISSON scan
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]

    handle = core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))
    settings = {"ralpha": 0.2, "clustersize": 3}
    res = core.ScanSlice(handle, observed, expected, settings)

    rate_ratio = res["Rate_Ratio"]

    # Export to CSV
    csv_path = tmp_path / "clusters_rr.csv"
    write_clusters_csv(res, str(csv_path), include_time=False)

    # Read back and verify RR column
    df = pd.read_csv(csv_path)
    assert "RR" in df.columns
    assert df.iloc[0]["RR"] == rate_ratio
