"""Tests for RR key consistency across C++ bindings and Python summaries.

Verifies that Rate_Ratio/Risk_Ratio from C++ bindings appear correctly
in all Python outputs (DataFrame, summarize_result, CSV export).
"""

import numpy as np
import pandas as pd

import flexscanx._core as core
from flexscanx.summary import clusters_to_dataframe, summarize_result, write_clusters_csv


def _poisson_handle():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    return core.BuildStatic(x, row_ptr, col_idx, 3, float("inf"))


def test_poisson_raw_result_has_rate_ratio_not_legacy_rr():
    handle = _poisson_handle()
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    settings = {"ralpha": 0.2, "clustersize": 3}
    result = core.ScanSlice(handle, observed, expected, settings)

    assert "Rate_Ratio" in result, "POISSON result must have 'Rate_Ratio' key"
    assert "RR" not in result, "Raw POISSON result should not expose legacy 'RR' key"


def test_poisson_rr_consistent_across_dataframe_summary_csv(tmp_path):
    handle = _poisson_handle()
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]
    settings = {"ralpha": 0.2, "clustersize": 3}
    result = core.ScanSlice(handle, observed, expected, settings)
    rate_ratio = result["Rate_Ratio"]

    df = clusters_to_dataframe(result, include_time=False)
    assert df.iloc[0]["RR"] == rate_ratio, "DataFrame RR mismatch"

    summary_df = summarize_result(result)
    assert summary_df.iloc[0]["RR"] == rate_ratio, "summarize_result RR mismatch"

    csv_path = tmp_path / "clusters.csv"
    write_clusters_csv(result, csv_path, include_time=False)
    csv_df = pd.read_csv(csv_path)
    assert csv_df.iloc[0]["RR"] == rate_ratio, "CSV export RR mismatch"


def test_binomial_raw_result_has_risk_ratio_not_legacy_rr():
    handle = _poisson_handle()
    observed = [8, 4, 1, 1]
    population = [10.0, 10.0, 10.0, 10.0]
    settings = {"model": "BINOMIAL", "ralpha": 0.2, "clustersize": 3}
    result = core.ScanSlice(handle, observed, population, settings)

    assert "Risk_Ratio" in result, "BINOMIAL result must have 'Risk_Ratio' key"
    assert "RR" not in result, "Raw BINOMIAL result should not expose legacy 'RR' key"


def test_binomial_rr_consistent_across_dataframe_and_summary():
    handle = _poisson_handle()
    observed = [8, 4, 1, 1]
    population = [10.0, 10.0, 10.0, 10.0]
    settings = {"model": "BINOMIAL", "ralpha": 0.2, "clustersize": 3}
    result = core.ScanSlice(handle, observed, population, settings)
    risk_ratio = result["Risk_Ratio"]

    df = clusters_to_dataframe(result, include_time=False)
    assert df.iloc[0]["RR"] == risk_ratio, "DataFrame RR mismatch"

    summary_df = summarize_result(result)
    assert summary_df.iloc[0]["RR"] == risk_ratio, "summarize_result RR mismatch"
