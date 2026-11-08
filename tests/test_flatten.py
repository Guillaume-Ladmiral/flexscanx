import pandas as pd

from flexscanx.summary import (
    clusters_to_dataframe,
    flatten_clusters,
    summarize_result,
    write_series_clusters_csv,
)


def test_flatten_single_and_both_and_series():
    # single-direction
    single = {
        "type": "HOT",
        "area": [1, 2],
        "n_case": 10,
        "expected": 5.0,
        "max_dist": 1.5,
        "Rate_Ratio": 2.0,
        "stats": 5.5,
        "pval": 0.01,
        "secondaries": [
            {
                "area": [3],
                "n_case": 4,
                "expected": 3.0,
                "max_dist": 0.0,
                "Rate_Ratio": 1.33,
                "stats": 1.2,
                "pval": 0.5,
            }
        ],
    }
    flat1 = flatten_clusters(single)
    assert len(flat1) == 2
    assert flat1[0]["rank"] == 1 and flat1[0]["direction"] == "HOT"
    assert flat1[0]["max_dist"] == 1.5
    assert flat1[1]["max_dist"] == 0.0

    # BOTH
    both = {
        "type": "BOTH",
        "hot": {
            "area": [1],
            "n_case": 6,
            "expected": 3.0,
            "max_dist": 0.0,
            "Rate_Ratio": 2.0,
            "stats": 3.2,
        },
        "cold": {
            "area": [2],
            "n_case": 1,
            "expected": 3.0,
            "max_dist": 0.0,
            "Rate_Ratio": 0.33,
            "stats": 2.8,
            "secondaries": [
                {
                    "area": [4],
                    "n_case": 0,
                    "expected": 2.0,
                    "max_dist": 0.0,
                    "Rate_Ratio": 0.0,
                    "stats": 0.5,
                }
            ],
        },
    }
    flat2 = flatten_clusters(both)
    assert len(flat2) == 3
    dirs = {d["direction"] for d in flat2}
    assert {"HOT", "COLD"}.issubset(dirs)

    # series
    series = {"per_time": [single, both]}
    flat3 = flatten_clusters(series)
    # equal to flat1 + flat2 in length
    assert len(flat3) == len(flat1) + len(flat2)

    # DataFrame conversion
    df1 = clusters_to_dataframe(single)
    assert isinstance(df1, pd.DataFrame) and len(df1) == len(flat1)
    # Ensure basic columns present
    for col in [
        "direction",
        "rank",
        "area",
        "n_case",
        "expected",
        "max_dist",
        "RR",
        "stats",
        "pval",
    ]:
        assert col in df1.columns

    df2 = clusters_to_dataframe(both)
    assert len(df2) == len(flat2)

    df3 = clusters_to_dataframe(series)
    assert len(df3) == len(flat3)
    # time should be present for series (even if some rows possibly None)
    assert "time" in df3.columns


def test_series_outputs_preserve_caller_supplied_time_labels(tmp_path):
    single = {
        "type": "HOT",
        "area": [1, 2],
        "n_case": 10,
        "expected": 5.0,
        "max_dist": 1.5,
        "Rate_Ratio": 2.0,
        "stats": 5.5,
        "pval": 0.01,
    }
    both = {
        "type": "BOTH",
        "hot": {
            "area": [1],
            "n_case": 6,
            "expected": 3.0,
            "max_dist": 0.0,
            "Rate_Ratio": 2.0,
            "stats": 3.2,
        },
        "cold": {
            "area": [2],
            "n_case": 1,
            "expected": 3.0,
            "max_dist": 0.0,
            "Rate_Ratio": 0.33,
            "stats": 2.8,
        },
    }
    series = {"per_time": [single, both], "times": ["2026-01", "2026-02"]}

    flat = flatten_clusters(series)
    assert [row["time"] for row in flat] == ["2026-01", "2026-02", "2026-02"]

    df = clusters_to_dataframe(series)
    assert list(df["time"]) == ["2026-01", "2026-02", "2026-02"]

    summary = summarize_result(series)
    assert list(summary.index.get_level_values("time").unique()) == ["2026-01", "2026-02"]

    out = tmp_path / "series_clusters.csv"
    write_series_clusters_csv(series, str(out))
    exported = pd.read_csv(out)
    assert list(exported["time"]) == ["2026-01", "2026-02", "2026-02"]
