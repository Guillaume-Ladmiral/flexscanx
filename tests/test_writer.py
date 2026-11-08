import pandas as pd

from flexscanx.summary import write_clusters_csv


def test_write_clusters_csv(tmp_path):
    single = {
        "type": "HOT",
        "area": [1, 2],
        "n_case": 10,
        "expected": 5.0,
        "Rate_Ratio": 2.0,
        "stats": 5.5,
        "pval": 0.01,
        "secondaries": [
            {
                "area": [3],
                "n_case": 4,
                "expected": 3.0,
                "Rate_Ratio": 1.33,
                "stats": 1.2,
                "pval": 0.5,
            }
        ],
    }
    out = tmp_path / "clusters.csv"
    write_clusters_csv(single, str(out))
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out)
    for col in [
        "direction",
        "rank",
        "area",
        "n_case",
        "expected",
        "RR",
        "stats",
        "pval",
    ]:
        assert col in df.columns
