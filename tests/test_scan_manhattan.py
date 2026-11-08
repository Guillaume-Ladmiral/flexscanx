import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from flexscanx.api import scan

# Require either pyogrio or fiona for GPKG IO
try:  # noqa: SIM105
    import pyogrio as _po  # type: ignore  # noqa: F401

    _HAS_GPKG_IO = True
except Exception:  # pragma: no cover - optional
    try:
        import fiona as _fi  # type: ignore  # noqa: F401

        _HAS_GPKG_IO = True
    except Exception:
        _HAS_GPKG_IO = False


pytestmark = pytest.mark.skipif(not _HAS_GPKG_IO, reason="geospatial IO deps not installed")

ROOT = Path(__file__).resolve().parents[1]
MANHATTAN_SMOKE_K = 8
MANHATTAN_SMOKE_SIMCOUNT = 19


def _load_manhattan():
    gpkg = ROOT / "vignettes" / "Manhattan_BreastCancer.gpkg"
    try:
        import pyogrio as po

        df = po.read_dataframe(str(gpkg), layer="manhattan_breast")
    except Exception:  # pragma: no cover - fallback
        import fiona
        import pandas as pd
        from shapely.geometry import shape

        records = []
        with fiona.open(str(gpkg), layer="manhattan_breast") as src:
            for feat in src:
                rec = dict(feat["properties"])
                rec["geometry"] = shape(feat["geometry"]) if feat.get("geometry") else None
                records.append(rec)
        df = pd.DataFrame.from_records(records)
    x = np.asarray(df["LONGITUDE"], dtype=float)
    y = np.asarray(df["LATITUDE"], dtype=float)
    coords = np.column_stack([x, y])
    observed = np.asarray(df["OBREAST"], dtype=int)
    expected = np.asarray(df["EBREAST"], dtype=float)
    # A zero expected count is a zero-exposure stratum, outside the package's
    # Poisson likelihood contract. The checked-in data contain one such row;
    # exclude it explicitly instead of silently asking scan() to divide by it.
    valid = np.isfinite(expected) & (expected > 0.0)
    return coords[valid], observed[valid], expected[valid]


def test_manhattan_smoke_preprocessing_enforces_positive_exposure():
    _, observed, expected = _load_manhattan()
    assert observed.size == expected.size
    assert expected.size == 981
    assert np.all(expected > 0.0)


def test_maintained_manhattan_driver_preprocesses_positive_exposure():
    namespace = runpy.run_path(str(ROOT / "scripts" / "test_manhattan_scan.py"))
    coords, observed, expected = namespace["load_data"]()
    assert coords.shape[0] == observed.size == expected.size == 981
    assert np.all(expected > 0.0)


@pytest.mark.parametrize("method", ["FLEXIBLE", "CIRCULAR"])
def test_mc_pvalues_present(method):
    coords, observed, expected = _load_manhattan()
    res = scan(
        coords=coords,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod=method,
        stattype="RESTRICTED",
        clustertype="HOT",
        # Keep the all-982-area integration smoke bounded: exact flexible
        # enumeration is exponential in K.  Higher-budget performance runs
        # belong in the opt-in benchmarks, not the default correctness suite.
        K=min(MANHATTAN_SMOKE_K, len(observed)),
        simcount=max(1, int(os.environ.get("SIMCOUNT", str(MANHATTAN_SMOKE_SIMCOUNT)))),
        seed=7,
        knn_method="geometry",
        n_jobs_centers=1,
    )
    assert isinstance(res, dict)
    # Core uses key 'pval' (with optional alias 'pvalue'). Accept either.
    p = res.get("pval", res.get("pvalue"))
    assert p is not None
    assert 0.0 <= float(p) <= 1.0


def test_both_parallel_structure():
    coords, observed, expected = _load_manhattan()
    res = scan(
        coords=coords,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="BOTH",
        K=min(MANHATTAN_SMOKE_K, len(observed)),
        simcount=0,
        seed=123,
        knn_method="geometry",
        both_parallel=True,
        n_jobs_centers=max(2, min((os.cpu_count() or 2), 4)),
    )
    assert isinstance(res, dict)
    assert res.get("type") == "BOTH"
    assert "hot" in res and "cold" in res and "primary" in res


@pytest.mark.parametrize("method", ["FLEXIBLE", "CIRCULAR"])
def test_scheduler_backends_equal_results(method):
    coords, observed, expected = _load_manhattan()
    res_a = scan(
        coords=coords,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod=method,
        stattype="RESTRICTED",
        clustertype="HOT",
        K=min(MANHATTAN_SMOKE_K, len(observed)),
        simcount=0,
        seed=42,
        knn_method="geometry",
        n_jobs_centers=4,
        scheduler_backend="ATOMIC",
    )
    res_w = scan(
        coords=coords,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod=method,
        stattype="RESTRICTED",
        clustertype="HOT",
        K=min(MANHATTAN_SMOKE_K, len(observed)),
        simcount=0,
        seed=42,
        knn_method="geometry",
        n_jobs_centers=4,
        scheduler_backend="WORK_STEAL",
    )
    assert isinstance(res_a, dict) and isinstance(res_w, dict)
    assert res_a.get("stats") == res_w.get("stats")
    assert list(res_a.get("area", []) or []) == list(res_w.get("area", []) or [])
