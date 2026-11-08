"""Regression and correctness tests for v0.0.1b audit remediation.

Covers:
  F3  — warm_start correctness (same areas as warm_start=False)
  F9  — csr_from_dense symmetry detection
  F1  — export_cluster_report smoke test (HTML, PNG)
  F2  — plot_consensus_frequency smoke test
  A4  — csr_from_dense now validates symmetry
  A5  — consensus_sweep length mismatch raises ValueError
  A6  — calibrate_llr_threshold n_sims < 1 raises ValueError
  A7  — seq_dist_hamming OOM warning fires for large N
  A8  — _Psort sentinel is now float('inf'), not 1.1
  D1  — seq_dist_gower is no longer accessible from flexscanx.sequence
  B4  — export_cluster_report is accessible from flexscanx root
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

import flexscanx as flx
from flexscanx import scan_series
from flexscanx.graph import csr_from_dense
from flexscanx.sequence import seq_dist_hamming

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_csr(n: int) -> tuple[list[int], list[int]]:
    """Build a line-graph CSR: 0—1—2—…—(n-1)."""
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        nbrs = []
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx


def _chain_coords(n: int) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n)])


# ---------------------------------------------------------------------------
# F3 — warm_start correctness
# ---------------------------------------------------------------------------


class TestWarmStartCorrectness:
    """warm_start=True must produce the same primary-cluster areas as warm_start=False."""

    def _make_series_data(self, n: int = 10, T: int = 4, seed: int = 7):
        rng = np.random.default_rng(seed)
        coords = _chain_coords(n)
        adjacency = _chain_csr(n)
        obs = rng.poisson(5, size=(T, n)).astype(np.int32)
        # Plant a hot spot in area 2-4 for all time slices
        obs[:, 2:5] += 10
        exp = np.full((T, n), 5.0)
        return coords, adjacency, obs, exp

    def test_warm_start_same_areas_as_cold_start(self):
        coords, adjacency, obs, exp = self._make_series_data()
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=obs,
            expected=exp,
            model="POISSON",
            K=8,
            clustersize=5,
            simcount=0,
            seed=42,
            crn=True,
            n_jobs_time=1,
        )
        result_cold = scan_series(**common, warm_start=False)
        result_warm = scan_series(**common, warm_start=True)

        for t in range(obs.shape[0]):
            cold_areas = sorted(result_cold["per_time"][t].get("area", []))
            warm_areas = sorted(result_warm["per_time"][t].get("area", []))
            assert cold_areas == warm_areas, (
                f"Time {t}: warm_start=True gave different areas "
                f"({warm_areas}) than warm_start=False ({cold_areas})"
            )

    def test_warm_start_same_stats(self):
        coords, adjacency, obs, exp = self._make_series_data()
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=obs,
            expected=exp,
            model="POISSON",
            K=8,
            clustersize=5,
            simcount=0,
            seed=42,
            crn=True,
            n_jobs_time=1,
        )
        result_cold = scan_series(**common, warm_start=False)
        result_warm = scan_series(**common, warm_start=True)

        for t in range(obs.shape[0]):
            s_cold = float(result_cold["per_time"][t].get("stats", 0.0))
            s_warm = float(result_warm["per_time"][t].get("stats", 0.0))
            assert (
                abs(s_cold - s_warm) < 1e-10
            ), f"Time {t}: LLR mismatch — cold={s_cold}, warm={s_warm}"


# ---------------------------------------------------------------------------
# F9 / A4 — csr_from_dense symmetry validation
# ---------------------------------------------------------------------------


class TestCsrFromDenseSymmetry:
    def test_symmetric_matrix_accepted(self):
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        row_ptr, col_idx = csr_from_dense(adj)
        assert row_ptr == [0, 1, 3, 4]
        assert col_idx == [1, 0, 2, 1]

    def test_asymmetric_matrix_raises(self):
        adj = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]  # [0][1]=1 but [1][0]=0
        with pytest.raises(ValueError, match="symmetric"):
            csr_from_dense(adj)

    def test_non_square_matrix_raises(self):
        adj = [[0, 1], [1, 0], [0, 1]]
        with pytest.raises(ValueError, match="square"):
            csr_from_dense(adj)

    def test_diagonal_is_ignored(self):
        adj = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]  # diagonal entries present
        row_ptr, col_idx = csr_from_dense(adj)
        # Diagonal entries should NOT appear in col_idx
        assert 0 not in col_idx[row_ptr[0] : row_ptr[1]]
        assert 1 not in col_idx[row_ptr[1] : row_ptr[2]]
        assert 2 not in col_idx[row_ptr[2] : row_ptr[3]]


# ---------------------------------------------------------------------------
# A5 — consensus_sweep input validation
# ---------------------------------------------------------------------------


class TestConsensusSweepValidation:
    def test_mismatched_lengths_raise(self):
        from flexscanx.validation import consensus_sweep

        n = 4
        coords = _chain_coords(n)
        adjacency = _chain_csr(n)
        observed = [1, 2, 3, 4]
        expected_wrong = [1.0, 2.0, 3.0]  # length 3, not 4

        with pytest.raises(ValueError, match="same length"):
            consensus_sweep(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected_wrong,
                clustersize=2,
                grid={"K": [3]},
            )

    def test_empty_observed_raises(self):
        from flexscanx.validation import consensus_sweep

        with pytest.raises(ValueError, match="at least one area"):
            consensus_sweep(
                coords=np.zeros((0, 2)),
                adjacency=([0], []),
                observed=[],
                expected=[],
                clustersize=1,
                grid={},
            )


# ---------------------------------------------------------------------------
# A6 — calibrate_llr_threshold n_sims validation
# ---------------------------------------------------------------------------


class TestCalibrateLLRThresholdValidation:
    def test_n_sims_zero_raises(self):
        from flexscanx.validation import calibrate_llr_threshold_impl

        with pytest.raises(ValueError, match="n_sims must be >= 1"):
            calibrate_llr_threshold_impl(
                observed=[1, 2, 3],
                expected=[1.0, 2.0, 3.0],
                K=2,
                clustersize=2,
                n_sims=0,
                ensure_static_handle=lambda *a, **kw: (object(), 0, None, None),
                cache_load=lambda: {},
                cache_save=lambda d: None,
                cache_schema_version=3,
                cache_format="test",
            )

    def test_n_sims_negative_raises(self):
        from flexscanx.validation import calibrate_llr_threshold_impl

        with pytest.raises(ValueError, match="n_sims must be >= 1"):
            calibrate_llr_threshold_impl(
                observed=[1, 2, 3],
                expected=[1.0, 2.0, 3.0],
                K=2,
                clustersize=2,
                n_sims=-5,
                ensure_static_handle=lambda *a, **kw: (object(), 0, None, None),
                cache_load=lambda: {},
                cache_save=lambda d: None,
                cache_schema_version=3,
                cache_format="test",
            )


# ---------------------------------------------------------------------------
# A7 — seq_dist_hamming OOM warning
# ---------------------------------------------------------------------------


class TestSeqDistHammingOOMWarning:
    def test_large_n_emits_warning(self):
        # 300 × 4 → 300² × 8 = 720 KB, below 500 MB threshold; use 9000 areas conceptually
        # We cannot actually allocate N=9000 in a test; instead patch the threshold
        # by making N large enough to trigger the warning (> 500 MB = N > ~7906 for float64).
        # Use N=8000 to stay memory-safe but trigger the check.
        N = 8001
        T = 2
        # Only allocate a tiny slice; mock the actual sequence computation
        seqs = np.zeros((N, T), dtype=np.int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                seq_dist_hamming(seqs)
            except MemoryError:
                pass  # expected on low-memory machines
        # There should be a UserWarning about memory
        oom_warnings = [
            x for x in w if issubclass(x.category, UserWarning) and "MB" in str(x.message)
        ]
        assert oom_warnings, "Expected OOM UserWarning from seq_dist_hamming for large N"

    def test_condensed_mode_returns_correct_shape(self):
        N = 5  # 5 areas x 3 time columns
        seqs = np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ],
            dtype=np.int32,
        )
        full = seq_dist_hamming(seqs, condensed=False)
        cond = seq_dist_hamming(seqs, condensed=True)
        assert full.shape == (N, N)
        assert cond.shape == (N * (N - 1) // 2,)
        # Verify values match: upper triangle row-major
        ptr = 0
        for i in range(N):
            for j in range(i + 1, N):
                assert abs(full[i, j] - cond[ptr]) < 1e-12, f"Mismatch at ({i},{j})"
                ptr += 1


# ---------------------------------------------------------------------------
# D1 — seq_dist_gower removed from public API
# ---------------------------------------------------------------------------


class TestSeqDistGowerRemoved:
    def test_not_in_sequence_namespace(self):
        import flexscanx.sequence as seq

        assert not hasattr(
            seq, "seq_dist_gower"
        ), "seq_dist_gower should have been removed from flexscanx.sequence"

    def test_not_in_root_namespace(self):
        assert not hasattr(flx, "seq_dist_gower")


# ---------------------------------------------------------------------------
# B4 — export_cluster_report accessible at root
# ---------------------------------------------------------------------------


class TestExportClusterReportAccessible:
    def test_accessible_from_root(self):
        assert hasattr(flx, "export_cluster_report")
        assert callable(flx.export_cluster_report)

    def test_in_all(self):
        assert "export_cluster_report" in flx.__all__


# ---------------------------------------------------------------------------
# F1 — export_cluster_report smoke test (requires matplotlib + geopandas)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("geopandas") is None or importlib.util.find_spec("matplotlib") is None,
    reason="requires the optional geopandas and matplotlib dependencies",
)
class TestExportClusterReportSmoke:
    """Smoke tests for export_cluster_report when optional dependencies exist."""

    def _make_synthetic_scan_result(self):
        return {
            "type": "HOT",
            "area": [1, 2],
            "n_case": 10.0,
            "expected": 5.0,
            "stats": 3.5,
            "rank": 1,
            "pval": 0.01,
            "max_dist": 1.0,
            "from_area": 1,
            "to_area": 2,
            "Rate_Ratio": 2.0,
            "Rate_Ratio_CI": None,
            "scan_settings": {},
            "secondaries": [],
        }

    def test_html_output(self, tmp_path):
        import geopandas as gpd
        import shapely.geometry as sg

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                sg.Point(0, 0).buffer(0.5),
                sg.Point(1, 0).buffer(0.5),
                sg.Point(2, 0).buffer(0.5),
            ],
            crs="EPSG:3857",
        )
        result = self._make_synthetic_scan_result()
        out = tmp_path / "report.html"
        flx.export_cluster_report(gdf, result, id_col="id", path=str(out))
        assert out.exists()
        assert out.stat().st_size > 100
        content = out.read_text(encoding="utf-8")
        assert "<img" in content

    def test_png_output(self, tmp_path):
        import geopandas as gpd
        import shapely.geometry as sg

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[
                sg.Point(0, 0).buffer(0.5),
                sg.Point(1, 0).buffer(0.5),
                sg.Point(2, 0).buffer(0.5),
            ],
            crs="EPSG:3857",
        )
        result = self._make_synthetic_scan_result()
        out = tmp_path / "report.png"
        flx.export_cluster_report(gdf, result, id_col="id", path=str(out))
        assert out.exists()
        assert out.stat().st_size > 100


# ---------------------------------------------------------------------------
# F2 — plot_consensus_frequency smoke test
# ---------------------------------------------------------------------------


class TestPlotConsensusSmokeNoGeopandas:
    def test_raises_importerror_without_matplotlib_when_absent(self, monkeypatch):
        """If matplotlib is absent the function raises ImportError (not AttributeError)."""
        import sys

        # Temporarily hide matplotlib from sys.modules
        saved = sys.modules.pop("matplotlib", None)
        saved_pyplot = sys.modules.pop("matplotlib.pyplot", None)
        saved_colors = sys.modules.pop("matplotlib.colors", None)
        # Make import fail
        sys.modules["matplotlib"] = None  # type: ignore
        sys.modules["matplotlib.pyplot"] = None  # type: ignore

        try:
            with pytest.raises(ImportError):
                flx.plot_consensus_frequency(
                    None,
                    [0.1, 0.5, 0.9],
                    id_col="id",
                )
        finally:
            if saved is not None:
                sys.modules["matplotlib"] = saved
            else:
                sys.modules.pop("matplotlib", None)
            if saved_pyplot is not None:
                sys.modules["matplotlib.pyplot"] = saved_pyplot
            else:
                sys.modules.pop("matplotlib.pyplot", None)
            if saved_colors is not None:
                sys.modules["matplotlib.colors"] = saved_colors
            else:
                sys.modules.pop("matplotlib.colors", None)


# ---------------------------------------------------------------------------
# A8 — _Psort sentinel is now float('inf')
# ---------------------------------------------------------------------------


class TestPsortSentinel:
    def test_nan_pval_sorts_last(self):
        """Clusters with NaN p-values should sort after all real p-values."""
        from flexscanx.summary import summarize_result

        # Two clusters: one with pval=0.9 (large but finite), one with no MC (pval=None).
        # The finite-pval cluster should sort before (or equal to) the NaN one at the same LLR.
        result = {
            "type": "HOT",
            "area": [1],
            "n_case": 5.0,
            "expected": 3.0,
            "stats": 2.0,
            "rank": 1,
            "pval": None,
            "max_dist": 0.5,
            "scan_settings": {},
            "secondaries": [
                {
                    "type": "HOT",
                    "area": [2],
                    "n_case": 4.0,
                    "expected": 3.0,
                    "stats": 2.0,
                    "rank": 2,
                    "pval": 0.9,
                    "max_dist": 0.3,
                }
            ],
        }
        df = summarize_result(result)
        # The row with pval=0.9 (smaller sentinel) should appear before pval=NaN (inf sentinel).
        pvals = df["P"].tolist()
        # Find positions of NaN and 0.9
        nan_pos = next(i for i, p in enumerate(pvals) if (p != p))  # p != p is True for NaN
        val_pos = next(i for i, p in enumerate(pvals) if p == pytest.approx(0.9))
        assert val_pos < nan_pos, "Cluster with p=0.9 should sort before cluster with p=NaN"


# ---------------------------------------------------------------------------
# B1 — validation.py facade re-exports all expected symbols
# ---------------------------------------------------------------------------


class TestValidationFacade:
    def test_public_functions_accessible(self):
        from flexscanx import validation

        expected = [
            "calibrate_llr_threshold",
            "silhouette_graph",
            "dunn_graph",
            "join_counts",
            "boundary_contrast",
            "morans_I_within",
            "conductance",
            "cv_glm_gain",
            "consensus_sweep",
            "consensus_to_series",
            "trend_level_slope",
            "stability_across_K",
        ]
        for name in expected:
            assert hasattr(validation, name), f"validation.{name} missing after B1 split"

    def test_internal_cache_symbols_accessible(self):
        """Test-facing internals must still be reachable via flexscanx.validation."""
        from flexscanx import validation

        internals = [
            "_CACHE_FILE",
            "_CACHE_FORMAT",
            "_CACHE_SCHEMA_VERSION",
            "_cache_load",
            "_cache_save",
            "_ensure_static_handle",
        ]
        for name in internals:
            assert hasattr(validation, name), f"validation.{name} missing after B1 split"

    def test_functions_are_callable(self):
        from flexscanx import validation

        assert callable(validation.calibrate_llr_threshold)
        assert callable(validation.consensus_sweep)
        assert callable(validation.cv_glm_gain)
