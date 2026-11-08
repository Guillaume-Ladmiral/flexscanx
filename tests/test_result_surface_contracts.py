from __future__ import annotations

import re
from pathlib import Path
from typing import get_type_hints

import numpy as np

import flexscanx as flx
import flexscanx._core as core
from flexscanx.types import (
    BothResultBase,
    ScanResultBase,
    ScanResultDict,
    SeriesEntryBothResultBase,
    SeriesEntryResultBase,
    SeriesEntryResultDict,
    SeriesResultBase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _typed_dict_keys(*typed_dicts: type) -> set[str]:
    keys: set[str] = set()
    for typed_dict in typed_dicts:
        keys.update(getattr(typed_dict, "__required_keys__", set()))
        keys.update(getattr(typed_dict, "__optional_keys__", set()))
    return keys


def _extract_scan_result_doc_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    heading = "### ScanResult"
    start = text.index(heading)
    code_start = text.index("```python", start)
    code_end = text.index("```", code_start + len("```python"))
    block = text[code_start:code_end]
    return set(re.findall(r"'([A-Za-z_]+)'\s*:", block))


def _extract_series_result_doc_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    heading = "### SeriesResult"
    start = text.index(heading)
    code_start = text.index("```python", start)
    code_end = text.index("```", code_start + len("```python"))
    block = text[code_start:code_end]
    return set(re.findall(r"'([A-Za-z_]+)'\s*:", block))


def _extract_stub_typed_dict_keys(path: Path, class_name: str) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    inside = False
    keys: set[str] = set()
    for line in lines:
        if line.startswith(f"class {class_name}("):
            inside = True
            continue
        if inside and line.startswith("class "):
            break
        if inside and line.startswith("#"):
            break
        if inside:
            match = re.match(r"\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
            if match:
                keys.add(match.group(1))
    return keys


DOC_SCAN_RESULT_KEYS = _extract_scan_result_doc_keys(REPO_ROOT / "API_REFERENCE.md")
DOC_SERIES_RESULT_KEYS = _extract_series_result_doc_keys(REPO_ROOT / "API_REFERENCE.md")
TYPED_SCAN_RESULT_KEYS = _typed_dict_keys(ScanResultBase, ScanResultDict)
REQUIRED_SCAN_RESULT_KEYS = set(getattr(ScanResultBase, "__required_keys__", set()))
TYPED_SERIES_RESULT_KEYS = _typed_dict_keys(SeriesResultBase)
TYPED_SERIES_ENTRY_KEYS = _typed_dict_keys(SeriesEntryResultBase, SeriesEntryResultDict)
TYPED_BOTH_RESULT_KEYS = _typed_dict_keys(BothResultBase)
TYPED_SERIES_BOTH_ENTRY_KEYS = _typed_dict_keys(SeriesEntryBothResultBase)
CORE_SCAN_RESULT_KEYS = _extract_stub_typed_dict_keys(
    REPO_ROOT / "src/flexscanx/_core.pyi", "CoreScanClusterBase"
) | _extract_stub_typed_dict_keys(REPO_ROOT / "src/flexscanx/_core.pyi", "CoreScanClusterDict")


def _chain_fixture() -> tuple[np.ndarray, list[int], list[int]]:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    return coords, row_ptr, col_idx


def _poisson_runtime_results() -> tuple[dict, dict]:
    coords, row_ptr, col_idx = _chain_fixture()
    observed = [10, 6, 1, 1]
    expected = [5.0, 5.0, 5.0, 5.0]

    handle = core.BuildStatic(coords, row_ptr, col_idx, 3, float("inf"))
    raw = core.ScanSlice(handle, observed, expected, {"ralpha": 0.2, "clustersize": 3})
    public = flx.scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        expected=expected,
        K=3,
        simcount=0,
    )
    return raw, public


def _binomial_runtime_results() -> tuple[dict, dict]:
    coords, row_ptr, col_idx = _chain_fixture()
    observed = [8, 4, 1, 1]
    population = [10.0, 10.0, 10.0, 10.0]

    handle = core.BuildStatic(coords, row_ptr, col_idx, 3, float("inf"))
    raw = core.ScanSlice(
        handle, observed, population, {"model": "BINOMIAL", "ralpha": 0.2, "clustersize": 3}
    )
    public = flx.scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        population=population,
        model="BINOMIAL",
        K=3,
        simcount=0,
    )
    return raw, public


def _series_runtime_results() -> tuple[dict, dict]:
    coords, row_ptr, col_idx = _chain_fixture()
    observed = np.asarray([[10, 6, 1, 1], [9, 5, 1, 1]], dtype=int)
    expected = np.asarray([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]], dtype=float)

    public_hot = flx.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=(row_ptr, col_idx),
        K=3,
        simcount=0,
    )
    public_both = flx.scan_series(
        observed=observed,
        expected=expected,
        coords=coords,
        adjacency=(row_ptr, col_idx),
        clustertype="BOTH",
        K=3,
        simcount=0,
    )
    return public_hot, public_both


def test_scan_result_docs_and_types_cover_runtime_poisson_keys():
    _raw, public = _poisson_runtime_results()

    assert set(public) <= DOC_SCAN_RESULT_KEYS
    assert set(public) <= TYPED_SCAN_RESULT_KEYS
    assert REQUIRED_SCAN_RESULT_KEYS <= set(public)
    assert {"Rate_Ratio", "Rate_Ratio_CI"} <= set(public)


def test_scan_result_docs_and_types_cover_runtime_binomial_keys():
    _raw, public = _binomial_runtime_results()

    assert set(public) <= DOC_SCAN_RESULT_KEYS
    assert set(public) <= TYPED_SCAN_RESULT_KEYS
    assert REQUIRED_SCAN_RESULT_KEYS <= set(public)
    assert {"Risk_Ratio", "Risk_Ratio_CI"} <= set(public)


def test_core_stub_covers_runtime_poisson_result_keys():
    raw, _public = _poisson_runtime_results()

    assert set(raw) <= CORE_SCAN_RESULT_KEYS
    assert {"Rate_Ratio", "Rate_Ratio_CI"} <= set(raw)


def test_core_stub_covers_runtime_binomial_result_keys():
    raw, _public = _binomial_runtime_results()

    assert set(raw) <= CORE_SCAN_RESULT_KEYS
    assert {"Risk_Ratio", "Risk_Ratio_CI"} <= set(raw)


def test_series_result_docs_and_types_cover_runtime_keys():
    public_hot, _public_both = _series_runtime_results()

    assert set(public_hot) <= DOC_SERIES_RESULT_KEYS
    assert set(public_hot) <= TYPED_SERIES_RESULT_KEYS
    first = public_hot["per_time"][0]
    assert "scan_settings" not in first
    assert set(first) <= TYPED_SERIES_ENTRY_KEYS


def test_series_both_entries_use_series_entry_contract_without_scan_settings():
    _public_hot, public_both = _series_runtime_results()

    first = public_both["per_time"][0]
    assert set(first) <= TYPED_SERIES_BOTH_ENTRY_KEYS
    assert "scan_settings" not in first
    assert "scan_settings" not in first["primary"]
    assert "scan_settings" not in first["hot"]
    assert "scan_settings" not in first["cold"]


def test_public_type_hints_capture_optional_ratios_and_series_entry_split():
    assert get_type_hints(ScanResultDict)["Rate_Ratio"] == (float | None)
    assert get_type_hints(ScanResultDict)["Risk_Ratio"] == (float | None)
    assert get_type_hints(SeriesEntryResultDict)["Rate_Ratio"] == (float | None)
    assert get_type_hints(SeriesEntryResultDict)["Risk_Ratio"] == (float | None)
    assert "scan_settings" in getattr(ScanResultBase, "__required_keys__", set())
    assert "scan_settings" in getattr(BothResultBase, "__required_keys__", set())
    assert "scan_settings" not in getattr(SeriesEntryResultBase, "__required_keys__", set())
    assert "scan_settings" not in getattr(SeriesEntryBothResultBase, "__required_keys__", set())


def test_secondary_cluster_keys_are_declared_in_the_cluster_type_and_docs():
    """FXA-167: every key on an enumerated secondary is declared (including simcount_actual)."""
    from flexscanx.types import ScanClusterDict

    coords, row_ptr, col_idx = _chain_fixture()
    public = flx.scan(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=[10, 1, 1, 9],
        expected=[5.0, 5.0, 5.0, 5.0],
        K=3,
        simcount=19,
        seed=3,
        secondary="ALL",
    )
    declared = _typed_dict_keys(ScanClusterDict)
    assert public["secondaries"], "fixture must enumerate at least one secondary"
    for secondary in public["secondaries"]:
        assert set(secondary) <= declared, sorted(set(secondary) - declared)
        assert "simcount_actual" in secondary
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "carries its own `simcount_actual`" in text
