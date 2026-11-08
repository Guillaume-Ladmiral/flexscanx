"""Regression contracts for the 2026-08-16 docs, scripts, and Quarto remediation."""

from __future__ import annotations

import re
from pathlib import Path

import flexscanx as flx

ROOT = Path(__file__).resolve().parents[1]


def test_api_reference_covers_all_root_exports_and_separates_both_results():
    api_reference = (ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")

    undocumented = [name for name in flx.__all__ if name not in api_reference]
    assert undocumented == []
    assert "### BothResult" in api_reference
    assert "### `summarize_result()`" in api_reference
    scan_result = api_reference.split("### ScanResult", 1)[1].split("### BothResult", 1)[0]
    assert "'BOTH'" not in scan_result


def test_api_reference_explains_radius_haversine_and_both_secondary_null_contracts():
    api_reference = (ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")

    assert "centre-radius" in api_reference
    assert "do not pass those longitude/latitude coordinates to `scan(coords=...)`" in api_reference
    assert "BOTH envelope primary" in api_reference
    assert "combined full-region two-sided null" in api_reference


def test_changelog_marks_the_mypy_scope_and_visit_once_claims_as_corrected():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "mypy over the release-sanity contract set" in changelog
    assert "Superseded by the 2026-06-17 ENUM-2 correction" in changelog


def test_docs_signature_gate_tracks_every_documented_signature_and_accepts_whitespace():
    script = (ROOT / "scripts" / "check_docs_signatures.py").read_text(encoding="utf-8")

    assert r"\)\s*->" in script
    assert "DOCUMENTED_SIGNATURES" in script
    assert "_normalise_default" in script
    assert "unregistered documented signature" in script
    assert 'finder.__class__.__module__ == "_flexscanx_editable"' in script


def test_docs_hygiene_uses_tracked_files_without_basename_masking():
    script = (ROOT / "scripts" / "check_docs_hygiene.py").read_text(encoding="utf-8")

    assert '"ls-files"' in script
    assert "return any(p.as_posix().endswith" not in script
    assert "return rel_ref.name in files_by_name" not in script


def test_metadata_drift_is_aggregated_without_a_traceback():
    script = (ROOT / "scripts" / "sync_docs_metadata.py").read_text(encoding="utf-8")

    assert "metadata drift detected" in script
    assert "drift_errors" in script
    assert 'raise ValueError(f"{path}: metadata drift' not in script


def test_heavy_rendering_and_note_execution_contracts_are_honest():
    heavy = (ROOT / "vignettes" / "workflows" / "Manhattan_Breast_Cancer.qmd").read_text(
        encoding="utf-8"
    )
    note = (
        ROOT / "vignettes" / "notes" / "HOT_ORIGINAL_Parity_Otani_Takahashi_2021.qmd"
    ).read_text(encoding="utf-8")
    renderer = (ROOT / "scripts" / "render_all.py").read_text(encoding="utf-8")

    assert "eval: false" not in heavy
    assert "--execute" in renderer
    assert "enabled: false" in note
    assert "nonpositive EBREAST rows" in heavy
    assert 'gdf["EBREAST"] > 0' in heavy
    assert "MANHATTAN_SMOKE_K" in heavy
    assert "MANHATTAN_SMOKE_CLUSTERSIZE" in heavy
    assert 'EXEC_SECONDARY = "ALL" if FULL_ANALYSIS else None' in heavy
    assert "secondary=EXEC_SECONDARY" in heavy


def test_script_cli_and_import_safety_contracts_are_explicit():
    for relative_path in (
        "scripts/perf_checks.py",
        "scripts/quickcheck_ot.py",
        "scripts/export_report.py",
    ):
        assert "SystemExit" in (ROOT / relative_path).read_text(encoding="utf-8")

    for relative_path in (
        "scripts/build_manhattan_breastcancer.py",
        "scripts/build_manhattan_breastcancer_light.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "def main(" in text
        assert "--output" in text
        assert 'if __name__ == "__main__"' in text


def test_platform_neutral_powershell_contract_and_small_script_fixes_are_present():
    powershell = (ROOT / "scripts" / "render_all.ps1").read_text(encoding="utf-8")
    smoke_tests = (ROOT / "tests" / "test_scripts_smoke.py").read_text(encoding="utf-8")
    graph_query = (ROOT / "scripts" / "graph_query.py").read_text(encoding="utf-8")
    quickcheck = (ROOT / "scripts" / "quickcheck_ot.py").read_text(encoding="utf-8")
    driver = (ROOT / "scripts" / "test_manhattan_scan.py").read_text(encoding="utf-8")
    example = (ROOT / "examples" / "explore_flexscanx.py").read_text(encoding="utf-8")

    assert "[IO.Path]::GetFullPath" in powershell
    assert "Path(proc.stdout.strip()).resolve() == Path(sys.executable).resolve()" in smoke_tests
    assert 'encoding="utf-8"' in graph_query
    assert 'knn_method="geometry"' not in quickcheck
    assert "__test__ = False" in driver
    assert re.search(r"clustersize\s*=\s*4", example)


def test_all_manhattan_workflows_filter_nonpositive_poisson_exposure():
    for filename in (
        "contiguity_queen_vs_rook.qmd",
        "restricted_llr_guidance.qmd",
        "stable_significant_workflow.qmd",
    ):
        text = (ROOT / "vignettes" / "workflows" / filename).read_text(encoding="utf-8")
        assert "np.isfinite(expected) & (expected > 0.0)" in text
        assert "gdf = gdf.loc[valid_expected].copy()" in text
