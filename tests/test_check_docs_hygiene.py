from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_docs_hygiene.py"


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "README.md", "# repo\n")
    _write(repo / "API_REFERENCE.md", "# api\n")
    _write(repo / "CHANGELOG.md", "# changelog\n")
    _write(
        repo / "DOCUMENTATION_INDEX.md",
        "# docs\n\n**Last Updated:** 2026-03-23\n\n"
        "- `DOCUMENTATION/README.md`\n"
        "- `RFLEXSCAN_ORIGINAL/README.md`\n"
        "- `archive/generated/reference_extracts/README.md`\n"
        "- `archive/generated/raw_dumps/README.md`\n"
        "- `archive/internal_notes/README.md`\n",
    )
    _write(repo / "LICENSE", "MIT License\n")
    _write(
        repo / "docs" / "README.md",
        "# docs readme\n\n**Last Updated:** 2026-03-23\n\n"
        "- [../DOCUMENTATION/README.md](../DOCUMENTATION/README.md)\n"
        "- [../RFLEXSCAN_ORIGINAL/README.md](../RFLEXSCAN_ORIGINAL/README.md)\n"
        "- [../archive/generated/reference_extracts/README.md]"
        "(../archive/generated/reference_extracts/README.md)\n"
        "- [../archive/generated/raw_dumps/README.md](../archive/generated/raw_dumps/README.md)\n"
        "- [../archive/internal_notes/README.md](../archive/internal_notes/README.md)\n",
    )
    _write(
        repo / "DOCUMENTATION" / "README.md",
        "# references\n\nThese are NOT tutorials or user guides.\n"
        "Active documentation should link to this `README.md` manifest.\n",
    )
    _write(
        repo / "DOCUMENTATION" / "FLEXSCANX_Python_API_Complete_Reference.qmd",
        "---\ntitle: ref\n---\n",
    )
    _write(
        repo / "vignettes" / "README.md",
        "# vignettes\n\n**Last Updated:** 2026-03-23\n\n"
        "## User Workflows\n\n## Method Notes and Validation Memos\n\n"
        "- `../RFLEXSCAN_ORIGINAL/README.md`\n",
    )
    _write(repo / "scripts" / "README.md", "# scripts\n\n**Last Updated:** 2026-03-23\n")
    _write(repo / "src" / "flexscanx_core" / "DEVELOPER.md", "# cxx dev\n")
    _write(repo / "src" / "flexscanx" / "sequence" / "NOTICE.txt", "NOTICES\n")
    _write(
        repo / "RFLEXSCAN_ORIGINAL" / "README.md",
        "# upstream reference\n\nReference materials.\n"
        "Active documentation should link to this `README.md` manifest.\n",
    )
    _write(repo / "archive" / "generated" / "README.md", "# generated\n")
    _write(
        repo / "archive" / "generated" / "reference_extracts" / "README.md",
        "# extracts\n\nnot part of the active documentation surface.\n"
        "If an active document needs to mention this area, it should link to this README.\n",
    )
    _write(
        repo / "archive" / "generated" / "raw_dumps" / "README.md",
        "# raw dumps\n\nshould not be treated as curated documentation.\n"
        "If an active document needs to mention this area, it should link to this README.\n",
    )
    _write(
        repo / "archive" / "internal_notes" / "README.md",
        "# internal\n\nnot part of user-facing docs.\n"
        "Active documentation should link to this `README.md` manifest.\n",
    )
    _write(
        repo / "archive" / "reports" / "2026" / "README.md",
        "# reports\n\n## Current High-Signal Report\n\n"
        "- `20260322T175938+0900/PROJECT_FLEXSCANX_AUDIT_REPORT.md`\n\n"
        "## Superseded Material\n\n- `superseded/`\n",
    )
    _write(
        repo
        / "archive"
        / "reports"
        / "2026"
        / "20260322T175938+0900"
        / "PROJECT_FLEXSCANX_AUDIT_REPORT.md",
        "# audit\n",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=flexscanx-tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def test_check_docs_hygiene_passes_for_clean_repo(tmp_path):
    repo = _init_repo(tmp_path)
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Documentation hygiene check passed." in proc.stdout


def test_check_docs_hygiene_fails_for_placeholder_upstream_url(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "src" / "flexscanx" / "sequence" / "NOTICE.txt",
        "See https://github.com/your-org/traminex for details.\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "github\\.com/your-org/" in proc.stdout


def test_check_docs_hygiene_rejects_an_untracked_link_target(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "assets" / "untracked.pdf", "scratch\n")
    _write(repo / "README.md", "# repo\n\n[Scratch](assets/untracked.pdf)\n")

    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)

    assert proc.returncode == 1
    assert "assets/untracked.pdf" in proc.stdout


def test_check_docs_hygiene_fails_for_dynamic_quarto_date(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "vignettes" / "workflows" / "workflow.qmd",
        "---\ntitle: workflow\ndate: today\n---\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "date:\\s*today" in proc.stdout


def test_check_docs_hygiene_fails_for_missing_last_updated_metadata(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "scripts" / "README.md", "# scripts\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "maintained markdown docs must include" in proc.stdout


def test_check_docs_hygiene_fails_for_missing_vignette_date(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "vignettes" / "workflows" / "workflow.qmd",
        "---\ntitle: workflow\n---\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "vignette qmd files must include a fixed front-matter date" in proc.stdout


def test_check_docs_hygiene_fails_for_root_level_vignette_qmd(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "vignettes" / "workflow.qmd",
        "---\ntitle: workflow\ndate: 2026-03-23\n---\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "move vignette qmd files under vignettes/workflows/ or vignettes/notes/" in proc.stdout


def test_check_docs_hygiene_fails_for_direct_reference_only_link_in_entrypoint_doc(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "vignettes" / "README.md",
        "# vignettes\n\n**Last Updated:** 2026-03-23\n\n"
        "## User Workflows\n\n## Method Notes and Validation Memos\n\n"
        "- `../RFLEXSCAN_ORIGINAL/rflexscan_examples_reference.md`\n",
    )
    _write(repo / "RFLEXSCAN_ORIGINAL" / "rflexscan_examples_reference.md", "# raw examples\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "should go through the manifest 'RFLEXSCAN_ORIGINAL/README.md'" in proc.stdout


def test_check_docs_hygiene_fails_when_vignette_index_loses_split_headings(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "vignettes" / "README.md", "# vignettes\n\n**Last Updated:** 2026-03-23\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "missing vignette-index heading" in proc.stdout


def test_check_docs_hygiene_fails_when_latest_audit_is_missing_from_manifest(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo
        / "archive"
        / "reports"
        / "2026"
        / "20260323T010203+0900"
        / "PROJECT_FLEXSCANX_AUDIT_REPORT.md",
        "# newer audit\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "latest timestamped audit" in proc.stdout
    assert "20260323T010203+0900/PROJECT_FLEXSCANX_AUDIT_REPORT.md" in proc.stdout


def test_check_docs_hygiene_ignores_superseded_timestamped_audits_for_latest_manifest(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo
        / "archive"
        / "reports"
        / "2026"
        / "superseded"
        / "20260324T010203+0900"
        / "PROJECT_FLEXSCANX_AUDIT_REPORT.md",
        "# superseded newer audit\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_check_docs_hygiene_fails_for_loose_year_report_file(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "archive" / "reports" / "2026" / "AUDIT_REPORT_20260209_040645.md", "# audit\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert (
        "move loose historical report files under archive/reports/2026/superseded/" in proc.stdout
    )


def test_check_docs_hygiene_fails_when_year_report_manifest_loses_status_labels(tmp_path):
    repo = _init_repo(tmp_path)
    _write(
        repo / "archive" / "reports" / "2026" / "README.md",
        "# reports\n- `20260322T175938+0900/PROJECT_FLEXSCANX_AUDIT_REPORT.md`\n",
    )
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "missing status heading" in proc.stdout


def test_check_docs_hygiene_fails_for_loose_development_audit_file(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "archive" / "development_audits" / "README.md", "# audits\n")
    _write(repo / "archive" / "development_audits" / "COMPREHENSIVE_AUDIT_REPORT.md", "# audit\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "move historical audit files into a category folder" in proc.stdout


def test_check_docs_hygiene_fails_when_reference_manifest_lacks_boundary_text(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "archive" / "generated" / "raw_dumps" / "README.md", "# raw dumps\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "missing required manifest guidance text" in proc.stdout


def test_check_docs_hygiene_fails_for_legacy_markdown_vignette(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "vignettes" / "legacy_examples.md", "# legacy\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "move legacy markdown reference files out of vignettes/" in proc.stdout
