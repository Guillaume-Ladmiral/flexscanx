"""Tests of this run's fixes that read the repository's development-only trees (``scripts/``),
which the source distribution intentionally omits.  ``tests/conftest.py`` declines this module
when ``scripts/`` is absent, exactly as it does for the other tooling test modules (FXA-170), so
the suite shipped in the sdist stays runnable.

Covered here: FXA-193 (signature checker validates derived-default comments), FXA-174 (the two
hand-dated documents are in the metadata-managed set) and U-1 (``scripts/render_all.ps1``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# FXA-193: the signature checker validates the derived-default comments too
# ---------------------------------------------------------------------------

_CHECKER_DRIVER = (
    "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]);"
    " import check_docs_signatures as c;"
    " c.DOC_SIGNATURES = {Path(sys.argv[1]): c.DOCUMENTED_SIGNATURES}; sys.exit(c.main())"
)


def _run_signature_checker_on(doc_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHECKER_DRIVER, str(doc_path), str(REPO_ROOT / "scripts")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )


_S_MIN_COMMENT = "S_min=None,              # Default: max(1, min(5, round(6% of N)))"


def test_signature_checker_accepts_the_committed_reference(tmp_path: Path) -> None:
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert _S_MIN_COMMENT in text
    copy = tmp_path / "API_REFERENCE.md"
    copy.write_text(text, encoding="utf-8")
    result = _run_signature_checker_on(copy)
    assert result.returncode == 0, result.stdout + result.stderr


def test_signature_checker_rejects_a_stale_derived_default_comment(tmp_path: Path) -> None:
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    stale = text.replace(_S_MIN_COMMENT, _S_MIN_COMMENT.replace("6% of N", "12% of N"))
    assert stale != text
    copy = tmp_path / "API_REFERENCE.md"
    copy.write_text(stale, encoding="utf-8")
    result = _run_signature_checker_on(copy)
    assert result.returncode != 0, "a stale derived default must fail the docs gate"
    assert "S_min" in result.stdout and "derived default" in result.stdout, result.stdout


def test_signature_checker_derives_the_expected_comments_from_the_tuning_constants() -> None:
    from flexscanx import tuning

    source = (REPO_ROOT / "scripts" / "check_docs_signatures.py").read_text(encoding="utf-8")
    assert "DERIVED_DEFAULT_COMMENTS" in source
    # The registry must be rendered from the live constants, not typed by hand.
    for constant in ("_DEFAULT_S_MIN_FRAC", "_DEFAULT_K_MIN", "_DEFAULT_K_CAP", "_DEFAULT_K_FLOOR"):
        assert constant in source, constant
        assert hasattr(tuning, constant)


# ---------------------------------------------------------------------------
# FXA-174 / D9: the two hand-dated documents join the metadata-managed set
# ---------------------------------------------------------------------------

_MANAGED_MARKER = "<!-- BEGIN GENERATED: maintained_last_updated -->"


@pytest.mark.parametrize(
    "relative", ["docs/MIGRATION_FROM_RFLEXSCAN.md", "docs/TRAMINEX_SEQUENCE_PROVENANCE.md"]
)
def test_hand_dated_documents_are_metadata_managed(relative: str) -> None:
    import tomllib

    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert _MANAGED_MARKER in text, f"{relative} lacks the managed last-updated block"
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    last_updated = meta["tool"]["flexscanx"]["docs"]["last_updated"]
    assert f"**Last Updated:** {last_updated}" in text, relative
    script = (REPO_ROOT / "scripts" / "sync_docs_metadata.py").read_text(encoding="utf-8")
    assert Path(relative).name in script, f"{relative} is not in sync_docs_metadata.py's targets"


# ---------------------------------------------------------------------------
# U-1: render_all.ps1 classifies notes by directory boundary, not by name prefix
# ---------------------------------------------------------------------------


RENDER_PS1 = REPO_ROOT / "scripts" / "render_all.ps1"
POWERSHELL_IMAGE = "mcr.microsoft.com/powershell:latest"


def _classifier_lines() -> list[str]:
    """The script's own note-directory and classifier statements, verbatim."""
    text = RENDER_PS1.read_text(encoding="utf-8")
    wanted = ("$noteDir = ", "$noteDirPrefix = ", "$shouldExecute = ")
    return [line.strip() for line in text.splitlines() if line.strip().startswith(wanted)]


def test_render_all_ps1_compares_against_the_directory_separator() -> None:
    text = RENDER_PS1.read_text(encoding="utf-8")
    assert "$noteDirPrefix = $noteDir.TrimEnd(" in text
    assert "+ [IO.Path]::DirectorySeparatorChar" in text
    assert "StartsWith($noteDirPrefix," in text
    assert "StartsWith($noteDir," not in text
    assert "\r\n" in RENDER_PS1.read_bytes().decode("utf-8")  # CRLF policy of .gitattributes


def _docker_powershell_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", POWERSHELL_IMAGE], capture_output=True, text=True, timeout=60
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    not _docker_powershell_available(),
    reason=f"Docker image {POWERSHELL_IMAGE} not available locally; Linux PowerShell probe skipped",
)
def test_render_all_ps1_classifier_ignores_a_sibling_sharing_the_notes_prefix(tmp_path) -> None:
    """Runs the script's exact classifier statements under Linux PowerShell (Docker).

    Windows-native path semantics (drive letters, UNC paths, backslash
    normalisation) are not exercised here and remain untested on this platform.
    """
    repo = tmp_path / "repo"
    for sub in ("vignettes/notes", "vignettes/notes_extra", "vignettes/workflows"):
        (repo / sub).mkdir(parents=True)
    lines = _classifier_lines()
    note_dir = [line for line in lines if line.startswith("$noteDir = ")]
    prefix = [line for line in lines if line.startswith("$noteDirPrefix = ")]
    classifier = [line for line in lines if line.startswith("$shouldExecute = ")]
    assert note_dir and prefix and classifier
    probe = "\n".join(
        [
            '$repoRoot = "/work/repo"',
            *note_dir,
            *prefix,
            "foreach ($f in @('/work/repo/vignettes/notes/a.qmd', "
            "'/work/repo/vignettes/notes_extra/b.qmd', '/work/repo/vignettes/workflows/c.qmd')) {",
            "  $normalizedFile = [IO.Path]::GetFullPath($f)",
            *classifier,
            '  Write-Output ("{0}={1}" -f $f, $shouldExecute)',
            "}",
        ]
    )
    (tmp_path / "probe.ps1").write_text(probe, encoding="utf-8")
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmp_path}:/work",
            POWERSHELL_IMAGE,
            "pwsh",
            "-NoProfile",
            "-File",
            "/work/probe.ps1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    verdict = dict(line.rsplit("=", 1) for line in proc.stdout.strip().splitlines())
    assert verdict["/work/repo/vignettes/notes/a.qmd"] == "False"
    assert verdict["/work/repo/vignettes/notes_extra/b.qmd"] == "True"
    assert verdict["/work/repo/vignettes/workflows/c.qmd"] == "True"
