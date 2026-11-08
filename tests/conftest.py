import pathlib
import sys

import pytest

pytest_plugins = ["tests.traminex_parity_utils"]

# Ensure installed flexscanx package is used instead of the source tree.
# This is needed when running tests with the source tree on sys.path
# (e.g. editable installs). For a cleaner approach, use pytest --import-mode=importlib.
_repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p and pathlib.Path(p).resolve() != _repo_root]

# Source distributions intentionally omit development-only scripts and rendered
# vignettes.  Keep the product tests shipped, while declining only tests that
# require those omitted trees when pytest runs from an extracted sdist.
collect_ignore: list[str] = []
if not (_repo_root / "scripts").is_dir():
    collect_ignore.extend(
        [
            "test_check_docs_hygiene.py",
            "test_check_docs_signatures.py",
            "test_check_release_tree.py",
            "test_render_all.py",
            "test_repository_guidance.py",
            "test_scripts_smoke.py",
            "test_sync_docs_metadata.py",
            "test_verify_cleanroom.py",
            "test_verify_release_artifacts.py",
            "test_whole_codebase_audit_20260801.py",
            "test_remediation_packaging_20260816.py",
            # Reads scripts/*.py and vignettes/**/*.qmd, neither of which ships (FXA-170).
            "test_remediation_docs_scripts_20260816.py",
            # Reads scripts/check_docs_signatures.py, scripts/sync_docs_metadata.py and
            # scripts/render_all.ps1 (2026-09-07 run: FXA-193, FXA-174, U-1).
            "test_fix20260907_scripts.py",
        ]
    )
if not (_repo_root / "vignettes").is_dir():
    collect_ignore.extend(["test_rflexscan_manual_vignette.py", "test_scan_manhattan.py"])

# Use a non-interactive backend for Matplotlib in CI/headless environments
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except ImportError:
    pass


@pytest.fixture(autouse=True)
def isolate_calibration_cache(monkeypatch, tmp_path):
    """Keep every test independent of persistent user-level calibration state."""
    import flexscanx.validation as validation

    monkeypatch.setattr(validation, "_CACHE_FILE", str(tmp_path / "calibration-cache.json"))
