from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
MANHATTAN_GPKG = ROOT / "vignettes" / "Manhattan_BreastCancer.gpkg"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=full_env)


def _load_script_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_binomial_script_smoke():
    proc = _run(
        [
            sys.executable,
            "scripts/bench.py",
            "--model",
            "BINOMIAL",
            "--grid",
            "2",
            "--K",
            "2",
            "--clustersize",
            "2",
            "--simcounts",
            "0",
            "--repeat",
            "1",
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    rows = json.loads(proc.stdout)
    assert rows[-1]["model"] == "BINOMIAL"
    assert rows[-1]["grid"] == 2


def test_perf_curves_binomial_script_smoke(tmp_path):
    out_csv = tmp_path / "perf_curves.csv"
    proc = _run(
        [
            sys.executable,
            "scripts/perf_curves.py",
            "--model",
            "BINOMIAL",
            "--grid",
            "2",
            "--K",
            "2",
            "--simcounts",
            "0",
            "--repeats",
            "1",
            "--out",
            str(out_csv),
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.exists()
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    assert lines[0].startswith("grid,N,K,simcount")


def test_perf_checks_binomial_script_smoke(tmp_path):
    out_json = tmp_path / "perf_checks.json"
    proc = _run(
        [
            sys.executable,
            "scripts/perf_checks.py",
            "--model",
            "BINOMIAL",
            "--grid",
            "4",
            "--K",
            "4",
            "--clustersize",
            "4",
            "--scanmethods",
            "FLEXIBLE",
            "--clustertypes",
            "HOT",
            "--quick",
            "--output",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_json.exists()
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert rows[-1]["model"] == "BINOMIAL"
    assert rows[-1]["simcount"] == 0


def test_perf_checks_rejects_too_small_grid():
    proc = _run(
        [
            sys.executable,
            "scripts/perf_checks.py",
            "--model",
            "POISSON",
            "--grid",
            "3",
            "--quick",
        ]
    )
    assert proc.returncode != 0
    assert "grid must be >= 4" in (proc.stderr + proc.stdout)


def test_otani_script_builds_model_specific_scan_kwargs():
    mod = _load_script_module("otani_takahashi_2021_test", "scripts/otani_takahashi_2021.py")
    poisson = mod._scan_data_kwargs("POISSON", [1.0, 2.0, 3.0])
    binomial = mod._scan_data_kwargs("BINOMIAL", [10.0, 20.0, 30.0])

    assert poisson == {"expected": [1.0, 2.0, 3.0], "rantype": "MULTINOMIAL"}
    assert binomial == {"population": [10.0, 20.0, 30.0], "rantype": "BINOMIAL"}


def test_canonical_manhattan_dataset_exists_and_has_expected_layer():
    pytest.importorskip("pyogrio")
    import pyogrio

    assert MANHATTAN_GPKG.exists()
    layers = pyogrio.list_layers(str(MANHATTAN_GPKG))
    assert any(row[0] == "manhattan_breast" for row in layers.tolist())


def test_quickcheck_ot_script_smoke(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("pyogrio")

    proc = _run(
        [
            sys.executable,
            "scripts/quickcheck_ot.py",
            "--gpkg",
            str(MANHATTAN_GPKG),
            "--simcount",
            "0",
            "--outdir",
            str(tmp_path),
        ],
        env={"MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (tmp_path / "quickcheck_hot.csv").exists()
    assert (tmp_path / "quickcheck_hot.svg").exists()


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")
def test_check_import_powershell_smoke_uses_env_fallback():
    proc = _run(
        [POWERSHELL, "-NoProfile", "-File", "scripts/check_import.ps1"],
        env={"QUARTO_PYTHON": sys.executable},
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "core_version:" in proc.stdout


def test_check_import_script_uses_repo_root_not_parent_of_repo():
    text = (ROOT / "scripts" / "check_import.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $PSScriptRoot "resolve_python.ps1"' in text
    assert "Split-Path (Split-Path $PSScriptRoot -Parent) -Parent" not in text


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")
def test_resolve_python_powershell_smoke_accepts_explicit_path():
    proc = _run(
        [
            POWERSHELL,
            "-NoProfile",
            "-File",
            "scripts/resolve_python.ps1",
            "-PythonPath",
            sys.executable,
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert Path(proc.stdout.strip()).resolve() == Path(sys.executable).resolve()


def test_pyproject_metadata_uses_current_project_name():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'authors = [{ name = "PROJECT_FLEXSCANX" }]' in text
    assert "PROJECT_PY_FLEXSCAN" not in text


def test_gen_compile_commands_script_uses_env_discovery_not_user_specific_paths():
    text = (ROOT / "scripts" / "gen_compile_commands.ps1").read_text(encoding="utf-8")
    assert "FLEXSCANX_PYTHON" in text
    assert "FLEXSCANX_PYBIND11_DIR" in text
    assert "FLEXSCANX_CL_EXE" in text
    assert "pybind11.get_cmake_dir()" in text
    assert "C:\\Users\\" not in text
    assert "14.44.35207" not in text


def test_historical_windows_scripts_share_resolver_and_env_name_contract():
    ci = (ROOT / "scripts" / "ci.ps1").read_text(encoding="utf-8")
    render_ps1 = (ROOT / "scripts" / "render_all.ps1").read_text(encoding="utf-8")
    render_bat = (ROOT / "scripts" / "render_all.bat").read_text(encoding="utf-8")
    env_yml = (ROOT / "environment.yml").read_text(encoding="utf-8")

    assert 'Join-Path $PSScriptRoot "resolve_python.ps1"' in ci
    assert 'Join-Path $PSScriptRoot "resolve_python.ps1"' in render_ps1
    assert (
        'powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve_python.ps1"'
        in render_bat
    )
    assert "vignettes/workflows" in render_ps1
    assert "vignettes/notes" in render_ps1
    assert "-IncludeHeavy" in render_ps1
    assert "--no-execute" in render_ps1
    assert "Manhattan_Breast_Cancer.qmd" in render_ps1
    assert "vignettes\\workflows" in render_bat
    assert "vignettes\\notes" in render_bat
    assert "--include-heavy" in render_bat
    assert "--no-execute" in render_bat
    assert "Manhattan_Breast_Cancer.qmd" in render_bat
    assert "name: flexscanx" in env_yml


def test_check_release_tree_help_smoke():
    proc = _run([sys.executable, "scripts/check_release_tree.py", "--help"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "release-relevant tree" in proc.stdout


def test_verify_cleanroom_help_smoke():
    proc = _run([sys.executable, "scripts/verify_cleanroom.py", "--help"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "temporary git repository" in proc.stdout


def test_verify_release_artifacts_help_smoke():
    proc = _run([sys.executable, "scripts/verify_release_artifacts.py", "--help"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "--python" in proc.stdout
    assert "--keep-temp" in proc.stdout


def test_render_all_help_smoke():
    proc = _run([sys.executable, "scripts/render_all.py", "--help"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "--include-heavy" in proc.stdout
    assert "--dry-run" in proc.stdout


def test_check_docs_hygiene_help_smoke():
    proc = _run([sys.executable, "scripts/check_docs_hygiene.py", "--help"])
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "archive layout" in proc.stdout


def test_release_sanity_includes_audit_fix_regressions():
    text = (ROOT / "scripts" / "release_sanity.py").read_text(encoding="utf-8")
    for rel_path in (
        "tests/test_check_docs_hygiene.py",
        "tests/test_check_docs_signatures.py",
        "tests/test_flatten.py",
        "tests/test_meta_version.py",
        "tests/test_regressions_audit_fixes.py",
        "tests/test_sync_docs_metadata.py",
        "tests/test_validation_comprehensive.py",
        "tests/test_verify_cleanroom.py",
        "tests/test_verify_release_artifacts.py",
    ):
        assert rel_path in text
    assert "scripts/verify_release_artifacts.py" in text


def test_scripts_readme_lists_historical_windows_tooling_helpers():
    text = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "Historical compatibility helpers" in text
    assert "not part of the active Ubuntu release gate" in text
    for script_name in (
        "ci.ps1",
        "resolve_python.ps1",
        "render_all.ps1",
        "render_all.bat",
    ):
        assert f"`{script_name}`" in text


def test_scripts_readme_registers_artifact_verifier():
    text = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "`verify_release_artifacts.py`" in text
    assert "build, twine, and auditwheel" in text
