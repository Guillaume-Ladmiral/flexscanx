"""Regression contracts for the 2026-08-16 packaging remediation."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _top_level_statement_index(tree: ast.Module, module: str) -> int | None:
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.Import) and any(
            alias.name == module for alias in statement.names
        ):
            return index
        if (
            isinstance(statement, ast.ImportFrom)
            and statement.module
            and (statement.module == module or statement.module.startswith(f"{module}."))
        ):
            return index
    return None


def _importorskip_index(tree: ast.Module, dependency: str) -> int:
    for index, statement in enumerate(tree.body):
        value = statement.value if isinstance(statement, (ast.Assign, ast.Expr)) else None
        if not isinstance(value, ast.Call):
            continue
        call = value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pytest"
            and call.func.attr == "importorskip"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and (
                call.args[0].value == dependency or call.args[0].value.startswith(f"{dependency}.")
            )
        ):
            return index
    raise AssertionError(f"pytest.importorskip({dependency!r}) is missing")


def test_public_knn_dependency_is_declared_and_documented_for_a_base_install():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    dependencies = pyproject["project"]["dependencies"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert any(dependency.startswith("scipy>=") for dependency in dependencies)
    assert "pip install -e ." in readme
    assert "SciPy" in readme


def test_artifact_smoke_uses_the_public_knn_builder():
    artifact_verifier = (ROOT / "scripts" / "verify_release_artifacts.py").read_text(
        encoding="utf-8"
    )

    assert "flexscanx.csr_knn_from_coords(" in artifact_verifier
    assert "row_ptr, col_idx = flexscanx.csr_knn_from_coords(" in artifact_verifier


def test_ci_has_a_base_environment_collection_gate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "name: Verify base test collection" in ci
    assert "pytest --collect-only -q" in ci


def test_optional_test_modules_skip_before_importing_optional_dependencies():
    test_dependencies = {
        "tests/test_helpers_and_graph.py": ("geopandas", "shapely"),
        "tests/test_traminex_parity_b5_plots.py": ("matplotlib",),
    }

    for relative_path, dependencies in test_dependencies.items():
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
        for dependency in dependencies:
            importorskip_index = _importorskip_index(tree, dependency)
            import_index = _top_level_statement_index(tree, dependency)
            assert import_index is None or importorskip_index < import_index


def test_collected_manhattan_driver_does_not_import_geo_extras_at_module_load():
    script = ROOT / "scripts" / "test_manhattan_scan.py"
    code = textwrap.dedent("""
        import builtins
        import runpy
        import sys

        original_import = builtins.__import__
        blocked = {"fiona", "geopandas", "pyogrio", "shapely"}

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in blocked:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        runpy.run_path(sys.argv[1])
        print("module-load-ok")
        """)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "module-load-ok"


def test_sdist_collection_ignores_only_tests_requiring_excluded_development_trees():
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")

    assert "collect_ignore" in conftest
    assert "test_scripts_smoke.py" in conftest
    assert "test_rflexscan_manual_vignette.py" in conftest
    assert 'not (_repo_root / "scripts").is_dir()' in conftest
    assert 'not (_repo_root / "vignettes").is_dir()' in conftest


def test_documented_public_composition_type_checks(tmp_path):
    example = tmp_path / "documented_composition.py"
    example.write_text(
        textwrap.dedent("""
            import numpy as np
            import flexscanx as flx

            coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
            observed = np.array([9, 7, 1, 1])
            expected = np.full(4, 4.5)
            row_ptr, col_idx = flx.csr_knn_from_coords(coords, k=1)
            result = flx.scan(
                coords=coords,
                adjacency=(row_ptr, col_idx),
                observed=observed,
                expected=expected,
                model="POISSON",
                K=4,
                clustersize=3,
                simcount=0,
            )
            table = flx.summarize_result(result)
            assert len(table) >= 1
            """),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--follow-imports=silent",
            "--ignore-missing-imports",
            str(example),
        ],
        cwd=ROOT,
        env={**os.environ, "MYPYPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_published_test_extra_has_no_unowned_traminex_distribution_dependency():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    test_dependencies = pyproject["project"]["optional-dependencies"]["test"]
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    repository_contract = (ROOT / "tests" / "test_repository_contracts.py").read_text(
        encoding="utf-8"
    )

    assert not any(dependency.startswith("traminex") for dependency in test_dependencies)
    assert any(dependency.startswith("mypy>=") for dependency in test_dependencies)
    assert 'python -m pip install -e ".[plot]" pytest' in ci
    assert re.search(r"TRAMINEX_REF:\s*[0-9a-f]{40}", ci)
    assert 'checkout --detach "${TRAMINEX_REF}"' in ci
    assert "traminex" not in test_dependencies
    assert 'assert "traminex" not in test_dependencies' in repository_contract
    assert "does not declare a traminex distribution" in readme


def test_ci_pins_actions_permissions_and_maintained_lint_scope():
    workflows = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "wheels.yml",
    ]
    uses_pattern = re.compile(
        r"^\s*(?:-\s+)?uses:\s+[^@]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE
    )

    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert "permissions:\n  contents: read" in text
        uses_lines = [line for line in text.splitlines() if "uses:" in line]
        assert uses_lines
        assert len(uses_pattern.findall(text)) == len(uses_lines)

    ci = workflows[0].read_text(encoding="utf-8")
    wheels = workflows[1].read_text(encoding="utf-8")
    assert "ruff check src tests scripts" in ci
    assert "black --check src tests scripts" in ci
    assert "CIBW_ARCHS_MACOS" not in wheels


def test_formatter_configuration_excludes_frozen_and_audit_trees_consistently():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    ruff_excludes = pyproject["tool"]["ruff"]["exclude"]
    black_excludes = pyproject["tool"]["black"]["extend-exclude"]
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    for directory in ("archive", "audit", "RFLEXSCAN_ORIGINAL"):
        assert directory in ruff_excludes
        assert directory in black_excludes
        assert directory in pre_commit
