"""Packaging and environment-switch regressions of the 2026-09-07 run (FXA-161..163)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import flexscanx as flx

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _extra(name: str) -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(rf"^{name} = \[(.*?)^\]", text, flags=re.S | re.M)
    assert block, f"extra {name!r} not declared in pyproject.toml"
    return re.findall(r'"([^"]+)"', block.group(1))


# ---------------------------------------------------------------------------
# FXA-162: scikit-learn is declared in an extra
# ---------------------------------------------------------------------------


def test_knn_extra_declares_scikit_learn_and_all_includes_it() -> None:
    knn = _extra("knn")
    assert any(req.startswith("scikit-learn") for req in knn), knn
    assert any(req.startswith("scikit-learn") for req in _extra("all"))


def test_reference_names_the_knn_extra_for_balltree_and_haversine() -> None:
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "flexscanx[knn]" in text


# ---------------------------------------------------------------------------
# FXA-163: the BallTree diagnostics name scikit-learn and the extra
# ---------------------------------------------------------------------------

_BLOCK_SKLEARN_SRC = """
    import json, sys
    sys.modules["sklearn"] = None
    sys.modules["sklearn.neighbors"] = None
    import numpy as np
    import flexscanx as flx
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = {}
    cases = (("balltree", dict(backend="balltree")), ("haversine", dict(metric="haversine")))
    for label, kw in cases:
        try:
            flx.csr_knn_from_coords(coords, 1, **kw)
            out[label] = None
        except ImportError as exc:
            out[label] = str(exc)
    print(json.dumps(out))
    """
_BLOCK_SKLEARN = textwrap.dedent(_BLOCK_SKLEARN_SRC)


def test_balltree_and_haversine_diagnostics_name_scikit_learn_and_the_extra() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_SKLEARN],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for label in ("balltree", "haversine"):
        message = out[label]
        assert message is not None, f"{label} did not raise ImportError with scikit-learn blocked"
        assert "scikit-learn" in message
        assert "flexscanx[knn]" in message
        assert "pip install scipy" not in message, message


def test_explicit_balltree_still_works_when_scikit_learn_is_present() -> None:
    pytest.importorskip("sklearn")
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert flx.csr_knn_from_coords(coords, 1, backend="balltree") == flx.csr_knn_from_coords(
        coords, 1, backend="kdtree"
    )


# ---------------------------------------------------------------------------
# FXA-161: FLEXSCANX_DISABLE_JIT actually disables the numba paths
# ---------------------------------------------------------------------------

_PROBE_JIT_SRC = """
    import json, warnings
    import flexscanx._graph_metrics as gm
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row_ptr = [0, 1, 3, 4]
        col_idx = [1, 0, 2, 1]
        gm.silhouette_graph(row_ptr, col_idx, [1, 1, 0])
    print(json.dumps({
        "has_numba": bool(gm._HAS_NUMBA),
        "disabled_by_env": bool(gm._JIT_DISABLED_BY_ENV),
        "fallback_warnings": sum("pure-Python fallback" in str(w.message) for w in caught),
    }))
    """
_PROBE_JIT = textwrap.dedent(_PROBE_JIT_SRC)


def _probe_jit(value: str | None) -> dict:
    env = dict(os.environ)
    env.pop("FLEXSCANX_DISABLE_JIT", None)
    if value is not None:
        env["FLEXSCANX_DISABLE_JIT"] = value
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_JIT],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_disable_jit_switch_turns_the_numba_paths_off_without_a_missing_numba_warning() -> None:
    pytest.importorskip("numba")
    on = _probe_jit(None)
    off = _probe_jit("1")
    assert on["has_numba"] is True and on["disabled_by_env"] is False
    assert off["has_numba"] is False and off["disabled_by_env"] is True
    # The user asked for the pure-Python path; it is not a missing dependency.
    assert off["fallback_warnings"] == 0


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_disable_jit_switch_ignores_falsy_values(value: str) -> None:
    pytest.importorskip("numba")
    probe = _probe_jit(value)
    assert probe["has_numba"] is True and probe["disabled_by_env"] is False


# ---------------------------------------------------------------------------
# FXA-170 / FXA-183: the sdist ships a self-sufficient suite with no private paths
# ---------------------------------------------------------------------------

_GOVERNANCE_DOCS = (
    "AGENTS.md",
    "ECOSYSTEM.md",
    "DUPLICATES.md",
    "MIGRATION_PLAN.md",
    "CLAUDE.md",
    "docs/DEVELOPER_GUIDE.md",
    "docs/README.md",
    "docs/SUPERVISOR_NOTES.md",
    "tests/test_repository_guidance.py",
)


def _sdist_exclude() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("[tool.scikit-build.sdist]")
    block = re.search(r"exclude = \[(.*?)\]", text[start:], flags=re.S)
    assert block
    return re.findall(r'"([^"]+)"', block.group(1))


def test_sdist_excludes_repository_governance_material_and_the_guidance_test() -> None:
    excluded = _sdist_exclude()
    for path in _GOVERNANCE_DOCS:
        assert path in excluded, path


def test_sdist_conftest_declines_the_docs_scripts_module_without_its_trees() -> None:
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "test_remediation_docs_scripts_20260816.py" in conftest
    # this run's scripts-reading tests live in one module declined the same way
    assert "test_fix20260907_scripts.py" in conftest


def test_shipped_fix_modules_do_not_read_the_unshipped_scripts_tree() -> None:
    """Every ``tests/test_fix20260907_*.py`` module that the sdist ships must not open
    ``scripts/``; the ones that do are collected in ``test_fix20260907_scripts.py``."""
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    for module in sorted((REPO_ROOT / "tests").glob("test_fix20260907_*.py")):
        if module.name in conftest:
            continue  # declined from the sdist run
        text = module.read_text(encoding="utf-8")
        # a path built from the repository root, or a quoted scripts/ path (the pattern is
        # assembled so that this test's own source does not match it)
        quoted_dir = "/" + r"\s*" + '"' + "scripts" + '"'
        quoted_path = '"' + "scripts" + "/"
        assert not re.search(f"{quoted_dir}|{quoted_path}", text), module.name


def test_shipped_test_support_carries_no_private_home_path() -> None:
    # Assembled at run time so this shipped file carries no home-path-shaped text itself.
    private = re.compile("/" + r"home/[^/\s]+(?:/|$)")
    for path in (
        REPO_ROOT / "tests" / "traminex_parity_utils.py",
        REPO_ROOT / "tests" / "conftest.py",
    ):
        assert not private.search(path.read_text(encoding="utf-8")), path.name


# ---------------------------------------------------------------------------
# FXA-196: the whole package type-checks (only missing third-party stubs may remain)
# ---------------------------------------------------------------------------


def test_package_type_checks_apart_from_missing_third_party_stubs() -> None:
    pytest.importorskip("mypy")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "src/flexscanx"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    errors = [line for line in proc.stdout.splitlines() if ": error:" in line]
    # A missing stub (import-untyped) or an absent optional third-party module such as numba
    # or geopandas (import-not-found) depends on the environment, not on the code; every error
    # class mypy raises about the package's own code is something else and fails the test.
    code_errors = [
        line
        for line in errors
        if "[import-untyped]" not in line and "[import-not-found]" not in line
    ]
    assert code_errors == [], "\n".join(code_errors)


# ---------------------------------------------------------------------------
# FXA-197: CI and pre-commit run the same pinned formatter versions
# ---------------------------------------------------------------------------


def _pre_commit_rev(repo_fragment: str) -> str:
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(
        rf"repo: https://github\.com/{repo_fragment}\s*\n\s*rev:\s*v?([0-9][0-9.]*)", text
    )
    assert match, repo_fragment
    return match.group(1)


def test_ci_lint_step_pins_black_and_ruff_to_the_pre_commit_versions() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    black_rev = _pre_commit_rev("psf/black")
    ruff_rev = _pre_commit_rev("astral-sh/ruff-pre-commit")
    assert "pip install -U ruff black" not in ci, "CI must not float the formatter versions"
    assert f"black=={black_rev}" in ci, f"CI must pin black to the pre-commit rev {black_rev}"
    assert f"ruff=={ruff_rev}" in ci, f"CI must pin ruff to the pre-commit rev {ruff_rev}"


# ---------------------------------------------------------------------------
# FXA-198: CI enforces the native formatting the pre-commit hook applies
# ---------------------------------------------------------------------------


def test_ci_checks_native_formatting_with_the_pre_commit_clang_format_version() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    rev = _pre_commit_rev("pre-commit/mirrors-clang-format")
    assert f"clang-format=={rev}" in ci, f"CI must install clang-format {rev} (the pre-commit rev)"
    assert "--dry-run" in ci and "--Werror" in ci, "CI must fail on unformatted native sources"
    assert "src/flexscanx_core" in ci.split("clang-format==")[1].split("- name:")[0]
