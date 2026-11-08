from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_cleanroom.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_cleanroom_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanroom_commands_use_new_venv_for_every_install_and_check(tmp_path):
    mod = _load_module()
    snapshot = tmp_path / "repo"
    env_dir = tmp_path / "cleanroom-env"

    commands = mod.cleanroom_commands("/opt/base-python", env_dir, snapshot_root=snapshot)

    env_python = str(env_dir / "bin" / "python")
    assert commands[0] == ["/opt/base-python", "-m", "venv", str(env_dir)]
    for command in commands[1:]:
        assert command[0] == env_python
    assert [env_python, "-m", "pip", "install", "-e", ".[all]"] in commands
    assert not any(
        "test" in argument
        for command in commands
        for argument in command
        if argument.startswith(".[")
    )
    assert commands[-3] == [env_python, "scripts/release_sanity.py"]
    assert commands[-2] == [env_python, "-m", "pytest", "-q"]
    assert commands[-1] == [
        env_python,
        "scripts/verify_release_artifacts.py",
        "--python",
        env_python,
    ]


def test_source_tree_is_checked_before_a_cleanroom_snapshot_is_created(tmp_path):
    mod = _load_module()

    assert mod.source_release_tree_command("/opt/base-python", tmp_path) == [
        "/opt/base-python",
        "scripts/check_release_tree.py",
        "--repo-root",
        str(tmp_path),
    ]


def test_copy_snapshot_excludes_local_build_cache_audit_and_planning_material(tmp_path):
    mod = _load_module()
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    keep = (
        "src/flexscanx/__init__.py",
        "docs/README.md",
        "archive/internal_notes/README.md",
        "archive/reports/2026/provenance.md",
    )
    exclude = (
        "build/local/state.txt",
        "build-debug/state.txt",
        "dist/package.whl",
        "outputs/scan.csv",
        "graphify-out/graph.json",
        ".pytest_cache/state",
        ".venv/lib/python",
        "audit/private.md",
        "audit_scratch/private.md",
        ".claude/session.json",
        "docs/superpowers/private-plan.md",
        "src/flexscanx/__pycache__/module.pyc",
    )
    for relative_path in keep + exclude:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")

    mod._copy_snapshot(source, snapshot)

    for relative_path in keep:
        assert (snapshot / relative_path).is_file()
    for relative_path in exclude:
        assert not (snapshot / relative_path).exists()


def test_copy_snapshot_rejects_symbolic_links_that_could_escape_source_root(tmp_path):
    mod = _load_module()
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    outside = tmp_path / "outside-secret.txt"
    source.mkdir()
    snapshot.mkdir()
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, source / "linked-secret.txt")

    with pytest.raises(ValueError, match="symbolic links"):
        mod._copy_snapshot(source, snapshot)


def test_snapshot_git_index_preserves_ignored_tracked_and_new_untracked_files(tmp_path):
    mod = _load_module()
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    source.mkdir()
    snapshot.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "cleanroom-test"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cleanroom-test@local.invalid"],
        cwd=source,
        check=True,
    )
    (source / ".gitignore").write_text("IGNORED.md\n", encoding="utf-8")
    (source / "IGNORED.md").write_text("tracked authority\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".gitignore", "IGNORED.md"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=source, check=True)
    (source / "NEW_TEST.py").write_text("assert True\n", encoding="utf-8")

    mod._copy_snapshot(source, snapshot)
    mod._init_git_snapshot(snapshot, source_root=source)

    tracked = subprocess.check_output(["git", "ls-files"], cwd=snapshot, text=True).splitlines()
    assert "IGNORED.md" in tracked
    assert "NEW_TEST.py" in tracked
