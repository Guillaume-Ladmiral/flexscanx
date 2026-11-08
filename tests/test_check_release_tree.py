from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release_tree.py"
GIT = shutil.which("git")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert GIT is not None
    return _run([GIT, *args], cwd=repo)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "src" / "flexscanx" / "__init__.py", "__version__ = '0.0.1'\n")
    _write(repo / "tests" / "test_smoke.py", "def test_smoke():\n    assert True\n")
    _write(repo / "scripts" / "tool.py", "print('tool')\n")
    _write(repo / ".gitignore", "__pycache__/\n")
    _write(repo / "README.md", "# repo\n")
    _write(repo / "DOCUMENTATION_INDEX.md", "# index\n")
    _write(repo / "LICENSE", "MIT License\n")
    _write(repo / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.15)\n")
    assert _git(repo, "init").returncode == 0
    assert _git(repo, "config", "user.name", "test-user").returncode == 0
    assert _git(repo, "config", "user.email", "test@example.com").returncode == 0
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-m", "init").returncode == 0
    return repo


@pytest.mark.skipif(GIT is None, reason="git not available")
def test_check_release_tree_passes_for_clean_repo(tmp_path):
    repo = _init_repo(tmp_path)
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "release tree is clean" in proc.stdout


@pytest.mark.skipif(GIT is None, reason="git not available")
def test_check_release_tree_fails_for_tracked_src_change(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "src" / "flexscanx" / "__init__.py", "__version__ = '0.0.2'\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "release-relevant paths are dirty" in proc.stderr
    assert "src/flexscanx/__init__.py" in proc.stderr


@pytest.mark.skipif(GIT is None, reason="git not available")
def test_check_release_tree_fails_for_untracked_test_file(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "tests" / "test_new_contract.py", "def test_new_contract():\n    assert True\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "blocker: untracked source/test/script files detected" in proc.stderr
    assert "tests/test_new_contract.py" in proc.stderr


@pytest.mark.skipif(GIT is None, reason="git not available")
def test_check_release_tree_ignores_untracked_non_release_artifacts(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "archive" / "reports" / "note.txt", "generated report\n")
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(GIT is None, reason="git not available")
@pytest.mark.parametrize(
    ("rel_path", "text"),
    [
        (".gitignore", "*.tmp\n"),
        ("DOCUMENTATION_INDEX.md", "# updated index\n"),
        ("LICENSE", "Updated license text\n"),
    ],
)
def test_check_release_tree_fails_for_tracked_release_metadata_changes(
    tmp_path, rel_path: str, text: str
):
    repo = _init_repo(tmp_path)
    _write(repo / rel_path, text)
    proc = _run([sys.executable, str(SCRIPT), "--repo-root", str(repo)], cwd=ROOT)
    assert proc.returncode == 1
    assert "release-relevant paths are dirty" in proc.stderr
    assert rel_path in proc.stderr
