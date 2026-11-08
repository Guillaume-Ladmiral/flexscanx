"""Keep the test-time traminex revision contract synchronized."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tested_traminex_commit_is_synchronized():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'traminex-tested-commit\s*=\s*"([0-9a-f]+)"', pyproject)
    assert match is not None
    commit = match.group(1)

    assert len(commit) == 40
    runtime_dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    test_dependencies = pyproject.split("test = [", 1)[1].split("]", 1)[0]
    assert "traminex" not in runtime_dependencies
    assert "traminex" not in test_dependencies
    for relative_path in (
        ".github/workflows/ci.yml",
        "README.md",
        "tests/traminex_parity_utils.py",
    ):
        assert commit in (ROOT / relative_path).read_text(encoding="utf-8")
