from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_FALLBACK_VERSION = "0.3.2"
_PROJECT_NAME_PATTERN = re.compile(r'^name\s*=\s*"([^"]+)"\s*$')
_PROJECT_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')


def _read_pyproject_version(pyproject_path: Path) -> str | None:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    in_project = False
    project_name: str | None = None
    project_version: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project:
            continue

        name_match = _PROJECT_NAME_PATTERN.match(line)
        if name_match:
            project_name = name_match.group(1)
            continue

        version_match = _PROJECT_VERSION_PATTERN.match(line)
        if version_match:
            project_version = version_match.group(1)
            continue

    if project_name == "flexscanx" and project_version:
        return project_version
    return None


def _fallback_repo_version(module_path: Path) -> str:
    """Walk up at most 5 parent directories looking for a matching pyproject.toml."""
    path = module_path.resolve()
    for _ in range(5):
        path = path.parent
        version = _read_pyproject_version(path / "pyproject.toml")
        if version is not None:
            return version
    return _FALLBACK_VERSION


try:
    __version__: str = _pkg_version("flexscanx")
except PackageNotFoundError:
    __version__ = _fallback_repo_version(Path(__file__))
