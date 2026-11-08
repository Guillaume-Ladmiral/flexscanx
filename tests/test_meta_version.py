from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "src" / "flexscanx" / "_meta.py"


def _load_meta_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, META_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {META_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_meta_falls_back_to_repo_pyproject_version_when_package_metadata_is_missing(monkeypatch):
    def _raise_package_not_found(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", _raise_package_not_found)

    mod = _load_meta_module("flexscanx_meta_missing_pkg")
    assert mod.__version__ == "0.3.3"


def test_read_pyproject_version_requires_matching_project_name(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "not-flexscanx"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )

    mod = _load_meta_module("flexscanx_meta_read_version")
    assert mod._read_pyproject_version(pyproject) is None
