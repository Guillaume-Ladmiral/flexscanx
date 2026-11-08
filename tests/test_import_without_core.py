from __future__ import annotations

import pytest


def test_core_version_falls_back_when_core_is_missing(monkeypatch):
    import flexscanx

    def _missing_core():
        raise ImportError("missing test core")

    monkeypatch.setattr(flexscanx, "_require_core", _missing_core)

    assert flexscanx.core_version() == "unavailable"


def test_require_core_error_message_mentions_build(monkeypatch):
    from flexscanx import _core_loader

    def _missing_import(name: str):
        assert name == "flexscanx._core"
        raise ImportError("missing test core")

    monkeypatch.setattr(_core_loader, "import_module", _missing_import)

    with pytest.raises(ImportError, match="compiled extension '_core' is not available"):
        _core_loader.require_core()
