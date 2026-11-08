"""Lazy loader for the compiled flexscanx core extension."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_CORE_MISSING_MESSAGE = (
    "flexscanx compiled extension '_core' is not available. "
    "Build or install the package before using scan/core APIs "
    "(for example: python -m pip install -e .)."
)


def require_core() -> ModuleType:
    """Return the compiled core module or raise an actionable ImportError."""
    try:
        return import_module("flexscanx._core")
    except ImportError as exc:
        raise ImportError(_CORE_MISSING_MESSAGE) from exc


class LazyCoreProxy:
    """Attribute proxy that imports the compiled core only when used."""

    def __getattr__(self, name: str) -> Any:
        return getattr(require_core(), name)
