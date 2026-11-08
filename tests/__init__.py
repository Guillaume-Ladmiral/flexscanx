# Ensure installed flexscanx is used (avoid local source-tree shadowing)
import pathlib
import sys

# Pre-import installed flexscanx to lock resolution (import first, before path surgery)
import flexscanx as _installed_flexscanx  # noqa: F401,E402

_repo_root = pathlib.Path(__file__).resolve().parents[1]
# Remove repo root if present so imports resolve to site-packages
sys.path = [p for p in sys.path if p and pathlib.Path(p).resolve() != _repo_root]

# Make the tests directory importable as a package for absolute imports
# like `from tests.test_plotting import _make_grid`.
