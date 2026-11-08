"""Test-only traminex bridge for the sequence parity tests (Phase 0.1).

Locked family decision (ECOSYSTEM.md / DUPLICATES.md Section B): flexscanx
installs and runs **without** traminex. The deliberate local copies in
``flexscanx.sequence`` must stay behaviour-identical to their traminex
originals (items B1-B5), proven by parity tests that use traminex at **test
time only**. When a parity test reveals drift, fix the flexscanx copy to match
traminex — never the reverse.

Consequences for this module:

- It must only ever be imported from ``tests/``; nothing under
  ``src/flexscanx`` may import traminex (guarded by a smoke test).
- When traminex is absent the parity tests skip, so the standalone suite stays
  green on a bare ``pip install flexscanx``. Set ``FLEXSCANX_PARITY_STRICT=1``
  to turn the skip into a failure in environments where traminex is expected.

traminex is a *local* dependency (not on PyPI), installed editable into the
shared environment — see MIGRATION_PLAN.md Phase 0.3 — and declared in this
project's optional ``test`` extra only (Phase 0.4).
"""

from __future__ import annotations

import inspect
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

TRAMINEX_TESTED_COMMIT = "2cbb3472736d0ccce32a291241983a12b8921122"

_TRAMINEX_MISSING = (
    "traminex is not importable; sequence parity tests need it at test time. "
    "It is a local dependency: install it editable into the shared environment "
    "from your PROJECT_TRAMINEX checkout (see MIGRATION_PLAN.md Phase 0.3): "
    "python -m pip install -e <path-to-PROJECT_TRAMINEX>"
)


def is_strict_parity_mode() -> bool:
    return str(os.environ.get("FLEXSCANX_PARITY_STRICT", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def import_traminex_or_skip() -> Any:
    """Return the traminex module, or skip (fail in strict mode) if absent."""
    try:
        import traminex
    except ImportError:
        if is_strict_parity_mode():
            pytest.fail(_TRAMINEX_MISSING)
        pytest.skip(_TRAMINEX_MISSING)
    if is_strict_parity_mode():
        _assert_tested_traminex_revision(traminex)
    return traminex


def _assert_tested_traminex_revision(traminex: Any) -> None:
    """Require strict parity runs to use the recorded traminex Git revision."""
    module_path = Path(traminex.__file__).resolve()
    repo_root = next((parent for parent in module_path.parents if (parent / ".git").exists()), None)
    if repo_root is None:
        pytest.fail(
            "strict traminex parity requires a Git checkout so the exact tested "
            f"revision {TRAMINEX_TESTED_COMMIT} can be verified; imported {module_path}"
        )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != TRAMINEX_TESTED_COMMIT:
        pytest.fail(
            "strict traminex parity revision mismatch: "
            f"expected {TRAMINEX_TESTED_COMMIT}, got {actual} from {repo_root}"
        )


@pytest.fixture(scope="module")
def traminex_mod() -> Any:
    return import_traminex_or_skip()


def to_traminex_sts(traminex: Any, sequences: np.ndarray) -> Any:
    """Build a traminex STSList from a flexscanx (N, T) state matrix.

    flexscanx sequence states are small ints (typically +1 HOT / -1 COLD /
    0 none, see ``flexscanx.sequence``). traminex encodes states as strings,
    so states are stringified and the alphabet is the sorted unique states —
    the same default ordering ``traminex.seqdef`` would infer.

    Missing values (``NaN`` in a float matrix, ``None`` in an object matrix) are
    passed to ``seqdef`` as genuine missing values, not as the string ``'nan'``
    (FXA-184).  Every missing value, trailing ones included, is kept as
    traminex's not-recorded state (``right=None``), because flexscanx counts a
    missing categorical value as an alphabet state wherever it occurs; this is
    the oracle relationship ``NOTICE.txt`` documents as
    ``seqstatd(include_missing=True)``.
    """
    mat = np.asarray(sequences)
    if mat.ndim != 2:
        raise ValueError(f"sequences must be a 2D (N, T) matrix, got shape {mat.shape}")
    missing = _missing_mask(mat)
    if not missing.any():
        mat_str = mat.astype(str)
        alphabet = sorted(np.unique(mat_str).tolist())
        return traminex.seqdef(mat_str, alphabet=alphabet)
    labelled = np.empty(mat.shape, dtype=object)
    for index, value in np.ndenumerate(mat):
        if missing[index]:
            labelled[index] = None
        else:
            number = float(value)
            labelled[index] = str(int(number)) if number.is_integer() else str(value)
    alphabet = sorted({str(v) for v in labelled.ravel() if v is not None})
    return traminex.seqdef(labelled, alphabet=alphabet, right=None)


def _missing_mask(mat: np.ndarray) -> np.ndarray:
    """True where a state matrix entry is a missing value (``NaN`` or ``None``)."""
    if mat.dtype.kind == "f":
        return np.isnan(mat)
    if mat.dtype.kind == "O":
        return np.array(
            [[v is None or (isinstance(v, float) and np.isnan(v)) for v in row] for row in mat],
            dtype=bool,
        ).reshape(mat.shape)
    return np.zeros(mat.shape, dtype=bool)


def traminex_seqtab_sts(traminex: Any, sts: Any, **kwargs: Any) -> Any:
    """Call ``seqtab`` with the pinned oracle's full-sequence label semantics.

    The exact tested revision predates the ``format`` parameter and always
    emits STS labels. Newer editable checkouts default to SPS labels, so an
    ordinary non-strict developer run must request STS explicitly when that
    parameter exists. Strict runs still verify and exercise the exact commit.
    """
    has_format = "format" in inspect.signature(traminex.seqtab).parameters
    if has_format:
        kwargs["format"] = "STS"
    # The fixtures include the negative state "-1", so the default "-" would
    # be an ambiguous separator. Both revisions expose ``sep``; use an
    # unambiguous delimiter for the semantic label comparison.
    kwargs.setdefault("sep", "|")
    if not has_format:
        return traminex.seqtab(sts, **kwargs)
    # Newer traminex warns whenever a state contains "-", even when ``sep`` is
    # not "-". With the safe separator above that warning is a false positive.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="^'-' character in state codes may cause invalid results$",
            category=UserWarning,
        )
        return traminex.seqtab(sts, **kwargs)
