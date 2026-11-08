"""Phase 0.1 smoke tests: the test-only traminex bridge works.

These tests validate harness mechanics only (runtime independence from
traminex, import gate, state-matrix conversion). The actual B1-B5 parity
tests (flexscanx.sequence vs traminex) are Phase 3 work (MIGRATION_PLAN.md)
and are deliberately NOT started here.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from .traminex_parity_utils import to_traminex_sts

# flexscanx-style membership sequences: +1 HOT / -1 COLD / 0 none
SEQS = np.array(
    [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [-1, -1, 0, 0],
    ],
    dtype=int,
)


def test_flexscanx_sequence_does_not_import_traminex():
    """flexscanx must run without traminex (locked decision, ECOSYSTEM.md).

    Run in a fresh interpreter so an already-imported traminex in this test
    process cannot mask a runtime dependency sneaking into src/flexscanx.
    """
    code = (
        "import sys; import flexscanx; import flexscanx.sequence; "
        "sys.exit(1 if 'traminex' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        "importing flexscanx.sequence pulled in traminex (or failed): "
        f"rc={result.returncode}\nstderr: {result.stderr}"
    )


def test_to_traminex_sts_builds_sequence_object(traminex_mod):
    sts = to_traminex_sts(traminex_mod, SEQS)
    assert sts.data.shape == SEQS.shape
    assert sts.alphabet == ["-1", "0", "1"]


def test_traminex_hamming_call_round_trip(traminex_mod):
    """traminex's Hamming (the B1 parity target) is callable on our states."""
    sts = to_traminex_sts(traminex_mod, SEQS)
    D = np.asarray(traminex_mod.seqdist(sts, method="HAM"), dtype=float)
    assert D.shape == (4, 4)
    np.testing.assert_array_equal(np.diag(D), np.zeros(4))
    np.testing.assert_array_equal(D, D.T)
    # rows 0 and 1 differ at exactly one position, rows 0 and 3 at three,
    # so D[0,1] < D[0,3] under any normalization.
    assert 0.0 < D[0, 1] < D[0, 3]
