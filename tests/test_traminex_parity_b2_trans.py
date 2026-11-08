"""B2 parity: flexscanx.sequence.seq_trans vs traminex (Phase 3).

flexscanx ``seq_trans`` aggregates state-to-state transitions across all
sequences and returns a ``from`` (rows) x ``to`` (columns) matrix. Its real
behavioural counterpart in traminex is the dataset-level transition matrix:

* ``normalize=True``  -> row-normalized transition *rates* ==
  ``traminex.seqtrate(count=False)``.
* ``normalize=False`` -> raw transition *counts* ==
  ``traminex.seqtrate(count=True)``.

In TraMineR/traminex, ``seqtransn(x, norm=...)`` is a separate statistic with
one transition-count value per sequence; it is not a matrix API. Only ``lag=1``
has a traminex matrix counterpart (``seqtrate``); the ``lag>1`` form is a
flexscanx extension and is not parity-tested.

traminex is used at test time only (skips when absent).
"""

from __future__ import annotations

import numpy as np

from flexscanx.sequence.analysis import seq_trans

from .traminex_parity_utils import to_traminex_sts

SYMMETRIC = np.array(
    [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, -1, -1],
        [-1, -1, 0, 1],
    ],
    dtype=int,
)

# Asymmetric: directed transitions are deliberately lopsided (e.g. 1->0 is
# common, 0->1 less so, -1 rarely an origin) so a transposed result would fail.
ASYMMETRIC = np.array(
    [
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 0, -1, 0, 0],
    ],
    dtype=int,
)

# Absent-source fixture: state +1 (HOT) appears *only* in the final column, so it
# is never a transition source (its count PA == 0). R's seqtrate zeroes that row
# (seqtrate.R: `if (PA == 0) tmat[x, ] <- 0`) rather than leaving it NaN from a
# 0/0 division, and traminex.seqtrate mirrors this (`where=row_sums > 0`). This
# is the exact convention Audit Phase B pinned as R-faithful in the oracle; the
# other fixtures never put a state in a pure-sink position, so without this
# fixture the zero-row branch of seq_trans was not load-bearing against the
# oracle. N=5 (not a multiple of the 3-state alphabet) keeps it asymmetric.
ABSENT_SOURCE = np.array(
    [
        [0, -1, 0, 1],
        [-1, 0, -1, 1],
        [0, 0, -1, 1],
        [-1, -1, 0, 1],
        [0, -1, -1, 1],
    ],
    dtype=int,
)

FIXTURES = {
    "symmetric": SYMMETRIC,
    "asymmetric": ASYMMETRIC,
    "absent_source": ABSENT_SOURCE,
}


def _assert_axes_align(fx_df, tx_df, name):
    """flexscanx labels integer states; traminex labels stringified states."""
    assert [str(s) for s in fx_df.index] == list(tx_df.index), name
    assert [str(s) for s in fx_df.columns] == list(tx_df.columns), name


class TestB2Transitions:
    def test_rates_equal_seqtrate(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_trans(seqs, lag=1, normalize=True)
            tx = traminex_mod.seqtrate(sts, count=False)  # row-normalized KxK DataFrame
            _assert_axes_align(fx, tx, name)
            np.testing.assert_allclose(fx.values, tx.values, atol=1e-12, err_msg=name)

    def test_seqtransn_is_a_per_sequence_statistic(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            raw = np.asarray(traminex_mod.seqtransn(sts, norm=False), dtype=float)
            normalized = np.asarray(traminex_mod.seqtransn(sts, norm=True), dtype=float)
            assert raw.shape == (seqs.shape[0],), name
            assert normalized.shape == (seqs.shape[0],), name
            np.testing.assert_allclose(
                normalized, raw / (seqs.shape[1] - 1), atol=1e-12, err_msg=name
            )

    def test_counts_equal_seqtrate_count_true(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_trans(seqs, lag=1, normalize=False)
            tx = traminex_mod.seqtrate(sts, count=True)
            _assert_axes_align(fx, tx, name)
            np.testing.assert_allclose(fx.values, tx.values, atol=1e-12, err_msg=name)

    def test_rows_are_origin_states(self, traminex_mod):
        """Guard the from->to orientation against an accidental transpose."""
        seqs = ASYMMETRIC
        sts = to_traminex_sts(traminex_mod, seqs)
        fx = seq_trans(seqs, lag=1, normalize=False)
        tx = np.asarray(traminex_mod.seqtrate(sts, count=True))
        # The asymmetric fixture is not symmetric, so transposing would diverge.
        assert not np.allclose(tx, tx.T)
        np.testing.assert_allclose(fx.values, tx, atol=1e-12)

    def test_absent_source_row_is_zeroed_like_seqtrate(self, traminex_mod):
        """A never-a-source state's rate row is all zeros (R seqtrate PA==0).

        This pins flexscanx's `where=row_sums > 0` branch to R's
        `if (PA == 0) tmat[x, ] <- 0` convention (via the R-faithful oracle),
        i.e. the row is zeroed, never NaN from a 0/0 division. State +1 is the
        absent source in the ABSENT_SOURCE fixture.
        """
        seqs = ABSENT_SOURCE
        # +1 must never appear as a transition origin for this test to bite.
        assert not (seqs[:, :-1] == 1).any()

        sts = to_traminex_sts(traminex_mod, seqs)
        fx = seq_trans(seqs, lag=1, normalize=True)
        tx = traminex_mod.seqtrate(sts, count=False)

        _assert_axes_align(fx, tx, "absent_source")
        assert np.all(np.isfinite(fx.values)), "rates must not be NaN/inf"
        # The +1 row is zeroed in both flexscanx and the R-faithful oracle.
        np.testing.assert_array_equal(fx.loc[1].values, np.zeros(fx.shape[1]))
        np.testing.assert_array_equal(tx.loc["1"].values, np.zeros(tx.shape[1]))
        # And every other (present-source) row is a probability row summing to 1.
        present_rows = fx.drop(index=1)
        np.testing.assert_allclose(present_rows.sum(axis=1).values, 1.0, atol=1e-12)
        np.testing.assert_allclose(fx.values, tx.values, atol=1e-12)
