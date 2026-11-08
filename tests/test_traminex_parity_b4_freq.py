"""B4 parity: flexscanx.sequence.seq_freq vs traminex.seqtab (Phase 3).

Both build a frequency table of distinct whole-sequence patterns sorted by
descending count. flexscanx labels each pattern with a tuple of integer states;
traminex.seqtab labels it with a delimited string over the (stringified) state
alphabet. The parity check maps the flexscanx tuple to the same safe "|"-joined key
and compares per-pattern counts and proportions (a dict comparison, so it is
insensitive to tie-break ordering between the two sorts).

traminex is used at test time only (skips when absent).
"""

from __future__ import annotations

import numpy as np

from flexscanx.sequence.analysis import seq_freq

from .traminex_parity_utils import to_traminex_sts, traminex_seqtab_sts

# Repeated patterns with ties (three patterns at count 2) plus singletons.
SYMMETRIC = np.array(
    [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, -1, -1],
        [0, 1, 0, 0],
        [1, 0, -1, -1],
        [1, 1, 1, 1],
    ],
    dtype=int,
)

# Asymmetric: distinct, unequal counts (one dominant pattern) and N=7.
ASYMMETRIC = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, -1, 0, 1, 0],
        [-1, 0, 0, 1, 1, 1],
        [1, 1, -1, 0, 1, 0],
        [0, 1, -1, -1, 0, 1],
    ],
    dtype=int,
)

FIXTURES = {"symmetric": SYMMETRIC, "asymmetric": ASYMMETRIC}


def _fx_key_to_count(df):
    return {"|".join(str(s) for s in seq): float(c) for seq, c in zip(df["sequence"], df["count"])}


class TestB4Freq:
    def test_counts_match_seqtab(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = _fx_key_to_count(seq_freq(seqs, top_n=None, normalize=True))
            tx_df = traminex_seqtab_sts(traminex_mod, sts, sort=True, normalize=True)
            tx = dict(zip(tx_df["sequence"], tx_df["count"].astype(float)))
            assert set(fx) == set(tx), name
            for k in fx:
                # Counts are the same integer arithmetic on both sides; pin tight
                # (rtol=0, atol=1e-12) rather than np.isclose's loose 1e-5 default.
                assert np.isclose(fx[k], tx[k], rtol=0, atol=1e-12), f"{name}:{k}"

    def test_proportions_match_seqtab(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx_df = seq_freq(seqs, top_n=None, normalize=True)
            fx = {
                "|".join(str(s) for s in seq): float(p)
                for seq, p in zip(fx_df["sequence"], fx_df["proportion"])
            }
            tx_df = traminex_seqtab_sts(traminex_mod, sts, sort=True, normalize=True)
            tx = dict(zip(tx_df["sequence"], tx_df["prop"].astype(float)))
            for k in fx:
                # count/total on both sides -> pin tight, not the 1e-5 default.
                assert np.isclose(fx[k], tx[k], rtol=0, atol=1e-12), f"{name}:{k}"
            # cumulative ends at 1.0 and counts are sorted descending
            assert np.isclose(fx_df["cumulative"].iloc[-1], 1.0, rtol=0, atol=1e-12), name
            assert np.all(np.diff(fx_df["count"].values) <= 0), name

    def test_top_n_matches_seqtab_head(self, traminex_mod):
        """top_n truncation keeps the same highest-count patterns as seqtab."""
        seqs = ASYMMETRIC
        sts = to_traminex_sts(traminex_mod, seqs)
        top_n = 2
        fx = seq_freq(seqs, top_n=top_n, normalize=True)
        tx_df = traminex_seqtab_sts(traminex_mod, sts, sort=True, normalize=True)
        assert len(fx) == top_n
        # The truncated counts are the top-n counts of the full traminex table.
        full_counts = np.sort(tx_df["count"].astype(float).values)[::-1][:top_n]
        np.testing.assert_allclose(np.sort(fx["count"].values)[::-1], full_counts, atol=1e-12)
