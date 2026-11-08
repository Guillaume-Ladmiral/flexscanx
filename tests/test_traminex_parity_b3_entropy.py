"""B3 parity: flexscanx.sequence.seq_entropy vs traminex.seqstatd (Phase 3).

The parity target is the *cross-sectional* per-position entropy. traminex
``seqstatd`` returns an ``Entropy`` row that is the natural-log Shannon entropy
of each time point's state distribution, **normalized** by ``log(K)`` (K = the
alphabet size, i.e. distinct states across the whole matrix), so it lies in
[0, 1] and is base-independent.

flexscanx ``seq_entropy(by_time=True)`` historically returned the *unnormalized*
entropy in bits, which drifted from seqstatd. Phase 3 fixes the flexscanx copy:
``seq_entropy(by_time=True, normalize=True)`` (now the default) matches
``seqstatd``'s Entropy row. ``normalize=False`` preserves the old unnormalized
``base``-unit values.

The longitudinal ``by_time=False`` form is flexscanx's own overall-trajectory
diversity statistic with no traminex counterpart, so it is not parity-tested
here (it is unaffected by the normalize flag).

traminex is used at test time only (skips when absent).
"""

from __future__ import annotations

import numpy as np

from flexscanx.sequence.analysis import seq_entropy

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

# Asymmetric: several columns are missing one of the three states, so the
# normalization denominator must be the full alphabet size (K=3), not the
# per-column count of present states — a wrong denominator would diverge here.
ASYMMETRIC = np.array(
    [
        [0, 0, 0, 1, 1, 0],
        [0, 1, -1, 1, 0, 0],
        [1, 1, 1, 0, -1, 0],
        [0, 0, 1, 1, 1, 1],
        [-1, 0, 0, 0, 1, -1],
    ],
    dtype=int,
)

FIXTURES = {"symmetric": SYMMETRIC, "asymmetric": ASYMMETRIC}

# Missing values in every position class (leading, internal, trailing).  flexscanx
# counts a missing value as one more alphabet state; the oracle relationship is
# seqstatd(include_missing=True) on an STS whose missing values are all kept as
# the not-recorded state (FXA-184).  include_missing=False must NOT match.
WITH_MISSING = np.array(
    [
        [0, 1, np.nan],
        [1, np.nan, 0],
        [0, 1, 1],
        [np.nan, 0, 1],
    ],
    dtype=float,
)


class TestB3Entropy:
    def test_by_time_matches_seqstatd_entropy(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_entropy(seqs, by_time=True)  # normalize=True by default
            tx = np.asarray(traminex_mod.seqstatd(sts)["Entropy"], dtype=float)
            assert fx.shape == (seqs.shape[1],), name
            np.testing.assert_allclose(fx, tx, atol=1e-9, err_msg=name)

    def test_missing_values_match_include_missing_true_only(self, traminex_mod):
        """NOTICE.txt: the oracle is seqstatd(include_missing=True), not the default."""
        sts = to_traminex_sts(traminex_mod, WITH_MISSING)
        assert "*" in list(sts.alphabet), "missing values must reach traminex as missing"
        fx = seq_entropy(WITH_MISSING, by_time=True)
        with_missing = np.asarray(
            traminex_mod.seqstatd(sts, include_missing=True)["Entropy"], dtype=float
        )
        without_missing = np.asarray(
            traminex_mod.seqstatd(sts, include_missing=False)["Entropy"], dtype=float
        )
        np.testing.assert_allclose(fx, with_missing, atol=1e-9)
        # The load-bearing half: the two oracle flags differ on this fixture, so the
        # assertion above pins the documented flag rather than either.
        assert not np.allclose(with_missing, without_missing, atol=1e-6)
        assert not np.allclose(fx, without_missing, atol=1e-6)

    def test_normalized_is_base_independent(self, traminex_mod):
        """Normalized entropy is a [0, 1] ratio, so `base` must not change it."""
        for name, seqs in FIXTURES.items():
            a = seq_entropy(seqs, by_time=True, base=2.0)
            b = seq_entropy(seqs, by_time=True, base=np.e)
            np.testing.assert_allclose(a, b, atol=1e-12, err_msg=name)
            assert np.all((a >= -1e-12) & (a <= 1.0 + 1e-12)), name

    def test_unnormalized_recovers_bits(self):
        """normalize=False keeps the old unnormalized per-position bits."""
        seqs = ASYMMETRIC
        K = len(np.unique(seqs))
        norm = seq_entropy(seqs, by_time=True, normalize=True)
        bits = seq_entropy(seqs, by_time=True, normalize=False, base=2.0)
        # bits = normalized * log2(K)  (definition of the normalization)
        np.testing.assert_allclose(bits, norm * np.log2(K), atol=1e-12)

    def test_uniform_column_is_one(self):
        """A column with all alphabet states equally present -> normalized 1.0."""
        seqs = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]], dtype=int)
        ent = seq_entropy(seqs, by_time=True)
        np.testing.assert_allclose(ent, np.ones(3), atol=1e-12)
