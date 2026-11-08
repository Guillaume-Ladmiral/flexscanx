"""B1 parity: flexscanx.sequence.seq_dist_hamming vs traminex (Phase 3).

Locked family decision (ECOSYSTEM.md / DUPLICATES.md Section B): flexscanx keeps
its own ``seq_dist_hamming`` (no runtime traminex dependency), but the numbers it
produces must stay identical to traminex's R-anchored Hamming distance. traminex
is used here at **test time only** (skips when absent; see traminex_parity_utils).

Parity targets
--------------
* Unweighted, ``norm=True``  ->
  ``traminex.seqdist(method="HAM", norm="maxlength")`` == mismatches / length.
* Unweighted, ``norm=False`` -> the raw/default
  ``traminex.seqdist(method="HAM")`` mismatch count == unit-weight DHD.
* Per-position ``weights``   -> the explicit-subm DHD path added in Phase 1:
  ``distances.pairwise(method="DHD", subm=subm, norm="none")`` with a
  time-varying ``(L, K, K)`` cost array ``subm[t][a, b] = w_t * (a != b)``.
  Plain traminex HAM has no per-position weight kwarg, so DHD is the faithful
  counterpart for the *raw* weighted distance.

Documented convention (not drift): flexscanx's ``weights`` + ``norm=True`` form
divides the weighted mismatch by ``sum(weights)`` (a [0, 1] weighted-average
mismatch). traminex's DHD ``norm="maxlength"`` divides by the sequence length
instead, so the two normalized weighted forms differ by design. Uniform weights
cancel from flexscanx's numerator and denominator, producing the same mismatch
proportion as explicitly length-normalized traminex HAM; that case is asserted
below.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform

from flexscanx.sequence.distances import seq_dist_hamming

from .traminex_parity_utils import to_traminex_sts

# Small +1 HOT / -1 COLD / 0 none state space flexscanx uses.
SYMMETRIC = np.array(
    [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, -1, -1],
        [-1, -1, 0, 1],
    ],
    dtype=int,
)

# Asymmetric fixture: unequal state frequencies and N=5 (not a multiple of the
# 3-state alphabet), so a transposed/normalized-wrong result cannot pass by
# symmetry.
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


def _unit_subm(L: int, K: int) -> np.ndarray:
    """Time-constant (L, K, K) cost array: 1 for a != b, else 0 (unit Hamming)."""
    base = (~np.eye(K, dtype=bool)).astype(np.float64)
    return np.broadcast_to(base, (L, K, K)).copy()


def _weighted_subm(weights: np.ndarray, K: int) -> np.ndarray:
    """(L, K, K) cost array with subm[t][a, b] = weights[t] * (a != b)."""
    base = (~np.eye(K, dtype=bool)).astype(np.float64)
    return weights[:, None, None] * base[None, :, :]


class TestB1HammingUnweighted:
    def test_norm_true_equals_traminex_ham(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_dist_hamming(seqs, norm=True)
            tx = np.asarray(traminex_mod.seqdist(sts, method="HAM", norm="maxlength"), dtype=float)
            np.testing.assert_allclose(fx, tx, atol=1e-12, err_msg=name)

    def test_norm_false_is_raw_count(self, traminex_mod):
        """norm=False is the raw/default traminex HAM mismatch count."""
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            raw = seq_dist_hamming(seqs, norm=False)
            ham = np.asarray(traminex_mod.seqdist(sts, method="HAM"), dtype=float)
            np.testing.assert_allclose(raw, ham, atol=1e-12, err_msg=name)

    def test_norm_false_equals_unit_weight_dhd(self, traminex_mod):
        from traminex.distances import pairwise

        for name, seqs in FIXTURES.items():
            L, K = seqs.shape[1], len(np.unique(seqs))
            sts = to_traminex_sts(traminex_mod, seqs)
            raw = seq_dist_hamming(seqs, norm=False)
            dhd = np.asarray(
                pairwise(sts.data, method="DHD", subm=_unit_subm(L, K), norm="none"),
                dtype=float,
            )
            np.testing.assert_allclose(raw, dhd, atol=1e-12, err_msg=name)

    def test_condensed_matches_traminex_upper_triangle(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            sts = to_traminex_sts(traminex_mod, seqs)
            ham = np.asarray(traminex_mod.seqdist(sts, method="HAM", norm="maxlength"), dtype=float)
            cond = seq_dist_hamming(seqs, norm=True, condensed=True)
            np.testing.assert_allclose(
                cond, squareform(ham, checks=False), atol=1e-12, err_msg=name
            )


class TestB1HammingWeighted:
    def test_weighted_raw_equals_dhd_subm(self, traminex_mod):
        """Per-position-weighted raw distance == explicit-subm DHD (Phase 1)."""
        from traminex.distances import pairwise

        for name, seqs in FIXTURES.items():
            L, K = seqs.shape[1], len(np.unique(seqs))
            w = np.linspace(0.5, 3.0, L)
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_dist_hamming(seqs, weights=w, norm=False)
            dhd = np.asarray(
                pairwise(sts.data, method="DHD", subm=_weighted_subm(w, K), norm="none"),
                dtype=float,
            )
            np.testing.assert_allclose(fx, dhd, atol=1e-12, err_msg=name)

    def test_uniform_weights_norm_true_equals_traminex_ham(self, traminex_mod):
        """Uniform weights collapse to explicitly length-normalized HAM."""
        for name, seqs in FIXTURES.items():
            T = seqs.shape[1]
            w = np.full(T, 2.5)  # any positive constant
            sts = to_traminex_sts(traminex_mod, seqs)
            fx = seq_dist_hamming(seqs, weights=w, norm=True)
            ham = np.asarray(traminex_mod.seqdist(sts, method="HAM", norm="maxlength"), dtype=float)
            np.testing.assert_allclose(fx, ham, atol=1e-12, err_msg=name)

    def test_weighted_norm_is_sum_of_weights_convention(self):
        """flexscanx's normalized weighted form divides by sum(weights)."""
        seqs = ASYMMETRIC
        T = seqs.shape[1]
        w = np.linspace(0.5, 3.0, T)
        raw = seq_dist_hamming(seqs, weights=w, norm=False)
        norm = seq_dist_hamming(seqs, weights=w, norm=True)
        np.testing.assert_allclose(norm, raw / w.sum(), atol=1e-12)
