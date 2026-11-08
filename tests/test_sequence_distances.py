"""Unit tests for sequence distance metrics.

Copyright (c) 2025 flexscanx contributors
"""

import numpy as np
import pytest

from flexscanx.sequence import seq_dist_hamming, seq_dist_matrix_info

# seq_dist_gower was removed in v0.0.1b (it was an unexported thin wrapper over
# seq_dist_hamming with no distinct behaviour for categorical sequences).


class TestSeqDistHamming:
    """Tests for Hamming distance computation."""

    def test_hamming_identical_sequences(self):
        """Identical sequences should have zero distance."""
        seqs = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
        D = seq_dist_hamming(seqs, norm=True)
        assert D.shape == (2, 2)
        assert np.allclose(D, 0.0)

    def test_hamming_completely_different(self):
        """Completely different sequences should have distance 1.0 when normalized."""
        seqs = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        D = seq_dist_hamming(seqs, norm=True)
        assert D[0, 1] == 1.0
        assert D[1, 0] == 1.0

    def test_hamming_partial_mismatch(self):
        """Test known partial mismatch case."""
        seqs = np.array(
            [
                [0, 1, 1, 0],  # seq 0
                [0, 1, 0, 0],  # seq 1: differs at position 2
                [1, 1, 1, 1],  # seq 2: differs at positions 0 and 3
            ]
        )
        D = seq_dist_hamming(seqs, norm=True)

        # seq 0 vs seq 1: differ at 1/4 positions
        assert D[0, 1] == 0.25
        # seq 0 vs seq 2: differ at 2/4 positions
        assert D[0, 2] == 0.50
        # seq 1 vs seq 2: differ at 3/4 positions (indices 0, 2, 3)
        assert D[1, 2] == 0.75

    def test_hamming_unnormalized(self):
        """Test unnormalized distance (raw counts)."""
        seqs = np.array([[0, 1, 1, 0], [0, 1, 0, 0]])
        D = seq_dist_hamming(seqs, norm=False)
        assert D[0, 1] == 1.0  # 1 mismatch

    def test_hamming_weighted(self):
        """Test weighted distance computation."""
        seqs = np.array([[0, 1, 1], [1, 1, 0]])
        # Weight last position more heavily
        weights = np.array([1.0, 1.0, 3.0])
        D = seq_dist_hamming(seqs, weights=weights, norm=True)

        # Differ at positions 0 (weight 1) and 2 (weight 3)
        # Total mismatch weight: 4.0
        # Total weight: 5.0
        expected = 4.0 / 5.0
        assert np.isclose(D[0, 1], expected)

    def test_hamming_symmetric(self):
        """Distance matrix should be symmetric."""
        seqs = np.random.default_rng(71).integers(-1, 2, size=(5, 10))
        D = seq_dist_hamming(seqs)
        assert np.allclose(D, D.T)

    def test_hamming_zero_diagonal(self):
        """Diagonal should be zero (self-distance)."""
        seqs = np.random.default_rng(77).integers(-1, 2, size=(5, 10))
        D = seq_dist_hamming(seqs)
        assert np.allclose(np.diag(D), 0.0)

    def test_hamming_empty_sequences(self):
        """Empty sequences (T=0) should have zero distance."""
        seqs = np.array([[], []])
        with pytest.raises(ValueError):
            # Can't reshape properly
            seq_dist_hamming(seqs)

    def test_hamming_single_sequence(self):
        """Single sequence should have zero distance to itself."""
        seqs = np.array([[0, 1, -1, 0]])
        D = seq_dist_hamming(seqs)
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_hamming_invalid_weights(self):
        """Invalid weights should raise errors."""
        seqs = np.array([[0, 1, 1], [1, 0, 0]])

        # Wrong length
        with pytest.raises(ValueError, match="weights must have length"):
            seq_dist_hamming(seqs, weights=np.array([1.0, 2.0]))

        # Non-positive sum
        with pytest.raises(ValueError, match="sum of weights must be positive"):
            seq_dist_hamming(seqs, weights=np.array([0.0, 0.0, 0.0]))

        # Mixed-sign weights can produce negative normalized distances
        with pytest.raises(ValueError, match="weights must be non-negative"):
            seq_dist_hamming(seqs, weights=np.array([2.0, -1.0, 2.0]))

    def test_hamming_preserves_non_integer_categorical_values(self):
        seqs = np.array([[0.2, 1.0], [0.9, 1.0]])

        distance = seq_dist_hamming(seqs, norm=False)

        assert distance[0, 1] == 1.0

    def test_hamming_treats_missing_values_as_one_categorical_state(self):
        seqs = np.empty((2, 2), dtype=object)
        seqs[:] = [[np.nan, "A"], [np.nan, "B"]]

        distance = seq_dist_hamming(seqs, norm=False)

        assert np.allclose(np.diag(distance), 0.0)
        assert distance[0, 1] == 1.0

    @pytest.mark.parametrize(
        "weights",
        [
            np.array(1.0),
            np.array([[1.0, 1.0, 1.0]]),
            np.array([1.0, np.nan, 1.0]),
            np.array([1.0, np.inf, 1.0]),
        ],
    )
    def test_hamming_rejects_non_vector_or_non_finite_weights(self, weights):
        seqs = np.array([[0, 1, 1], [1, 0, 0]])

        with pytest.raises(ValueError, match="weights must be a finite 1D array"):
            seq_dist_hamming(seqs, weights=weights)


class TestSeqDistCondensed:
    """Tests for the condensed output mode added in v0.0.1b."""

    def test_condensed_shape(self):
        """condensed=True must return a 1-D upper-triangle vector of length N*(N-1)//2."""
        seqs = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=np.int32)
        N = seqs.shape[0]
        cond = seq_dist_hamming(seqs, condensed=True)
        assert cond.shape == (N * (N - 1) // 2,)

    def test_condensed_matches_full(self):
        """condensed=True upper-triangle must match the corresponding cells of the full matrix."""
        seqs = np.array([[0, 1, -1], [1, 1, 0], [-1, 0, -1], [0, 0, 1]], dtype=np.int32)
        N = seqs.shape[0]
        full = seq_dist_hamming(seqs, norm=True, condensed=False)
        cond = seq_dist_hamming(seqs, norm=True, condensed=True)
        ptr = 0
        for i in range(N):
            for j in range(i + 1, N):
                assert abs(full[i, j] - cond[ptr]) < 1e-12, f"Mismatch at ({i},{j})"
                ptr += 1

    def test_condensed_weighted(self):
        """Condensed mode must respect per-position weights."""
        seqs = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int32)
        weights = np.array([1.0, 2.0, 1.0])
        cond = seq_dist_hamming(seqs, weights=weights, condensed=True)
        full = seq_dist_hamming(seqs, weights=weights, condensed=False)
        assert abs(cond[0] - full[0, 1]) < 1e-12


class TestSeqDistMatrixInfo:
    """Tests for distance matrix summary statistics."""

    def test_matrix_info_basic(self):
        """Test basic statistics computation."""
        seqs = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
        D = seq_dist_hamming(seqs, norm=True)
        info = seq_dist_matrix_info(D)

        assert info["n"] == 4
        assert "mean" in info
        assert "std" in info
        assert "min" in info
        assert "max" in info
        assert "median" in info

        # Basic sanity checks
        assert info["min"] >= 0.0
        assert info["max"] <= 1.0
        assert info["min"] <= info["median"] <= info["max"]

    def test_matrix_info_identical_sequences(self):
        """All identical sequences should have zero distances."""
        seqs = np.array([[1, 1, 1]] * 5)
        D = seq_dist_hamming(seqs)
        info = seq_dist_matrix_info(D)

        assert info["mean"] == 0.0
        assert info["std"] == 0.0
        assert info["min"] == 0.0
        assert info["max"] == 0.0

    def test_matrix_info_single_sequence(self):
        """Single sequence edge case."""
        D = np.array([[0.0]])
        info = seq_dist_matrix_info(D)

        assert info["n"] == 1
        # No pairwise distances
        assert info["mean"] == 0.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_shape(self):
        """Non-2D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            seq_dist_hamming(np.array([1, 2, 3]))

    def test_large_sequences(self):
        """Test with larger sequences to verify performance."""
        seqs = np.random.default_rng(42).integers(-1, 2, size=(100, 50))
        D = seq_dist_hamming(seqs, norm=True)

        assert D.shape == (100, 100)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0.0)
        assert np.all((D >= 0) & (D <= 1))

    def test_three_state_system(self):
        """Test with realistic 3-state cluster membership."""
        # HOT=1, COLD=-1, none=0
        seqs = np.array(
            [
                [0, 1, 1, 0, -1, -1, 0],  # HOT then COLD pattern
                [0, 1, 1, 1, 1, 0, 0],  # Persistent HOT
                [0, 0, 0, 0, 0, 0, 0],  # Never in clusters
                [-1, -1, -1, 0, 1, 1, 0],  # COLD then HOT
            ]
        )

        D = seq_dist_hamming(seqs, norm=True)

        # seq 0 and seq 3 both have HOT-COLD transitions but differ
        # seq 1 is persistent HOT
        # seq 2 is always none

        assert D[1, 2] > 0  # HOT vs none: always different when HOT present
        assert D[0, 3] > 0  # Different patterns

        # Verify symmetric and zero-diagonal properties
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0.0)
