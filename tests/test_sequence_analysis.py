"""Unit tests for sequence analysis functions.

Copyright (c) 2025 flexscanx contributors
"""

import numpy as np
import pandas as pd
import pytest

import flexscanx.sequence.analysis as sequence_analysis
from flexscanx.sequence import seq_entropy, seq_freq, seq_trans


class TestSeqFreq:
    """Tests for sequence frequency analysis."""

    def test_freq_basic(self):
        """Test basic frequency counting."""
        seqs = np.array(
            [
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 1],  # duplicate of row 0
                [1, 0, 0],
                [0, 1, 0],  # duplicate of row 1
            ]
        )

        freq = seq_freq(seqs, top_n=None, normalize=True)

        assert isinstance(freq, pd.DataFrame)
        assert "sequence" in freq.columns
        assert "count" in freq.columns
        assert "proportion" in freq.columns
        assert "cumulative" in freq.columns

        # Should have 3 unique sequences
        assert len(freq) == 3

        # Most frequent should be (0,1,1) and (0,1,0) with count 2 each
        top_counts = freq["count"].values[:2]
        assert np.allclose(top_counts, [2.0, 2.0])

        # Proportions should sum to 1
        assert np.isclose(freq["proportion"].sum(), 1.0)

    def test_freq_top_n(self):
        """Test limiting to top N sequences."""
        seqs = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1]])  # duplicate

        freq = seq_freq(seqs, top_n=2, normalize=True)
        assert len(freq) == 2

        # Top sequence should have count 2
        assert freq.iloc[0]["count"] == 2.0

    def test_freq_without_normalization(self):
        """Test frequency table without normalization."""
        seqs = np.array([[0, 1], [1, 0], [0, 1]])
        freq = seq_freq(seqs, normalize=False)

        assert "sequence" in freq.columns
        assert "count" in freq.columns
        assert "proportion" not in freq.columns
        assert "cumulative" not in freq.columns

    def test_freq_all_unique(self):
        """Test with all unique sequences."""
        seqs = np.array([[0, 0], [1, 1], [0, 1]])

        freq = seq_freq(seqs, normalize=True)

        # All should have proportion 1/3
        assert np.allclose(freq["proportion"].values, 1 / 3)

        # Cumulative should end at 1.0
        assert np.isclose(freq["cumulative"].iloc[-1], 1.0)

    def test_freq_all_identical(self):
        """Test with all identical sequences."""
        seqs = np.array([[1, 1, 1]] * 5)
        freq = seq_freq(seqs, normalize=True)

        assert len(freq) == 1
        assert freq.iloc[0]["count"] == 5.0
        assert freq.iloc[0]["proportion"] == 1.0

    def test_freq_supports_mixed_object_state_labels(self):
        seqs = np.empty((3, 2), dtype=object)
        seqs[:] = [[0, "A"], [0, "A"], ["0", "B"]]

        freq = seq_freq(seqs, top_n=None)

        assert list(freq["count"]) == [2.0, 1.0]
        assert set(freq["sequence"]) == {(0, "A"), ("0", "B")}

    def test_freq_coalesces_missing_values_and_has_stable_tie_order(self):
        seqs = np.empty((3, 2), dtype=object)
        seqs[:] = [[np.nan, "A"], [np.nan, "A"], [0, "B"]]

        freq = seq_freq(seqs, top_n=None)

        assert list(freq["count"]) == [2.0, 1.0]
        assert pd.isna(freq.iloc[0]["sequence"][0])
        assert freq.iloc[1]["sequence"] == (0, "B")

    def test_freq_empty_input_preserves_declared_schema(self):
        freq = seq_freq(np.empty((0, 2)), top_n=None)

        assert list(freq.columns) == ["sequence", "count", "proportion", "cumulative"]
        assert freq.empty


class TestSeqTrans:
    """Tests for transition matrix computation."""

    def test_trans_basic(self):
        """Test basic transition matrix."""
        seqs = np.array(
            [
                [0, 1, 1, 0],  # 0→1, 1→1, 1→0
                [0, 1, 0, 0],  # 0→1, 1→0, 0→0
                [1, 0, -1, -1],  # 1→0, 0→-1, -1→-1
            ]
        )

        trans = seq_trans(seqs, lag=1, normalize=True)

        assert isinstance(trans, pd.DataFrame)
        assert trans.index.name == "from"
        assert trans.columns.name == "to"

        # Should have 3 states: -1, 0, 1
        assert set(trans.index) == {-1, 0, 1}
        assert set(trans.columns) == {-1, 0, 1}

        # Rows should sum to 1 (probabilities)
        row_sums = trans.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)

    def test_trans_unnormalized(self):
        """Test transition counts (unnormalized)."""
        seqs = np.array([[0, 1, 1, 0], [0, 1, 0, 0]])

        trans = seq_trans(seqs, lag=1, normalize=False)

        # Count 0→1 transitions
        assert trans.loc[0, 1] == 2.0  # both sequences start with 0→1

        # Count 1→1 transitions
        assert trans.loc[1, 1] == 1.0  # first sequence has 1→1

        # Count 1→0 transitions
        assert trans.loc[1, 0] == 2.0  # both have 1→0

    def test_trans_lag2(self):
        """Test transitions with lag=2."""
        seqs = np.array([[0, 1, -1, 0, 1]])

        trans = seq_trans(seqs, lag=2, normalize=False)

        # Position 0→2: 0→-1
        # Position 1→3: 1→0
        # Position 2→4: -1→1

        assert trans.loc[0, -1] == 1.0
        assert trans.loc[1, 0] == 1.0
        assert trans.loc[-1, 1] == 1.0

    def test_trans_persistent_state(self):
        """Test transition probabilities for persistent state."""
        # All sequences stay in state 1
        seqs = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

        trans = seq_trans(seqs, lag=1, normalize=True)

        # 1→1 should have probability 1.0
        assert trans.loc[1, 1] == 1.0

    def test_trans_insufficient_length(self):
        """Test with sequences too short for given lag."""
        seqs = np.array([[0, 1], [1, 0]])

        trans = seq_trans(seqs, lag=2, normalize=True)

        # No transitions possible with lag=2 for length 2
        # Should return empty/zero matrix
        assert trans.shape[0] == trans.shape[1]

    def test_trans_rejects_non_integer_coded_states(self):
        with pytest.raises(ValueError, match="integer-coded states"):
            seq_trans(np.array([[0.2, 1.0], [0.9, 1.0]]))

    def test_trans_rejects_non_integer_lag(self):
        with pytest.raises(ValueError, match="lag must be an integer"):
            seq_trans(np.array([[0, 1], [1, 0]]), lag=1.5)

    def test_trans_sparse_integer_codes_do_not_allocate_by_label_span(self, monkeypatch):
        original_full = sequence_analysis.np.full

        def guarded_full(shape, *args, **kwargs):
            if isinstance(shape, int) and shape > 1000:
                raise AssertionError("state lookup allocated by numeric label span")
            return original_full(shape, *args, **kwargs)

        monkeypatch.setattr(sequence_analysis.np, "full", guarded_full)
        seqs = np.array([[0, 1_000_000], [1_000_000, 0]], dtype=np.int64)

        trans = seq_trans(seqs, normalize=False)

        assert list(trans.index) == [0, 1_000_000]
        assert trans.loc[0, 1_000_000] == 1.0
        assert trans.loc[1_000_000, 0] == 1.0


class TestSeqEntropy:
    """Tests for entropy computation."""

    def test_entropy_overall_all_unique(self):
        """Maximum entropy when all sequences are unique."""
        seqs = np.array([[0, 0], [1, 1], [0, 1]])

        ent = seq_entropy(seqs, by_time=False, base=2.0)

        # 3 unique sequences → entropy = log2(3)
        expected = np.log2(3)
        assert np.isclose(ent, expected)

    def test_entropy_overall_all_identical(self):
        """Zero entropy when all sequences are identical."""
        seqs = np.array([[1, 1, 1]] * 5)

        ent = seq_entropy(seqs, by_time=False)

        # All identical → entropy = 0
        assert np.isclose(ent, 0.0)

    def test_entropy_by_time(self):
        """Test per-position entropy."""
        seqs = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])

        ent = seq_entropy(seqs, by_time=True, base=2.0)

        assert len(ent) == 3  # 3 time points

        # Position 0: all same (0) → entropy ≈ 0
        assert np.isclose(ent[0], 0.0)

        # Position 1: 2 states (1,1,0,0) → entropy > 0
        assert ent[1] > 0

        # Position 2: 2 states (1,0,0,0) → entropy > 0 but < position 1
        assert ent[2] > 0

    def test_entropy_base_e(self):
        """Test entropy with base e (nats)."""
        seqs = np.array([[0, 0], [1, 1]])

        ent_bits = seq_entropy(seqs, base=2.0)
        ent_nats = seq_entropy(seqs, base=np.e)

        # Conversion: nats = bits * ln(2)
        assert np.isclose(ent_nats, ent_bits * np.log(2))

    def test_entropy_by_time_uniform(self):
        """Test per-position entropy with uniform distribution.

        Since v0.0.2, by_time entropy is normalized by log(K) by default
        (matching traminex.seqstatd), so a uniform full-alphabet column is 1.0.
        The unnormalized bits (log2(3)) remain available via normalize=False.
        """
        # At each position, all states equally represented
        seqs = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        # Default (normalize=True): each position is maximally diverse -> 1.0
        ent = seq_entropy(seqs, by_time=True, base=2.0)
        assert np.allclose(ent, 1.0)

        # normalize=False recovers the unnormalized bits: log2(3) per position
        ent_bits = seq_entropy(seqs, by_time=True, base=2.0, normalize=False)
        assert np.allclose(ent_bits, np.log2(3))

    def test_entropy_invalid_base(self):
        """Test invalid base parameter."""
        seqs = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError, match="base must be"):
            seq_entropy(seqs, base=1.0)

        with pytest.raises(ValueError, match="base must be"):
            seq_entropy(seqs, base=-1.0)

    def test_entropy_supports_mixed_object_state_labels(self):
        seqs = np.empty((3, 2), dtype=object)
        seqs[:] = [[0, "A"], [0, "A"], ["0", "B"]]

        overall = seq_entropy(seqs)
        by_time = seq_entropy(seqs, by_time=True)

        assert np.isclose(overall, -(2 / 3 * np.log2(2 / 3) + 1 / 3 * np.log2(1 / 3)))
        assert by_time.shape == (2,)
        assert np.all(np.isfinite(by_time))

    def test_entropy_coalesces_missing_values_as_one_state(self):
        seqs = np.array([[np.nan, 0.0], [np.nan, 1.0]])

        overall = seq_entropy(seqs)
        by_time = seq_entropy(seqs, by_time=True)

        assert np.isclose(overall, 1.0)
        assert np.isclose(by_time[0], 0.0)

    def test_entropy_rejects_empty_sequence_collection(self):
        with pytest.raises(ValueError, match="at least one sequence"):
            seq_entropy(np.empty((0, 2)))


class TestIntegration:
    """Integration tests combining multiple analysis functions."""

    def test_realistic_cluster_sequences(self):
        """Test with realistic cluster membership patterns."""
        # Simulate 10 areas over 6 time points
        # HOT=1, COLD=-1, none=0
        seqs = np.array(
            [
                [0, 1, 1, 1, 0, 0],  # Early HOT then fade
                [0, 0, 1, 1, 1, 0],  # Mid HOT
                [0, 0, 0, 1, 1, 1],  # Late HOT
                [-1, -1, 0, 0, 0, 0],  # Early COLD then none
                [0, 0, 0, 0, 0, 0],  # Never in cluster
                [0, 0, 0, 0, 0, 0],  # Never in cluster
                [1, 1, 1, 1, 1, 1],  # Always HOT
                [-1, -1, -1, -1, -1, -1],  # Always COLD
                [0, 1, 0, 1, 0, 1],  # Intermittent HOT
                [1, 0, -1, 0, 1, 0],  # Mixed pattern
            ]
        )

        # Test frequency
        freq = seq_freq(seqs, top_n=5)
        assert len(freq) <= 5
        assert freq["count"].sum() <= 10

        # Test transitions
        trans = seq_trans(seqs, lag=1, normalize=True)

        # Check that all probabilities are valid
        assert np.all((trans.values >= 0) & (trans.values <= 1))

        # Test overall entropy
        ent_overall = seq_entropy(seqs, by_time=False)
        assert ent_overall > 0  # Not all identical

        # Test per-time entropy
        ent_time = seq_entropy(seqs, by_time=True)
        assert len(ent_time) == 6
        assert np.all(ent_time >= 0)
