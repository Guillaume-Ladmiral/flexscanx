"""Unit tests for sequence extraction utilities.

Copyright (c) 2025 flexscanx contributors
"""

import numpy as np
import pytest

from flexscanx.sequence import extract_sequences


class TestExtractSequences:
    """Tests for extract_sequences() function."""

    def test_extract_basic_hot(self):
        """Test basic extraction with HOT clusters only."""
        result = {
            "per_time": [
                {"area": [1, 2], "type": "HOT"},
                {"area": [1, 3], "type": "HOT"},
                {"area": [2, 3], "type": "HOT"},
            ]
        }

        seqs, ids = extract_sequences(result, area_ids=[1, 2, 3], include_type="hot")

        assert seqs.shape == (3, 3)  # 3 areas, 3 time points
        assert list(ids) == [1, 2, 3]

        # Area 1: in clusters at t=0,1
        assert list(seqs[0]) == [1, 1, 0]
        # Area 2: in clusters at t=0,2
        assert list(seqs[1]) == [1, 0, 1]
        # Area 3: in clusters at t=1,2
        assert list(seqs[2]) == [0, 1, 1]

    def test_extract_both_types(self):
        """Test extraction with both HOT and COLD."""
        result = {
            "per_time": [
                {"area": [1, 2], "type": "HOT"},
                {"area": [3], "type": "COLD"},
                {"area": [1], "type": "COLD"},
            ]
        }

        seqs, ids = extract_sequences(result, area_ids=[1, 2, 3], include_type="both")

        assert seqs.shape == (3, 3)

        # Area 1: HOT at t=0, COLD at t=2
        assert list(seqs[0]) == [1, 0, -1]
        # Area 2: HOT at t=0
        assert list(seqs[1]) == [1, 0, 0]
        # Area 3: COLD at t=1
        assert list(seqs[2]) == [0, -1, 0]

    def test_extract_cold_only(self):
        """Test extraction with COLD clusters only."""
        result = {
            "per_time": [
                {"area": [1], "type": "HOT"},  # should be ignored
                {"area": [2], "type": "COLD"},
                {"area": [1, 2], "type": "COLD"},
            ]
        }

        seqs, ids = extract_sequences(result, area_ids=[1, 2], include_type="cold")

        # HOT at t=0 is ignored
        assert list(seqs[0]) == [0, 0, -1]  # Area 1: only COLD at t=2
        assert list(seqs[1]) == [0, -1, -1]  # Area 2: COLD at t=1,2

    def test_extract_auto_area_ids(self):
        """Test automatic extraction of all area IDs."""
        result = {"per_time": [{"area": [5, 10], "type": "HOT"}, {"area": [3, 5], "type": "HOT"}]}

        seqs, ids = extract_sequences(result, area_ids=None, include_type="hot")

        # Should extract areas 3, 5, 10 (sorted)
        assert sorted(ids) == [3, 5, 10]
        assert seqs.shape[0] == 3
        assert seqs.shape[1] == 2

    def test_extract_no_clusters(self):
        """Test with time points that have no significant clusters."""
        result = {
            "per_time": [
                {"area": [1], "type": "HOT"},
                {"area": None, "type": "HOT"},  # No cluster at t=1
                {"area": [2], "type": "HOT"},
            ]
        }

        seqs, ids = extract_sequences(result, area_ids=[1, 2], include_type="hot")

        # t=1 should be all zeros
        assert np.all(seqs[:, 1] == 0)
        assert seqs[0, 0] == 1  # Area 1 at t=0
        assert seqs[1, 2] == 1  # Area 2 at t=2

    def test_extract_area_not_in_any_cluster(self):
        """Test with area that never appears in clusters."""
        result = {"per_time": [{"area": [1], "type": "HOT"}, {"area": [1], "type": "HOT"}]}

        seqs, ids = extract_sequences(result, area_ids=[1, 2, 3], include_type="hot")

        # Areas 2 and 3 never in clusters
        assert list(seqs[1]) == [0, 0]  # Area 2: all zeros
        assert list(seqs[2]) == [0, 0]  # Area 3: all zeros

    def test_extract_empty_per_time(self):
        """Test with empty per_time list."""
        result = {"per_time": []}

        with pytest.raises(ValueError, match="must contain 'per_time'"):
            extract_sequences(result, area_ids=[1, 2])

    def test_extract_missing_per_time(self):
        """Test with missing per_time key."""
        result = {}

        with pytest.raises(ValueError, match="must contain 'per_time'"):
            extract_sequences(result, area_ids=[1, 2])

    def test_extract_invalid_include_type(self):
        """Test with invalid include_type parameter."""
        result = {"per_time": [{"area": [1], "type": "HOT"}]}

        with pytest.raises(ValueError, match="include_type must be"):
            extract_sequences(result, include_type="invalid")

    def test_extract_no_areas_found(self):
        """Test when no areas appear in any clusters."""
        result = {"per_time": [{"area": None, "type": "HOT"}, {"area": [], "type": "HOT"}]}

        with pytest.raises(ValueError, match="No areas found"):
            extract_sequences(result, area_ids=None)

    def test_extract_preserves_order(self):
        """Test that area IDs are sorted consistently."""
        result = {"per_time": [{"area": [10, 5, 3], "type": "HOT"}]}

        seqs, ids = extract_sequences(result, area_ids=[10, 5, 3])

        # IDs should be sorted
        assert list(ids) == [3, 5, 10]

        # Rows should correspond to sorted IDs
        assert seqs[0, 0] == 1  # Area 3
        assert seqs[1, 0] == 1  # Area 5
        assert seqs[2, 0] == 1  # Area 10

    def test_extract_dtype(self):
        """Test output dtype is int8."""
        result = {"per_time": [{"area": [1, 2], "type": "HOT"}, {"area": [3], "type": "COLD"}]}

        seqs, ids = extract_sequences(result, area_ids=[1, 2, 3])

        assert seqs.dtype == np.int8
