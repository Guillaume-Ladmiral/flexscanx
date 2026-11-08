"""Regression tests for the 2026-08-16 sequence remediation tranche."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from flexscanx.sequence import seq_freq, seq_iplot


@pytest.mark.parametrize(
    ("max_display", "message"),
    [
        (0, "max_display must be >= 1"),
        (-1, "max_display must be >= 1"),
        (True, "max_display must be an integer"),
        (1.5, "max_display must be an integer"),
        (np.nan, "max_display must be an integer"),
    ],
)
def test_seq_iplot_requires_positive_integer_max_display(max_display: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        seq_iplot(np.array([[0, 1], [1, 0]]), max_display=max_display)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("shape", "message"),
    [((0, 2), "at least one sequence"), ((2, 0), "at least one time point")],
)
def test_seq_iplot_rejects_empty_sequence_matrix(shape: tuple[int, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        seq_iplot(np.empty(shape, dtype=int))


def test_seq_iplot_validates_before_loading_optional_plot_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "matplotlib":
            raise AssertionError("matplotlib loaded before public input validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="max_display must be >= 1"):
        seq_iplot(np.array([[0, 1], [1, 0]]), max_display=0)


def test_seq_freq_uses_joined_label_order_at_tied_prefix_cutoff() -> None:
    sequences = np.array([[1, 2], [10, 1], [2, 0]], dtype=int)

    result = seq_freq(sequences, top_n=1, normalize=False)

    assert result.loc[0, "sequence"] == (10, 1)


def test_missing_state_parity_contract_names_explicit_oracle_flag() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notice = (project_root / "src/flexscanx/sequence/NOTICE.txt").read_text(encoding="utf-8")

    from flexscanx.sequence import seq_dplot, seq_entropy

    assert "seqstatd(include_missing=True)" in notice
    assert "include_missing=True" in (seq_entropy.__doc__ or "")
    assert "include_missing=True" in (seq_dplot.__doc__ or "")


def test_seq_iplot_valid_positive_limit_still_truncates() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = seq_iplot(np.array([[0, 1], [1, 0]]), max_display=1)
    try:
        assert np.asarray(ax.images[0].get_array()).shape == (1, 2, 3)
    finally:
        plt.close(fig)
