"""B5 parity: flexscanx.sequence plotting *data* vs traminex (Phase 3).

Per the migration plan, plots are compared at the data level, not pixel level:
assert the matrices/orderings that drive the figures, and align styling
separately. The three flexscanx plotters and their data oracles:

* ``seq_dplot``  -> stacked-bar heights reconstruct the per-position state
  distribution == ``traminex.seqstatd``'s ``Frequencies`` (L x K).
* ``seq_fplot``  -> horizontal-bar widths are the pattern proportions from
  ``seq_freq`` (which B4 pins to ``traminex.seqtab``).
* ``seq_iplot``  -> one image row per input sequence in input order (row r is
  sequence r, row 0 at top) — the same convention as
  ``traminex.sequence_index_plot``, which imshows ``x.data`` directly. flexscanx
  maps states to colors and traminex maps state codes through a colormap (a
  styling difference), but the row->sequence assignment is identical.

Figures are closed to keep the suite clean. traminex is used at test time only.
"""

from __future__ import annotations

import numpy as np  # noqa: E402
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
plt = pytest.importorskip("matplotlib.pyplot")
to_rgb = pytest.importorskip("matplotlib.colors").to_rgb

from flexscanx.sequence.analysis import seq_freq  # noqa: E402
from flexscanx.sequence.plotting import seq_dplot, seq_fplot, seq_iplot  # noqa: E402

from .traminex_parity_utils import to_traminex_sts  # noqa: E402

SYMMETRIC = np.array(
    [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, -1, -1],
        [0, 0, 0, 1],
        [1, 1, -1, 0],
        [-1, -1, 0, 1],
        [0, 1, 1, 1],
    ],
    dtype=int,
)

# Asymmetric: a column missing a state and unequal pattern counts.
ASYMMETRIC = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, -1, 0, 1],
        [1, 1, -1, 0, 1],
        [-1, 0, 1, 1, 0],
    ],
    dtype=int,
)

FIXTURES = {"symmetric": SYMMETRIC, "asymmetric": ASYMMETRIC}

# Missing values in leading, internal and trailing positions (FXA-184).
WITH_MISSING = np.array(
    [
        [0, 1, np.nan],
        [1, np.nan, 0],
        [0, 1, 1],
        [np.nan, 0, 1],
    ],
    dtype=float,
)


class TestB5DistributionPlot:
    def test_dplot_heights_match_seqstatd_frequencies(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            uniq = np.unique(seqs)
            sts = to_traminex_sts(traminex_mod, seqs)
            # state ordering must line up: ascending ints vs string-sorted alphabet
            assert [str(s) for s in uniq] == list(sts.alphabet), name

            T, K = seqs.shape[1], len(uniq)
            fig, ax = seq_dplot(seqs)
            heights = np.zeros((T, K))
            for i, container in enumerate(ax.containers):
                for t, patch in enumerate(container):
                    heights[t, i] = patch.get_height()
            plt.close(fig)

            freqs = np.asarray(traminex_mod.seqstatd(sts)["Frequencies"], dtype=float)
            np.testing.assert_allclose(heights, freqs, atol=1e-12, err_msg=name)

    def test_dplot_missing_state_matches_seqstatd_include_missing_true_only(self, traminex_mod):
        """NOTICE.txt: heights == seqstatd(include_missing=True) Frequencies, per state label.

        flexscanx draws the missing state as its own labelled bar; traminex appends the
        not-recorded state as the last alphabet entry.  Only the labelled per-state
        values are compared, and the include_missing=False oracle must differ.
        """
        sts = to_traminex_sts(traminex_mod, WITH_MISSING)
        alphabet = list(sts.alphabet)
        assert alphabet[-1] == "*"
        fig, ax = seq_dplot(WITH_MISSING)
        heights_by_label = {}
        for container in ax.containers:
            heights_by_label[container.get_label()] = np.array(
                [patch.get_height() for patch in container], dtype=float
            )
        plt.close(fig)
        flexscanx_label = {"0": "None", "1": "HOT", "-1": "COLD", "*": "Missing"}
        with_missing = np.asarray(
            traminex_mod.seqstatd(sts, include_missing=True)["Frequencies"], dtype=float
        )
        without_missing = np.asarray(
            traminex_mod.seqstatd(sts, include_missing=False)["Frequencies"], dtype=float
        )
        # seqstatd's Frequencies matrix is (T positions x K states).
        assert with_missing.shape == (WITH_MISSING.shape[1], len(alphabet))
        for state_index, state in enumerate(alphabet):
            fx = heights_by_label[flexscanx_label[state]]
            np.testing.assert_allclose(fx, with_missing[:, state_index], atol=1e-12, err_msg=state)
        assert not np.allclose(with_missing, without_missing, atol=1e-6)


class TestB5FrequencyPlot:
    def test_fplot_widths_match_seq_freq_proportions(self, traminex_mod):
        for name, seqs in FIXTURES.items():
            top_n = 5
            fig, ax = seq_fplot(seqs, top_n=top_n)
            widths = np.array([p.get_width() for p in ax.patches], dtype=float)
            plt.close(fig)

            props = seq_freq(seqs, top_n=top_n, normalize=True)["proportion"].to_numpy(float)
            # seq_freq is pinned to traminex.seqtab in B4; here we confirm the
            # plot draws exactly those proportions (order-insensitive).
            np.testing.assert_allclose(np.sort(widths), np.sort(props), atol=1e-12, err_msg=name)


class TestB5IndexPlot:
    def test_iplot_preserves_input_row_order(self, traminex_mod):
        """Row r of the image is input sequence r (matches sequence_index_plot)."""
        default_colors = {-1: "#4575b4", 0: "#f7f7f7", 1: "#d73027"}
        for name, seqs in FIXTURES.items():
            N, T = seqs.shape
            fig, ax = seq_iplot(seqs)
            img = np.asarray(ax.images[0].get_array(), dtype=float)
            plt.close(fig)
            assert img.shape == (N, T, 3), name
            expected = np.zeros((N, T, 3))
            for state, color in default_colors.items():
                expected[seqs == state] = to_rgb(color)
            np.testing.assert_allclose(img, expected, atol=1e-12, err_msg=name)

    def test_traminex_index_plot_uses_same_row_order(self, traminex_mod):
        """traminex.sequence_index_plot imshows x.data in input order too."""
        seqs = SYMMETRIC
        sts = to_traminex_sts(traminex_mod, seqs)
        fig, ax = traminex_mod.sequence_index_plot(sts)
        data = np.asarray(ax.images[0].get_array())
        plt.close(fig)
        # The imshown code matrix is x.data, row r == input sequence r.
        np.testing.assert_array_equal(data, sts.data)
