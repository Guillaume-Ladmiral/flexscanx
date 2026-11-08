import numpy as np

from flexscanx.validation import _alias_build, _alias_multinomial_counts


def test_alias_matches_probabilities_small_K():
    rng = np.random.default_rng(123)
    # Include zeros and skew to test edge handling
    p = np.array([0.0, 0.1, 0.0, 0.9], dtype=float)
    prob, alias = _alias_build(p)
    n = 200_000
    counts = _alias_multinomial_counts(prob, alias, n, rng)
    freqs = np.array(counts, dtype=float) / float(n)
    # Within ~5e-3 tolerance due to randomness
    assert np.allclose(freqs, p, atol=5e-3)


def test_alias_large_K_sum_and_shape():
    rng = np.random.default_rng(1)
    K = 1000
    p = rng.random(K)
    p /= p.sum()
    prob, alias = _alias_build(p)
    n = 5000
    counts = _alias_multinomial_counts(prob, alias, n, rng)
    counts = np.asarray(counts)
    assert counts.shape == (K,)
    assert counts.sum() == n
    assert (counts >= 0).all()


def test_alias_vs_numpy_multinomial_mean():
    rng = np.random.default_rng(7)
    K = 20
    p = rng.random(K)
    p /= p.sum()
    prob, alias = _alias_build(p)
    n = 100_000
    counts_alias = np.asarray(_alias_multinomial_counts(prob, alias, n, rng))
    # Use a fresh RNG with the same seed for numpy multinomial comparison
    rng2 = np.random.default_rng(7)
    counts_np = rng2.multinomial(n, p, size=1)[0]
    freq_alias = counts_alias / float(n)
    freq_np = counts_np.astype(float) / float(n)
    # Both should be close to true p; and close to each other
    assert np.allclose(freq_alias, p, atol=5e-3)
    assert np.allclose(freq_np, p, atol=5e-3)
    assert np.allclose(freq_alias, freq_np, atol=8e-3)
