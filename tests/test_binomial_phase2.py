import numpy as np
import pytest

from flexscanx.validation import calibrate_llr_threshold, cv_glm_gain, morans_I_within


def test_calibrate_binomial_smoke():
    # Small synthetic population and cases
    pop = np.array([50, 60, 70, 80, 90], dtype=float)
    p = 0.1
    rng = np.random.default_rng(123)
    obs = rng.binomial(pop.astype(int), p).astype(int).tolist()
    # Dummy CSR for 5 nodes (line)
    row_ptr = [0, 1, 3, 5, 7, 8]
    col_idx = [1, 0, 2, 1, 3, 2, 4, 3]
    # Coords are irrelevant for calibration (handled by handle)
    coords = np.column_stack([np.arange(5), np.zeros(5)])
    cal = calibrate_llr_threshold(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=obs,
        population=pop.tolist(),
        model="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        n_sims=19,
        alpha=0.2,
        seed=7,
    )
    assert "tau" in cal and np.isfinite(cal["tau"])
    independent = calibrate_llr_threshold(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=obs,
        population=pop.tolist(),
        model="BINOMIAL",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustertype="HOT",
        K=3,
        clustersize=3,
        n_sims=19,
        alpha=0.2,
        seed=8,
    )
    assert any(cal[key] != independent[key] for key in ("tau", "mu", "sigma"))


def test_morans_I_within_binomial_smoke():
    pop = np.array([40, 50, 60, 70], dtype=float)
    rng = np.random.default_rng(7)
    obs = rng.binomial(pop.astype(int), 0.12).astype(int)
    # 4-node line
    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]
    membership = [1, 1, 0, 0]
    moran_I = morans_I_within(obs, pop, row_ptr, col_idx, membership, model="BINOMIAL")
    assert moran_I == pytest.approx(-1.0)


def test_cv_glm_gain_binomial_smoke():
    pop = np.array([100, 120, 80, 90, 110], dtype=float)
    rng = np.random.default_rng(9)
    obs = rng.binomial(pop.astype(int), 0.08).astype(int)
    membership = [1, 0, 1, 0, 0]
    cv = cv_glm_gain(obs, pop, membership, k=3, seed=3, model="BINOMIAL")
    assert np.isfinite(cv["d_dev"])
