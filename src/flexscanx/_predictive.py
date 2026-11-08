"""Predictive (cross-validation) cluster validation.

Moved from validation.py (B1 split) — all external imports go through
validation.py which re-exports the public names.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._utils import (
    validate_binomial_population,
    validate_expected_array,
    validate_integer_parameter,
    validate_observed_array,
)


def cv_glm_gain(
    observed: Sequence[int],
    expected: Sequence[float],
    membership: Sequence[int],
    k: int = 5,
    seed: int | None = None,
    model: str = "POISSON",
) -> dict[str, Any]:
    """Estimate out-of-fold log-likelihood gain from cluster membership.

    Parameters
    ----------
    observed : sequence of int
        Observed counts, length N.
    expected : sequence of float
        Expected counts (POISSON) or population sizes (BINOMIAL), length N.
    membership : sequence of int
        Binary membership vector (1 = in-cluster, 0 = out-of-cluster), length N.
    k : int, default 5
        Number of cross-validation folds.
    seed : int, optional
        RNG seed for fold assignment reproducibility.
    model : str, default "POISSON"
        ``'POISSON'`` or ``'BINOMIAL'``.

    Returns
    -------
    dict
        Keys: ``'d_ll'`` (mean log-likelihood gain), ``'d_dev'`` (deviance gain),
        ``'folds'`` (list of per-fold records with fold index and per-fold gains).
    """
    model_u = str(model).strip().upper()
    if model_u not in {"POISSON", "BINOMIAL"}:
        raise ValueError(f"model must be one of ['BINOMIAL', 'POISSON'], got {model!r}")

    k_valid = validate_integer_parameter(k, "k", minimum=2)
    seed_valid = None if seed is None else validate_integer_parameter(seed, "seed", minimum=0)
    obs = validate_observed_array(observed).astype(np.float64)
    if model_u == "BINOMIAL":
        exp = validate_binomial_population(expected)
    else:
        exp = validate_expected_array(expected, allow_zero=True)
    membership_raw = np.asarray(membership)
    if membership_raw.dtype.kind not in "biu":
        raise ValueError("membership must contain only 0 or 1")
    x = membership_raw.astype(np.int8, copy=False)
    if obs.ndim != 1 or exp.ndim != 1 or x.ndim != 1:
        raise ValueError("observed, expected, and membership must be 1D arrays")
    if not (len(obs) == len(exp) == len(x)):
        raise ValueError(
            "observed, expected, and membership must have the same length "
            f"(got {len(obs)}, {len(exp)}, {len(x)})"
        )
    if np.any((x != 0) & (x != 1)):
        raise ValueError("membership must contain only 0 or 1")
    if model_u == "BINOMIAL" and np.any(obs > exp):
        raise ValueError("observed counts cannot exceed population for BINOMIAL model")

    n_obs = len(obs)
    if k_valid > n_obs:
        raise ValueError(f"k must be <= number of observations ({n_obs}), got {k_valid}")
    rng = np.random.default_rng(seed_valid)
    idx = np.arange(n_obs)
    rng.shuffle(idx)
    folds = np.array_split(idx, k_valid)
    records: list[dict[str, float]] = []
    eps = 1e-12

    for fold_idx, test_idx in enumerate(folds):
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[test_idx] = False
        in_cluster = (x == 1) & train_mask
        out_cluster = (x == 0) & train_mask
        n_a, m_a = float(obs[in_cluster].sum()), float(exp[in_cluster].sum())
        n_b, m_b = float(obs[out_cluster].sum()), float(exp[out_cluster].sum())
        n_t, m_t = float(obs[train_mask].sum()), float(exp[train_mask].sum())

        if model_u == "BINOMIAL":
            p0 = (n_t / m_t) if m_t > 0 else 0.0
            p_b = (n_b / m_b) if m_b > 0 else p0
            p_a = (n_a / m_a) if m_a > 0 else p0
            p0 = min(max(p0, eps), 1.0 - eps)
            p_b = min(max(p_b, eps), 1.0 - eps)
            p_a = min(max(p_a, eps), 1.0 - eps)
            y_test = obs[test_idx]
            m_test = exp[test_idx]
            ll0 = float(np.sum(y_test * np.log(p0) + (m_test - y_test) * np.log(1.0 - p0)))
            p1 = np.where(x[test_idx] == 1, p_a, p_b)
            ll1 = float(np.sum(y_test * np.log(p1) + (m_test - y_test) * np.log(1.0 - p1)))
        else:
            rate0 = (n_t / m_t) if m_t > 0 else 0.0
            rate_b = (n_b / m_b) if m_b > 0 else rate0
            rate_a = (n_a / m_a) if m_a > 0 else rate0
            mu0 = exp[test_idx] * rate0
            mu1 = exp[test_idx] * np.where(x[test_idx] == 1, rate_a, rate_b)
            y_test = obs[test_idx]
            ll0 = float(np.sum(y_test * np.log(mu0 + eps) - mu0))
            ll1 = float(np.sum(y_test * np.log(mu1 + eps) - mu1))

        records.append(
            {
                "fold": int(fold_idx + 1),
                "ll_with": ll1,
                "ll_base": ll0,
                "d_ll": ll1 - ll0,
                "d_dev": 2.0 * (ll1 - ll0),
            }
        )

    d_ll = float(np.mean([record["d_ll"] for record in records])) if records else float("nan")
    return {"d_ll": d_ll, "d_dev": 2.0 * d_ll, "folds": records}
