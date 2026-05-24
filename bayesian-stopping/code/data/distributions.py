"""
V3 sampling distributions.

Five generators producing per-sequence (X, sigma_i, mu_i):

    D_1: sigma = 1     (data-dominant)
    D_2: sigma = 10    (balanced)
    D_3: sigma = 100   (prior-dominant)
    D_disc: sigma ~ Uniform{1, 10, 100} per sequence
    D_logu: log_10 sigma ~ Uniform(0, 2) per sequence
            (uniform on log-decade across [1, 100])

All distributions: mu_i ~ N(0, tau_0^2) with tau_0=10. Conditional on
(mu_i, sigma_i), X[t] ~ N(mu_i, sigma_i^2) iid for t=1..n=256.

sigma_i is returned per-sequence so the loss in `v3/model/losses.py` can
normalize by it. The model never sees sigma_i.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Spec constants (Section 3 of v3 spec)
# ---------------------------------------------------------------------------

N = 256
MU_0 = 0.0
TAU0_2 = 100.0
TAU_0 = 10.0

SIGMA_REGIMES = {1: 1.0, 2: 10.0, 3: 100.0}

SIGMA_DISC = np.array([1.0, 10.0, 100.0])
LOG10_SIGMA_LOW = 0.0
LOG10_SIGMA_HIGH = 2.0

STATIC_DISTRIBUTIONS = ('D_1', 'D_2', 'D_3')
RANDOM_DISTRIBUTIONS = ('D_disc', 'D_logu')
ALL_DISTRIBUTIONS = STATIC_DISTRIBUTIONS + RANDOM_DISTRIBUTIONS


def is_static(distribution: str) -> bool:
    return distribution in STATIC_DISTRIBUTIONS


def is_random(distribution: str) -> bool:
    return distribution in RANDOM_DISTRIBUTIONS


def static_sigma(distribution: str) -> float:
    """sigma value for a static-variance distribution; raises for random."""
    if distribution not in STATIC_DISTRIBUTIONS:
        raise ValueError(f"{distribution} is not a static-sigma distribution")
    regime_id = int(distribution.split('_')[1])
    return SIGMA_REGIMES[regime_id]


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample(
    distribution: str, N_seq: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw N_seq sequences from `distribution`.

    Args:
        distribution: one of D_1, D_2, D_3, D_disc, D_logu.
        N_seq: number of sequences.
        rng: numpy.random.Generator.

    Returns:
        X:       (N_seq, n) float64
        sigma_i: (N_seq,) float64 per-sequence true sigma
        mu_i:    (N_seq,) float64 per-sequence true mu
    """
    if distribution in STATIC_DISTRIBUTIONS:
        sigma_i = np.full(N_seq, static_sigma(distribution), dtype=np.float64)
    elif distribution == 'D_disc':
        sigma_i = rng.choice(SIGMA_DISC, size=N_seq).astype(np.float64)
    elif distribution == 'D_logu':
        log10_sigma = rng.uniform(LOG10_SIGMA_LOW, LOG10_SIGMA_HIGH, size=N_seq)
        sigma_i = (10.0 ** log10_sigma).astype(np.float64)
    else:
        raise ValueError(f"unknown distribution: {distribution!r}")

    mu_i = rng.normal(MU_0, TAU_0, size=N_seq)
    noise = rng.standard_normal(size=(N_seq, N))
    X = mu_i[:, None] + sigma_i[:, None] * noise
    return X, sigma_i, mu_i
