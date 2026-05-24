"""
Closed-form Normal-Normal posterior helpers (Section 3.1 of v3 spec).

This module is the boundary that baselines and evaluation code import from.
It does not depend on either ADP module; it provides the per-step posterior
mathematics that downstream callers (labeling, baselines, evaluation) reuse
without pulling in DP machinery.

Contents:
  posterior_update           one-step conjugate update at known sigma.
  posterior_path_batch       full posterior-mean path at known sigma.
  compute_eta                known-mu standardized continuation values
                             (eq. eta_{n-1}=0; eta_t=psi(eta_{t+1})).
  marginal_log_likelihood    log p(X_{1:t} | sigma) at unknown sigma
                             (eq. (5) of v3 spec; derived in Appendix A).
  interp_uniform             linear interp on a uniform grid with
                             boundary clipping.

Indexing convention. The notes use 1-indexed time t in {1, ..., n}. Arrays
returned by `compute_eta` are 0-indexed on decision steps i in {0, ..., n-2}
with eta[i] = eta_{i+1}; this matches v2/code/oracle.py.
"""

from typing import Tuple

import numpy as np
from scipy.special import ndtr


# ---------------------------------------------------------------------------
# Posterior update at known sigma (Proposition / Eq. 3 of v3 spec)
# ---------------------------------------------------------------------------

def posterior_update(
    mu_prev: float, tau2_prev: float, x: float, sigma2: float,
) -> Tuple[float, float]:
    """One-step conjugate update: (mu_{t-1}, tau2_{t-1}, X_t) -> (mu_t, tau2_t)."""
    tau2_new = 1.0 / (1.0 / tau2_prev + 1.0 / sigma2)
    mu_new = tau2_new * (mu_prev / tau2_prev + x / sigma2)
    return mu_new, tau2_new


def posterior_path_batch(
    X_batch: np.ndarray, mu_0: float, tau0_2: float, sigma2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized closed-form posterior path at known sigma.

    Args:
        X_batch: (B, n) observations.
        mu_0, tau0_2: prior parameters.
        sigma2: known noise variance (scalar).

    Returns:
        mu_path:   (B, n) where mu_path[b, i] = posterior mean after observing
                   X_batch[b, :i+1]   (i.e. mu_{i+1} in 1-indexed notation).
        tau2_path: (n,) posterior variance after i+1 observations; data-free.
    """
    B, n = X_batch.shape
    t = np.arange(1, n + 1, dtype=float)
    tau2_path = 1.0 / (1.0 / tau0_2 + t / sigma2)
    S = np.cumsum(X_batch, axis=1)
    mu_path = tau2_path[None, :] * (mu_0 / tau0_2 + S / sigma2)
    return mu_path, tau2_path


# ---------------------------------------------------------------------------
# Known-mu eta_t recursion (Section 4.1 of v3 spec)
# ---------------------------------------------------------------------------

def _psi(z: np.ndarray) -> np.ndarray:
    """psi(z) = z * Phi(z) + phi(z)."""
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return z * ndtr(z) + phi


def compute_eta(n: int) -> np.ndarray:
    """Standardized known-mu thresholds eta_t for the plug-in / prior-only rules.

    Returns an array of length n-1 with eta[i] = eta_{i+1} (1-indexed). The base
    case is eta[n-2] = 0 (i.e. eta_{n-1} = 0); earlier entries are filled by
    eta[i] = psi(eta[i+1]). At step n-1 the agent must accept, so no eta is
    needed there.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    eta = np.zeros(n - 1)
    for i in range(n - 3, -1, -1):
        eta[i] = _psi(eta[i + 1])
    return eta


# ---------------------------------------------------------------------------
# Marginal log-likelihood log p(X_{1:t} | sigma) (Eq. 5 of v3 spec)
# ---------------------------------------------------------------------------

def marginal_log_likelihood(
    t, S, Q, sigma, tau0_2,
):
    """log p(X_{1:t} | sigma) up to const(t).

    Per Eq. (5) of v3 spec (derived in Appendix A):

        log p(X_{1:t} | sigma)
            = const(t)
              - (t/2) log sigma^2
              - (1/2) log(1 + t * tau_0^2 / sigma^2)
              - (1/2) [ Q / sigma^2
                        - tau_0^2 * S^2 / sigma^4 / (1 + t * tau_0^2 / sigma^2) ]

    The const(t) = -(t/2) log(2 pi) term is dropped: it cancels in any
    sigma-posterior normalization.

    Arguments may broadcast. t is a scalar (or array broadcastable to S, Q).
    sigma may be a scalar or any-shape array; the result broadcasts in the
    standard NumPy way.
    """
    sigma2 = sigma * sigma
    r = t * tau0_2 / sigma2
    return (
        -0.5 * t * np.log(sigma2)
        - 0.5 * np.log1p(r)
        - 0.5 * (Q / sigma2 - (tau0_2 * S * S / (sigma2 * sigma2)) / (1.0 + r))
    )


# ---------------------------------------------------------------------------
# Uniform-grid linear interpolation with boundary clipping
# ---------------------------------------------------------------------------

def interp_uniform(
    x: np.ndarray, grid: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    """Linear interp of `values` defined on a uniform `grid`, with boundary
    clipping (returns the boundary value for x outside the grid). Vectorized:
    `x` may have any shape; `grid` and `values` must be 1-D of equal length.
    """
    K = grid.shape[0]
    h = (grid[-1] - grid[0]) / (K - 1)
    idx = (x - grid[0]) / h
    idx_clip = np.clip(idx, 0.0, K - 1.0)
    lo = np.floor(idx_clip).astype(np.int64)
    lo = np.clip(lo, 0, K - 2)
    frac = idx_clip - lo
    return (1.0 - frac) * values[lo] + frac * values[lo + 1]
