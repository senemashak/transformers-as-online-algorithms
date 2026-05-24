"""
Static-variance ADP (Algorithm 1, Section 3.1 of v3 spec).

Verbatim port of v2/code/oracle.py:solve_adp, with the conjugate-math
helpers moved to v3/oracle/conjugate.py and the known-mu eta_t recursion
factored out so it does not pull in DP machinery.

State: posterior mean mu_t (1D). Per-stage uniform grid of K points on
[mu_0 - 5 s_t, mu_0 + 5 s_t] with s_t = sqrt(tau_0^2 - tau_t^2); J-point
Gauss-Hermite for the inner Gaussian expectation; linear interpolation
with boundary clipping for off-grid lookup.

Indexing convention. Decision step i in {0, ..., n-2} maps to 1-indexed
stage t = i+1. Returned arrays are indexed by i.
"""

from typing import Tuple

import numpy as np
from scipy.special import roots_hermite

from .conjugate import interp_uniform


def solve_adp(
    n: int,
    mu_0: float,
    sigma2: float,
    tau0_2: float,
    K: int = 2048,
    J: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the approximate backward induction of Algorithm 1.

    Returns:
        C_hat: (n-1, K) threshold table; C_hat[i, k] approximates
               C_{i+1}(m_{i+1}^{(k+1)}).
        grids: (n-1, K) per-stage grids m_t.
    """
    if n < 2 or K < 2 or J < 1:
        raise ValueError("require n >= 2, K >= 2, J >= 1")

    t_arr = np.arange(1, n, dtype=float)                       # 1, ..., n-1
    tau2 = 1.0 / (1.0 / tau0_2 + t_arr / sigma2)               # (n-1,)
    v = tau2 + sigma2                                          # predictive var
    alpha = sigma2 / v                                         # update weight

    half_w = np.sqrt(np.maximum(tau0_2 - tau2, 0.0))           # (n-1,)
    grids = np.empty((n - 1, K))
    for i in range(n - 1):
        grids[i] = np.linspace(mu_0 - 5.0 * half_w[i], mu_0 + 5.0 * half_w[i], K)

    z_nodes, w_nodes = roots_hermite(J)
    C_hat = np.empty((n - 1, K))
    C_hat[n - 2] = grids[n - 2]                                # C_{n-1}(mu) = mu

    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    for i in range(n - 3, -1, -1):
        m = grids[i]
        s2v = np.sqrt(2.0 * v[i])
        x = m[:, None] + s2v * z_nodes[None, :]                # (K, J)
        mu_prime = alpha[i] * m[:, None] + (1.0 - alpha[i]) * x
        C_next = interp_uniform(mu_prime, grids[i + 1], C_hat[i + 1])
        S = (w_nodes[None, :] * np.maximum(x, C_next)).sum(axis=1)
        C_hat[i] = inv_sqrt_pi * S

    return C_hat, grids


def C_hat_lin(
    t: int, mu_prime: np.ndarray, C_hat: np.ndarray, grids: np.ndarray,
) -> np.ndarray:
    """Off-grid lookup at decision step `t` (0-indexed in {0, ..., n-2}) with
    boundary clipping. `mu_prime` may be a scalar or any-shape array.
    """
    return interp_uniform(np.asarray(mu_prime), grids[t], C_hat[t])


def query(
    table: dict, t_one_indexed: int, mu_t: np.ndarray,
) -> np.ndarray:
    """Static ADP query interface. Returns Ĉ_t at posterior-mean state(s) mu_t.

    `t_one_indexed` is the 1-indexed time step in {1, ..., n-1}; we convert
    internally. `table` is a dict with keys 'C_hat' and 'grids'.
    """
    return C_hat_lin(t_one_indexed - 1, mu_t, table['C_hat'], table['grids'])
