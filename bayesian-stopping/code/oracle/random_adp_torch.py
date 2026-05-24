"""
Random-variance ADP (Algorithm 2) — PyTorch / GPU implementation.

One-for-one port of v3/oracle/random_adp.py. Same math, same returned table
format (numpy arrays in a dict), drop-in compatible with the numpy `query`,
`save_table`, `load_table` helpers in `random_adp.py`. The DP runs on GPU;
the result is moved back to CPU/numpy at the end.

Bilinear lookup is hand-written (4-corner gather + blend) since
`scipy.ndimage.map_coordinates` has no torch equivalent — `grid_sample`
exists but normalizes coordinates and assumes image-axis conventions
that don't match our setup.

Memory note. At reference resolution (K1=K2=512, M=J_sigma=128, J=128) the
full (K1, K2, M, J) tensor is 4.3G float64 ≈ 34 GB. We chunk along K1 to
keep peak working memory well under the GPU's 48 GB; default chunk for
reference is 64.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_hermite


SIGMA_DISC = np.array([1.0, 10.0, 100.0])
DEFAULT_DEVICE = 'cuda:0'
DEFAULT_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# sigma grid construction (numpy; called once per solve)
# ---------------------------------------------------------------------------

def make_sigma_grid(
    distribution: str, J_sigma: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if distribution == 'disc':
        return SIGMA_DISC.copy(), np.zeros(SIGMA_DISC.shape[0])
    if distribution == 'logu':
        a, b = 0.0, np.log(100.0)
        nodes, weights = leggauss(J_sigma)
        log_sigma_grid = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        sigma_grid = np.exp(log_sigma_grid)
        omega = 0.5 * (b - a) * weights
        return sigma_grid, np.log(omega)
    raise ValueError(f"unknown distribution: {distribution!r}")


def make_xbar_grids(
    n: int, tau0_2: float, sigma_max: float, K1: int,
) -> np.ndarray:
    t_arr = np.arange(1, n, dtype=np.float64)
    half_w = 5.0 * np.sqrt(tau0_2 + sigma_max * sigma_max / t_arr)
    grids = np.empty((n - 1, K1), dtype=np.float64)
    for i in range(n - 1):
        grids[i] = np.linspace(-half_w[i], +half_w[i], K1)
    return grids


# ---------------------------------------------------------------------------
# torch helpers
# ---------------------------------------------------------------------------

def _marginal_log_likelihood_torch(t, S, Q, sigma2, tau0_2):
    """log p(X_{1:t} | sigma) up to const(t); torch version of Eq. 5."""
    r = t * tau0_2 / sigma2
    return (
        -0.5 * t * torch.log(sigma2)
        - 0.5 * torch.log1p(r)
        - 0.5 * (Q / sigma2 - (tau0_2 * S * S / (sigma2 * sigma2)) / (1.0 + r))
    )


def _bilinear_lookup_torch(
    values: torch.Tensor,           # (K1, K2)
    Xbar: torch.Tensor, L: torch.Tensor,
    Xbar0: float, Xbar_step: float,
    L0: float, L_step: float,
    K1: int, K2: int,
) -> torch.Tensor:
    """Bilinear interpolation with boundary clipping (mode='nearest')."""
    out_shape = torch.broadcast_shapes(Xbar.shape, L.shape)
    row = (Xbar.expand(out_shape) - Xbar0) / Xbar_step
    col = (L.expand(out_shape) - L0) / L_step
    r = torch.clamp(row, 0.0, K1 - 1.0)
    c = torch.clamp(col, 0.0, K2 - 1.0)
    r_lo = torch.floor(r).long()
    c_lo = torch.floor(c).long()
    r_hi = torch.clamp(r_lo + 1, max=K1 - 1)
    c_hi = torch.clamp(c_lo + 1, max=K2 - 1)
    fr = r - r_lo
    fc = c - c_lo
    v00 = values[r_lo, c_lo]
    v10 = values[r_hi, c_lo]
    v01 = values[r_lo, c_hi]
    v11 = values[r_hi, c_hi]
    return ((1 - fr) * (1 - fc) * v00 + fr * (1 - fc) * v10
            + (1 - fr) * fc * v01 + fr * fc * v11)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_random_adp_torch(
    sigma_grid: np.ndarray,
    log_omega: np.ndarray,
    n: int = 256,
    mu_0: float = 0.0,
    tau0_2: float = 100.0,
    sigma_max: float = None,
    K1: int = 256,
    K2: int = 256,
    J: int = 64,
    L_lim: Tuple[float, float] = None,
    chunk: int = None,
    device: str = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    verbose: bool = False,
) -> Dict:
    """Algorithm 2 backward induction on GPU.

    Returns a dict with the same keys as the numpy `solve_random_adp`
    (C_hat, Xbar_grids, L_grid, sigma_grid, log_omega, sigma_max, n, mu_0,
    tau0_2, K1, K2, J), all as numpy arrays / Python scalars on the way out.
    """
    if L_lim is None:
        L_lim = (np.log(0.1), np.log(1e5))
    if n < 2:
        raise ValueError("n must be >= 2")
    if sigma_max is None:
        sigma_max = float(np.asarray(sigma_grid).max())

    # Default chunk: tight for high-M reference solves so the largest per-chunk
    # tensor (K1c * K2 * M * J float64) stays under ~5 GB. At reference
    # (K1=K2=512, M=J=128) that means chunk <= 32. At production we don't
    # chunk because the full tensor is already small.
    M_provisional = int(np.asarray(sigma_grid).size)
    if chunk is None:
        if K1 >= 512 and M_provisional >= 64:
            chunk = 32
        elif K1 >= 512:
            chunk = 128
        else:
            chunk = K1

    # numpy setup (single-shot; small)
    Xbar_grids_np = make_xbar_grids(n, tau0_2, sigma_max, K1)
    L_grid_np = np.linspace(L_lim[0], L_lim[1], K2)
    z_nodes_np, w_nodes_np = roots_hermite(J)

    # to torch on the chosen device
    Xbar_grids = torch.tensor(Xbar_grids_np, dtype=dtype, device=device)
    L_grid = torch.tensor(L_grid_np, dtype=dtype, device=device)
    expL = torch.exp(L_grid)
    sigma_grid_t = torch.tensor(np.asarray(sigma_grid), dtype=dtype, device=device)
    sigma2_grid = sigma_grid_t * sigma_grid_t
    log_omega_t = torch.tensor(np.asarray(log_omega), dtype=dtype, device=device)
    z_nodes = torch.tensor(z_nodes_np, dtype=dtype, device=device)
    w_nodes = torch.tensor(w_nodes_np, dtype=dtype, device=device)

    M = sigma_grid_t.shape[0]
    inv_sqrt_pi = 1.0 / float(np.sqrt(np.pi))

    C_hat = torch.empty((n - 1, K1, K2), dtype=dtype, device=device)

    Xbar0_per_stage = Xbar_grids[:, 0]
    Xbar_step_per_stage = (Xbar_grids[:, -1] - Xbar_grids[:, 0]) / (K1 - 1)
    L0 = float(L_grid_np[0])
    L_step = float((L_grid_np[-1] - L_grid_np[0]) / (K2 - 1))

    chunk_ranges = [
        (k1_lo, min(k1_lo + chunk, K1)) for k1_lo in range(0, K1, chunk)
    ]

    def _terminal_chunk(k1_lo, k1_hi, t_term, Xbar_grid_term, tau2_term):
        Xbar_chunk = Xbar_grid_term[k1_lo:k1_hi]                                # (k1c,)
        S_chunk = t_term * Xbar_chunk                                            # (k1c,)
        Q_chunk = (
            (t_term - 1) * expL[None, :]
            + t_term * (Xbar_chunk * Xbar_chunk)[:, None]
        )                                                                        # (k1c, K2)
        S_b = S_chunk[:, None, None]
        Q_b = Q_chunk[:, :, None]
        log_marg = _marginal_log_likelihood_torch(
            t_term, S_b, Q_b, sigma2_grid, tau0_2,
        )
        log_w_unnorm = log_marg + log_omega_t
        log_w = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)
        w_sigma = torch.exp(log_w)                                               # (k1c, K2, M)
        mu_term = tau2_term * (mu_0 / tau0_2 + S_chunk[:, None] / sigma2_grid)
        return (w_sigma * mu_term[:, None, :]).sum(dim=-1)                       # (k1c, K2)

    def _recursion_chunk(k1_lo, k1_hi, t, Xbar_grid_t, Xbar_grid_next_idx,
                         tau2_t, sqrt_2v, C_next):
        k1c = k1_hi - k1_lo
        Xbar_chunk = Xbar_grid_t[k1_lo:k1_hi]
        S_chunk = t * Xbar_chunk
        Q_chunk = (t - 1) * expL[None, :] + t * (Xbar_chunk * Xbar_chunk)[:, None]
        S_b = S_chunk[:, None, None]
        Q_b = Q_chunk[:, :, None]
        log_marg = _marginal_log_likelihood_torch(
            t, S_b, Q_b, sigma2_grid, tau0_2,
        )
        log_w_unnorm = log_marg + log_omega_t
        log_w = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)
        w_sigma = torch.exp(log_w)                                               # (k1c, K2, M)

        mu_t_sig = tau2_t * (mu_0 / tau0_2 + S_chunk[:, None] / sigma2_grid)     # (k1c, M)
        # x: (k1c, M, J) — independent of K2, broadcast at lookup time
        x_km = mu_t_sig[:, :, None] + sqrt_2v[None, :, None] * z_nodes
        # successor S, Xbar at (k1c, M, J)
        S_next_km = S_chunk[:, None, None] + x_km
        Xbar_next_km = S_next_km / (t + 1)
        # Q depends on K2: (k1c, K2, M, J)
        Q_next = Q_chunk[:, :, None, None] + (x_km * x_km)[:, None, :, :]
        sigma_hat2_next = torch.clamp(
            (Q_next - (t + 1) * Xbar_next_km[:, None, :, :] ** 2) / t,
            min=1e-300,
        )
        L_next = torch.log(sigma_hat2_next)                                      # (k1c, K2, M, J)
        # Bilinear lookup at stage-(t+1)'s Xbar grid
        Xbar0_next = float(Xbar0_per_stage[Xbar_grid_next_idx].item())
        Xbar_step_next = float(Xbar_step_per_stage[Xbar_grid_next_idx].item())
        # Broadcast Xbar_next_km (k1c, M, J) -> (k1c, K2, M, J) via L_next's K2 axis
        Xbar_next_full = Xbar_next_km[:, None, :, :].expand(k1c, K2, M, J)
        C_lookup = _bilinear_lookup_torch(
            C_next, Xbar_next_full, L_next,
            Xbar0_next, Xbar_step_next, L0, L_step, K1, K2,
        )
        x_full = x_km[:, None, :, :].expand(k1c, K2, M, J)
        max_x_C = torch.maximum(x_full, C_lookup)
        gh_sum = inv_sqrt_pi * (w_nodes * max_x_C).sum(dim=-1)                   # (k1c, K2, M)
        return (w_sigma * gh_sum).sum(dim=-1)                                    # (k1c, K2)

    # ---- Terminal stage: t = n-1 -----
    t_term = n - 1
    Xbar_grid_term = Xbar_grids[t_term - 1]
    tau2_term = 1.0 / (1.0 / tau0_2 + t_term / sigma2_grid)
    for k1_lo, k1_hi in chunk_ranges:
        C_hat[t_term - 1, k1_lo:k1_hi] = _terminal_chunk(
            k1_lo, k1_hi, t_term, Xbar_grid_term, tau2_term,
        )
    if verbose:
        print(f'[random_adp_torch] terminal stage t={t_term} done')

    # ---- Recursion -----
    for i_target in range(t_term - 2, -1, -1):
        t = i_target + 1
        C_next = C_hat[i_target + 1]
        Xbar_grid_t = Xbar_grids[i_target]
        tau2_t = 1.0 / (1.0 / tau0_2 + t / sigma2_grid)
        v_t = tau2_t + sigma2_grid
        sqrt_2v = torch.sqrt(2.0 * v_t)
        for k1_lo, k1_hi in chunk_ranges:
            C_hat[i_target, k1_lo:k1_hi] = _recursion_chunk(
                k1_lo, k1_hi, t, Xbar_grid_t, i_target + 1,
                tau2_t, sqrt_2v, C_next,
            )
        if verbose and (t % 32 == 0 or t == 1):
            print(f'[random_adp_torch] stage t={t} done')

    return {
        'C_hat': C_hat.cpu().numpy(),
        'Xbar_grids': Xbar_grids.cpu().numpy(),
        'L_grid': L_grid.cpu().numpy(),
        'sigma_grid': sigma_grid_t.cpu().numpy(),
        'log_omega': log_omega_t.cpu().numpy(),
        'sigma_max': float(sigma_max),
        'n': n, 'mu_0': mu_0, 'tau0_2': tau0_2,
        'K1': K1, 'K2': K2, 'J': J,
    }
