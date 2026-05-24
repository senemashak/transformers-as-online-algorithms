"""
Random-variance ADP (Algorithm 2, Section 3.2 of v3 spec).

State: (S_t, Q_t = sum X_s^2) discretized in transformed coordinates
(X_bar_t, L_t = log sigma_hat_t^2) on a 2D uniform grid with bilinear
interpolation. The bijection between (X_bar_t, L_t) and (S_t, Q_t) is

    S_t = t * X_bar_t,
    Q_t = (t - 1) * exp(L_t) + t * X_bar_t^2.

For each (X_bar, L) grid point we recover (S_t, Q_t), compute the
sigma-posterior via the marginal log-likelihood (Eq. 5 of v3 spec),
mix-quadrature over sigma (M components), and J Gauss-Hermite nodes per
sigma-component for the inner Gaussian expectation over X_{t+1}.

Per-stage adaptive X_bar bounds. Var(X_bar_t) = tau_0^2 + sigma^2 / t, so
the marginal SD only converges down to tau_0; at small t and large sigma
it is much larger. We use per-stage half-width

    s_Xbar_t = sqrt(tau_0^2 + sigma_max^2 / t),

where sigma_max is the maximum of the prior's support. The X_bar grid at
stage t is K1 uniform points on [-5 * s_Xbar_t, +5 * s_Xbar_t].
The L_t grid is the same at every stage.

For D_disc the sigma-grid is fixed at {1, 10, 100} with log_omega = 0.
For D_logu the sigma-grid is J_sigma Gauss-Legendre nodes on
log sigma in [0, log 100] with log_omega = log(GL weights on that interval).
The mixture weight at the k-th node is w_k = omega_k * post_unnorm_k / Z_t
(prior cancels in normalization for both priors).

Off-grid lookup uses 2D bilinear interpolation via
scipy.ndimage.map_coordinates(order=1, mode='nearest') for boundary clipping.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import map_coordinates
from scipy.special import logsumexp, roots_hermite

from .conjugate import marginal_log_likelihood


# scipy.ndimage.map_coordinates and most numpy elementwise ops release the
# GIL, so threading the per-chunk work parallelizes nearly linearly with
# the thread count. 8 keeps memory pressure manageable on workstation
# configurations with ~16 CPU cores.
DEFAULT_N_THREADS = 8


SIGMA_DISC = np.array([1.0, 10.0, 100.0])


# ---------------------------------------------------------------------------
# sigma grid construction
# ---------------------------------------------------------------------------

def make_sigma_grid(
    distribution: str, J_sigma: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma_grid, log_omega) for the sigma-prior of `distribution`.

    'disc' returns the 3-point discrete grid with log_omega = 0.
    'logu' returns J_sigma Gauss-Legendre nodes on log sigma in [0, log 100]
    with log_omega = log of the GL weights on that interval (the (b-a)/2
    Jacobian is folded into the weights). The prior on log sigma is constant
    on this interval; it cancels in the mixture normalization.
    """
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


# ---------------------------------------------------------------------------
# Adaptive X_bar bounds
# ---------------------------------------------------------------------------

def make_xbar_grids(
    n: int, tau0_2: float, sigma_max: float, K1: int,
) -> np.ndarray:
    """Per-stage uniform X_bar grids of K1 points each.

    Half-width at stage t (1-indexed) is 5 * sqrt(tau_0^2 + sigma_max^2 / t).
    Returns an (n-1, K1) array indexed by 0-indexed decision step i = t - 1.
    """
    t_arr = np.arange(1, n, dtype=np.float64)               # 1, ..., n-1
    half_w = 5.0 * np.sqrt(tau0_2 + sigma_max * sigma_max / t_arr)
    grids = np.empty((n - 1, K1), dtype=np.float64)
    for i in range(n - 1):
        grids[i] = np.linspace(-half_w[i], +half_w[i], K1)
    return grids


# ---------------------------------------------------------------------------
# 2D bilinear lookup with boundary clipping (one stage)
# ---------------------------------------------------------------------------

def _bilinear_lookup(
    values: np.ndarray,                 # (K1, K2) — C_hat at one stage
    Xbar: np.ndarray, L: np.ndarray,    # query points; broadcast-compatible
    Xbar_grid_t: np.ndarray,            # (K1,) — that stage's X_bar grid
    L_grid: np.ndarray,                 # (K2,) — global L grid
) -> np.ndarray:
    K1 = Xbar_grid_t.shape[0]
    K2 = L_grid.shape[0]
    Xbar0 = Xbar_grid_t[0]
    Xbar_step = (Xbar_grid_t[-1] - Xbar0) / (K1 - 1)
    L0 = L_grid[0]
    L_step = (L_grid[-1] - L0) / (K2 - 1)
    out_shape = np.broadcast_shapes(Xbar.shape, L.shape)
    row_idx = (np.broadcast_to(Xbar, out_shape) - Xbar0) / Xbar_step
    col_idx = (np.broadcast_to(L, out_shape) - L0) / L_step
    coords = np.stack([row_idx.ravel(), col_idx.ravel()], axis=0)
    flat = map_coordinates(values, coords, order=1, mode='nearest', prefilter=False)
    return flat.reshape(out_shape)


# ---------------------------------------------------------------------------
# sigma-posterior at grid points
# ---------------------------------------------------------------------------

def _sigma_posterior_weights(
    t: int,
    S_chunk: np.ndarray,          # (k1c, 1)
    Q_chunk: np.ndarray,          # (k1c, K2)
    sigma2_grid: np.ndarray,      # (M,)
    tau0_2: float,
    log_omega: np.ndarray,        # (M,)
) -> np.ndarray:
    """w_sigma of shape (k1c, K2, M) summing to 1 along the last axis.

    Computes log w_unnorm_k = marginal_log_likelihood_k + log_omega_k via
    Eq. (5), then normalizes via logsumexp.
    """
    S_b = S_chunk[..., None]              # (k1c, 1, 1)
    Q_b = Q_chunk[..., None]              # (k1c, K2, 1)
    sigma2_b = sigma2_grid                # (M,)

    log_marg = marginal_log_likelihood(t, S_b, Q_b, np.sqrt(sigma2_b), tau0_2)
    log_w_unnorm = log_marg + log_omega
    log_w = log_w_unnorm - logsumexp(log_w_unnorm, axis=-1, keepdims=True)
    return np.exp(log_w)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_random_adp(
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
    chunk: int = 32,
    n_threads: int = DEFAULT_N_THREADS,
    verbose: bool = False,
) -> Dict:
    """Algorithm 2 backward induction with per-stage adaptive X_bar bounds.

    Args:
        sigma_grid:  (M,) sigma mixture nodes.
        log_omega:   (M,) log of mixture weights.
        sigma_max:   sup of the prior's support; used to size the per-stage
                     X_bar bounds. Defaults to max(sigma_grid).
        Other args follow the v3 spec defaults (n=256, mu_0=0, tau_0^2=100,
        K1=K2=256, J=64, L_lim=(log 0.1, log 1e5), chunk=32).

    Returns dict with keys:
        C_hat:       (n-1, K1, K2) threshold table; C_hat[i] approximates
                     C_{i+1} on the i-th stage's grid.
        Xbar_grids:  (n-1, K1) per-stage X_bar grids.
        L_grid:      (K2,) uniform L grid (stage-invariant).
        sigma_grid:  (M,) sigma mixture nodes.
        log_omega:   (M,) log of mixture weights.
        sigma_max:   sup of the prior's support that sized the X_bar bounds.
        n, mu_0, tau0_2, K1, K2, J: solver hyperparameters.
    """
    if L_lim is None:
        L_lim = (np.log(0.1), np.log(1e5))
    if n < 2:
        raise ValueError("n must be >= 2")

    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
    log_omega = np.asarray(log_omega, dtype=np.float64)
    sigma2_grid = sigma_grid * sigma_grid
    M = sigma_grid.shape[0]
    if sigma_max is None:
        sigma_max = float(sigma_grid.max())

    Xbar_grids = make_xbar_grids(n, tau0_2, sigma_max, K1)      # (n-1, K1)
    L_grid = np.linspace(L_lim[0], L_lim[1], K2)

    z_nodes, w_nodes = roots_hermite(J)
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

    C_hat = np.empty((n - 1, K1, K2), dtype=np.float64)
    expL = np.exp(L_grid)                          # (K2,)

    chunk_ranges = [
        (k1_lo, min(k1_lo + chunk, K1)) for k1_lo in range(0, K1, chunk)
    ]

    # ---- Per-chunk worker for the terminal stage --------------------------
    def _terminal_chunk(k1_range, t_term, Xbar_grid_term, tau2_term):
        k1_lo, k1_hi = k1_range
        Xbar_chunk = Xbar_grid_term[k1_lo:k1_hi]
        S_chunk = t_term * Xbar_chunk
        Q_chunk = (
            (t_term - 1) * expL[None, :]
            + t_term * (Xbar_chunk * Xbar_chunk)[:, None]
        )
        w_sigma = _sigma_posterior_weights(
            t_term, S_chunk[:, None], Q_chunk, sigma2_grid, tau0_2, log_omega,
        )
        mu_term = tau2_term * (mu_0 / tau0_2 + S_chunk[:, None] / sigma2_grid)
        return k1_lo, k1_hi, (w_sigma * mu_term[:, None, :]).sum(axis=-1)

    # ---- Per-chunk worker for the recursion ------------------------------
    def _recursion_chunk(
        k1_range, t, Xbar_grid_t, Xbar_grid_next,
        tau2_t, sqrt_2v, C_next,
    ):
        k1_lo, k1_hi = k1_range
        k1c = k1_hi - k1_lo
        Xbar_chunk = Xbar_grid_t[k1_lo:k1_hi]
        S_chunk = t * Xbar_chunk
        Q_chunk = (t - 1) * expL[None, :] + t * (Xbar_chunk * Xbar_chunk)[:, None]
        w_sigma = _sigma_posterior_weights(
            t, S_chunk[:, None], Q_chunk, sigma2_grid, tau0_2, log_omega,
        )
        mu_t_sig = tau2_t * (mu_0 / tau0_2 + S_chunk[:, None] / sigma2_grid)
        x = (
            mu_t_sig[:, None, :, None]
            + sqrt_2v[None, None, :, None] * z_nodes
        )
        x = np.broadcast_to(x, (k1c, K2, M, J))
        S_next = S_chunk[:, None, None, None] + x
        Q_next = Q_chunk[:, :, None, None] + x * x
        Xbar_next = S_next / (t + 1)
        sigma_hat2_next = np.maximum(
            (Q_next - (t + 1) * Xbar_next * Xbar_next) / t, 1e-300,
        )
        L_next = np.log(sigma_hat2_next)
        C_lookup = _bilinear_lookup(
            C_next, Xbar_next, L_next, Xbar_grid_next, L_grid,
        )
        max_x_C = np.maximum(x, C_lookup)
        gh_sum = inv_sqrt_pi * (w_nodes * max_x_C).sum(axis=-1)
        return k1_lo, k1_hi, (w_sigma * gh_sum).sum(axis=-1)

    # ---- Run the DP; threaded across chunks within each stage -------------
    n_workers = max(1, min(n_threads, len(chunk_ranges)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Terminal stage.
        t_term = n - 1
        Xbar_grid_term = Xbar_grids[t_term - 1]
        tau2_term = 1.0 / (1.0 / tau0_2 + t_term / sigma2_grid)
        futs = [
            ex.submit(_terminal_chunk, r, t_term, Xbar_grid_term, tau2_term)
            for r in chunk_ranges
        ]
        for fut in futs:
            k1_lo, k1_hi, C_chunk = fut.result()
            C_hat[t_term - 1, k1_lo:k1_hi] = C_chunk
        if verbose:
            print(f'[random_adp] terminal stage t={t_term} done')

        # Recursion.
        for i_target in range(t_term - 2, -1, -1):
            t = i_target + 1
            C_next = C_hat[i_target + 1]
            Xbar_grid_next = Xbar_grids[i_target + 1]
            Xbar_grid_t = Xbar_grids[i_target]
            tau2_t = 1.0 / (1.0 / tau0_2 + t / sigma2_grid)
            v_t = tau2_t + sigma2_grid
            sqrt_2v = np.sqrt(2.0 * v_t)

            futs = [
                ex.submit(
                    _recursion_chunk, r, t, Xbar_grid_t, Xbar_grid_next,
                    tau2_t, sqrt_2v, C_next,
                )
                for r in chunk_ranges
            ]
            for fut in futs:
                k1_lo, k1_hi, C_chunk = fut.result()
                C_hat[i_target, k1_lo:k1_hi] = C_chunk

            if verbose and (t % 32 == 0 or t == 1):
                print(f'[random_adp] stage t={t} done')

    return {
        'C_hat': C_hat,
        'Xbar_grids': Xbar_grids,
        'L_grid': L_grid,
        'sigma_grid': sigma_grid,
        'log_omega': log_omega,
        'sigma_max': float(sigma_max),
        'n': n, 'mu_0': mu_0, 'tau0_2': tau0_2,
        'K1': K1, 'K2': K2, 'J': J,
    }


# ---------------------------------------------------------------------------
# Off-grid lookup at sequence states (used for labeling and convergence checks)
# ---------------------------------------------------------------------------

def query(table: Dict, t_one_indexed, S_t, Q_t):
    """Look up Ĉ_t at sequence states (S_t, Q_t).

    `t_one_indexed` is the 1-indexed time step in {1, ..., n-1}.
    S_t, Q_t are arrays with the same shape; the result has that shape.

    At t = 1 the within-sequence sample variance is undefined; the table
    has been built so that Ĉ_1(X_bar, L) is independent of L (the
    sigma-posterior at t=1 depends on X_bar_1 only, since Q_1 = X_1^2 is
    a deterministic function of X_bar_1). We pick the grid midpoint for
    L_1 and let the lookup proceed.
    """
    t = int(t_one_indexed)
    if t < 1 or t > table['n'] - 1:
        raise ValueError(f"t must be in [1, {table['n'] - 1}], got {t}")

    Xbar_grid_t = table['Xbar_grids'][t - 1]
    L_grid = table['L_grid']
    K2 = table['K2']

    S_t = np.asarray(S_t, dtype=np.float64)
    Q_t = np.asarray(Q_t, dtype=np.float64)
    Xbar = S_t / t

    if t == 1:
        L = np.full_like(Xbar, L_grid[K2 // 2])
    else:
        sigma_hat2 = np.maximum((Q_t - t * Xbar * Xbar) / (t - 1), 1e-300)
        L = np.log(sigma_hat2)

    return _bilinear_lookup(
        table['C_hat'][t - 1], Xbar, L, Xbar_grid_t, L_grid,
    )


# ---------------------------------------------------------------------------
# Serialize / deserialize
# ---------------------------------------------------------------------------

def save_table(table: Dict, path) -> None:
    np.savez_compressed(
        path,
        C_hat=table['C_hat'],
        Xbar_grids=table['Xbar_grids'],
        L_grid=table['L_grid'],
        sigma_grid=table['sigma_grid'],
        log_omega=table['log_omega'],
        meta=np.array([
            table['n'], table['mu_0'], table['tau0_2'], table['sigma_max'],
            table['K1'], table['K2'], table['J'],
        ], dtype=np.float64),
    )


def load_table(path) -> Dict:
    z = np.load(path, allow_pickle=False)
    n, mu_0, tau0_2, sigma_max, K1, K2, J = z['meta']
    return {
        'C_hat': z['C_hat'],
        'Xbar_grids': z['Xbar_grids'],
        'L_grid': z['L_grid'],
        'sigma_grid': z['sigma_grid'],
        'log_omega': z['log_omega'],
        'sigma_max': float(sigma_max),
        'n': int(n), 'mu_0': float(mu_0), 'tau0_2': float(tau0_2),
        'K1': int(K1), 'K2': int(K2), 'J': int(J),
    }
