"""
Probing dataset: D_logu sequences + the seven target quantities per (seq, t).

Targets (all computed analytically from X_{1:t} and known oracle parameters):
    X_bar          — sample mean.
    sigma_hat2     — sample variance σ̂_t² = (Q_t - t·X̄_t²) / (t-1).
    sigma_MAP      — MAP-σ under the D_logu prior (64-pt GL grid on log σ).
    sigma_MLE      — sqrt(σ̂_t²).
    C_star         — random-ADP oracle threshold under D_logu (the headline target).
    C_plugin_MAP   — X̄_t + sigma_MAP · η_t (data-only baseline threshold).
    C_plugin_MLE   — X̄_t + sigma_MLE · η_t.

Conventions:
    - Sequence length n = 256 (matches the trained model).
    - We supervise probes on positions t ∈ {2, ..., n-1} (1-indexed), matching the
      random-distribution cv_mask. Targets are NaN outside that range; the probe
      training code masks NaN positions out of the loss.
    - C_star at t=1 is also NaN (the random-ADP table is not defined there).

Disk layout under `results/probing/logu-act/data/`:
    X_<split>.npy            (N, n)         float32
    targets_<split>.npz      keys = target names, each (N, n) float32
    sigma_i_<split>.npy      (N,)           float32  (true σ; not used by probes)

Splits: 'train' (N=50000, seed=7001), 'val' (N=5000, seed=7002), 'test' (N=10000, seed=7003).
These seeds are disjoint from the training (1xxx, 2xxx) and test (4xxx, 5xxx) seed pools.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import MU_0, TAU0_2, TAU_0, N, sample
from oracle.conjugate import compute_eta, marginal_log_likelihood
from oracle.random_adp import load_table as load_random_table, query as query_random
from oracle.random_adp_torch import make_sigma_grid
from train.configs import ORACLE_TABLES
from interp.probe_paths import (
    DATA_ROOT as PROBE_DATA_ROOT,
    SPLITS,
    DISTRIBUTION,
    SIGMA_GRID_KIND,
    SIGMA_GRID_J,
    RANDOM_ADP_TABLE,
)

# Indices that the probe is supervised on. 0-indexed t' = t-1, so the
# 1-indexed [2, n-1] band corresponds to 0-indexed [1, n-2].
SUPERVISE_T_LO = 1            # 0-indexed; 1-indexed t = 2
SUPERVISE_T_HI = N - 2        # 0-indexed; 1-indexed t = n - 1 = 255


def _compute_targets_for_split(
    X: np.ndarray, sigma_i: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all eleven target arrays for sequences in X.

    Args:
        X: (N, n) float64 — sampled sequences.
        sigma_i: (N,) float64 — true latent σ per sequence; used to
            evaluate target `log_lik_true_sigma` = log p(X_{1:t} | σ_i).
    Returns:
        dict of (N, n) float32 arrays, NaN outside the supervised range.
    """
    N_seq, n = X.shape
    sigma_i = sigma_i.astype(np.float64)                                  # (N,)
    eta = compute_eta(n)                                                  # (n-1,) — η_t for t ∈ {1..n-1}
    S = np.cumsum(X, axis=1)                                              # (N, n)
    Q = np.cumsum(X * X, axis=1)                                          # (N, n)
    t_arr = np.arange(1, n + 1, dtype=np.float64)                         # 1-indexed t

    # Cumulative sum and sum-of-squares (defined for all t ≥ 1). The
    # probe-side comparison is informative: S_t and Q_t are the model's
    # raw running cumulants; X_bar and σ̂² are derived from them. If S_t
    # decodes shallower than X_bar, the X_bar circuit is reading off a
    # division by t. If Q_t decodes shallower than σ̂², the σ̂² circuit
    # is doing the (Q − t·X̄²)/(t−1) algebra on top.
    S_t_target = S.copy()                                                # alias for clarity
    Q_t_target = Q.copy()

    # Sample mean (defined for all t ≥ 1).
    Xbar = S / t_arr

    # Sample variance σ̂_t² = (Q - t * Xbar²) / (t - 1), defined for t ≥ 2.
    sigma_hat2 = np.full((N_seq, n), np.nan, dtype=np.float64)
    sigma_hat2[:, 1:] = np.maximum(
        (Q[:, 1:] - t_arr[1:] * Xbar[:, 1:] ** 2) / (t_arr[1:] - 1),
        1e-300,
    )
    sigma_MLE = np.full((N_seq, n), np.nan, dtype=np.float64)
    sigma_MLE[:, 1:] = np.sqrt(sigma_hat2[:, 1:])

    # MAP-σ under the variant's σ prior (D_logu = 64-pt GL grid; D_disc = 3-pt discrete).
    sg_grid, log_omega_grid = make_sigma_grid(SIGMA_GRID_KIND, J_sigma=SIGMA_GRID_J)
    sigma_MAP = np.full((N_seq, n), np.nan, dtype=np.float64)
    for t_one in range(2, n):                                             # t = 2..n-1
        S_t = S[:, t_one - 1]
        Q_t = Q[:, t_one - 1]
        log_marg = np.empty((N_seq, sg_grid.shape[0]))
        for k, sg in enumerate(sg_grid):
            log_marg[:, k] = marginal_log_likelihood(t_one, S_t, Q_t, sg, TAU0_2)
        log_post_unnorm = log_marg + log_omega_grid[None, :]
        sigma_MAP[:, t_one - 1] = sg_grid[np.argmax(log_post_unnorm, axis=1)]
    # σ_MAP at t=1: posterior dominated by prior; set to prior median (log-uniform).
    sigma_MAP[:, 0] = float(np.exp(0.5 * (np.log(1.0) + np.log(100.0))))  # 10.0

    # Oracle continuation value Ĉ*_t under the variant's training distribution.
    table = load_random_table(ORACLE_TABLES / RANDOM_ADP_TABLE)
    C_star = np.full((N_seq, n), np.nan, dtype=np.float64)
    for t_one in range(2, n):
        C_star[:, t_one - 1] = query_random(table, t_one, S[:, t_one - 1], Q[:, t_one - 1])

    # Plug-in thresholds: Xbar_t + σ̂_t · η_t. η has shape (n-1,) for t ∈ {1..n-1}.
    C_plugin_MAP = np.full((N_seq, n), np.nan, dtype=np.float64)
    C_plugin_MLE = np.full((N_seq, n), np.nan, dtype=np.float64)
    # t = 2..n-1 (0-indexed 1..n-2):
    C_plugin_MAP[:, 1:n - 1] = Xbar[:, 1:n - 1] + sigma_MAP[:, 1:n - 1] * eta[None, 1:]
    C_plugin_MLE[:, 1:n - 1] = Xbar[:, 1:n - 1] + sigma_MLE[:, 1:n - 1] * eta[None, 1:]

    # Log marginal likelihoods log p(X_{1:t} | sigma). Includes the const(t)
    # = -(t/2)*log(2*pi) term explicitly so the four log-lik targets are
    # comparable on the same scale.
    t_arr_int = np.arange(1, n + 1, dtype=np.float64)
    const_t = -0.5 * t_arr_int * np.log(2.0 * np.pi)                      # (n,)
    log_lik_sigma_1, log_lik_sigma_10, log_lik_sigma_100 = (
        np.full((N_seq, n), np.nan, dtype=np.float64) for _ in range(3)
    )
    log_lik_true_sigma = np.full((N_seq, n), np.nan, dtype=np.float64)
    for t_one in range(2, n):
        S_t = S[:, t_one - 1]
        Q_t = Q[:, t_one - 1]
        ct = const_t[t_one - 1]
        log_lik_sigma_1[:, t_one - 1]   = ct + marginal_log_likelihood(t_one, S_t, Q_t,   1.0, TAU0_2)
        log_lik_sigma_10[:, t_one - 1]  = ct + marginal_log_likelihood(t_one, S_t, Q_t,  10.0, TAU0_2)
        log_lik_sigma_100[:, t_one - 1] = ct + marginal_log_likelihood(t_one, S_t, Q_t, 100.0, TAU0_2)
        log_lik_true_sigma[:, t_one - 1] = (
            ct + marginal_log_likelihood(t_one, S_t, Q_t, sigma_i, TAU0_2)
        )

    # Mask all targets to the supervised range [SUPERVISE_T_LO, SUPERVISE_T_HI].
    def _mask(a: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan)
        out[:, SUPERVISE_T_LO : SUPERVISE_T_HI + 1] = a[:, SUPERVISE_T_LO : SUPERVISE_T_HI + 1]
        return out.astype(np.float32)

    return {
        'S_t':                _mask(S_t_target),
        'Q_t':                _mask(Q_t_target),
        'X_bar':              _mask(Xbar),
        'sigma_hat2':         _mask(sigma_hat2),
        'sigma_MAP':          _mask(sigma_MAP),
        'sigma_MLE':          _mask(sigma_MLE),
        'C_star':             _mask(C_star),
        'C_plugin_MAP':       _mask(C_plugin_MAP),
        'C_plugin_MLE':       _mask(C_plugin_MLE),
        'log_lik_sigma_1':    _mask(log_lik_sigma_1),
        'log_lik_sigma_10':   _mask(log_lik_sigma_10),
        'log_lik_sigma_100':  _mask(log_lik_sigma_100),
        'log_lik_true_sigma': _mask(log_lik_true_sigma),
    }


def build_all_splits() -> None:
    PROBE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for split, (N_split, seed) in SPLITS.items():
        rng = np.random.default_rng(seed)
        X, sigma_i, _ = sample(DISTRIBUTION, N_split, rng)
        targets = _compute_targets_for_split(X, sigma_i)
        np.save(PROBE_DATA_ROOT / f'X_{split}.npy', X.astype(np.float32))
        np.save(PROBE_DATA_ROOT / f'sigma_i_{split}.npy', sigma_i.astype(np.float32))
        np.savez_compressed(PROBE_DATA_ROOT / f'targets_{split}.npz', **targets)
        print(f'[probe-data] wrote {split} (N={N_split}, seed={seed})')


# ---------------------------------------------------------------------------
# Selectivity / shuffled-target control
# ---------------------------------------------------------------------------

# Per-split selectivity seeds — disjoint from all data-sampling seeds.
SELECTIVITY_SEEDS = {'train': 9001, 'val': 9002, 'test': 9003}


def build_shuffled_targets() -> None:
    """Generate a shuffled-target control for the Hewitt-Liang selectivity test.

    For each split, draw a random permutation of sequence indices (seeds in
    SELECTIVITY_SEEDS, disjoint from every other v3 seed pool) and apply the
    *same* permutation to every target column. This breaks the
    `H[i, t, :] → target[i, t]` alignment a probe needs to read off, while
    preserving the per-`t` marginal distribution of every target. A probe
    that fits these shuffled labels well is using its own capacity rather
    than reading structure from the model's representations.

    Output: `targets_shuffled_{split}.npz`, sibling to `targets_{split}.npz`,
    with the same 11 keys.
    """
    for split, (N_split, _) in SPLITS.items():
        z = np.load(PROBE_DATA_ROOT / f'targets_{split}.npz')
        rng = np.random.default_rng(SELECTIVITY_SEEDS[split])
        perm = rng.permutation(N_split)
        out = {k: z[k][perm] for k in z.files}
        np.savez_compressed(
            PROBE_DATA_ROOT / f'targets_shuffled_{split}.npz', **out,
        )
        print(f'[probe-data] wrote shuffled targets for {split} '
              f'(perm seed={SELECTIVITY_SEEDS[split]})')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--shuffled', action='store_true',
                   help='Build shuffled-target control (selectivity test) '
                        'instead of the default sample-and-compute pipeline.')
    args = p.parse_args()
    if args.shuffled:
        build_shuffled_targets()
    else:
        build_all_splits()
