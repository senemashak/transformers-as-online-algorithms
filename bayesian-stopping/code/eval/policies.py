"""
V3 evaluation policies — all decision rules, vectorized over a test set.

Every policy takes (X, ...) and returns:
    action:    (N_seq, n) bool, accept/reject per (seq, t).
               Convention: t = n is forced acceptance (set to True), so
               action[:, n-1] = True for every policy.
    threshold: (N_seq, n) float, the threshold value used at each (seq, t),
               or NaN where the policy is non-threshold-form. Trajectory
               plots only consume this; payoff and agreement use action.

Threshold-form policies all share `threshold_to_action(X, threshold)`.
The trained-model output (cv head) plugs in directly as a threshold;
the trained-model act head produces logits and goes straight to action.

Ports from v2/code/baselines.py + eval_common.py — vectorized over the
test set (V2's per-sequence loop replaced by numpy-vectorized cumsums and
per-stage ADP queries).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from oracle.conjugate import (
    compute_eta,
    marginal_log_likelihood,
    posterior_path_batch,
)
from oracle.random_adp import query as query_random
from oracle.static_adp import C_hat_lin


MU_0 = 0.0
TAU0_2 = 100.0


# ---------------------------------------------------------------------------
# action / payoff helpers
# ---------------------------------------------------------------------------

def threshold_to_action(X: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """X, threshold: (N_seq, n). Returns (N_seq, n) bool.

    action[seq, t] = (X[seq, t] >= threshold[seq, t]) for t < n;
    action[seq, n-1] = True (forced acceptance).
    """
    N, n = X.shape
    action = np.zeros((N, n), dtype=bool)
    action[:, : n - 1] = X[:, : n - 1] >= threshold[:, : n - 1]
    action[:, n - 1] = True
    return action


def stop_index(action: np.ndarray) -> np.ndarray:
    """Returns (N_seq,) 0-indexed stop time. action_t True means accept."""
    N, n = action.shape
    any_acc = action[:, : n - 1].any(axis=1)
    first_idx = action[:, : n - 1].argmax(axis=1)        # 0 if all-False
    return np.where(any_acc, first_idx, n - 1)


def payoff(action: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Returns (N_seq,) payoff X_τ for each sequence."""
    s = stop_index(action)
    return X[np.arange(X.shape[0]), s]


# ---------------------------------------------------------------------------
# Per-regime Bayes-optimal oracle (static ADP)
# ---------------------------------------------------------------------------

def static_oracle(
    X: np.ndarray, sigma: float, table_static: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-regime Bayes-optimal oracle. Static ADP at known sigma."""
    N, n = X.shape
    sigma2 = sigma * sigma
    mu_path, _ = posterior_path_batch(X, MU_0, TAU0_2, sigma2)
    threshold = np.zeros((N, n), dtype=np.float64)
    for i in range(n - 1):
        threshold[:, i] = C_hat_lin(
            i, mu_path[:, i], table_static['C_hat'], table_static['grids'],
        )
    return threshold_to_action(X, threshold), threshold


# ---------------------------------------------------------------------------
# Per-training-distribution Bayes-optimal oracle (random ADP)
# ---------------------------------------------------------------------------

def random_oracle(
    X: np.ndarray, table_random: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-training-distribution Bayes-optimal oracle. Random ADP."""
    N, n = X.shape
    S = np.cumsum(X, axis=1)
    Q = np.cumsum(X * X, axis=1)
    threshold = np.zeros((N, n), dtype=np.float64)
    for t_one in range(1, n):
        threshold[:, t_one - 1] = query_random(
            table_random, t_one, S[:, t_one - 1], Q[:, t_one - 1],
        )
    return threshold_to_action(X, threshold), threshold


# ---------------------------------------------------------------------------
# Known-sigma baselines
# ---------------------------------------------------------------------------

def plug_in(X: np.ndarray, sigma: float, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1[X_t >= X̄_t + sigma · eta_t]. eta has shape (n-1,)."""
    N, n = X.shape
    S = np.cumsum(X, axis=1)
    t_arr = np.arange(1, n + 1, dtype=float)
    Xbar = S / t_arr
    threshold = np.zeros((N, n), dtype=np.float64)
    threshold[:, : n - 1] = Xbar[:, : n - 1] + sigma * eta[None, :]
    return threshold_to_action(X, threshold), threshold


def prior_only(X: np.ndarray, sigma: float, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1[X_t >= mu_0 + sigma · eta_t]. Constant across sequences."""
    N, n = X.shape
    threshold = np.zeros((N, n), dtype=np.float64)
    threshold[:, : n - 1] = MU_0 + sigma * eta[None, :]
    return threshold_to_action(X, threshold), threshold


def myopic(X: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """1[X_t >= mu_t]. mu_t is conjugate posterior mean at known sigma."""
    N, n = X.shape
    mu_path, _ = posterior_path_batch(X, MU_0, TAU0_2, sigma * sigma)
    threshold = np.zeros((N, n), dtype=np.float64)
    threshold[:, : n - 1] = mu_path[:, : n - 1]
    return threshold_to_action(X, threshold), threshold


# ---------------------------------------------------------------------------
# Data-only baselines (V3, §4.3)
# ---------------------------------------------------------------------------

def map_sigma_plugin(
    X: np.ndarray,
    sigma_grid: np.ndarray,
    log_omega: np.ndarray,
    eta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """MAP-sigma plug-in.

    σ̂_MAP_t = argmax_σ p(σ | X_{1:t}) over `sigma_grid`. Plug in:
    1[X_t >= X̄_t + σ̂_MAP_t · eta_t]. The `prior` is encoded in
    log_omega: 0 for D_disc (uniform on the 3 grid points); GL log-weights
    for D_logu (uniform on log σ).
    """
    N, n = X.shape
    S = np.cumsum(X, axis=1)
    Q = np.cumsum(X * X, axis=1)
    t_arr = np.arange(1, n + 1, dtype=float)
    Xbar = S / t_arr
    threshold = np.zeros((N, n), dtype=np.float64)

    M = sigma_grid.shape[0]
    for t_one in range(1, n):
        S_t = S[:, t_one - 1]
        Q_t = Q[:, t_one - 1]
        # log p(X_{1:t} | sigma_k) for each (seq, sigma_k); shape (N, M)
        log_marg = np.empty((N, M))
        for k, sg in enumerate(sigma_grid):
            log_marg[:, k] = marginal_log_likelihood(t_one, S_t, Q_t, sg, TAU0_2)
        log_post_unnorm = log_marg + log_omega[None, :]
        map_idx = np.argmax(log_post_unnorm, axis=1)
        sigma_hat = sigma_grid[map_idx]
        threshold[:, t_one - 1] = Xbar[:, t_one - 1] + sigma_hat * eta[t_one - 1]
    return threshold_to_action(X, threshold), threshold


def mle_sigma_plugin(X: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MLE-sigma plug-in: σ̂_MLE_t = sqrt(σ̂_t²) (within-sequence sample SD)
    plugged into 1[X_t >= X̄_t + σ̂_MLE_t · eta_t]. Undefined at t = 1
    (set threshold to +inf so action is always reject)."""
    N, n = X.shape
    S = np.cumsum(X, axis=1)
    Q = np.cumsum(X * X, axis=1)
    t_arr = np.arange(1, n + 1, dtype=float)
    Xbar = S / t_arr
    sigma_hat2 = np.zeros((N, n))
    sigma_hat2[:, 1:] = np.maximum(
        (Q[:, 1:] - t_arr[1:] * Xbar[:, 1:] ** 2) / (t_arr[1:] - 1),
        1e-300,
    )
    sigma_hat = np.sqrt(sigma_hat2)
    threshold = np.full((N, n), np.inf, dtype=np.float64)
    threshold[:, 1 : n - 1] = (
        Xbar[:, 1 : n - 1] + sigma_hat[:, 1 : n - 1] * eta[None, 1:]
    )
    return threshold_to_action(X, threshold), threshold


# ---------------------------------------------------------------------------
# Scale-free
# ---------------------------------------------------------------------------

def secretary(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Skip first r = floor(n/e); accept first X_t > max(X[:r]); else accept X_n.
    For threshold form: threshold_t = +inf for t <= r, max(X[:r]) for t > r."""
    N, n = X.shape
    r = int(n / np.e)
    M_r = X[:, :r].max(axis=1, keepdims=True)
    threshold = np.full((N, n), np.inf, dtype=np.float64)
    threshold[:, r : n - 1] = M_r                                   # broadcast
    return threshold_to_action(X, threshold), threshold


# ---------------------------------------------------------------------------
# Trained model (cv or act head)
# ---------------------------------------------------------------------------

def model_action(
    model, X: np.ndarray, head: str, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a trained model forward; return (action, threshold).

    For cv: threshold = Ĉ_t (model output); action = 1[X_t >= Ĉ_t].
    For act: action = 1[logit_t > 0]; threshold = NaN (act has no
    interpretable threshold for trajectory plots).
    """
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(Xt)
    N, n = X.shape
    if head == 'cv':
        threshold = out['cv'].cpu().numpy().astype(np.float64)
        action = threshold_to_action(X, threshold)
    elif head == 'act':
        logits = out['act'].cpu().numpy().astype(np.float64)
        action = np.zeros((N, n), dtype=bool)
        action[:, : n - 1] = logits[:, : n - 1] > 0.0
        action[:, n - 1] = True
        threshold = np.full((N, n), np.nan, dtype=np.float64)
    else:
        raise ValueError(f'unknown head {head!r}')
    return action, threshold
