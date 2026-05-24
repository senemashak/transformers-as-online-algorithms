"""
V3 labelers.

Two functions, one signature per family:

    label_static(X, sigma, table_static):
        For D_1, D_2, D_3. Posterior path at known sigma; lookup in the
        1D static-ADP table at mu_t.
    label_random(X, table_random):
        For D_disc, D_logu. (S_t, Q_t) -> (X̄_t, L_t) -> bilinear lookup
        in the 2D random-ADP table. Crucially does NOT take sigma — the
        oracle infers sigma in-context, just like the model is meant to.

Output convention (both labelers, both target streams):

    y_cv:  (N_seq, n) float64 continuation values. Index n-1 (t=n) is a
           placeholder of 0 (terminal step has no continuation value);
           cv_mask in `v3/data/streaming.py` is False there.
    y_act: (N_seq, n) float64 action labels in {0, 1}. Index n-1 (t=n)
           is set to 1 (forced acceptance, but masked by act_mask).

The random labeler additionally hardcodes the t=1 entry:

    y_cv[:, 0] = 0       (placeholder; cv_mask is False at t=1 for random)
    y_act[:, 0] = 0      (oracle never accepts at t=1 except in extreme
                          tails; threshold is well above 0 for n=256)

The cv_mask returned by the streamer ensures the loss never sees the
placeholder positions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from data.distributions import MU_0, N, TAU0_2
from oracle.conjugate import posterior_path_batch
from oracle.random_adp import query as query_random
from oracle.static_adp import C_hat_lin


def label_static(
    X: np.ndarray, sigma: float, table_static: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label sequences from a static-sigma regime.

    Args:
        X: (N_seq, n) sequences from D_i.
        sigma: scalar sigma for that regime (1, 10, or 100).
        table_static: dict with keys 'C_hat' (n-1, K) and 'grids' (n-1, K).

    Returns:
        y_cv: (N_seq, n)
        y_act: (N_seq, n)
    """
    N_seq, n = X.shape
    sigma2 = float(sigma) ** 2

    mu_path, _ = posterior_path_batch(X, MU_0, TAU0_2, sigma2)            # (N_seq, n)

    C_hat = table_static['C_hat']
    grids = table_static['grids']
    y_cv = np.zeros((N_seq, n), dtype=np.float64)
    for i in range(n - 1):
        y_cv[:, i] = C_hat_lin(i, mu_path[:, i], C_hat, grids)
    y_act = np.zeros((N_seq, n), dtype=np.float64)
    y_act[:, : n - 1] = (X[:, : n - 1] >= y_cv[:, : n - 1]).astype(np.float64)
    y_act[:, n - 1] = 1.0                                                 # forced acceptance
    return y_cv, y_act


def label_random(
    X: np.ndarray, table_random: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label sequences from a random-sigma distribution.

    Does NOT take sigma — the random ADP table itself encodes the
    sigma-posterior at every (S_t, Q_t).

    Args:
        X: (N_seq, n) sequences from D_disc or D_logu.
        table_random: dict from random_adp.solve_random_adp() (numpy or
                      torch backend; both produce the same dict format).

    Returns:
        y_cv: (N_seq, n). y_cv[:, 0] = 0 (placeholder, cv_mask False at t=1
              for random). y_cv[:, n-1] = 0 (placeholder, terminal).
        y_act: (N_seq, n). y_act[:, 0] = 0 (oracle rejects at t=1 by
              construction). y_act[:, n-1] = 1 (forced acceptance).
    """
    N_seq, n = X.shape
    S = np.cumsum(X, axis=1)                                              # (N_seq, n)
    Q = np.cumsum(X * X, axis=1)                                          # (N_seq, n)

    y_cv = np.zeros((N_seq, n), dtype=np.float64)
    y_act = np.zeros((N_seq, n), dtype=np.float64)

    # t = 2..n-1: query the random ADP at (S_t, Q_t).
    for t in range(2, n):
        y_cv[:, t - 1] = query_random(table_random, t, S[:, t - 1], Q[:, t - 1])
    # Action label at t=2..n-1: 1[X_t >= Ĉ_t]. (t=1 already 0.)
    y_act[:, 1 : n - 1] = (X[:, 1 : n - 1] >= y_cv[:, 1 : n - 1]).astype(np.float64)
    y_act[:, n - 1] = 1.0
    return y_cv, y_act
