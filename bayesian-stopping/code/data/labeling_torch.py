"""
GPU-side random labeler.

label_random_torch(X, table_torch) is the torch counterpart of
data.labeling.label_random; it produces identical y_cv / y_act on the GPU,
fully vectorized over t. Used by the trainer for D_disc / D_logu runs to
keep per-step wall under the 150 ms gate; the numpy `label_random` in
data/labeling.py is unchanged and still used by smoke tests / eval.

Workflow:
    table = oracle.random_adp.load_table(...)
    table_t = build_random_table_torch(table, device)   # one-time copy to GPU
    # per batch:
    y_cv, y_act = label_random_torch(X_t, table_t)      # X_t on GPU

Bilinear math is the same hand-written 4-corner gather as
oracle/random_adp_torch.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def build_random_table_torch(
    table: dict, device: torch.device, dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """Move the random-ADP table to GPU once. Returned dict mirrors `table`
    but values are torch tensors plus precomputed grid steps.
    """
    C_hat = torch.as_tensor(table['C_hat'], dtype=dtype, device=device)
    Xbar_grids = torch.as_tensor(table['Xbar_grids'], dtype=dtype, device=device)
    L_grid = torch.as_tensor(table['L_grid'], dtype=dtype, device=device)
    K1 = int(table['K1'])
    K2 = int(table['K2'])
    Xbar0 = Xbar_grids[:, 0]                                 # (n-1,)
    Xbar_step = (Xbar_grids[:, -1] - Xbar0) / (K1 - 1)       # (n-1,)
    L0 = L_grid[0].item()
    L_step = ((L_grid[-1] - L_grid[0]) / (K2 - 1)).item()
    return {
        'C_hat': C_hat,
        'Xbar_grids': Xbar_grids,
        'Xbar0': Xbar0, 'Xbar_step': Xbar_step,
        'L_grid': L_grid, 'L0': L0, 'L_step': L_step,
        'K1': K1, 'K2': K2,
        'n': int(table['n']),
    }


def label_random_torch(
    X: torch.Tensor, table_t: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized random labeler on GPU.

    Args:
        X: (B, n) torch tensor on the same device as table_t.
        table_t: dict from build_random_table_torch.

    Returns:
        y_cv: (B, n). Index 0 (t=1) and index n-1 (t=n) are 0 placeholders;
              cv_mask zeros them.
        y_act: (B, n). Index 0 (t=1) is 0 (oracle rejects in practice);
              index n-1 (t=n) is 1 (forced acceptance).
    """
    device = X.device
    B, n = X.shape
    if n != table_t['n']:
        raise ValueError(f"sequence length mismatch: X has n={n}, "
                         f"table is for n={table_t['n']}")

    Xf = X.to(table_t['C_hat'].dtype)
    S = torch.cumsum(Xf, dim=1)                                       # (B, n)
    Q = torch.cumsum(Xf * Xf, dim=1)                                  # (B, n)

    # We compute Ĉ_t for t=2..n-1 → indices 1..n-2 in y_cv.
    # Stage indices i = 1..n-2 (0-indexed); 1-indexed t = i+1 = 2..n-1.
    n_stage = n - 2
    t_arr = torch.arange(2, n, dtype=Xf.dtype, device=device)         # (n_stage,) = [2..n-1]
    stage_idx = torch.arange(1, n - 1, device=device)                  # i = 1..n-2 (0-indexed in C_hat)

    # State at each (t, b) — shapes (n_stage, B).
    S_t = S[:, 1:n - 1].T                                              # (n_stage, B)
    Q_t = Q[:, 1:n - 1].T                                              # (n_stage, B)
    Xbar_t = S_t / t_arr[:, None]                                      # (n_stage, B)
    sigma_hat2_t = torch.clamp(
        (Q_t - t_arr[:, None] * Xbar_t * Xbar_t) / (t_arr[:, None] - 1),
        min=1e-300,
    )
    L_t = torch.log(sigma_hat2_t)                                      # (n_stage, B)

    # Per-stage Xbar grid params (we want stages 1..n-2 of Xbar_grids).
    Xbar0_t = table_t['Xbar0'][stage_idx][:, None]                     # (n_stage, 1)
    Xbar_step_t = table_t['Xbar_step'][stage_idx][:, None]             # (n_stage, 1)
    L0 = table_t['L0']
    L_step = table_t['L_step']
    K1 = table_t['K1']
    K2 = table_t['K2']

    row = (Xbar_t - Xbar0_t) / Xbar_step_t                              # (n_stage, B)
    col = (L_t - L0) / L_step
    r = torch.clamp(row, 0.0, K1 - 1.0)
    c = torch.clamp(col, 0.0, K2 - 1.0)
    r_lo = torch.floor(r).long()
    r_hi = torch.clamp(r_lo + 1, max=K1 - 1)
    c_lo = torch.floor(c).long()
    c_hi = torch.clamp(c_lo + 1, max=K2 - 1)
    fr = r - r_lo
    fc = c - c_lo

    # Advanced indexing: stage_idx_b broadcasts to (n_stage, B) over the stage axis.
    stage_b = stage_idx[:, None].expand(n_stage, B)
    C_hat_full = table_t['C_hat']                                       # (n-1, K1, K2)
    v00 = C_hat_full[stage_b, r_lo, c_lo]
    v10 = C_hat_full[stage_b, r_hi, c_lo]
    v01 = C_hat_full[stage_b, r_lo, c_hi]
    v11 = C_hat_full[stage_b, r_hi, c_hi]
    vals = ((1 - fr) * (1 - fc) * v00 + fr * (1 - fc) * v10
            + (1 - fr) * fc * v01 + fr * fc * v11)                       # (n_stage, B)

    y_cv = torch.zeros((B, n), dtype=Xf.dtype, device=device)
    y_cv[:, 1:n - 1] = vals.T

    y_act = torch.zeros((B, n), dtype=Xf.dtype, device=device)
    y_act[:, 1:n - 1] = (X[:, 1:n - 1].to(Xf.dtype) >= y_cv[:, 1:n - 1]).to(Xf.dtype)
    y_act[:, n - 1] = 1.0
    return y_cv, y_act
