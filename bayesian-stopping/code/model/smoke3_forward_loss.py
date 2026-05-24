"""
Smoke 3 — model forward + loss + backward numerical sanity.

Tests, on a freshly-initialized GPTStopper:

    1. Forward pass on a batch of 64 sequences from D_1 produces finite
       (no NaN/Inf) Ĉ_t and logits.
    2. Same for D_disc.
    3. cv_loss returns a finite scalar of order unity for both D_1 (sigma=1)
       and D_disc (mixed sigma in {1,10,100}). Without the per-sequence
       1/sigma_i^2 normalization, D_disc's sigma=100 sequences would push
       gradients ~1e4x larger; the regime-invariance check confirms the
       loss machinery does the right thing.
    4. act_loss returns a finite scalar near log 2 ≈ 0.693 (random init,
       balanced labels).
    5. Backward pass: gradients are finite. Specifically, the cv-loss
       gradient norm on D_disc must NOT scale with sigma_max in the batch
       — this is the regime-invariance test; if per-sequence normalization
       is wrong, the gradient norm would be ~10^4 larger on a sigma=100
       batch than a sigma=1 batch.

Writes v3/results/phase3/smoke3.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from data.distributions import N, sample, static_sigma
from data.labeling import label_random, label_static
from data.streaming import make_act_mask, make_cv_mask
from model.losses import act_loss, cv_loss
from model.transformer import GPTStopper
from oracle.random_adp_torch import make_sigma_grid, solve_random_adp_torch
from oracle.static_adp import solve_adp


BATCH = 64
SEED = 99999
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = V3_ROOT / 'results' / 'phase3'


def _grad_norm(model: GPTStopper) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq))


def _to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Build a fresh model.
    model = GPTStopper(n=N, d_emb=128, n_layers=8, n_heads=4).to(DEVICE)
    n_params = model.num_params()

    # ---- D_1 batch ----
    X1, sigma_i1, _ = sample('D_1', BATCH, rng)
    sigma_static_1 = static_sigma('D_1')
    C_hat_s1, g_s1 = solve_adp(N, 0.0, sigma_static_1 ** 2, 100.0, K=2048, J=128)
    table_static_1 = {'C_hat': C_hat_s1, 'grids': g_s1}
    y_cv_1, y_act_1 = label_static(X1, sigma_static_1, table_static_1)
    cv_mask_static = make_cv_mask('D_1')
    act_mask = make_act_mask()

    # Forward + losses + backward on D_1.
    Xt = _to_torch(X1, DEVICE)
    out = model(Xt)
    finite_cv_d1 = bool(torch.isfinite(out['cv']).all().item())
    finite_act_d1 = bool(torch.isfinite(out['act']).all().item())

    sigma_t = _to_torch(sigma_i1, DEVICE)
    y_cv_t = _to_torch(y_cv_1, DEVICE)
    y_act_t = _to_torch(y_act_1, DEVICE)
    cv_mask_t = torch.as_tensor(cv_mask_static, dtype=torch.bool, device=DEVICE)
    act_mask_t = torch.as_tensor(act_mask, dtype=torch.bool, device=DEVICE)

    loss_cv_1 = cv_loss(out['cv'], y_cv_t, sigma_t, cv_mask_t)
    loss_act_1 = act_loss(out['act'], y_act_t, act_mask_t)

    model.zero_grad()
    loss_cv_1.backward(retain_graph=True)
    grad_cv_d1 = _grad_norm(model)
    model.zero_grad()
    loss_act_1.backward()
    grad_act_d1 = _grad_norm(model)

    # ---- D_disc batch ----
    sigma_grid_disc, log_omega_disc = make_sigma_grid('disc')
    table_disc = solve_random_adp_torch(
        sigma_grid_disc, log_omega_disc, sigma_max=100.0,
        n=N, K1=256, K2=256, J=64,
    )
    rng2 = np.random.default_rng(SEED + 1)
    Xd, sigma_id, _ = sample('D_disc', BATCH, rng2)
    y_cv_d, y_act_d = label_random(Xd, table_disc)
    cv_mask_random = make_cv_mask('D_disc')

    Xt_d = _to_torch(Xd, DEVICE)
    out_d = model(Xt_d)
    finite_cv_dd = bool(torch.isfinite(out_d['cv']).all().item())
    finite_act_dd = bool(torch.isfinite(out_d['act']).all().item())

    sigma_t_d = _to_torch(sigma_id, DEVICE)
    y_cv_t_d = _to_torch(y_cv_d, DEVICE)
    y_act_t_d = _to_torch(y_act_d, DEVICE)
    cv_mask_t_d = torch.as_tensor(cv_mask_random, dtype=torch.bool, device=DEVICE)

    loss_cv_d = cv_loss(out_d['cv'], y_cv_t_d, sigma_t_d, cv_mask_t_d)
    loss_act_d = act_loss(out_d['act'], y_act_t_d, act_mask_t)

    model.zero_grad()
    loss_cv_d.backward(retain_graph=True)
    grad_cv_dd = _grad_norm(model)
    model.zero_grad()
    loss_act_d.backward()
    grad_act_dd = _grad_norm(model)

    # ---- Regime-invariance check: D_disc cv-grad must NOT scale with sigma_max ----
    # Compare grad-norm ratio: D_disc / D_1 should be ~unity (within ~10x),
    # not 1e4-1e8 which is what unnormalized loss would produce.
    grad_ratio = grad_cv_dd / max(grad_cv_d1, 1e-30)
    sigma_max_in_batch = float(sigma_id.max())

    # Pass conditions.
    finite_pass = finite_cv_d1 and finite_act_d1 and finite_cv_dd and finite_act_dd
    cv_pass = (np.isfinite(loss_cv_1.item()) and np.isfinite(loss_cv_d.item())
               and 1e-3 < loss_cv_1.item() < 1e3
               and 1e-3 < loss_cv_d.item() < 1e3)
    act_pass = (np.isfinite(loss_act_1.item()) and np.isfinite(loss_act_d.item())
                and 0.4 < loss_act_1.item() < 1.0
                and 0.4 < loss_act_d.item() < 1.0)
    grad_pass = (np.isfinite(grad_cv_d1) and np.isfinite(grad_cv_dd)
                 and np.isfinite(grad_act_d1) and np.isfinite(grad_act_dd)
                 and grad_ratio < 100.0)  # << sigma_max^2 = 1e4 if normalization wrong

    overall_pass = finite_pass and cv_pass and act_pass and grad_pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append('# Smoke 3 — model forward + loss + backward sanity\n\n')
    md.append(f'Setting: GPTStopper (n={N}, d_emb=128, L=8, M=4 heads), '
              f'random init. Batch size {BATCH}. Device: {DEVICE}.\n')
    md.append(f'Param count: {n_params:,} (target ~1.6M).\n\n')

    md.append('## Forward + loss\n\n')
    md.append('| batch | finite outputs | cv loss | act loss |\n|---|---|---|---|\n')
    md.append(f'| D_1   | cv={finite_cv_d1}, act={finite_act_d1} | '
              f'{loss_cv_1.item():.4e} | {loss_act_1.item():.4e} |\n')
    md.append(f'| D_disc | cv={finite_cv_dd}, act={finite_act_dd} | '
              f'{loss_cv_d.item():.4e} | {loss_act_d.item():.4e} |\n')

    md.append('\n## Gradient norms (backward pass)\n\n')
    md.append('| batch | sigma in batch | cv-grad norm | act-grad norm |\n|---|---|---|---|\n')
    md.append(f'| D_1   | {sigma_static_1}                      | '
              f'{grad_cv_d1:.4e} | {grad_act_d1:.4e} |\n')
    md.append(f'| D_disc | sigma_max={sigma_max_in_batch:g} | '
              f'{grad_cv_dd:.4e} | {grad_act_dd:.4e} |\n')
    md.append(
        f'\nRegime-invariance check (cv-grad ratio D_disc / D_1): '
        f'**{grad_ratio:.2f}**.\n'
        f'Without per-sequence 1/sigma_i^2 normalization, this would be '
        f'~sigma_max^2 / sigma_D1^2 = {sigma_max_in_batch**2:.0f}; '
        'observed ratio close to unity confirms the loss is regime-invariant.\n'
    )

    md.append('\n## Gates\n\n')
    md.append(f'- finite outputs (D_1 + D_disc, cv + act): '
              f'**{"PASS" if finite_pass else "FAIL"}**\n')
    md.append(f'- cv loss in [1e-3, 1e3] for both batches: '
              f'**{"PASS" if cv_pass else "FAIL"}**\n')
    md.append(f'- act loss in [0.4, 1.0] for both batches '
              f'(near log 2 ≈ 0.693): **{"PASS" if act_pass else "FAIL"}**\n')
    md.append(f'- gradient norms finite, cv-grad ratio < 100: '
              f'**{"PASS" if grad_pass else "FAIL"}**\n')
    md.append(f'\n**Overall: {"PASS" if overall_pass else "FAIL"}**\n')
    (RESULTS_DIR / 'smoke3.md').write_text(''.join(md))
    print(''.join(md))

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
