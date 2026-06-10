"""Eta_t probes: simple-linear and attention-pooled, trained with raw MSE.

Target. eta_t is the standardized plug-in threshold from the known-mu, known-sigma
Gaussian stopping recursion (see oracle/conjugate.py::compute_eta):
    eta_{n-1} = 0,  eta_t = psi(eta_{t+1}), for t = n-2, ..., 1
where psi(z) = z * Phi(z) + phi(z).

Eta_t depends only on the timestep, not on X_{1:t} or sigma_i. As a probe target
it diagnoses whether the representation makes the plug-in multiplier linearly
accessible from absolute-position information; controls (perm_glob) are
*expected* to do well at layers where positional embeddings dominate (esp. L0).

Probe parameterization. Both architectures emit an unconstrained scalar a_hat;
we predict eta_pred = a_hat directly (no exp, no normalization).

Loss: raw MSE
    L = mean((eta_pred - eta_t)^2)

Metrics on train/val/test:
    eta_mse  = mean((eta_pred - eta_t)^2)
    eta_rmse = sqrt(eta_mse)
    eta_mae  = mean(|eta_pred - eta_t|)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from interp.prelim_v2.probe import _load_layer
from interp.prelim_v2.probe_attn import (
    TimestepSharedAttentionPooledProbe, _build_mc_pairs,
)
from oracle.conjugate import compute_eta


TARGET_NAME = 'eta_t_raw_mse'
PROBE_TYPE_SIMPLE = 'timestep_shared_simple_linear_eta_t_raw_mse'
PROBE_TYPE_ATTN   = 'timestep_shared_attention_pooled_eta_t_raw_mse'


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def build_eta_target(M: int, n: int) -> np.ndarray:
    """Returns Y (M, n) with eta_t at column t-1 (i.e. col 0..n-2); col n-1 = NaN.

    eta has length n-1 representing eta_1..eta_{n-1}. At step n the agent must
    accept and no eta is needed.
    """
    eta = compute_eta(n)                                       # (n-1,)
    Y = np.full((M, n), np.nan, dtype=np.float32)
    Y[:, :n - 1] = eta.astype(np.float32)[None, :]             # broadcast
    return Y


# ---------------------------------------------------------------------------
# Loss + diagnostics
# ---------------------------------------------------------------------------

def raw_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


@torch.no_grad()
def eta_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = (pred - target).pow(2).mean().item()
    mae = (pred - target).abs().mean().item()
    return {'eta_mse': mse, 'eta_rmse': float(np.sqrt(mse)), 'eta_mae': mae}


# ---------------------------------------------------------------------------
# Data assembly (simple): flatten across all valid (m, t) pairs
# ---------------------------------------------------------------------------

def _flatten_for_simple(H: np.ndarray, Y: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    M, n, d = H.shape
    H_flat = H.reshape(-1, d).astype(np.float32)
    Y_flat = Y.reshape(-1).astype(np.float32)
    valid = np.isfinite(Y_flat)
    return H_flat[valid], Y_flat[valid]


# ---------------------------------------------------------------------------
# Simple probe
# ---------------------------------------------------------------------------

def train_simple_probe(*,
    H_train: np.ndarray, Y_train: np.ndarray,
    H_val:   np.ndarray, Y_val:   np.ndarray,
    H_test:  np.ndarray, Y_test:  np.ndarray,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 1,
    seed: int = 0,
    val_chunks_per_epoch: int = 8,
) -> dict:
    torch.manual_seed(seed)
    Ht, Yt = _flatten_for_simple(H_train, Y_train)
    Hv, Yv = _flatten_for_simple(H_val,   Y_val)
    He, Ye = _flatten_for_simple(H_test,  Y_test)

    probe = nn.Linear(Ht.shape[1], 1, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    Xt = torch.from_numpy(Ht).to(device);  yt = torch.from_numpy(Yt).to(device)
    Xv = torch.from_numpy(Hv).to(device);  yv = torch.from_numpy(Yv).to(device)
    Xe = torch.from_numpy(He).to(device);  ye = torch.from_numpy(Ye).to(device)

    ds = TensorDataset(Xt, yt)
    g = torch.Generator(device='cpu').manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=g, drop_last=False)

    @torch.no_grad()
    def _eval_loss(X_, y_) -> float:
        probe.eval()
        L = raw_mse(probe(X_).squeeze(-1), y_).item()
        probe.train()
        return L

    @torch.no_grad()
    def _eval_all(X_, y_) -> dict:
        probe.eval()
        out = eta_metrics(probe(X_).squeeze(-1), y_)
        probe.train()
        return out

    total_steps = max_epochs * max(1, (len(ds) + batch_size - 1) // batch_size)
    val_every = max(1, total_steps // val_chunks_per_epoch)
    step = 0
    best_val = float('inf')
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    final_train_loss = float('nan')

    probe.train()
    for _epoch in range(max_epochs):
        for Xb, yb in loader:
            step += 1
            pred = probe(Xb).squeeze(-1)
            loss = raw_mse(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_train_loss = loss.item()
            if step % val_every == 0 or step == total_steps:
                vL = _eval_loss(Xv, yv)
                if vL < best_val:
                    best_val = vL
                    best_state = {k: v.detach().clone()
                                  for k, v in probe.state_dict().items()}
                    best_step = step
    probe.load_state_dict(best_state)
    test = _eval_all(Xe, ye)
    return {
        'n_train_examples': int(Ht.shape[0]),
        'train_loss_final': final_train_loss,
        'val_loss_best': best_val,
        'val_loss_step_best': best_step,
        'test_loss':     test['eta_mse'],
        'test_eta_rmse': test['eta_rmse'],
        'test_eta_mae':  test['eta_mae'],
    }


# ---------------------------------------------------------------------------
# Attention-pooled probe
# ---------------------------------------------------------------------------

def train_attn_probe(*,
    H_train: np.ndarray, Y_train: np.ndarray,
    H_val:   np.ndarray, Y_val:   np.ndarray,
    H_test:  np.ndarray, Y_test:  np.ndarray,
    t_start: int,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 1,
    seed: int = 0,
    val_chunks_per_epoch: int = 8,
) -> dict:
    torch.manual_seed(seed)
    M_tr, n, d = H_train.shape

    m_train_np, c_train_np = _build_mc_pairs(Y_train, t_start)
    m_val_np,   c_val_np   = _build_mc_pairs(Y_val,   t_start)
    m_test_np,  c_test_np  = _build_mc_pairs(Y_test,  t_start)

    H_train_t = torch.as_tensor(H_train, device=device, dtype=torch.float32)
    Y_train_t = torch.as_tensor(Y_train, device=device, dtype=torch.float32)
    H_val_t   = torch.as_tensor(H_val,   device=device, dtype=torch.float32)
    Y_val_t   = torch.as_tensor(Y_val,   device=device, dtype=torch.float32)
    H_test_t  = torch.as_tensor(H_test,  device=device, dtype=torch.float32)
    Y_test_t  = torch.as_tensor(Y_test,  device=device, dtype=torch.float32)

    m_train = torch.as_tensor(m_train_np, device=device, dtype=torch.long)
    c_train = torch.as_tensor(c_train_np, device=device, dtype=torch.long)
    Y_train_flat = Y_train_t[m_train, c_train].clone()

    cols_val_unique  = torch.unique(torch.as_tensor(c_val_np,  device=device, dtype=torch.long))
    cols_test_unique = torch.unique(torch.as_tensor(c_test_np, device=device, dtype=torch.long))
    valid_mask_val   = torch.isfinite(Y_val_t[:,  cols_val_unique])
    valid_mask_test  = torch.isfinite(Y_test_t[:, cols_test_unique])

    probe = TimestepSharedAttentionPooledProbe(n=n, d_emb=d).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    P = m_train.shape[0]
    total_steps = max_epochs * max(1, (P + batch_size - 1) // batch_size)
    val_every = max(1, total_steps // val_chunks_per_epoch)
    step = 0
    best_val = float('inf')
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    final_train_loss = float('nan')

    g = torch.Generator(device='cpu').manual_seed(seed)
    probe.train()
    for _epoch in range(max_epochs):
        perm = torch.randperm(P, generator=g).to(device)
        for start in range(0, P, batch_size):
            step += 1
            idx = perm[start:start + batch_size]
            m_b = m_train[idx];  c_b = c_train[idx]
            H_b = H_train_t[m_b]
            pred = probe(H_b, c_b)
            target = Y_train_flat[idx]
            loss = raw_mse(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_train_loss = loss.item()

            if step % val_every == 0 or step == total_steps:
                probe.eval()
                with torch.no_grad():
                    p_val = probe.forward_all_prefixes(H_val_t, cols_val_unique)
                    tgt_val = Y_val_t[:, cols_val_unique]
                    vL = raw_mse(p_val[valid_mask_val], tgt_val[valid_mask_val]).item()
                if vL < best_val:
                    best_val = vL
                    best_state = {k: v.detach().clone()
                                  for k, v in probe.state_dict().items()}
                    best_step = step
                probe.train()

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        p_test = probe.forward_all_prefixes(H_test_t, cols_test_unique)
        tgt_test = Y_test_t[:, cols_test_unique]
        test = eta_metrics(p_test[valid_mask_test], tgt_test[valid_mask_test])

    s_vec = best_state['s'].detach().float().cpu().numpy()
    attn_summary = {
        's_position_scores': s_vec.tolist(),
        's_max_position': int(np.argmax(s_vec)),
        's_min_position': int(np.argmin(s_vec)),
        's_range': float(s_vec.max() - s_vec.min()),
    }
    return {
        'n_train_examples': int(P),
        'train_loss_final': final_train_loss,
        'val_loss_best': best_val,
        'val_loss_step_best': best_step,
        'test_loss':     test['eta_mse'],
        'test_eta_rmse': test['eta_rmse'],
        'test_eta_mae':  test['eta_mae'],
        'attn_summary':  attn_summary,
    }


# ---------------------------------------------------------------------------
# Per-member sweep
# ---------------------------------------------------------------------------

def run_eta_t_sweep(*,
    src_member_dir: Path,
    dst_member_dir: Path,
    rep_name: str,
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024),
    device: torch.device,
    seed: int = 0,
    probe_families: tuple[str, ...] = ('simple', 'attn'),
) -> dict:
    cache_dir = src_member_dir / 'hidden_cache' / rep_name
    cache_meta = json.loads((cache_dir / 'meta.json').read_text())
    L_plus_1 = cache_meta['n_layers_total']
    n = int(cache_meta['n'])
    if layers is None:
        layers = tuple(range(L_plus_1))

    # Eta target arrays per split. Eta is timestep-only so all sequences share
    # the same target row.
    M_train_full = cache_meta['splits']['train']['M']
    M_val        = cache_meta['splits']['val']['M']
    M_test       = cache_meta['splits']['test']['M']
    Y_train_full = build_eta_target(M_train_full, n)
    Y_val        = build_eta_target(M_val, n)
    Y_test       = build_eta_target(M_test, n)
    print(f'  [eta_t] {rep_name}: target shape per split: '
          f'train={Y_train_full.shape}, val={Y_val.shape}, test={Y_test.shape}; '
          f'eta range=[{np.nanmin(Y_test):.4f}, {np.nanmax(Y_test):.4f}]')

    rows_simple: list[dict] = []
    rows_attn:   list[dict] = []
    t_start = 0                                                 # eta valid at t=1..n-1 → col 0..n-2

    for layer in layers:
        H_train_full = np.asarray(_load_layer(cache_dir, layer, 'train'))
        H_val        = np.asarray(_load_layer(cache_dir, layer, 'val'))
        H_test       = np.asarray(_load_layer(cache_dir, layer, 'test'))

        for M_train in M_train_budgets:
            if M_train > Y_train_full.shape[0]:
                raise ValueError(f'M_train={M_train} > available {Y_train_full.shape[0]}')
            Y_tr = Y_train_full[:M_train]
            H_tr = H_train_full[:M_train]

            if 'simple' in probe_families:
                t1 = time.perf_counter()
                res = train_simple_probe(
                    H_train=H_tr, Y_train=Y_tr,
                    H_val=H_val,  Y_val=Y_val,
                    H_test=H_test, Y_test=Y_test,
                    device=device,
                    seed=seed + 1000 * layer + M_train,
                )
                rows_simple.append(dict(
                    target=TARGET_NAME, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE_SIMPLE,
                    M_train_sequences=int(M_train),
                    wall_s=time.perf_counter() - t1,
                    **res,
                ))

            if 'attn' in probe_families:
                t1 = time.perf_counter()
                res = train_attn_probe(
                    H_train=H_tr, Y_train=Y_tr,
                    H_val=H_val,  Y_val=Y_val,
                    H_test=H_test, Y_test=Y_test,
                    t_start=t_start,
                    device=device,
                    seed=seed + 1000 * layer + M_train,
                )
                rows_attn.append(dict(
                    target=TARGET_NAME, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE_ATTN,
                    M_train_sequences=int(M_train),
                    wall_s=time.perf_counter() - t1,
                    **res,
                ))

    out_paths = {}
    if rows_simple:
        out_dir = dst_member_dir / 'probe_runs' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results.json'
        p.write_text(json.dumps(rows_simple, indent=2))
        out_paths['simple'] = str(p)
        print(f'  [eta_t] wrote {p} ({len(rows_simple)} cells)')
    if rows_attn:
        out_dir = dst_member_dir / 'probe_runs_attn' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results.json'
        p.write_text(json.dumps(rows_attn, indent=2))
        out_paths['attn'] = str(p)
        print(f'  [eta_t] wrote {p} ({len(rows_attn)} cells)')

    return {'simple_rows': rows_simple, 'attn_rows': rows_attn, 'out_paths': out_paths}
