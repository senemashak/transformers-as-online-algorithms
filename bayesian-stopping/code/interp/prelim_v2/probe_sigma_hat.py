"""Sigma-hat exp-output probes: simple-linear and attention-pooled, trained
with relative sigma-space MSE.

Target. For each sequence i and prefix t >= 2,
    mean_t      = mean(X_{i,1:t})
    Q_t         = sum_s X_{i,s}^2
    sigma_hat_t = sqrt(max(eps, (Q_t - t * mean_t^2) / (t - 1))).
t=1 is invalid (NaN target). This is the unbiased sample std (Bessel-corrected),
distinct from the existing 'sigma_mle_sqrt' target which divides by t.

Probe parameterization. Both probe architectures output an unconstrained scalar
a_hat per (sequence, prefix). We interpret it as log(sigma) and predict
    sigma_pred = exp(a_hat).

Loss / val / test selection (rel_sigma_mse):
    L = mean( ((sigma_pred - sigma_hat_t) / sigma_hat_t)^2 ).

We additionally report on the test split:
    test_raw_rmse = sqrt(mean((sigma_pred - sigma_hat_t)^2))
    test_raw_mae  = mean(|sigma_pred - sigma_hat_t|)
    test_log_mse  = mean((a_hat - log(sigma_hat_t))^2)
    test_log_mae  = mean(|a_hat - log(sigma_hat_t)|)
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


TARGET_NAME = 'sigma_hat_exp_relmse'
PROBE_TYPE_SIMPLE = 'timestep_shared_simple_linear_sigma_hat_exp_relmse'
PROBE_TYPE_ATTN   = 'timestep_shared_attention_pooled_sigma_hat_exp_relmse'
EPS_VAR = 1e-12                                                # variance clamp before sqrt


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def compute_sigma_hat(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return Y (M, n) with sigma_hat_t at column t-1; NaN at column 0.

    X: (M, n) float observation sequences.

    Also returns a tiny dict with the number of cells where var < EPS_VAR
    (which would have produced sqrt(neg) without the clamp).
    """
    M, n = X.shape
    X64 = X.astype(np.float64, copy=False)
    S = X64.cumsum(axis=1)                                     # (M, n) running sum
    Q = (X64 * X64).cumsum(axis=1)                             # (M, n) running sum of squares
    t = np.arange(1, n + 1, dtype=np.float64)                  # 1..n
    mean = S / t                                               # (M, n)
    var_num = Q - t * (mean * mean)                            # = sum((X_s - mean_t)^2)
    var = var_num / np.maximum(t - 1.0, 1.0)                   # divide by (t-1); t=1 placeholder
    clamp = int((var < EPS_VAR).sum())
    var = np.maximum(var, EPS_VAR)
    sigma_hat = np.sqrt(var).astype(np.float32)
    Y = np.full((M, n), np.nan, dtype=np.float32)
    Y[:, 1:] = sigma_hat[:, 1:]                                # valid only at t >= 2
    return Y, {'clamp_count': clamp,
               'total_cells_above_t1': int(M * (n - 1))}


# ---------------------------------------------------------------------------
# Loss + diagnostic metrics
# ---------------------------------------------------------------------------

A_CLAMP = 12.0                                                  # exp(12) ≈ 1.6e5; well above max sigma_hat (~300)
RESIDUAL_CAP = 50.0                                             # per-sample relative residual cap (train only)
SIGMA_FLOOR_FOR_METRIC = 1e-2                                   # robustness for test rel-MSE on heavy lower tails


def relative_sigma_mse(a_hat: torch.Tensor, sigma_target: torch.Tensor,
                       ) -> torch.Tensor:
    """L = mean(((exp(a_hat) - sigma_target) / sigma_target)^2). Unclipped --
    used for val/test reporting; we still pre-clamp a_hat so exp does not
    overflow fp32 (which would propagate to inf metrics).
    """
    a_safe = a_hat.clamp(min=-A_CLAMP, max=A_CLAMP)
    sigma_pred = torch.exp(a_safe)
    return ((sigma_pred - sigma_target) / sigma_target).pow(2).mean()


def relative_sigma_mse_train(a_hat: torch.Tensor, sigma_target: torch.Tensor,
                              residual_cap: float = RESIDUAL_CAP) -> torch.Tensor:
    """Training-stable variant: clamp a_hat to a safe range BEFORE exp (otherwise
    exp(>~88) overflows fp32, and the backward pass through clamp(residual)
    produces 0*inf = NaN), and cap per-sample relative residuals at ±cap so a
    single sample with sigma_target ~ 3e-4 cannot dominate the gradient.
    Typical samples (|residual| ~ 1) are untouched by either clamp.
    """
    a_safe = a_hat.clamp(min=-A_CLAMP, max=A_CLAMP)
    sigma_pred = torch.exp(a_safe)
    r = (sigma_pred - sigma_target) / sigma_target
    return r.clamp(min=-residual_cap, max=residual_cap).pow(2).mean()


@torch.no_grad()
def all_metrics(a_hat: torch.Tensor, sigma_target: torch.Tensor) -> dict:
    """Test metrics: clamp a_hat for exp safety and floor sigma_target for
    relative-MSE denominator stability (test set has rare sigma_hat ~ 3e-4
    tail values at t=2 that, without flooring, dominate the mean rel-MSE
    even when the probe's prediction is otherwise good).
    """
    a_safe = a_hat.clamp(min=-A_CLAMP, max=A_CLAMP)
    sigma_pred = torch.exp(a_safe)
    sigma_for_rel = sigma_target.clamp(min=SIGMA_FLOOR_FOR_METRIC)
    rel_mse = ((sigma_pred - sigma_for_rel) / sigma_for_rel).pow(2).mean().item()
    raw_rmse = (sigma_pred - sigma_target).pow(2).mean().sqrt().item()
    raw_mae = (sigma_pred - sigma_target).abs().mean().item()
    log_t = sigma_target.log()
    log_mse = (a_hat - log_t).pow(2).mean().item()
    log_mae = (a_hat - log_t).abs().mean().item()
    return {'rel_sigma_mse': rel_mse, 'raw_rmse': raw_rmse, 'raw_mae': raw_mae,
            'log_mse': log_mse, 'log_mae': log_mae}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _flatten_for_simple(H: np.ndarray, Y: np.ndarray, t_start: int = 1
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (M, n, d) hiddens + (M, n) target across all valid (m, t)
    pairs at columns t >= t_start with finite Y. Returns (P, d), (P,)."""
    M, n, d = H.shape
    H = H[:, t_start:]
    Y = Y[:, t_start:]
    n_prime = Y.shape[1]
    H_flat = H.reshape(-1, d).astype(np.float32)
    Y_flat = Y.reshape(-1).astype(np.float32)
    valid = np.isfinite(Y_flat)
    return H_flat[valid], Y_flat[valid]


# ---------------------------------------------------------------------------
# Simple probe trainer
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
    """Train a timestep-shared simple linear probe with rel_sigma_mse."""
    torch.manual_seed(seed)
    Ht, Yt = _flatten_for_simple(H_train, Y_train)
    Hv, Yv = _flatten_for_simple(H_val,   Y_val)
    He, Ye = _flatten_for_simple(H_test,  Y_test)
    d = Ht.shape[1]

    probe = nn.Linear(d, 1, bias=True).to(device)
    # Initialize bias to log(median sigma_hat) so exp(a_hat) starts at the
    # right scale; otherwise the rel-MSE loss can overflow before the probe
    # has any chance to fit.
    log_med = float(np.log(np.median(Yt)))
    with torch.no_grad():
        probe.bias.data.fill_(log_med)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    Xt = torch.from_numpy(Ht).to(device);   yt = torch.from_numpy(Yt).to(device)
    Xv = torch.from_numpy(Hv).to(device);   yv = torch.from_numpy(Yv).to(device)
    Xe = torch.from_numpy(He).to(device);   ye = torch.from_numpy(Ye).to(device)

    ds = TensorDataset(Xt, yt)
    g = torch.Generator(device='cpu').manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=g, drop_last=False)

    @torch.no_grad()
    def _eval_loss(X_, y_) -> float:
        probe.eval()
        a = probe(X_).squeeze(-1)
        L = relative_sigma_mse(a, y_).item()
        probe.train()
        return L

    @torch.no_grad()
    def _eval_all(X_, y_) -> dict:
        probe.eval()
        a = probe(X_).squeeze(-1)
        out = all_metrics(a, y_)
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
            a = probe(Xb).squeeze(-1)
            loss = relative_sigma_mse_train(a, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
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
        'val_loss_best':    best_val,
        'val_loss_step_best': best_step,
        'test_loss':         test['rel_sigma_mse'],
        'test_raw_rmse':     test['raw_rmse'],
        'test_raw_mae':      test['raw_mae'],
        'test_log_mse':      test['log_mse'],
        'test_log_mae':      test['log_mae'],
    }


# ---------------------------------------------------------------------------
# Attention-pooled probe trainer
# ---------------------------------------------------------------------------

def _eval_attn_split(probe, H_t, Y_t, cols_t, valid_mask_t, metric_fn):
    pred = probe.forward_all_prefixes(H_t, cols_t)              # (M, C)
    target = Y_t[:, cols_t]
    valid = valid_mask_t
    return metric_fn(pred[valid], target[valid])


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
    """Train a timestep-shared attention-pooled probe with rel_sigma_mse."""
    torch.manual_seed(seed)
    M_tr, n, d = H_train.shape

    m_train_np, c_train_np = _build_mc_pairs(Y_train, t_start)
    m_val_np,   c_val_np   = _build_mc_pairs(Y_val,   t_start)
    m_test_np,  c_test_np  = _build_mc_pairs(Y_test,  t_start)
    if m_train_np.size == 0:
        raise RuntimeError(f'no valid train pairs (M_tr={M_tr}, t_start={t_start})')

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
    # Same scale-init for the attn probe's final linear readout.
    Yt_flat = Y_train_flat.detach().cpu().numpy()
    log_med = float(np.log(np.median(Yt_flat[np.isfinite(Yt_flat)])))
    with torch.no_grad():
        probe.ff.bias.data.fill_(log_med)
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
            m_b = m_train[idx]
            c_b = c_train[idx]
            H_b = H_train_t[m_b]
            pred = probe(H_b, c_b)
            target = Y_train_flat[idx]
            loss = relative_sigma_mse_train(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            opt.step()
            final_train_loss = loss.item()

            if step % val_every == 0 or step == total_steps:
                probe.eval()
                with torch.no_grad():
                    p_val = probe.forward_all_prefixes(H_val_t, cols_val_unique)
                    tgt_val = Y_val_t[:, cols_val_unique]
                    vL = relative_sigma_mse(p_val[valid_mask_val],
                                             tgt_val[valid_mask_val]).item()
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
        test = all_metrics(p_test[valid_mask_test], tgt_test[valid_mask_test])

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
        'val_loss_best':    best_val,
        'val_loss_step_best': best_step,
        'test_loss':         test['rel_sigma_mse'],
        'test_raw_rmse':     test['raw_rmse'],
        'test_raw_mae':      test['raw_mae'],
        'test_log_mse':      test['log_mse'],
        'test_log_mae':      test['log_mae'],
        'attn_summary':      attn_summary,
    }


# ---------------------------------------------------------------------------
# Per-member sweep
# ---------------------------------------------------------------------------

def run_sigma_hat_sweep(*,
    src_member_dir: Path,                                       # has hidden_cache/, probe_data/
    dst_member_dir: Path,                                       # outputs land here
    rep_name: str,
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024),
    device: torch.device,
    seed: int = 0,
    probe_families: tuple[str, ...] = ('simple', 'attn'),
) -> dict:
    """Train sigma_hat probes (both families by default) for one rep."""
    probe_data_dir = src_member_dir / 'probe_data'
    cache_dir = src_member_dir / 'hidden_cache' / rep_name

    cache_meta = json.loads((cache_dir / 'meta.json').read_text())
    L_plus_1 = cache_meta['n_layers_total']
    if layers is None:
        layers = tuple(range(L_plus_1))

    # Materialize sigma_hat target arrays from X (in memory only).
    X_train = np.load(probe_data_dir / 'X_train.npy')
    X_val   = np.load(probe_data_dir / 'X_val.npy')
    X_test  = np.load(probe_data_dir / 'X_test.npy')
    Y_train, st_tr = compute_sigma_hat(X_train)
    Y_val,   st_v  = compute_sigma_hat(X_val)
    Y_test,  st_te = compute_sigma_hat(X_test)
    clamp_stats = {'train': st_tr, 'val': st_v, 'test': st_te}
    print(f'  [sigma-hat] {rep_name}: target materialized; clamp triggers '
          f'tr={st_tr["clamp_count"]} v={st_v["clamp_count"]} te={st_te["clamp_count"]}')

    rows_simple: list[dict] = []
    rows_attn:   list[dict] = []
    t_start = 1                                                  # sigma_hat valid at t >= 2

    for layer in layers:
        H_train_full = np.asarray(_load_layer(cache_dir, layer, 'train'))
        H_val        = np.asarray(_load_layer(cache_dir, layer, 'val'))
        H_test       = np.asarray(_load_layer(cache_dir, layer, 'test'))

        for M_train in M_train_budgets:
            if M_train > Y_train.shape[0]:
                raise ValueError(
                    f'M_train={M_train} > available {Y_train.shape[0]}')
            Y_tr = Y_train[:M_train]
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
                row = dict(
                    target=TARGET_NAME, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE_SIMPLE,
                    M_train_sequences=int(M_train),
                    clamp_stats={k: v['clamp_count'] for k, v in clamp_stats.items()},
                    wall_s=time.perf_counter() - t1,
                    **res,
                )
                rows_simple.append(row)

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
                row = dict(
                    target=TARGET_NAME, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE_ATTN,
                    M_train_sequences=int(M_train),
                    clamp_stats={k: v['clamp_count'] for k, v in clamp_stats.items()},
                    wall_s=time.perf_counter() - t1,
                    **res,
                )
                rows_attn.append(row)

    out_paths = {}
    if rows_simple:
        out_dir = dst_member_dir / 'probe_runs' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results.json'
        p.write_text(json.dumps(rows_simple, indent=2))
        out_paths['simple'] = str(p)
        print(f'  [sigma-hat] wrote {p} ({len(rows_simple)} cells)')
    if rows_attn:
        out_dir = dst_member_dir / 'probe_runs_attn' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results.json'
        p.write_text(json.dumps(rows_attn, indent=2))
        out_paths['attn'] = str(p)
        print(f'  [sigma-hat] wrote {p} ({len(rows_attn)} cells)')

    return {'simple_rows': rows_simple, 'attn_rows': rows_attn,
            'clamp_stats': clamp_stats, 'out_paths': out_paths}
