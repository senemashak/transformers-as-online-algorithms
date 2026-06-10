"""Probe target: cumulative Gaussian log-likelihood of the prefix under the
true (oracle) latent parameters.

For each sequence i and prefix length t in 1..n,
    target_{i,t}  =  sum_{s=1}^{t}  log N(X_{i,s} ; mu_i, sigma_i^2)
                  =  -t/2 * log(2*pi*sigma_i^2)
                     - (1/(2*sigma_i^2)) * sum_{s=1}^{t} (X_{i,s} - mu_i)^2.

The latent (mu_i, sigma_i) is known to us (we sampled from it) but never seen
by the model -- probing this asks how well the trained representation infers
the data's Bayes-optimal sufficient log-likelihood, which the model is given
no oracle target for during base training. Valid for all t >= 1 (no NaN).

Probe parameterization: a single unconstrained scalar v_hat. No exp. Raw MSE
loss; diagnostic RMSE and MAE on the test split.

Re-sampling note: probe_data/X is stored on disk, but mu_i was discarded
during materialization (sample_probe_X drops _mu_i). Because
data.distributions.sample is deterministic in its RNG, we re-call it with the
SAME per-member-per-split seed (see probe_data._seed_for) to recover mu_i and
verify the regenerated X matches the stored array bit-for-bit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.distributions import sample as sample_distribution
from interp.prelim_v2.probe import _load_layer
from interp.prelim_v2.probe_attn import (
    TimestepSharedAttentionPooledProbe, _build_mc_pairs,
)
from interp.prelim_v2.probe_data import _seed_for


TARGET_NAME = 'loglik_true_params_raw_mse'
PROBE_TYPE_SIMPLE = 'timestep_shared_simple_linear_loglik_raw_mse'
PROBE_TYPE_ATTN   = 'timestep_shared_attention_pooled_loglik_raw_mse'


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def _resample_mu_and_verify(ensemble_idx: int, split: str,
                             X_loaded: np.ndarray) -> np.ndarray:
    """Re-sample (X, sigma_i, mu_i) deterministically from the stored
    per-split seed, verify X matches the on-disk array, and return mu_i."""
    rng = np.random.default_rng(_seed_for(ensemble_idx, split))
    X_re, _sigma_re, mu_i = sample_distribution('D_logu', X_loaded.shape[0], rng)
    if not np.allclose(X_re.astype(np.float32), X_loaded.astype(np.float32),
                       atol=0.0, rtol=0.0):
        max_abs = float(np.max(np.abs(X_re.astype(np.float32)
                                      - X_loaded.astype(np.float32))))
        raise RuntimeError(
            f'mu re-sample sanity check FAILED: regenerated X != stored X '
            f'(max abs diff = {max_abs}). seed mismatch?')
    return mu_i.astype(np.float64)


def build_loglik_target(X: np.ndarray, mu_i: np.ndarray, sigma_i: np.ndarray,
                        ) -> np.ndarray:
    """Returns Y (M, n), Y[m, c] = sum_{s=1..c+1} log N(X[m, s]; mu_i[m], sigma_i[m]^2).
    All entries finite (valid for all t >= 1)."""
    M, n = X.shape
    X64 = X.astype(np.float64)
    mu = mu_i.astype(np.float64)[:, None]                       # (M, 1)
    sigma = sigma_i.astype(np.float64)[:, None]                 # (M, 1)
    var = sigma * sigma
    log_pdf = -0.5 * np.log(2.0 * np.pi * var) - (X64 - mu) ** 2 / (2.0 * var)
    return log_pdf.cumsum(axis=1).astype(np.float32)            # (M, n)


# ---------------------------------------------------------------------------
# Loss + diagnostics
# ---------------------------------------------------------------------------

def raw_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


@torch.no_grad()
def loglik_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = (pred - target).pow(2).mean().item()
    mae = (pred - target).abs().mean().item()
    return {'loglik_mse': mse, 'loglik_rmse': float(np.sqrt(mse)),
            'loglik_mae': mae}


# ---------------------------------------------------------------------------
# Data assembly (simple): flatten across all valid (m, t) pairs (t >= 1)
# ---------------------------------------------------------------------------

def _flatten_for_simple(H: np.ndarray, Y: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (M, n, d) hiddens + (M, n) target. All columns are valid (no NaN)."""
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

    Xt = torch.from_numpy(Ht).to(device); yt = torch.from_numpy(Yt).to(device)
    Xv = torch.from_numpy(Hv).to(device); yv = torch.from_numpy(Yv).to(device)
    Xe = torch.from_numpy(He).to(device); ye = torch.from_numpy(Ye).to(device)

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
        out = loglik_metrics(probe(X_).squeeze(-1), y_)
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
        'test_loss':         test['loglik_mse'],
        'test_loglik_rmse':  test['loglik_rmse'],
        'test_loglik_mae':   test['loglik_mae'],
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
            m_b = m_train[idx]; c_b = c_train[idx]
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
                    vL = raw_mse(p_val[valid_mask_val],
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
        test = loglik_metrics(p_test[valid_mask_test],
                               tgt_test[valid_mask_test])

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
        'test_loss':         test['loglik_mse'],
        'test_loglik_rmse':  test['loglik_rmse'],
        'test_loglik_mae':   test['loglik_mae'],
        'attn_summary':      attn_summary,
    }


# ---------------------------------------------------------------------------
# Per-member, per-rep sweep
# ---------------------------------------------------------------------------

def run_loglik_sweep(
    *,
    member_dir: Path,
    ensemble_idx: int,
    rep_name: str,                       # 'true' / 'perm_glob' / 'init'
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024),
    device: torch.device,
    seed: int = 0,
    probe_families: tuple[str, ...] = ('simple', 'attn'),
) -> dict:
    cache_dir = member_dir / 'hidden_cache' / rep_name
    cache_meta = json.loads((cache_dir / 'meta.json').read_text())
    L_plus_1 = cache_meta['n_layers_total']
    n = int(cache_meta['n'])
    if layers is None:
        layers = tuple(range(L_plus_1))

    # Build per-split loglik targets from the stored X, the on-disk sigma_i,
    # and the seed-resampled mu_i. The X verification inside
    # _resample_mu_and_verify catches any seed drift.
    probe_data_dir = member_dir / 'probe_data'
    Y_by_split: dict[str, np.ndarray] = {}
    for split in cache_meta['splits']:
        X = np.load(probe_data_dir / f'X_{split}.npy')
        sigma_i = np.load(probe_data_dir / f'sigma_{split}.npy')
        mu_i = _resample_mu_and_verify(ensemble_idx, split, X)
        Y_by_split[split] = build_loglik_target(X, mu_i, sigma_i)
    print(f'  [loglik] {rep_name}: target built; Y range across splits = '
          f'[{min(Y_by_split[s].min() for s in Y_by_split):.1f}, '
          f'{max(Y_by_split[s].max() for s in Y_by_split):.1f}]')

    rows_simple: list[dict] = []
    rows_attn:   list[dict] = []
    t_start = 0                                                 # loglik valid at t >= 1 -> col 0..n-1

    Y_train = Y_by_split['train']
    Y_val   = Y_by_split['val']
    Y_test  = Y_by_split['test']

    for layer in layers:
        H_train_full = np.asarray(_load_layer(cache_dir, layer, 'train'))
        H_val        = np.asarray(_load_layer(cache_dir, layer, 'val'))
        H_test       = np.asarray(_load_layer(cache_dir, layer, 'test'))

        for M_train in M_train_budgets:
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
                rows_simple.append(dict(
                    target=TARGET_NAME, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE_SIMPLE,
                    M_train_sequences=int(M_train),
                    wall_s=time.perf_counter() - t1, **res,
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
                    wall_s=time.perf_counter() - t1, **res,
                ))

    out_paths = {}
    if rows_simple:
        out_dir = member_dir / 'probe_runs' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results_loglik.json'
        p.write_text(json.dumps(rows_simple, indent=2))
        out_paths['simple'] = str(p)
        print(f'  [loglik] wrote {p} ({len(rows_simple)} cells)')
    if rows_attn:
        out_dir = member_dir / 'probe_runs_attn' / rep_name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'probe_results_loglik.json'
        p.write_text(json.dumps(rows_attn, indent=2))
        out_paths['attn'] = str(p)
        print(f'  [loglik] wrote {p} ({len(rows_attn)} cells)')

    return {'simple_rows': rows_simple, 'attn_rows': rows_attn,
            'out_paths': out_paths}
