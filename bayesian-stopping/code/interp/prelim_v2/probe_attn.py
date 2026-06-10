"""Timestep-shared attention-pooled linear probe on cached hidden states.

Probe form (§7.4 of research-notes, "Timestep-shared attention-pooled probe"):
    alpha_c[i] = softmax(s[:c+1])[i]   for i in 0..c, else 0
    z_c        = sum_{i<=c} alpha_c[i] * W h_i^{(l)}
    v_hat_c    = ff(z_c)

where s in R^n is a learned position-score vector (one per probe), W in
R^{d_emb x d_emb} is a learned linear projection, ff is a linear readout
R^{d_emb} -> R, and c is the 0-indexed prefix column (positions 0..c
inclusive are visible). Linear FF only; matches the headline simple-probe
experiment.

Loss / reporting: sigma-normalized MSE (same as interp/prelim_v2/probe.py),
    sMSE_v(E) = mean_{(X_{1:t}, v_t, sigma_i) in E} ((v_hat - v_t) / sigma_i)^2,
no epsilon clip (sigma_i in [1, 100] under D_logu).

Optimizer/schedule: Adam(lr=1e-3, batch=256), capped at one epoch,
best-validation checkpoint selected -- identical to the simple-probe run.
Budget grid M_train in {64, 128, 256, 512, 1024} is also identical.

Hidden states, targets, and sigma are loaded from an existing simple-probe
run directory (read-only) and never recomputed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from interp.prelim_v2.probe import sigma_norm_mse, t_start_for, _load_layer


PROBE_TYPE = 'timestep_shared_attention_pooled'


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class TimestepSharedAttentionPooledProbe(nn.Module):
    """Single learned position-score vector s in R^n, single learned
    projection W in R^{d x d}, and a linear scalar readout. The same s
    is used at every prefix; the mask m_c (positions <= c) changes alpha.
    """

    def __init__(self, n: int, d_emb: int):
        super().__init__()
        self.n = int(n)
        self.d_emb = int(d_emb)
        self.s = nn.Parameter(torch.zeros(n))                # position scores
        self.W = nn.Linear(d_emb, d_emb, bias=False)
        self.ff = nn.Linear(d_emb, 1, bias=True)

    def _alpha(
        self, c: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """alpha of shape (B, n): softmax over s with positions > c masked.

        c: int64 tensor of shape (B,), values in [0, n-1].
        Positions 0..c[b] are unmasked.
        """
        B = c.shape[0]
        positions = torch.arange(self.n, device=device)
        mask = positions[None, :] <= c[:, None]              # (B, n) bool
        scores = self.s[None, :].expand(B, -1).to(dtype)
        scores = scores.masked_fill(~mask, float('-inf'))
        return torch.softmax(scores, dim=-1)

    def forward(self, H: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward at training time (stochastic mini-batches).

        H: (B, n, d) hidden stacks.
        c: (B,) int64 prefix column index.

        Returns (B,) predictions.
        """
        alpha = self._alpha(c, H.device, H.dtype)            # (B, n)
        Z = self.W(H)                                         # (B, n, d)
        z = torch.einsum('bn,bnd->bd', alpha, Z)              # (B, d)
        return self.ff(z).squeeze(-1)                         # (B,)

    @torch.no_grad()
    def forward_all_prefixes(
        self, H: torch.Tensor, cols: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized eval: predict v_hat[m, c] for all m and all c in `cols`.

        H:    (M, n, d) hidden stack.
        cols: (C,) int64 of distinct prefix columns (0-indexed).

        Returns (M, C) predictions.
        """
        device, dtype = H.device, H.dtype
        positions = torch.arange(self.n, device=device)
        mask = positions[None, :] <= cols[:, None]           # (C, n) bool
        scores = self.s[None, :].expand(cols.shape[0], -1).to(dtype)
        scores = scores.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(scores, dim=-1)                # (C, n)
        Z = self.W(H)                                         # (M, n, d)
        z = torch.einsum('cn,mnd->mcd', alpha, Z)             # (M, C, d)
        return self.ff(z).squeeze(-1)                         # (M, C)


# ---------------------------------------------------------------------------
# Data assembly: (m, c) pair index per split (no per-example H copy)
# ---------------------------------------------------------------------------

def _build_mc_pairs(
    Y: np.ndarray, t_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (m_idx, c_idx) int64 arrays for all valid (sequence, prefix-col)
    pairs at columns c >= t_start with finite Y[m, c].
    """
    M, n = Y.shape
    cols = np.arange(t_start, n, dtype=np.int64)
    m_grid, c_grid = np.meshgrid(
        np.arange(M, dtype=np.int64), cols, indexing='ij',
    )
    m_flat = m_grid.reshape(-1)
    c_flat = c_grid.reshape(-1)
    Y_flat = Y[m_flat, c_flat]
    valid = np.isfinite(Y_flat)
    return m_flat[valid], c_flat[valid]


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def _eval_split(
    probe: TimestepSharedAttentionPooledProbe,
    H_t: torch.Tensor, Y_t: torch.Tensor, sigma_t: torch.Tensor,
    cols_t: torch.Tensor, valid_mask_t: torch.Tensor,
) -> float:
    """Evaluate sigma-normalized MSE over all valid (m, c) pairs.

    H_t:     (M, n, d) on GPU.
    Y_t:     (M, n) on GPU; non-finite entries are masked out via valid_mask_t.
    sigma_t: (M,) on GPU.
    cols_t:  (C,) int64 prefix columns to evaluate (c >= t_start).
    valid_mask_t: (M, C) bool, True where Y[m, cols[k]] is finite.
    """
    pred = probe.forward_all_prefixes(H_t, cols_t)            # (M, C)
    target = Y_t[:, cols_t]                                   # (M, C)
    sigma_b = sigma_t[:, None].expand_as(target)              # (M, C)
    sq = ((pred - target) / sigma_b).pow(2)
    return float(sq[valid_mask_t].mean().item())


def train_one_attn_probe(
    *,
    H_train: np.ndarray, Y_train: np.ndarray, sigma_train: np.ndarray,
    H_val:   np.ndarray, Y_val:   np.ndarray, sigma_val:   np.ndarray,
    H_test:  np.ndarray, Y_test:  np.ndarray, sigma_test:  np.ndarray,
    t_start: int,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 1,
    seed: int = 0,
    val_chunks_per_epoch: int = 8,
) -> tuple[float, float, int, float, int, dict]:
    """Train one attention-pooled probe with sigma-normalized MSE.

    Returns (final_train_loss, best_val_loss, val_loss_step_best, test_loss,
             n_train_examples, learned_attention_summary_dict).
    """
    torch.manual_seed(seed)

    M_tr, n, d = H_train.shape

    # Build (m, c) pair indices for each split.
    m_train_np, c_train_np = _build_mc_pairs(Y_train, t_start)
    m_val_np,   c_val_np   = _build_mc_pairs(Y_val,   t_start)
    m_test_np,  c_test_np  = _build_mc_pairs(Y_test,  t_start)
    if m_train_np.size == 0:
        raise RuntimeError(f'no valid (m, c) training pairs (M_tr={M_tr}, t_start={t_start})')

    # Move splits to GPU (float32 for compute).
    H_train_t = torch.as_tensor(H_train, device=device, dtype=torch.float32)
    Y_train_t = torch.as_tensor(Y_train, device=device, dtype=torch.float32)
    s_train_t = torch.as_tensor(sigma_train, device=device, dtype=torch.float32)
    H_val_t   = torch.as_tensor(H_val, device=device, dtype=torch.float32)
    Y_val_t   = torch.as_tensor(Y_val, device=device, dtype=torch.float32)
    s_val_t   = torch.as_tensor(sigma_val, device=device, dtype=torch.float32)
    H_test_t  = torch.as_tensor(H_test, device=device, dtype=torch.float32)
    Y_test_t  = torch.as_tensor(Y_test, device=device, dtype=torch.float32)
    s_test_t  = torch.as_tensor(sigma_test, device=device, dtype=torch.float32)

    # Training pair tensors.
    m_train = torch.as_tensor(m_train_np, device=device, dtype=torch.long)
    c_train = torch.as_tensor(c_train_np, device=device, dtype=torch.long)
    Y_train_flat = Y_train_t[m_train, c_train].clone()
    s_train_flat = s_train_t[m_train].clone()

    # Validation / test column sets and validity masks for the vectorized eval.
    cols_val_unique = torch.unique(torch.as_tensor(c_val_np, device=device,
                                                   dtype=torch.long))
    cols_test_unique = torch.unique(torch.as_tensor(c_test_np, device=device,
                                                    dtype=torch.long))
    valid_mask_val  = torch.isfinite(Y_val_t[:,  cols_val_unique])
    valid_mask_test = torch.isfinite(Y_test_t[:, cols_test_unique])

    probe = TimestepSharedAttentionPooledProbe(n=n, d_emb=d).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    P = m_train.shape[0]
    total_steps = max_epochs * max(1, (P + batch_size - 1) // batch_size)
    val_every = max(1, total_steps // val_chunks_per_epoch)
    step = 0
    final_train_loss = float('nan')
    best_val = float('inf')
    best_val_step = 0
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    g = torch.Generator(device='cpu').manual_seed(seed)

    probe.train()
    for _epoch in range(max_epochs):
        perm = torch.randperm(P, generator=g).to(device)
        for start in range(0, P, batch_size):
            step += 1
            idx = perm[start:start + batch_size]
            m_b = m_train[idx]
            c_b = c_train[idx]
            H_b = H_train_t[m_b]                              # (B, n, d)
            pred = probe(H_b, c_b)                            # (B,)
            target = Y_train_flat[idx]
            sigma_b = s_train_flat[idx]
            loss = sigma_norm_mse(pred, target, sigma_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_train_loss = loss.item()

            if step % val_every == 0 or step == total_steps:
                probe.eval()
                vL = _eval_split(probe, H_val_t, Y_val_t, s_val_t,
                                 cols_val_unique, valid_mask_val)
                if vL < best_val:
                    best_val = vL
                    best_state = {k: v.detach().clone()
                                  for k, v in probe.state_dict().items()}
                    best_val_step = step
                probe.train()

    probe.load_state_dict(best_state)
    probe.eval()
    test_loss = _eval_split(probe, H_test_t, Y_test_t, s_test_t,
                            cols_test_unique, valid_mask_test)

    # Learned attention summary: store the best-state position scores so we
    # can later inspect the alpha profile at any prefix without rerunning.
    s_vec = best_state['s'].detach().float().cpu().numpy()
    summary = {
        's_position_scores': s_vec.tolist(),
        's_max_position': int(np.argmax(s_vec)),
        's_min_position': int(np.argmin(s_vec)),
        's_range': float(s_vec.max() - s_vec.min()),
    }
    return final_train_loss, best_val, best_val_step, test_loss, P, summary


# ---------------------------------------------------------------------------
# Layer x target x M_train sweep, per base rep
# ---------------------------------------------------------------------------

def run_attn_probe_sweep(
    *,
    src_member_dir: Path,
    dst_member_dir: Path,
    rep_name: str,
    targets: tuple[str, ...] = ('sigma_mle_sqrt', 'C_star'),
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024),
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Run all (target, layer, M_train) attention-pooled probes for one rep.

    Inputs come from <src_member_dir> (read-only). Outputs go to
    <dst_member_dir>/probe_runs/<rep_name>/probe_results.json.
    """
    probe_data_dir = src_member_dir / 'probe_data'
    cache_dir = src_member_dir / 'hidden_cache' / rep_name

    cache_meta = json.loads((cache_dir / 'meta.json').read_text())
    L_plus_1 = cache_meta['n_layers_total']
    if layers is None:
        layers = tuple(range(L_plus_1))

    out_dir = dst_member_dir / 'probe_runs_attn' / rep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'probe_results.json'

    sigma_train_full = np.load(probe_data_dir / 'sigma_train.npy')
    sigma_val        = np.load(probe_data_dir / 'sigma_val.npy')
    sigma_test       = np.load(probe_data_dir / 'sigma_test.npy')

    rows: list[dict] = []
    for tgt in targets:
        t0_target = time.perf_counter()
        t_start = t_start_for(tgt)
        Y_train_full = np.load(probe_data_dir / f'Y_{tgt}_train.npy')
        Y_val        = np.load(probe_data_dir / f'Y_{tgt}_val.npy')
        Y_test       = np.load(probe_data_dir / f'Y_{tgt}_test.npy')

        for layer in layers:
            H_train_full = np.asarray(_load_layer(cache_dir, layer, 'train'))
            H_val        = np.asarray(_load_layer(cache_dir, layer, 'val'))
            H_test       = np.asarray(_load_layer(cache_dir, layer, 'test'))

            for M_train in M_train_budgets:
                if M_train > Y_train_full.shape[0]:
                    raise ValueError(
                        f'M_train={M_train} > available {Y_train_full.shape[0]}')
                t1 = time.perf_counter()
                Y_tr = Y_train_full[:M_train]
                H_tr = H_train_full[:M_train]
                s_tr = sigma_train_full[:M_train]
                (final_tr, best_val, best_step, test_loss,
                 n_examples, attn_summary) = train_one_attn_probe(
                    H_train=H_tr, Y_train=Y_tr, sigma_train=s_tr,
                    H_val=H_val, Y_val=Y_val, sigma_val=sigma_val,
                    H_test=H_test, Y_test=Y_test, sigma_test=sigma_test,
                    t_start=t_start,
                    device=device,
                    seed=seed + 1000 * layer + M_train,
                )
                rows.append(dict(
                    target=tgt, layer=int(layer), base_rep=rep_name,
                    probe_type=PROBE_TYPE,
                    M_train_sequences=int(M_train),
                    n_train_examples=int(n_examples),
                    train_loss_final=float(final_tr),
                    val_loss_best=float(best_val),
                    val_loss_step_best=int(best_step),
                    test_loss=float(test_loss),
                    wall_s=time.perf_counter() - t1,
                    attn_summary=attn_summary,
                ))
        print(f'  [attn-probe] target={tgt} done in {time.perf_counter() - t0_target:.1f}s')

    out_path.write_text(json.dumps(rows, indent=2))
    print(f'  [attn-probe] wrote {out_path}  ({len(rows)} cells)')
    return {'rows': rows, 'out_path': str(out_path)}
