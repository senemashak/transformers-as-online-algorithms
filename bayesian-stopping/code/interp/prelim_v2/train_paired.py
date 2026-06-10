"""Paired base-model training for the preliminary mechanistic-interpretability run.

Trains one D_logu action-supervised base model under one of two label sources:

  - 'true'      : oracle ADP action labels (the normal pipeline).
  - 'perm_glob' : oracle labels with a per-batch global shuffle across the
                  (batch, prefix-position) plane. Each batch's labels are
                  flattened to shape (B*(n-1),), permuted by a fresh RNG draw,
                  and reshaped. Marginal preserved exactly; (X, y) alignment
                  destroyed within the batch.

This is the streaming-pipeline approximation to the paper's "global permutation
across all training (seq, t) pairs" — see runs/prelim_mech_interp_v2/.../README.

The function mirrors ``train.loop.train_one`` for the (D_logu, act) case but
takes the output dirs explicitly so callers can pin runs/.../members/i/{true,perm_glob}/
without touching CHECKPOINT_ROOT or the default phase4 log paths.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from data.labeling_torch import build_random_table_torch, label_random_torch
from data.streaming import load_cache, make_act_mask, stream_batches
from model.losses import act_loss
from model.transformer import GPTStopper
from oracle.random_adp import load_table as load_random_table
from train.configs import ORACLE_TABLES
from train.io import build_payload, save_checkpoint
from train.loop import _env_fingerprint, _grad_norm, _make_lr_lambda


# ---------------------------------------------------------------------------
# Defaults matching the D_logu_act full-stage RunConfig.
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    distribution='D_logu',
    supervision='act',
    n=256,
    d_emb=128,
    n_layers=8,
    n_heads=4,
    batch_size=64,
    step_count=int(1.5e5),
    lr=1e-4,
    warmup_frac=0.20,
    val_every=2500,
    train_log_every=100,
)


def _global_shuffle_labels(
    y_act: torch.Tensor, act_mask: torch.Tensor, perm_rng: torch.Generator,
) -> torch.Tensor:
    """Per-batch global shuffle of action labels across the supervised
    (batch, prefix-position) plane.

    Args:
      y_act:    (B, n) float oracle action labels.
      act_mask: (n,)  bool mask, True at supervised positions (t = 1..n-1).
      perm_rng: torch.Generator for the permutation draw.

    Returns:
      (B, n) float labels equal to y_act at unsupervised positions and a
      uniform random permutation of the supervised entries elsewhere.

    Approximation note: the paper's "global permutation across all training
    (seq, t) pairs" is interpreted as a per-batch shuffle because the
    streaming pipeline has no fixed training set to permute across. The
    permutation is fresh every step.
    """
    B, n = y_act.shape
    pos = act_mask.nonzero(as_tuple=False).flatten()        # (T_sup,)
    T_sup = pos.numel()
    M = B * T_sup
    # Flatten just the supervised positions, permute, scatter back.
    sup_vals = y_act.index_select(1, pos).reshape(M)
    perm = torch.randperm(M, generator=perm_rng, device=sup_vals.device)
    shuffled = sup_vals[perm].view(B, T_sup)
    out = y_act.clone()
    out.index_copy_(1, pos, shuffled)
    return out


def train_paired(
    *,
    seed: int,
    label_perm: Literal['true', 'perm_glob'],
    output_dir: Path,
    device: torch.device,
    step_count: int | None = None,
    val_every: int | None = None,
    perm_seed: int | None = None,
) -> dict:
    """Train one D_logu action-supervised base model.

    Args:
      seed:        Init + input-stream seed. Two paired calls with the same
                   ``seed`` share both initialization and the stream of X.
      label_perm:  'true' uses oracle labels; 'perm_glob' applies the
                   per-batch global shuffle described above.
      output_dir:  Directory for config.json, log.jsonl, best.pt, final.pt.
                   Must not exist or must be empty.
      device:      torch.device('cuda:N') or torch.device('cpu').
      step_count:  Override DEFAULTS['step_count'] (e.g. for smoke tests).
      val_every:   Override DEFAULTS['val_every'].
      perm_seed:   RNG seed for the label permutation. Defaults to
                   ``seed + 10**6``; this keeps the permutation reproducible
                   from (seed, label_perm) alone.

    Returns:
      metadata dict (also written to <output_dir>/result.json).
    """
    if label_perm not in ('true', 'perm_glob'):
        raise ValueError(f"label_perm must be 'true' or 'perm_glob', got {label_perm!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)          # fail loudly if exists

    cfg = dict(DEFAULTS)
    if step_count is not None:
        cfg['step_count'] = int(step_count)
    if val_every is not None:
        cfg['val_every'] = int(val_every)
    cfg['seed'] = int(seed)
    cfg['label_perm'] = label_perm
    cfg['perm_seed'] = int(seed + 1_000_000) if perm_seed is None else int(perm_seed)

    log_path = output_dir / 'log.jsonl'
    if log_path.exists():
        raise FileExistsError(f'log already exists: {log_path}')

    (output_dir / 'config.json').write_text(json.dumps(cfg, indent=2))

    # ----- seed init + stream -----
    rng = np.random.default_rng(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Separate generator for the label permutation so it does not perturb
    # init or input-stream RNGs.
    perm_rng = torch.Generator(device=device)
    perm_rng.manual_seed(cfg['perm_seed'])

    # ----- model + optimizer + scheduler -----
    model = GPTStopper(
        n=cfg['n'], d_emb=cfg['d_emb'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    warmup_steps = max(1, int(cfg['step_count'] * cfg['warmup_frac']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(cfg['step_count'], warmup_steps),
    )

    # ----- oracle table for labeling (always built; we always need true labels) -----
    table = load_random_table(ORACLE_TABLES / 'D_logu_K256_J64_Js64.npz')
    table_t = build_random_table_torch(table, device, dtype=torch.float64)

    # ----- validation cache (always uses true labels, never permuted) -----
    X_val_np, sigma_val_np, _ = load_cache(cfg['distribution'], 'val')
    X_val_t = torch.as_tensor(X_val_np, dtype=torch.float32, device=device)
    sigma_val_t = torch.as_tensor(sigma_val_np, dtype=torch.float32, device=device)
    _, y_act_val_t = label_random_torch(X_val_t.to(torch.float64), table_t)
    y_act_val_t = y_act_val_t.to(torch.float32)

    act_mask_np = make_act_mask()
    act_mask_t = torch.as_tensor(act_mask_np, dtype=torch.bool, device=device)

    streamer = stream_batches(
        cfg['distribution'], cfg['batch_size'], table, rng,
        compute_labels=False,                               # GPU-side oracle labeling
    )

    # ----- logging -----
    env = _env_fingerprint(device)
    train_start = time.perf_counter()
    log_f = log_path.open('a', buffering=1)
    log_f.write(json.dumps({
        'kind': 'start', 'step': 0,
        'config': cfg, 'env': env,
        'wall_clock_start': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }) + '\n')

    best_val_loss = float('inf')
    best_step = -1
    train_smoothed: list[float] = []

    model.train()
    for step in range(1, cfg['step_count'] + 1):
        X_np, _sigma_np, _y_cv_np, _y_act_np, _cv_mask, _act_mask = next(streamer)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        # GPU oracle labels.
        _y_cv_t, y_act_t = label_random_torch(X.to(torch.float64), table_t)
        y_act_t = y_act_t.to(torch.float32)
        if label_perm == 'perm_glob':
            y_act_t = _global_shuffle_labels(y_act_t, act_mask_t, perm_rng)

        out = model(X)
        loss = act_loss(out['act'], y_act_t, act_mask_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = _grad_norm(model)
        optimizer.step()
        scheduler.step()

        train_smoothed.append(loss.item())
        if len(train_smoothed) > 100:
            train_smoothed.pop(0)

        if step % cfg['train_log_every'] == 0:
            log_f.write(json.dumps({
                'kind': 'train', 'step': step,
                'train_loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'grad_norm': gnorm,
                'wall_s_since_start': time.perf_counter() - train_start,
            }) + '\n')

        if step % cfg['val_every'] == 0 or step == cfg['step_count']:
            model.eval()
            with torch.no_grad():
                val_batch = 256
                N_val = X_val_t.shape[0]
                losses: list[torch.Tensor] = []
                for i in range(0, N_val, val_batch):
                    s = slice(i, i + val_batch)
                    out_b = model(X_val_t[s])
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        out_b['act'], y_act_val_t[s], reduction='none',
                    )
                    masked = bce * act_mask_t.to(bce.dtype)
                    per_seq = masked.sum(dim=1) / act_mask_t.to(bce.dtype).sum()
                    losses.append(per_seq)
                per_seq_loss = torch.cat(losses, dim=0)
            val_loss = float(per_seq_loss.mean().item())
            model.train()

            log_f.write(json.dumps({
                'kind': 'val', 'step': step, 'val_loss': val_loss,
                'train_loss_smoothed_100': float(np.mean(train_smoothed)),
                'wall_s_since_start': time.perf_counter() - train_start,
            }) + '\n')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                payload = build_payload(
                    model, optimizer, scheduler, step, val_loss,
                    is_best=True, is_periodic=False, trained_head='act',
                    model_config=dict(
                        n=cfg['n'], d_emb=cfg['d_emb'],
                        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
                    ),
                    config_dict=cfg,
                    rng_states={
                        'torch': torch.get_rng_state(),
                        'numpy': rng.bit_generator.state,
                    },
                )
                save_checkpoint(output_dir / 'best.pt', payload, overwrite=True)

    # ----- final checkpoint -----
    final_payload = build_payload(
        model, optimizer, scheduler, cfg['step_count'], val_loss,
        is_best=False, is_periodic=False, trained_head='act',
        model_config=dict(
            n=cfg['n'], d_emb=cfg['d_emb'],
            n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        ),
        config_dict=cfg,
        rng_states={
            'torch': torch.get_rng_state(),
            'numpy': rng.bit_generator.state,
        },
    )
    save_checkpoint(output_dir / 'final.pt', final_payload, overwrite=True)

    result = {
        'label_perm': label_perm,
        'seed': cfg['seed'],
        'perm_seed': cfg['perm_seed'],
        'step_count': cfg['step_count'],
        'best_step': best_step,
        'best_val_loss': best_val_loss,
        'final_val_loss': val_loss,
        'wall_s_total': time.perf_counter() - train_start,
        'env': env,
    }
    (output_dir / 'result.json').write_text(json.dumps(result, indent=2))
    log_f.write(json.dumps({'kind': 'end', **result}) + '\n')
    log_f.close()
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--label-perm', choices=['true', 'perm_glob'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--step-count', type=int, default=None,
                   help='Override default 1.5e5 steps for smoke tests.')
    p.add_argument('--val-every', type=int, default=None)
    args = p.parse_args()

    if torch.cuda.is_available():
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_paired(
        seed=args.seed,
        label_perm=args.label_perm,
        output_dir=args.output_dir,
        device=device,
        step_count=args.step_count,
        val_every=args.val_every,
    )
