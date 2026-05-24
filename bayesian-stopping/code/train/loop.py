"""
Step 4 single-model training loop.

train_one(config, device) drives one (distribution, supervision) run end
to end: build model + optimizer + scheduler, stream training batches,
compute the chosen loss, backprop, validate at the configured cadence,
save best/final/periodic checkpoints, and emit a JSONL event log.

Per-step structure:
    1. Sample (X, sigma_i, mu_i) from the streamer (compute_labels=False
       for random distributions; compute_labels=True for static).
    2. For random: label X on GPU via label_random_torch.
    3. Forward through GPTStopper -> (cv, act) outputs.
    4. cv_loss or act_loss; the model never sees sigma_i.
    5. Backward, optimizer step, scheduler step.

Validation: pre-computed labels (one labeling pass at run start) on the
cached val set; loss-only pass with model.eval() and torch.no_grad().

Logging: one JSONL line per event in v3/results/phase4{,_pilot}/<run>/log.jsonl.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch

from data.distributions import (
    MU_0,
    RANDOM_DISTRIBUTIONS,
    STATIC_DISTRIBUTIONS,
    TAU0_2,
    static_sigma,
)
from data.labeling import label_random, label_static
from data.labeling_offline import label_offline, label_offline_torch
from data.labeling_torch import build_random_table_torch, label_random_torch
from data.streaming import load_cache, make_act_mask, make_cv_mask, stream_batches
from model.losses import act_loss, cv_loss
from model.transformer import GPTStopper
from oracle.random_adp import load_table as load_random_table
from oracle.static_adp import solve_adp
from train.configs import ORACLE_TABLES, RunConfig
from train.io import build_payload, save_checkpoint


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _make_lr_lambda(total_steps: int, warmup_steps: int):
    def fn(epoch: int) -> float:
        step = epoch + 1
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def _grad_norm(model: GPTStopper) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq))


def _env_fingerprint(device: torch.device) -> dict:
    return {
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'hostname': socket.gethostname(),
        'python_version': platform.python_version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'pid': os.getpid(),
    }


def _load_oracle_table(distribution: str) -> dict:
    """Load the appropriate ADP table for `distribution`."""
    if distribution in STATIC_DISTRIBUTIONS:
        sigma = static_sigma(distribution)
        # Static ADP solves quickly (~1 sec); resolve fresh for each run rather
        # than caching per-distribution to keep this self-contained.
        from data.distributions import N
        C_hat, grids = solve_adp(N, MU_0, sigma * sigma, TAU0_2, K=2048, J=128)
        return {'C_hat': C_hat, 'grids': grids}
    if distribution == 'D_disc':
        path = ORACLE_TABLES / 'D_disc_K256_J64.npz'
    elif distribution == 'D_logu':
        path = ORACLE_TABLES / 'D_logu_K256_J64_Js64.npz'
    else:
        raise ValueError(f'Unknown distribution {distribution!r}')
    return load_random_table(path)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_one(config: RunConfig, device: torch.device) -> dict:
    """Train one model. Returns the metadata dict (also written to disk)."""
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to run if a previous run's log already exists in this dir.
    log_path = config.log_dir / 'log.jsonl'
    if log_path.exists():
        raise FileExistsError(
            f'log already exists: {log_path}. Move or delete first.'
        )

    config_dict = config.to_dict()
    (config.checkpoint_dir / 'config.json').write_text(
        json.dumps(config_dict, indent=2, default=str)
    )

    # ----- model + optimizer -----
    model_kwargs = config.model_kwargs()
    model = GPTStopper(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    warmup_steps = max(1, int(config.step_count * config.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(config.step_count, warmup_steps),
    )

    # ----- oracle table + val cache + masks -----
    label_source = getattr(config, 'label_source', 'oracle')
    use_offline = (label_source == 'offline')
    use_control = (label_source == 'control_zero')
    # Oracle table is only needed for oracle-supervised runs; for offline
    # and control runs we skip the table build entirely.
    table = None if (use_offline or use_control) else _load_oracle_table(config.distribution)

    X_val_np, sigma_val_np, _ = load_cache(config.distribution, 'val')
    if use_control:
        # Degenerate task: every label is 0. The cv label is unused (act-only).
        y_cv_val_np = np.zeros_like(X_val_np)
        y_act_val_np = np.zeros_like(X_val_np)
    elif use_offline:
        y_cv_val_np, y_act_val_np = label_offline(X_val_np)
    elif config.is_random:
        y_cv_val_np, y_act_val_np = label_random(X_val_np, table)
    else:
        y_cv_val_np, y_act_val_np = label_static(
            X_val_np, static_sigma(config.distribution), table,
        )

    cv_mask_np = make_cv_mask(config.distribution)
    act_mask_np = make_act_mask()

    # Move val tensors to device (one-time).
    X_val_t = torch.as_tensor(X_val_np, dtype=torch.float32, device=device)
    sigma_val_t = torch.as_tensor(sigma_val_np, dtype=torch.float32, device=device)
    y_cv_val_t = torch.as_tensor(y_cv_val_np, dtype=torch.float32, device=device)
    y_act_val_t = torch.as_tensor(y_act_val_np, dtype=torch.float32, device=device)
    cv_mask_t = torch.as_tensor(cv_mask_np, dtype=torch.bool, device=device)
    act_mask_t = torch.as_tensor(act_mask_np, dtype=torch.bool, device=device)

    # GPU-side random table (only built for random distributions and
    # only when oracle labeling is in use; offline / control labels need no table).
    table_t = (
        build_random_table_torch(table, device, dtype=torch.float64)
        if (config.is_random and not use_offline and not use_control)
        else None
    )

    # When offline- or control-labeling, skip the CPU oracle label pass
    # entirely. The streamer just produces X/sigma_i; labels are computed
    # on GPU in the per-step block below.
    streamer = stream_batches(
        config.distribution, config.batch_size, table, rng,
        compute_labels=(not config.is_random) and (not use_offline) and (not use_control),
    )

    # ----- logging -----
    env = _env_fingerprint(device)
    train_start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    train_start = time.perf_counter()
    log_f = log_path.open('a', buffering=1)
    log_f.write(json.dumps({
        'kind': 'start',
        'step': 0,
        'config': config_dict,
        'env': env,
        'wall_clock_start': train_start_iso,
    }) + '\n')

    train_smoothed: list[float] = []
    best_val_loss = float('inf')
    best_step = -1
    last_val_loss = float('nan')

    # ----- training loop -----
    model.train()
    for step in range(1, config.step_count + 1):
        X_np, sigma_i_np, y_cv_np_b, y_act_np_b, _cv_mask, _act_mask = next(streamer)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        sigma_i = torch.as_tensor(sigma_i_np, dtype=torch.float32, device=device)

        if use_control:
            # All-zero degenerate label — no σ-aware signal.
            y_cv_t = torch.zeros_like(X)
            y_act_t = torch.zeros_like(X)
        elif use_offline:
            # Hindsight (prophet) labels: O(n) suffix-max per sequence on GPU.
            y_cv_t, y_act_t = label_offline_torch(X.to(torch.float64))
            y_cv_t = y_cv_t.to(torch.float32)
            y_act_t = y_act_t.to(torch.float32)
        elif config.is_random:
            # GPU-side oracle labeling (single big advanced-indexing call).
            y_cv_t, y_act_t = label_random_torch(X.to(torch.float64), table_t)
            y_cv_t = y_cv_t.to(torch.float32)
            y_act_t = y_act_t.to(torch.float32)
        else:
            y_cv_t = torch.as_tensor(y_cv_np_b, dtype=torch.float32, device=device)
            y_act_t = torch.as_tensor(y_act_np_b, dtype=torch.float32, device=device)

        out = model(X)
        if config.supervision == 'cv':
            loss = cv_loss(out['cv'], y_cv_t, sigma_i, cv_mask_t,
                           sigma_power=config.sigma_power)
        else:
            loss = act_loss(out['act'], y_act_t, act_mask_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = _grad_norm(model)
        optimizer.step()
        scheduler.step()

        train_smoothed.append(loss.item())
        if len(train_smoothed) > 100:
            train_smoothed.pop(0)

        if step % config.train_log_every == 0:
            log_f.write(json.dumps({
                'kind': 'train', 'step': step,
                'train_loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'grad_norm': gnorm,
                'wall_s_since_start': time.perf_counter() - train_start,
            }) + '\n')

        # ----- validation -----
        if step % config.val_every == 0 or step == config.step_count:
            model.eval()
            per_seq_loss_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                val_batch = 256
                N_val = X_val_t.shape[0]
                for i in range(0, N_val, val_batch):
                    s = slice(i, i + val_batch)
                    out_b = model(X_val_t[s])
                    if config.supervision == 'cv':
                        sq_err = (out_b['cv'] - y_cv_val_t[s]).pow(2)
                        masked = sq_err * cv_mask_t.to(sq_err.dtype)
                        per_seq_mse = masked.sum(dim=1) / cv_mask_t.to(sq_err.dtype).sum()
                        sigma_b = sigma_val_t[s].to(sq_err.dtype)
                        sigma_pow = (sigma_b if config.sigma_power == 1
                                     else sigma_b * sigma_b)
                        per_seq = per_seq_mse / sigma_pow
                    else:
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(
                            out_b['act'], y_act_val_t[s], reduction='none',
                        )
                        masked = bce * act_mask_t.to(bce.dtype)
                        per_seq = masked.sum(dim=1) / act_mask_t.to(bce.dtype).sum()
                    per_seq_loss_chunks.append(per_seq)
            per_seq_loss = torch.cat(per_seq_loss_chunks, dim=0)
            last_val_loss = float(per_seq_loss.mean().item())

            per_sigma_fields = _per_sigma_val_fields(
                config.distribution, sigma_val_t, per_seq_loss,
            )
            model.train()

            val_record = {
                'kind': 'val', 'step': step,
                'val_loss': last_val_loss,
                'train_loss_smoothed_100': float(np.mean(train_smoothed)),
                'wall_s_since_start': time.perf_counter() - train_start,
                **per_sigma_fields,
            }
            log_f.write(json.dumps(val_record) + '\n')

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_step = step
                payload = build_payload(
                    model, optimizer, scheduler, step, last_val_loss,
                    is_best=True, is_periodic=False,
                    trained_head=config.supervision,
                    model_config=model_kwargs, config_dict=config_dict,
                    rng_states={'torch': torch.get_rng_state(),
                                'numpy': rng.bit_generator.state},
                )
                save_checkpoint(
                    config.checkpoint_dir / 'best.pt', payload, overwrite=True,
                )
                log_f.write(json.dumps({
                    'kind': 'checkpoint', 'step': step,
                    'path': str(config.checkpoint_dir / 'best.pt'),
                    'val_loss': last_val_loss,
                    'is_best': True, 'is_periodic': False,
                }) + '\n')

        # ----- periodic checkpoint -----
        if config.periodic_every and step % config.periodic_every == 0:
            ckpt_name = f'step_{step // 1000}k.pt'
            ckpt_path = config.checkpoint_dir / ckpt_name
            payload = build_payload(
                model, optimizer, scheduler, step, last_val_loss,
                is_best=False, is_periodic=True,
                trained_head=config.supervision,
                model_config=model_kwargs, config_dict=config_dict,
                rng_states={'torch': torch.get_rng_state(),
                            'numpy': rng.bit_generator.state},
            )
            save_checkpoint(ckpt_path, payload, overwrite=False)
            log_f.write(json.dumps({
                'kind': 'checkpoint', 'step': step,
                'path': str(ckpt_path), 'val_loss': last_val_loss,
                'is_best': False, 'is_periodic': True,
            }) + '\n')

    # ----- final checkpoint -----
    final_payload = build_payload(
        model, optimizer, scheduler, config.step_count, last_val_loss,
        is_best=False, is_periodic=False,
        trained_head=config.supervision,
        model_config=model_kwargs, config_dict=config_dict,
        rng_states={'torch': torch.get_rng_state(),
                    'numpy': rng.bit_generator.state},
    )
    save_checkpoint(config.checkpoint_dir / 'final.pt', final_payload, overwrite=True)

    train_wall_s = time.perf_counter() - train_start
    final_train_loss = float(loss.item())
    log_f.write(json.dumps({
        'kind': 'end',
        'total_steps': config.step_count,
        'total_wall_s': train_wall_s,
        'final_train_loss': final_train_loss,
        'final_val_loss': last_val_loss,
        'best_val_loss': best_val_loss,
        'best_step': best_step,
    }) + '\n')
    log_f.close()

    _emit_curves_npz(log_path, config.log_dir / 'curves.npz')

    metadata = {
        'run_name': config.run_name,
        'training_start': train_start_iso,
        'training_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_wall_s': train_wall_s,
        'best_val_loss': best_val_loss,
        'best_step': best_step,
        'final_train_loss': final_train_loss,
        'final_val_loss': last_val_loss,
        'environment': env,
    }
    (config.checkpoint_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return metadata


def _emit_curves_npz(log_path: Path, out_path: Path) -> None:
    train_step: list[int] = []
    train_loss: list[float] = []
    val_step: list[int] = []
    val_loss: list[float] = []
    val_sigma_fields: dict[str, list[float]] = {}
    with log_path.open('r') as f:
        for line in f:
            ev = json.loads(line)
            if ev['kind'] == 'train':
                train_step.append(ev['step'])
                train_loss.append(ev['train_loss'])
            elif ev['kind'] == 'val':
                val_step.append(ev['step'])
                val_loss.append(ev['val_loss'])
                for k, v in ev.items():
                    if k.startswith('val_loss_sigma_'):
                        val_sigma_fields.setdefault(k, []).append(float(v))
    payload = {
        'train_step': np.array(train_step, dtype=np.int64),
        'train_loss': np.array(train_loss, dtype=np.float64),
        'val_step': np.array(val_step, dtype=np.int64),
        'val_loss': np.array(val_loss, dtype=np.float64),
    }
    for k, vs in val_sigma_fields.items():
        payload[k] = np.array(vs, dtype=np.float64)
    np.savez(out_path, **payload)


def _per_sigma_val_fields(
    distribution: str,
    sigma_val: torch.Tensor,         # (N_val,)
    per_seq_loss: torch.Tensor,      # (N_val,)
) -> dict[str, float]:
    """Per-σ-group mean of the per-sequence val loss.

    For static D_1 / D_2 / D_3 the val set has a single σ; we emit
    val_loss_sigma_{int(σ)} as a duplicate of the aggregate val loss.
    For D_disc we group by σ ∈ {1, 10, 100}. For D_logu we bin by
    log_10 σ on [0, 2/3), [2/3, 4/3), [4/3, 2] (low / mid / high).
    """
    fields: dict[str, float] = {}
    if distribution in ('D_1', 'D_2', 'D_3'):
        sigma_unique = float(sigma_val.unique().item())
        fields[f'val_loss_sigma_{int(sigma_unique)}'] = float(per_seq_loss.mean().item())
    elif distribution == 'D_disc':
        for s in (1.0, 10.0, 100.0):
            mask = (sigma_val == s)
            if mask.any():
                fields[f'val_loss_sigma_{int(s)}'] = float(per_seq_loss[mask].mean().item())
    elif distribution == 'D_logu':
        log10_sigma = torch.log10(sigma_val)
        bins = (
            (0.0,        2.0 / 3.0, 'low'),
            (2.0 / 3.0,  4.0 / 3.0, 'mid'),
            (4.0 / 3.0,  2.0,        'high'),
        )
        for lo, hi, name in bins:
            if name == 'high':
                mask = (log10_sigma >= lo) & (log10_sigma <= hi)
            else:
                mask = (log10_sigma >= lo) & (log10_sigma < hi)
            if mask.any():
                fields[f'val_loss_sigma_{name}'] = float(per_seq_loss[mask].mean().item())
    return fields
