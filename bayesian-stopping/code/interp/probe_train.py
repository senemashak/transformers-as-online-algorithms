"""
Probe training loop.

A single `train_probe(layer, target, variant)` call:
  - loads the layer-ℓ activation cache (memmap) and the corresponding target,
  - trains an `AttentionPooledProbe` with Adam at lr=1e-3,
  - reports raw MSE and normalized MSE = MSE / Var(target) on train/val/test.

NaN target positions (outside the supervised t-range) are masked out of the
loss and the metric.

For efficiency the orchestrator (`probe_run.py`) groups probes by layer so
the memmap-backed activation cache is read once per layer rather than once
per probe.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from interp.probe_data import PROBE_DATA_ROOT, SPLITS
from interp.probe_model import AttentionPooledProbe, PerTimestepProbe
from interp.probe_paths import CACHE_ROOT, RESULTS_ROOT  # noqa: F401 — re-exported


def _build_probe(probe_kind: str, n: int, d_emb: int, variant: str):
    if probe_kind == 'attn':
        return AttentionPooledProbe(n=n, d_emb=d_emb, variant=variant)
    if probe_kind == 'noattn':
        return PerTimestepProbe(n=n, d_emb=d_emb, variant=variant)
    raise ValueError(f'probe_kind must be attn or noattn, got {probe_kind!r}')

# Adam + epochs per spec.
LR = 1e-3
EPOCHS_LINEAR = 3
EPOCHS_MLP = 5
BATCH_SIZE_DEFAULT = 256
WEIGHT_DECAY = 0.0


@dataclass
class ProbeResult:
    target: str
    layer: int
    variant: str
    n_epochs: int
    train_mse: float
    val_mse: float
    test_mse: float
    train_norm_mse: float
    val_norm_mse: float
    test_norm_mse: float
    target_var: float
    wall_s: float
    best_epoch: int
    attention_argmax_per_t_mean: float       # mean over t of |argmax_α_t - t| (positions back)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        return d


def _load_activations(cache_dir: Path, layer: int) -> Dict[str, np.memmap]:
    return {
        split: np.load(cache_dir / f'H_layer_{layer}_{split}.npy', mmap_mode='r')
        for split in SPLITS
    }


def _load_target(target_name: str, shuffled: bool = False) -> Dict[str, np.ndarray]:
    name = 'targets_shuffled' if shuffled else 'targets'
    out = {}
    for split in SPLITS:
        z = np.load(PROBE_DATA_ROOT / f'{name}_{split}.npz')
        out[split] = z[target_name]                                     # (N, n) fp32, NaN outside band
    return out


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    # Replace NaN targets with 0 so the difference is finite; then mask out.
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    diff = (pred - safe_target) * mask.to(pred.dtype)
    return diff.pow(2).sum() / mask.sum().clamp_min(1.0).to(pred.dtype)


def _evaluate(
    probe: AttentionPooledProbe,
    H_mm,                                              # np.memmap | np.ndarray | torch.Tensor (cuda)
    y,                                                 # np.ndarray | torch.Tensor
    device: torch.device,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> float:
    probe.eval()
    sum_sq = 0.0
    cnt = 0
    H_on_gpu = isinstance(H_mm, torch.Tensor) and H_mm.is_cuda
    if H_on_gpu and not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, device=device)
    with torch.no_grad():
        for i in range(0, H_mm.shape[0], batch_size):
            j = min(i + batch_size, H_mm.shape[0])
            if H_on_gpu:
                H_b = H_mm[i:j].to(torch.float32)
                y_b = y[i:j]
            else:
                H_b = torch.from_numpy(np.ascontiguousarray(H_mm[i:j])).to(
                    device=device, dtype=torch.float32,
                )
                y_b = torch.from_numpy(np.ascontiguousarray(y[i:j])).to(device)
            v_hat = probe(H_b)
            mask = ~torch.isnan(y_b)
            safe_target = torch.where(mask, y_b, torch.zeros_like(y_b))
            diff = (v_hat - safe_target) * mask.to(v_hat.dtype)
            sum_sq += float(diff.pow(2).sum().item())
            cnt += int(mask.sum().item())
    probe.train()
    return sum_sq / max(cnt, 1)


def train_probe(
    layer: int,
    target_name: str,
    variant: str,
    cache_dir: Path,
    device: torch.device,
    target_arrs: Optional[Dict[str, np.ndarray]] = None,
    H_arrs: Optional[Dict[str, np.memmap]] = None,
    batch_size: int = BATCH_SIZE_DEFAULT,
    seed: int = 13,
    n_epochs: Optional[int] = None,
    probe_kind: str = 'attn',
) -> ProbeResult:
    if H_arrs is None:
        H_arrs = _load_activations(cache_dir, layer)
    if target_arrs is None:
        target_arrs = _load_target(target_name)

    H_train = H_arrs['train']
    H_val = H_arrs['val']
    H_test = H_arrs['test']
    y_train = target_arrs['train']
    y_val = target_arrs['val']
    y_test = target_arrs['test']

    n = H_train.shape[1]
    d_emb = H_train.shape[2]

    if n_epochs is None:
        n_epochs = EPOCHS_LINEAR if variant == 'linear' else EPOCHS_MLP

    # Variance of the target on the test set (over non-NaN positions).
    y_test_valid = y_test[~np.isnan(y_test)]
    target_var = float(y_test_valid.var())

    probe = _build_probe(probe_kind, n=n, d_emb=d_emb, variant=variant).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    best_val_mse = float('inf')
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    best_epoch = -1

    N_train = H_train.shape[0]
    # If H_train is already a torch tensor on GPU, use it directly. Otherwise
    # the orchestrator hands us a numpy ndarray and we do per-batch copy.
    H_train_on_gpu = isinstance(H_train, torch.Tensor) and H_train.is_cuda
    if H_train_on_gpu:
        y_train_gpu = torch.as_tensor(y_train, device=device)
    for epoch in range(n_epochs):
        perm = rng.permutation(N_train)
        for i in range(0, N_train, batch_size):
            idx = perm[i : i + batch_size]
            if H_train_on_gpu:
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                H_b = H_train[idx_t].to(torch.float32)
                y_b = y_train_gpu[idx_t]
            else:
                H_b = torch.from_numpy(np.ascontiguousarray(H_train[idx])).to(
                    device=device, dtype=torch.float32,
                )
                y_b = torch.from_numpy(np.ascontiguousarray(y_train[idx])).to(device)
            v_hat = probe(H_b)
            loss = _masked_mse(v_hat, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_mse = _evaluate(probe, H_val, y_val, device, batch_size=batch_size)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch

    probe.load_state_dict(best_state)
    test_mse = _evaluate(probe, H_test, y_test, device, batch_size=batch_size)
    train_mse_final = _evaluate(probe, H_train, y_train, device, batch_size=batch_size)
    val_mse_final = best_val_mse

    # Position-attention summary: argmax(α_t) tells us which past timestep
    # the probe leaned on; report mean (t - argmax) over t in [1, n-1].
    with torch.no_grad():
        alpha = probe.attention_weights().cpu().numpy()                  # (n, n)
        # For t = 1..n-1: argmax over [0..t], then offset.
        offsets = []
        for t_idx in range(1, n - 1):
            arg = int(np.argmax(alpha[t_idx, : t_idx + 1]))
            offsets.append(t_idx - arg)
        attn_arg_mean = float(np.mean(offsets))

    return ProbeResult(
        target=target_name, layer=layer, variant=variant,
        n_epochs=n_epochs,
        train_mse=train_mse_final, val_mse=val_mse_final, test_mse=test_mse,
        train_norm_mse=train_mse_final / max(target_var, 1e-30),
        val_norm_mse=val_mse_final / max(target_var, 1e-30),
        test_norm_mse=test_mse / max(target_var, 1e-30),
        target_var=target_var,
        wall_s=time.perf_counter() - t0,
        best_epoch=best_epoch,
        attention_argmax_per_t_mean=attn_arg_mean,
    )
