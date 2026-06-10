"""Timestep-shared simple linear probe trained on cached hidden states.

Loss / reporting: sigma-normalized MSE
    sMSE_v(E) = mean_{(X_{1:t}, v_t, sigma_i) in E} ((v_hat - v_t) / sigma_i)^2,
where sigma_i is the per-sequence true latent (constant within a sequence,
varies across sequences). Used for both probe targets in the preliminary
run (sqrt(hat_sigma^2) and Bayes continuation value); see report §7.5.
No epsilon clip is needed since sigma_i in [1, 100] under D_logu.

Probe form (timestep-shared, simple, linear):
    v_hat_t = w^T h_t^{(l)} + b.

Optimizer: Adam(lr=1e-3, batch=256), capped at one epoch, best-validation
checkpoint selected. The same probe configuration is used for the
budget-sweep sub-runs at M_train in {64, 128, 256, 512, 1024}; M_val and
M_test are fixed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sigma_norm_mse(
    pred: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """Mean of ((pred - target) / sigma_i)^2 over the batch.

    `sigma` is the per-sequence true latent (>= 1 under D_logu); no clipping
    needed since it is bounded away from zero by construction.
    """
    return ((pred - target) / sigma).pow(2).mean()


# ---------------------------------------------------------------------------
# Data assembly: flatten (M, n) hidden states + targets into supervised pairs
# ---------------------------------------------------------------------------

def _flatten_valid(
    H: np.ndarray, Y: np.ndarray, sigma_i: np.ndarray, t_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (M, n, d) hiddens + (M, n) targets + (M,) per-sequence sigma
    across all valid (m, t) pairs.

    Validity: NaN-mask the target; also drop t < t_start (e.g. t_start=1 to
    cover only t >= 2 for variance-based targets). sigma_i is constant within
    a sequence and broadcasts across t. Returns float32 (P, d), (P,), (P,).
    """
    M, n, d = H.shape
    if Y.shape != (M, n):
        raise ValueError(f'shape mismatch: H {H.shape} vs Y {Y.shape}')
    if sigma_i.shape != (M,):
        raise ValueError(f'sigma shape {sigma_i.shape} != (M={M},)')
    H = H[:, t_start:]                                    # (M, n', d)
    Y = Y[:, t_start:]                                    # (M, n')
    n_prime = Y.shape[1]
    sigma_b = np.broadcast_to(sigma_i[:, None], (M, n_prime))
    H_flat = H.reshape(-1, d).astype(np.float32)
    Y_flat = Y.reshape(-1).astype(np.float32)
    sigma_flat = sigma_b.reshape(-1).astype(np.float32)
    valid = np.isfinite(Y_flat)
    return H_flat[valid], Y_flat[valid], sigma_flat[valid]


def t_start_for(target_name: str) -> int:
    """Index of the first valid prefix for a given target.

    sigma_mle_sqrt requires t >= 2 (Bessel correction; t=1 yields NaN).
    C_star is defined for t = 1..n-1; the t=n column is NaN.
    """
    if target_name == 'sigma_mle_sqrt':
        return 1                                          # t=2 is the first valid prefix (0-indexed col 1)
    if target_name == 'C_star':
        return 0
    raise ValueError(target_name)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    target: str
    layer: int
    base_rep: str
    M_train_sequences: int
    n_train_examples: int
    train_loss_final: float
    val_loss_best: float
    val_loss_step_best: int
    test_loss: float
    wall_s: float


def train_one_probe(
    *,
    H_train: np.ndarray, Y_train: np.ndarray, sigma_train: np.ndarray,
    H_val:   np.ndarray, Y_val:   np.ndarray, sigma_val:   np.ndarray,
    H_test:  np.ndarray, Y_test:  np.ndarray, sigma_test:  np.ndarray,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 1,
    seed: int = 0,
    val_chunks_per_epoch: int = 8,
) -> tuple[float, float, int, float]:
    """Train a timestep-shared simple linear probe with sigma-normalized MSE.

    Returns (final_train_loss, best_val_loss, val_loss_step_best, test_loss).
    """
    torch.manual_seed(seed)

    d = H_train.shape[1]
    probe = torch.nn.Linear(d, 1, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    Xt = torch.from_numpy(H_train).to(device)
    yt = torch.from_numpy(Y_train).to(device)
    st = torch.from_numpy(sigma_train).to(device)
    ds = TensorDataset(Xt, yt, st)
    g = torch.Generator(device='cpu').manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False,
    )

    Xv = torch.from_numpy(H_val).to(device)
    yv = torch.from_numpy(Y_val).to(device)
    sv = torch.from_numpy(sigma_val).to(device)
    Xte = torch.from_numpy(H_test).to(device)
    yte = torch.from_numpy(Y_test).to(device)
    ste = torch.from_numpy(sigma_test).to(device)

    @torch.no_grad()
    def _eval(X_, y_, s_) -> float:
        probe.eval()
        out = probe(X_).squeeze(-1)
        L = sigma_norm_mse(out, y_, s_).item()
        probe.train()
        return L

    best_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    best_val_step = 0
    final_train_loss = float('nan')

    total_steps = max_epochs * max(1, (len(ds) + batch_size - 1) // batch_size)
    val_every = max(1, total_steps // val_chunks_per_epoch)
    step = 0

    probe.train()
    for _epoch in range(max_epochs):
        for Xb, yb, sb in loader:
            step += 1
            out = probe(Xb).squeeze(-1)
            loss = sigma_norm_mse(out, yb, sb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_train_loss = loss.item()
            if step % val_every == 0 or step == total_steps:
                vL = _eval(Xv, yv, sv)
                if vL < best_val:
                    best_val = vL
                    best_state = {k: v.detach().clone()
                                  for k, v in probe.state_dict().items()}
                    best_val_step = step

    probe.load_state_dict(best_state)
    test_loss = _eval(Xte, yte, ste)
    return final_train_loss, best_val, best_val_step, test_loss


# ---------------------------------------------------------------------------
# Layer x target x M_train sweep, per base rep
# ---------------------------------------------------------------------------

def _load_layer(
    cache_dir: Path, layer: int, split: str,
) -> np.ndarray:
    """Memory-mapped load of one layer/split hidden cache."""
    path = cache_dir / f'H_layer_{layer}_{split}.npy'
    return np.load(path, mmap_mode='r')                  # (M, n, d) fp16


def run_probe_sweep(
    *,
    member_dir: Path,
    rep_name: str,
    targets: tuple[str, ...] = ('sigma_mle_sqrt', 'C_star'),
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] = (64, 128, 256, 512, 1024),
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Run all (target, layer, M_train) probes for one base representation.

    Hidden caches at <member_dir>/hidden_cache/<rep_name>/.
    Targets at <member_dir>/probe_data/Y_<tgt>_<split>.npy.
    """
    probe_data_dir = member_dir / 'probe_data'
    cache_dir = member_dir / 'hidden_cache' / rep_name

    cache_meta = json.loads((cache_dir / 'meta.json').read_text())
    L_plus_1 = cache_meta['n_layers_total']
    if layers is None:
        layers = tuple(range(L_plus_1))

    out_dir = member_dir / 'probe_runs' / rep_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_budget = max(M_train_budgets)
    rows: list[dict] = []

    # Per-sequence sigma_i (shared across targets and layers).
    sigma_train_full = np.load(probe_data_dir / 'sigma_train.npy')           # (M_full,)
    sigma_val        = np.load(probe_data_dir / 'sigma_val.npy')
    sigma_test       = np.load(probe_data_dir / 'sigma_test.npy')

    for tgt in targets:
        t0_target = time.perf_counter()
        t_start = t_start_for(tgt)
        # ---- targets, all splits, NumPy in RAM (small) ----
        Y_train_full = np.load(probe_data_dir / f'Y_{tgt}_train.npy')        # (M_full, n)
        Y_val        = np.load(probe_data_dir / f'Y_{tgt}_val.npy')
        Y_test       = np.load(probe_data_dir / f'Y_{tgt}_test.npy')

        for layer in layers:
            # Hidden states for this layer, all splits.
            H_train_full = _load_layer(cache_dir, layer, 'train')            # (M_full, n, d) fp16
            H_val        = _load_layer(cache_dir, layer, 'val')
            H_test       = _load_layer(cache_dir, layer, 'test')

            # Materialize val/test flattened triples once per (target, layer).
            Hv_flat, Yv_flat, Sv_flat = _flatten_valid(
                np.asarray(H_val),  Y_val,  sigma_val,  t_start)
            Hte_flat, Yte_flat, Ste_flat = _flatten_valid(
                np.asarray(H_test), Y_test, sigma_test, t_start)

            for M_train in M_train_budgets:
                if M_train > Y_train_full.shape[0]:
                    raise ValueError(
                        f'M_train={M_train} > available {Y_train_full.shape[0]}')
                t1 = time.perf_counter()
                # nested prefix: deterministic across budgets
                Y_train_sub = Y_train_full[:M_train]
                H_train_sub = np.asarray(H_train_full[:M_train])
                sigma_train_sub = sigma_train_full[:M_train]
                Ht_flat, Yt_flat, St_flat = _flatten_valid(
                    H_train_sub, Y_train_sub, sigma_train_sub, t_start)
                final_tr, best_val, best_step, test_loss = train_one_probe(
                    H_train=Ht_flat, Y_train=Yt_flat, sigma_train=St_flat,
                    H_val=Hv_flat,   Y_val=Yv_flat,   sigma_val=Sv_flat,
                    H_test=Hte_flat, Y_test=Yte_flat, sigma_test=Ste_flat,
                    device=device,
                    seed=seed + 1000 * layer + M_train,
                )
                rows.append(dict(
                    target=tgt, layer=layer, base_rep=rep_name,
                    M_train_sequences=M_train,
                    n_train_examples=Ht_flat.shape[0],
                    train_loss_final=final_tr,
                    val_loss_best=best_val,
                    val_loss_step_best=best_step,
                    test_loss=test_loss,
                    wall_s=time.perf_counter() - t1,
                ))
        print(f'  [probe] target={tgt} done in {time.perf_counter() - t0_target:.1f}s')

    out_path = out_dir / 'probe_results.json'
    out_path.write_text(json.dumps(rows, indent=2))
    print(f'  [probe] wrote {out_path}  ({len(rows)} cells)')
    return {'rows': rows, 'out_path': str(out_path)}
