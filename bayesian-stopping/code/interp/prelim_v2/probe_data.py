"""Probe data for the preliminary mechanistic interpretability run.

Per ensemble member i we generate:
  - X_probe_{train,val,test}.npy  : D_logu sequences for that member's probes.
                                    Seeds disjoint from base training/validation.
  - Y_sigmaMLE_{split}.npy        : sqrt(hat_sigma_t^2), valid t >= 2.
  - Y_Cstar_{split}.npy           : Bayes continuation value (D_logu ADP).
  - H_layer_<L>_<split>.npy       : fp16 per-layer prefix hidden states from a
                                    given frozen checkpoint, written per base
                                    representation under cache_dir/<rep>/.

The X splits are SHARED across the two base representations within a member
(true and perm_glob see the same probe data); targets only depend on X, so
they are computed once per member and reused across reps.

We use ``M_train=1024``, ``M_val=512``, ``M_test=10^4`` for the full-budget run.
The probe-label budget sweep at ``M_train in {64,128,256,512,1024}`` reuses
nested prefixes of X_probe_train.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from data.distributions import sample as sample_distribution
from model.transformer import GPTStopper
from oracle.random_adp import load_table as load_random_table, query as query_random
from train.configs import ORACLE_TABLES


# ---------------------------------------------------------------------------
# Splits and target validity
# ---------------------------------------------------------------------------

DEFAULT_SPLITS = {
    'train': 1024,
    'val':    512,
    'test':   10_000,
}

# Seed offsets used to make per-member probe splits disjoint from base
# training (which uses seed `1000 + ensemble_idx` in the original code path;
# here we use ``base_seed = 2_000_000 + ensemble_idx`` and probe seeds far
# above). 'train'/'val' base caches sit at seeds ~42 / ~43; we pin probes at
# 5_000_000+ to keep all three datasets disjoint.
PROBE_SEED_BASE = 5_000_000


def _seed_for(ensemble_idx: int, split: str) -> int:
    offsets = {'train': 0, 'val': 1, 'test': 2}
    if split not in offsets:
        raise KeyError(split)
    return PROBE_SEED_BASE + 10 * ensemble_idx + offsets[split]


def sample_probe_X(ensemble_idx: int, split: str, M: int, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, sigma_i) for one probe split. RNG seed is deterministic
    in (ensemble_idx, split). Note: data.distributions.sample uses the fixed
    module-level ``N=256``; the ``n`` argument here is documentary only.
    """
    if n != 256:
        raise NotImplementedError('probe sampler is pinned to n=256 by data.distributions.N')
    rng = np.random.default_rng(_seed_for(ensemble_idx, split))
    X, sigma_i, _mu_i = sample_distribution('D_logu', M, rng)
    return X.astype(np.float32), sigma_i.astype(np.float32)


# ---------------------------------------------------------------------------
# Targets: sqrt(sigma_hat^2) and C_star  (no probe-rep dependence)
# ---------------------------------------------------------------------------

def _running_sums(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S_t and Q_t for t = 1..n, shape (M, n)."""
    S = np.cumsum(X.astype(np.float64), axis=1)
    Q = np.cumsum((X.astype(np.float64))**2, axis=1)
    return S, Q


def sigma_mle_sqrt(X: np.ndarray) -> np.ndarray:
    """Bessel-corrected running sample SD sqrt(hat_sigma_t^2), shape (M, n).

    Defined for t >= 2; t = 1 entry is filled with NaN.
    """
    M, n = X.shape
    S, Q = _running_sums(X)
    t = np.arange(1, n + 1, dtype=np.float64)
    bar = S / t
    var = (Q - t * bar**2) / np.maximum(t - 1.0, 1.0)
    var = np.clip(var, 0.0, None)                                # numerical floor
    out = np.sqrt(var).astype(np.float32)
    out[:, 0] = np.nan
    return out


def C_star_random(X: np.ndarray, table: dict) -> np.ndarray:
    """Bayes continuation value C_hat_t^* under the D_logu ADP table.

    Returns (M, n) with C_star at decision timesteps t = 1..n-1, NaN at t=n.
    """
    M, n = X.shape
    S, Q = _running_sums(X)
    out = np.full((M, n), np.nan, dtype=np.float32)
    # Query expects 1-indexed t and (M,) S_t, Q_t per call.
    for t in range(1, n):  # decision timesteps 1..n-1
        out[:, t - 1] = query_random(table, t_one_indexed=t,
                                     S_t=S[:, t - 1], Q_t=Q[:, t - 1])
    return out


def build_targets(X: np.ndarray, table: dict) -> dict[str, np.ndarray]:
    """Return {'sigma_mle_sqrt': (M, n), 'C_star': (M, n)} for one split."""
    return {
        'sigma_mle_sqrt': sigma_mle_sqrt(X),
        'C_star': C_star_random(X, table),
    }


# ---------------------------------------------------------------------------
# Member-level data layout
# ---------------------------------------------------------------------------

def materialize_probe_data(
    member_dir: Path,
    ensemble_idx: int,
    splits: dict[str, int] | None = None,
    n: int = 256,
) -> dict:
    """Sample probe X for one ensemble member, compute targets, write to disk.

    Layout under <member_dir>/probe_data/:
      X_<split>.npy, sigma_<split>.npy, Y_sigma_mle_sqrt_<split>.npy,
      Y_C_star_<split>.npy, meta.json.

    Returns the meta dict.
    """
    splits = splits or DEFAULT_SPLITS
    out_dir = member_dir / 'probe_data'
    out_dir.mkdir(parents=True, exist_ok=False)

    table = load_random_table(ORACLE_TABLES / 'D_logu_K256_J64_Js64.npz')

    meta = {
        'ensemble_idx': ensemble_idx,
        'n': n,
        'splits': {},
        'targets': ['sigma_mle_sqrt', 'C_star'],
        'oracle_table': 'D_logu_K256_J64_Js64.npz',
    }
    for split, M in splits.items():
        t0 = time.perf_counter()
        X, sigma_i = sample_probe_X(ensemble_idx, split, M, n=n)
        np.save(out_dir / f'X_{split}.npy', X)
        np.save(out_dir / f'sigma_{split}.npy', sigma_i)
        targets = build_targets(X, table)
        for tname, arr in targets.items():
            np.save(out_dir / f'Y_{tname}_{split}.npy', arr)
        meta['splits'][split] = {
            'M': M, 'seed': _seed_for(ensemble_idx, split),
            'wall_s': time.perf_counter() - t0,
        }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Hidden caching (per base representation)
# ---------------------------------------------------------------------------

def _load_paired_checkpoint(ckpt_path: Path, device: torch.device) -> GPTStopper:
    """Load a checkpoint saved by train_paired.train_paired."""
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mcfg = payload['model_config']
    model = GPTStopper(
        n=mcfg['n'], d_emb=mcfg['d_emb'],
        n_layers=mcfg['n_layers'], n_heads=mcfg['n_heads'],
    )
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval()


@torch.no_grad()
def build_hidden_cache(
    *,
    ckpt_path: Path,
    member_dir: Path,
    rep_name: str,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Forward-pass the frozen base model on the member's probe splits,
    dump per-layer fp16 hidden states under <member_dir>/hidden_cache/<rep_name>/.
    """
    out_dir = member_dir / 'hidden_cache' / rep_name
    out_dir.mkdir(parents=True, exist_ok=False)

    model = _load_paired_checkpoint(ckpt_path, device)
    n_layers_total = model.n_layers + 1                 # depths 0..L
    n = model.n
    d_emb = model.d_emb

    probe_data_dir = member_dir / 'probe_data'
    splits_meta = json.loads((probe_data_dir / 'meta.json').read_text())['splits']

    meta = {
        'ckpt_path': str(ckpt_path),
        'rep_name': rep_name,
        'n_layers_total': n_layers_total,
        'n': n, 'd_emb': d_emb,
        'splits': {},
    }
    for split in splits_meta:
        X = np.load(probe_data_dir / f'X_{split}.npy')              # (M, n) fp32
        M = X.shape[0]
        # Pre-allocate per-layer memmaps.
        memmaps = []
        for layer in range(n_layers_total):
            path = out_dir / f'H_layer_{layer}_{split}.npy'
            mm = np.lib.format.open_memmap(
                path, mode='w+', dtype=np.float16,
                shape=(M, n, d_emb),
            )
            memmaps.append(mm)
        t0 = time.perf_counter()
        for i in range(0, M, batch_size):
            j = min(i + batch_size, M)
            Xt = torch.as_tensor(X[i:j], dtype=torch.float32, device=device)
            out = model(Xt, return_hidden=True)
            hidden = out['hidden']                                  # (B, L+1, n, d_emb)
            for layer in range(n_layers_total):
                arr = hidden[:, layer].to(torch.float16).cpu().numpy()
                memmaps[layer][i:j] = arr
        for mm in memmaps:
            mm.flush()
            del mm
        meta['splits'][split] = {'M': M, 'wall_s': time.perf_counter() - t0}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta
