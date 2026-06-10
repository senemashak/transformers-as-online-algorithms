"""Build hidden-state caches for the *untrained* (step-0) baseline.

For each ensemble member, reconstructs the deterministic init state of the
GPTStopper from the same seed used in train_paired (seed = BASE_SEED + i),
then forwards the existing probe_data X splits through it and writes the
per-layer hidden states under <member_dir>/hidden_cache/init/. This is the
third "rep" alongside `true` and `perm_glob`; the model is never trained.

Init weights are bit-reproducible from the seed alone (see train_paired.py
line 148: torch.manual_seed(cfg['seed']) is called before model creation).
The init is shared by the `true` and `perm_glob` arms of each member by
design, so a single init cache per member is correct.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from model.transformer import GPTStopper
from interp.prelim_v2.train_paired import DEFAULTS as TRAIN_DEFAULTS


REP_NAME = 'init'


def _build_init_model(seed: int, device: torch.device) -> GPTStopper:
    """Reconstruct the step-0 GPTStopper for the given seed; bit-identical to
    what training would have started from.
    """
    torch.manual_seed(int(seed))
    model = GPTStopper(
        n=TRAIN_DEFAULTS['n'], d_emb=TRAIN_DEFAULTS['d_emb'],
        n_layers=TRAIN_DEFAULTS['n_layers'], n_heads=TRAIN_DEFAULTS['n_heads'],
    )
    return model.to(device).eval()


@torch.no_grad()
def build_init_hidden_cache(
    *,
    member_dir: Path,
    seed: int,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Cache per-layer hidden states from the seed-deterministic init model.

    Writes <member_dir>/hidden_cache/init/H_layer_<L>_<split>.npy (fp16,
    shape (M, n, d_emb)) and a meta.json. Refuses to overwrite an existing
    init cache so re-runs are idempotent.
    """
    out_dir = member_dir / 'hidden_cache' / REP_NAME
    if out_dir.exists() and (out_dir / 'meta.json').exists():
        print(f'[init cache] {member_dir.name}: exists, skipping')
        return json.loads((out_dir / 'meta.json').read_text())
    out_dir.mkdir(parents=True, exist_ok=False)

    model = _build_init_model(seed, device)
    n_layers_total = model.n_layers + 1                     # depths 0..L
    n = model.n
    d_emb = model.d_emb

    probe_data_dir = member_dir / 'probe_data'
    splits_meta = json.loads((probe_data_dir / 'meta.json').read_text())['splits']

    meta = {
        'rep_name': REP_NAME,
        'seed': int(seed),
        'source': 'seed-deterministic init (no training)',
        'n_layers_total': n_layers_total,
        'n': n, 'd_emb': d_emb,
        'splits': {},
    }
    for split in splits_meta:
        X = np.load(probe_data_dir / f'X_{split}.npy')               # (M, n) fp32
        M = X.shape[0]
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
            hidden = out['hidden']                                   # (B, L+1, n, d_emb)
            for layer in range(n_layers_total):
                memmaps[layer][i:j] = hidden[:, layer].to(torch.float16).cpu().numpy()
        for mm in memmaps:
            mm.flush(); del mm
        meta['splits'][split] = {'M': int(M), 'wall_s': time.perf_counter() - t0}

    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta
