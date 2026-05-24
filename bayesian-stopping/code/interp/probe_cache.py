"""
Activation cache for a frozen GPTStopper checkpoint.

One forward pass per (split, sequence) batch, with hidden=True; for each
layer ℓ ∈ {0, ..., L} (L=8 so 9 layers total) we dump the residual-stream
states to a per-layer fp16 .npy file. All subsequent probe training reads
from these caches.

Disk layout under `<cache_root>/<run_name>/`:
    H_layer_{0..8}_train.npy   (N_train, n, d_emb)  fp16
    H_layer_{0..8}_val.npy     (N_val,   n, d_emb)  fp16
    H_layer_{0..8}_test.npy    (N_test,  n, d_emb)  fp16
    meta.json                  run_name, n, d_emb, L, splits, sequence seeds.

Storage (at our defaults n=256, d_emb=128, fp16):
  - per layer per split: N · 256 · 128 · 2 bytes = N · 64 KB
  - 65k sequences, 9 layers: ~37 GB total

Per-split per-layer dump uses a streaming fp32→fp16 write so peak RAM
stays at a few hundred MB.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from train.io import load_checkpoint
from interp.probe_data import PROBE_DATA_ROOT, SPLITS
from interp.probe_paths import CACHE_ROOT


@torch.no_grad()
def build_cache(
    run_name: str,
    cache_root: Path | None = None,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache_root = Path(cache_root) if cache_root is not None else (CACHE_ROOT / run_name)
    cache_root.mkdir(parents=True, exist_ok=True)

    model, head, _ = load_checkpoint(run_name, which='best')
    model = model.to(device).eval()
    L_plus_1 = model.n_layers + 1
    n = model.n
    d_emb = model.d_emb

    meta = {
        'run_name': run_name,
        'head': head,
        'n': n,
        'd_emb': d_emb,
        'n_layers_total_hidden': L_plus_1,
        'splits': {},
    }

    for split, (N_split, seed) in SPLITS.items():
        X_np = np.load(PROBE_DATA_ROOT / f'X_{split}.npy')               # (N, n) fp32
        if X_np.shape[0] != N_split:
            raise ValueError(f'{split}: expected N={N_split}, got {X_np.shape[0]}')
        # Pre-allocate per-layer memmaps for streaming writes.
        memmaps = []
        for layer in range(L_plus_1):
            path = cache_root / f'H_layer_{layer}_{split}.npy'
            mm = np.lib.format.open_memmap(
                path, mode='w+', dtype=np.float16,
                shape=(N_split, n, d_emb),
            )
            memmaps.append(mm)

        print(f'[cache] {split}: forwarding N={N_split} in batches of {batch_size}')
        t0 = time.perf_counter()
        for i in range(0, N_split, batch_size):
            j = min(i + batch_size, N_split)
            Xt = torch.as_tensor(
                X_np[i:j], dtype=torch.float32, device=device,
            )
            out = model(Xt, return_hidden=True)
            hidden = out['hidden']                                       # (B, L+1, n, d_emb)
            for layer in range(L_plus_1):
                arr = hidden[:, layer].to(torch.float16).cpu().numpy()
                memmaps[layer][i:j] = arr
            if (i // batch_size) % 50 == 0:
                rate = j / (time.perf_counter() - t0 + 1e-6)
                eta_s = (N_split - j) / rate
                print(f'  {split} {j}/{N_split}  rate={rate:.0f}/s  '
                      f'ETA={eta_s:.0f}s')
        # Flush memmaps.
        for mm in memmaps:
            mm.flush()
            del mm
        dt = time.perf_counter() - t0
        meta['splits'][split] = {
            'N': N_split, 'seed': seed, 'wall_s': dt,
        }
        print(f'  {split} done in {dt:.1f}s')

    (cache_root / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[cache] wrote {cache_root / "meta.json"}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='D_logu_act')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    build_cache(a.run, batch_size=a.batch_size, device=torch.device(a.device))
