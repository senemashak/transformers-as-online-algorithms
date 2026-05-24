"""
Materialize a random-init GPTStopper with the same architecture as
`D_logu_act` and save it as a checkpoint under `checkpoints/<run>/best.pt`,
so the existing `train.io.load_checkpoint` API can load it unchanged for
the probing infrastructure.

No training — weights come from PyTorch's default `nn.Linear` and
`nn.LayerNorm` initialization. The seed is fixed so the random-init
baseline is reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from model.transformer import GPTStopper
from train.configs import CHECKPOINT_ROOT
from train.io import build_payload, save_checkpoint


import argparse                                       # noqa: E402 — for CLI override

DEFAULT_RUN_NAME = 'D_logu_act_random_init'
DEFAULT_SEED = 31415                  # disjoint from every other seed used in v3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=DEFAULT_RUN_NAME)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--distribution', default='D_logu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_kwargs = {'n': 256, 'd_emb': 128, 'n_layers': 8, 'n_heads': 4}
    model = GPTStopper(**model_kwargs)
    # Move to CPU explicitly; load_checkpoint will move it where it needs to go.
    model = model.cpu().eval()

    ckpt_dir = CHECKPOINT_ROOT / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        'run_name': args.run,
        'distribution': args.distribution,
        'supervision': 'act',
        'label_source': 'random_init',
        'stage': 'random_init',
        'step_count': 0,
        'seed': args.seed,
        'note': 'Fresh random-init weights; no training. Used as the '
                'strict noise-floor baseline for the probing sweep.',
    }
    (ckpt_dir / 'config.json').write_text(json.dumps(config_dict, indent=2))

    payload = build_payload(
        model=model, optimizer=None, scheduler=None,
        step=0, val_loss=float('nan'),
        is_best=True, is_periodic=False,
        trained_head='act',
        model_config=model_kwargs,
        config_dict=config_dict,
        rng_states={'torch': torch.get_rng_state(), 'numpy': None},
    )
    save_checkpoint(ckpt_dir / 'best.pt', payload, overwrite=True)
    print(f'[random-init] wrote {ckpt_dir / "best.pt"}')
    print(f'[random-init] seed={args.seed}, params={model.num_params()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
