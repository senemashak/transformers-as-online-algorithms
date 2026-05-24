"""
Step 4 checkpoint I/O — the single reload entry point used by Steps 5+.

Conventions:
    - Each run lives under v3/checkpoints/<run_name>/.
    - best.pt and final.pt overwrite freely. step_<N>k.pt is periodic
      and refuses to overwrite (FileExistsError) — the spec is explicit.
    - config.json is written once at training start (human-readable).
      metadata.json is written at training end with timing + env info.
    - Each .pt payload contains state_dict (both heads), optimizer state,
      scheduler state, step, val_loss, RNG state, model_config, and the
      canonical `trained_head` field. Filenames are hints; `trained_head`
      is authoritative.

Usage:

    from train.io import load_checkpoint
    model, head_name, metadata = load_checkpoint('D_disc_cv', which='best')
    # model is on CPU in eval() mode; caller .cuda() if needed
    # head_name is 'cv' or 'act' — Step 5+ should query only this head
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from model.transformer import GPTStopper
from train.configs import CHECKPOINT_ROOT


def load_checkpoint(
    run_name: str,
    which: str = 'best',
    ckpt_root: Path | None = None,
) -> Tuple[GPTStopper, str, dict]:
    """Reload a saved run.

    Args:
        run_name: 'D_disc_cv', 'D_1_cv_pilot', etc.
        which: 'best', 'final', or a periodic step like 'step_50k'.
        ckpt_root: defaults to v3/checkpoints/.

    Returns:
        model: GPTStopper instance with both heads, state_dict loaded,
               on CPU, in eval() mode.
        head_name: 'cv' or 'act' — the trained head. Steps 5+ should
                   only query this head; the other head's parameters are
                   random init and meaningless.
        metadata: full config + training stats, including the on-disk
                  config.json and metadata.json contents if present.
    """
    root = Path(ckpt_root) if ckpt_root is not None else CHECKPOINT_ROOT
    ckpt_dir = root / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'No checkpoint dir at {ckpt_dir}')
    ckpt_path = ckpt_dir / f'{which}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'No checkpoint at {ckpt_path}')

    config_path = ckpt_dir / 'config.json'
    config_dict = json.loads(config_path.read_text()) if config_path.exists() else {}
    metadata_path = ckpt_dir / 'metadata.json'
    metadata_extra = (json.loads(metadata_path.read_text())
                      if metadata_path.exists() else {})

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_kwargs = ckpt.get(
        'model_config',
        {'n': 256, 'd_emb': 128, 'n_layers': 8, 'n_heads': 4},
    )
    model = GPTStopper(**model_kwargs)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    head_name = ckpt['trained_head']
    metadata = {
        'run_name': run_name,
        'which': which,
        'step': ckpt['step'],
        'val_loss': ckpt['val_loss'],
        'is_best': ckpt.get('is_best'),
        'is_periodic': ckpt.get('is_periodic'),
        'config': config_dict,
        'environment': metadata_extra,
    }
    return model, head_name, metadata


def save_checkpoint(
    path: Path,
    payload: dict,
    overwrite: bool,
) -> None:
    """Save a checkpoint payload. Periodic checkpoints (overwrite=False)
    refuse to overwrite an existing file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'Refusing to overwrite {path}. '
            'Move or delete first; checkpoint is read-only by convention.'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_payload(
    model: GPTStopper,
    optimizer,
    scheduler,
    step: int,
    val_loss: float,
    is_best: bool,
    is_periodic: bool,
    trained_head: str,
    model_config: dict,
    config_dict: dict,
    rng_states: dict,
) -> dict:
    return {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'step': step,
        'val_loss': val_loss,
        'is_best': is_best,
        'is_periodic': is_periodic,
        'trained_head': trained_head,
        'model_config': model_config,
        'config': config_dict,
        'rng_state_torch': rng_states.get('torch'),
        'rng_state_numpy': rng_states.get('numpy'),
    }
