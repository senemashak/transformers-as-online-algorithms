"""
Step 4 sweep — single-run entry point.

CLI used by run_sweep.sh:
    python -m train.sweep --run D_1_cv --gpu 0 --stage pilot
    python -m train.sweep --run D_disc_cv --gpu 4 --stage full

The bash driver loops over (run, gpu) tuples per wave and launches each
as a subprocess with CUDA_VISIBLE_DEVICES set to the physical GPU id.
Inside each subprocess only one GPU is visible — we always use cuda:0.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from train.configs import get_run_config, parse_run_name      # noqa: E402
from train.loop import train_one                              # noqa: E402


def _install_sigint_handler():
    def _handler(signum, frame):
        sys.stderr.write(f'\n[sweep] received signal {signum}, flushing and exiting...\n')
        sys.stderr.flush()
        # Let Python's default SIGINT handler take over after one shot
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True,
                   help='Base run name like D_disc_cv (no _pilot suffix).')
    p.add_argument('--gpu', type=int, required=True,
                   help='Physical GPU id; sets CUDA_VISIBLE_DEVICES inside the parent.')
    p.add_argument('--stage', choices=['pilot', 'full'], required=True)
    args = p.parse_args()

    _install_sigint_handler()

    # Validate run name parses; this also fails fast on typos.
    parse_run_name(args.run)

    # Inside this process we always use cuda:0. The shell sets
    # CUDA_VISIBLE_DEVICES to the requested physical GPU before launch.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not torch.cuda.is_available():
        sys.exit('FATAL: CUDA not available')
    if torch.cuda.device_count() < 1:
        sys.exit('FATAL: no CUDA device visible to this process')
    device = torch.device('cuda:0')
    print(f'[sweep] run={args.run} stage={args.stage} '
          f'physical_gpu={args.gpu} CUDA_VISIBLE_DEVICES={visible} '
          f'device={device}', flush=True)

    cfg = get_run_config(args.run, args.stage)
    print(f'[sweep] run_name={cfg.run_name} steps={cfg.step_count} '
          f'val_every={cfg.val_every} periodic_every={cfg.periodic_every}',
          flush=True)

    t0 = time.perf_counter()
    metadata = train_one(cfg, device)
    elapsed = time.perf_counter() - t0
    print(f'[sweep] DONE run={cfg.run_name} wall={elapsed:.1f}s '
          f'best_val_loss={metadata["best_val_loss"]:.4e} '
          f'best_step={metadata["best_step"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
