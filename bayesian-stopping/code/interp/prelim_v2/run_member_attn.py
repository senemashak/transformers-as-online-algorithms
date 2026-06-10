"""Per-member driver: train the timestep-shared attention-pooled probe on
already-cached hidden states from an existing simple-probe run.

Treats the source run directory as read-only: never writes under it, never
modifies probe_data/, hidden_cache/, true/, perm_glob/, probe_runs/.

Outputs are written to <attn_run_dir>/members/<i>/probe_runs/<rep>/probe_results.json
plus a tiny <attn_run_dir>/members/<i>/member_result.json log.

Compute estimate (single A6000): ~10-15 min per member (2 reps x 2 targets
x 9 layers x 5 budgets = 180 cells). No base-model training, no
hidden-state caching: probe training only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from interp.prelim_v2.probe_attn import run_attn_probe_sweep, PROBE_TYPE


M_TRAIN_BUDGETS = (64, 128, 256, 512, 1024)


def run_member_attn(
    *,
    src_run_dir: Path,
    attn_run_dir: Path,
    ensemble_idx: int,
    device: torch.device,
    smoke: bool = False,
    targets: tuple[str, ...] = ('sigma_mle_sqrt', 'C_star'),
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] | None = None,
    reps: tuple[str, ...] = ('true', 'perm_glob'),
) -> dict:
    src_member = src_run_dir  / 'members' / f'{ensemble_idx:03d}'
    dst_member = attn_run_dir / 'members' / f'{ensemble_idx:03d}'

    # ---- preflight: source artifacts must exist and be read-only friendly
    required = [
        src_member / 'probe_data' / 'meta.json',
        src_member / 'hidden_cache' / 'true'      / 'meta.json',
        src_member / 'hidden_cache' / 'perm_glob' / 'meta.json',
        src_member / 'true'      / 'best.pt',
        src_member / 'perm_glob' / 'best.pt',
        src_member / 'probe_runs' / 'true'      / 'probe_results.json',
        src_member / 'probe_runs' / 'perm_glob' / 'probe_results.json',
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f'expected source artifact missing: {p}')

    dst_member.mkdir(parents=True, exist_ok=True)

    if M_train_budgets is None:
        M_train_budgets = (64, 128, 256) if smoke else M_TRAIN_BUDGETS

    summary: dict = {
        'ensemble_idx': ensemble_idx,
        'smoke': smoke,
        'probe_type': PROBE_TYPE,
        'src_member_dir': str(src_member),
        'attn_member_dir': str(dst_member),
        'targets': list(targets),
        'layers': list(layers) if layers is not None else None,
        'M_train_budgets': list(M_train_budgets),
        'reps': list(reps),
    }

    for rep in reps:
        out_path = dst_member / 'probe_runs' / rep / 'probe_results.json'
        if out_path.exists():
            print(f'[member {ensemble_idx}] attn probes for {rep} '
                  f'already at {out_path}, skipping')
            continue
        print(f'[member {ensemble_idx}] training attn probes for {rep} '
              f'(targets x layers x budgets = {len(targets)} x '
              f'{len(layers) if layers else 9} x {len(M_train_budgets)})')
        t0 = time.perf_counter()
        out = run_attn_probe_sweep(
            src_member_dir=src_member,
            dst_member_dir=dst_member,
            rep_name=rep,
            targets=targets,
            layers=layers,
            M_train_budgets=M_train_budgets,
            device=device,
            seed=ensemble_idx,
        )
        summary.setdefault('probes', {})[rep] = {
            'n_rows': len(out['rows']),
            'wall_s': time.perf_counter() - t0,
        }

    (dst_member / 'member_result.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--src-run-dir', type=Path, required=True,
                   help='Existing simple-probe run dir (read-only).')
    p.add_argument('--attn-run-dir', type=Path, required=True,
                   help='New output run dir for the attention-pooled sweep.')
    p.add_argument('--ensemble-idx', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--targets', nargs='*', default=None)
    p.add_argument('--layers', nargs='*', type=int, default=None)
    p.add_argument('--M-train', nargs='*', type=int, default=None,
                   help='Override M_train budget grid; default is 64..1024.')
    p.add_argument('--reps', nargs='*', default=None,
                   choices=['true', 'perm_glob'])
    a = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    device = (torch.device('cuda:0') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'[run_member_attn] member {a.ensemble_idx} on {device} '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    kwargs = dict(
        src_run_dir=a.src_run_dir, attn_run_dir=a.attn_run_dir,
        ensemble_idx=a.ensemble_idx, device=device, smoke=a.smoke,
    )
    if a.targets:           kwargs['targets'] = tuple(a.targets)
    if a.layers is not None: kwargs['layers'] = tuple(a.layers)
    if a.M_train:           kwargs['M_train_budgets'] = tuple(a.M_train)
    if a.reps:              kwargs['reps'] = tuple(a.reps)
    run_member_attn(**kwargs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
