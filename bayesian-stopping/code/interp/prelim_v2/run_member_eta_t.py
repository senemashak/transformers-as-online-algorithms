"""Per-member driver for the eta_t raw-MSE probing sweep.

Reads hidden_cache from a source run dir; writes simple + attention-pooled
probe outputs under a separate run dir. Existing outputs are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from interp.prelim_v2.probe_eta_t import (
    run_eta_t_sweep, PROBE_TYPE_SIMPLE, PROBE_TYPE_ATTN, TARGET_NAME,
)


M_TRAIN_BUDGETS = (64, 128, 256, 512, 1024)


def run_member_eta_t(*,
    src_run_dir: Path,
    dst_run_dir: Path,
    ensemble_idx: int,
    device: torch.device,
    smoke: bool = False,
    layers: tuple[int, ...] | None = None,
    M_train_budgets: tuple[int, ...] | None = None,
    reps: tuple[str, ...] = ('true', 'perm_glob'),
    probe_families: tuple[str, ...] = ('simple', 'attn'),
) -> dict:
    src_member = src_run_dir / 'members' / f'{ensemble_idx:03d}'
    dst_member = dst_run_dir / 'members' / f'{ensemble_idx:03d}'

    required = [
        src_member / 'hidden_cache' / 'true'      / 'meta.json',
        src_member / 'hidden_cache' / 'perm_glob' / 'meta.json',
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
        'target': TARGET_NAME,
        'probe_types': {'simple': PROBE_TYPE_SIMPLE, 'attn': PROBE_TYPE_ATTN},
        'src_member_dir': str(src_member),
        'dst_member_dir': str(dst_member),
        'layers': list(layers) if layers is not None else None,
        'M_train_budgets': list(M_train_budgets),
        'reps': list(reps),
        'probe_families': list(probe_families),
    }

    for rep in reps:
        simple_out = dst_member / 'probe_runs'      / rep / 'probe_results.json'
        attn_out   = dst_member / 'probe_runs_attn' / rep / 'probe_results.json'
        need_simple = 'simple' in probe_families and not simple_out.exists()
        need_attn   = 'attn'   in probe_families and not attn_out.exists()
        if not (need_simple or need_attn):
            print(f'[eta_t member {ensemble_idx}] {rep}: outputs exist, skipping')
            continue
        families = []
        if need_simple: families.append('simple')
        if need_attn:   families.append('attn')
        t0 = time.perf_counter()
        out = run_eta_t_sweep(
            src_member_dir=src_member,
            dst_member_dir=dst_member,
            rep_name=rep,
            layers=layers,
            M_train_budgets=M_train_budgets,
            device=device,
            seed=ensemble_idx,
            probe_families=tuple(families),
        )
        summary.setdefault('runs', {})[rep] = {
            'n_simple': len(out['simple_rows']),
            'n_attn':   len(out['attn_rows']),
            'wall_s':   time.perf_counter() - t0,
        }

    (dst_member / 'member_result.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--src-run-dir', type=Path, required=True)
    p.add_argument('--dst-run-dir', type=Path, required=True)
    p.add_argument('--ensemble-idx', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--layers', nargs='*', type=int, default=None)
    p.add_argument('--M-train', nargs='*', type=int, default=None)
    p.add_argument('--reps', nargs='*', default=None, choices=['true', 'perm_glob'])
    p.add_argument('--probe-families', nargs='*', default=None,
                   choices=['simple', 'attn'])
    a = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    device = (torch.device('cuda:0') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'[run_member_eta_t] member {a.ensemble_idx} on {device}')

    kwargs = dict(
        src_run_dir=a.src_run_dir, dst_run_dir=a.dst_run_dir,
        ensemble_idx=a.ensemble_idx, device=device, smoke=a.smoke,
    )
    if a.layers is not None:  kwargs['layers'] = tuple(a.layers)
    if a.M_train:             kwargs['M_train_budgets'] = tuple(a.M_train)
    if a.reps:                kwargs['reps'] = tuple(a.reps)
    if a.probe_families:      kwargs['probe_families'] = tuple(a.probe_families)
    run_member_eta_t(**kwargs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
