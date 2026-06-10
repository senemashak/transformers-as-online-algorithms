"""Per-member driver for the cumulative Gaussian log-likelihood target.

Trains simple + attention-pooled probes per layer per rep
(true, perm_glob, init) on the existing hidden caches; writes outputs to
<member_dir>/probe_runs/<rep>/probe_results_loglik.json (and the attn
sibling). Existing probe outputs are NOT mutated. Idempotent per rep:
re-runs skip if both probe_results_loglik.json files exist for that rep.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from interp.prelim_v2.probe_loglik import (
    run_loglik_sweep, PROBE_TYPE_SIMPLE, PROBE_TYPE_ATTN, TARGET_NAME,
)
from interp.prelim_v2.run_member import BASE_SEED, M_TRAIN_BUDGETS


REPS = ('true', 'perm_glob', 'init')


def run_member_loglik(
    *,
    run_dir: Path,
    ensemble_idx: int,
    device: torch.device,
    smoke: bool = False,
    M_train_budgets: tuple[int, ...] | None = None,
    layers: tuple[int, ...] | None = None,
    reps: tuple[str, ...] = REPS,
) -> dict:
    member_dir = run_dir / 'members' / f'{ensemble_idx:03d}'
    if not (member_dir / 'probe_data' / 'meta.json').exists():
        raise FileNotFoundError(f'{member_dir}/probe_data missing')
    seed = BASE_SEED + ensemble_idx
    if M_train_budgets is None:
        M_train_budgets = (64, 128, 256) if smoke else M_TRAIN_BUDGETS

    summary: dict = {
        'ensemble_idx': ensemble_idx,
        'seed': seed,
        'target': TARGET_NAME,
        'probe_types': {'simple': PROBE_TYPE_SIMPLE, 'attn': PROBE_TYPE_ATTN},
        'smoke': smoke,
        'reps': list(reps),
        'M_train_budgets': list(M_train_budgets),
    }

    for rep in reps:
        # Verify the hidden_cache for this rep exists.
        cache_meta = member_dir / 'hidden_cache' / rep / 'meta.json'
        if not cache_meta.exists():
            raise FileNotFoundError(
                f'{cache_meta} missing for rep={rep}; run cache_init or the '
                f'simple+attn pipeline first.')
        simple_out = member_dir / 'probe_runs'      / rep / 'probe_results_loglik.json'
        attn_out   = member_dir / 'probe_runs_attn' / rep / 'probe_results_loglik.json'
        if simple_out.exists() and attn_out.exists():
            print(f'[loglik member {ensemble_idx}] {rep}: both files exist, skipping')
            continue
        families = []
        if not simple_out.exists(): families.append('simple')
        if not attn_out.exists():   families.append('attn')
        t0 = time.perf_counter()
        print(f'[loglik member {ensemble_idx}] {rep}: training '
              f'{",".join(families)} (9 layers x {len(M_train_budgets)} budgets)')
        out = run_loglik_sweep(
            member_dir=member_dir, ensemble_idx=ensemble_idx, rep_name=rep,
            layers=layers, M_train_budgets=M_train_budgets,
            device=device, seed=ensemble_idx, probe_families=tuple(families),
        )
        summary.setdefault('runs', {})[rep] = {
            'n_simple': len(out['simple_rows']),
            'n_attn':   len(out['attn_rows']),
            'wall_s':   time.perf_counter() - t0,
        }

    (member_dir / 'loglik_baseline_result.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--ensemble-idx', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--layers', nargs='*', type=int, default=None)
    p.add_argument('--M-train', nargs='*', type=int, default=None)
    p.add_argument('--reps', nargs='*', default=None, choices=list(REPS))
    a = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    device = (torch.device('cuda:0') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'[run_member_loglik] member {a.ensemble_idx} on {device}')

    kwargs = dict(
        run_dir=a.run_dir, ensemble_idx=a.ensemble_idx,
        device=device, smoke=a.smoke,
    )
    if a.layers is not None: kwargs['layers'] = tuple(a.layers)
    if a.M_train:           kwargs['M_train_budgets'] = tuple(a.M_train)
    if a.reps:              kwargs['reps'] = tuple(a.reps)
    run_member_loglik(**kwargs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
