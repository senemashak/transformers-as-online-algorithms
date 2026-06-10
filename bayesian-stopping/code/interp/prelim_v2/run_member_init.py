"""Per-member driver for the untrained-init probing baseline.

For each member: rebuild the seed-deterministic init model, cache hidden
states under hidden_cache/init/, then train the existing simple + attn
probes on those states for the primary targets (sigma_mle_sqrt, C_star).
Outputs land in members/<i>/probe_runs/init/ and probe_runs_attn/init/,
alongside the existing true/ and perm_glob/ outputs.

Trained checkpoints are NOT touched. Existing init outputs are skipped (idempotent).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from interp.prelim_v2.cache_init_hidden import build_init_hidden_cache, REP_NAME
from interp.prelim_v2.probe import run_probe_sweep
from interp.prelim_v2.probe_attn import run_attn_probe_sweep
from interp.prelim_v2.run_member import BASE_SEED, M_TRAIN_BUDGETS


PRIMARY_TARGETS = ('sigma_mle_sqrt', 'C_star')


def run_member_init(
    *,
    run_dir: Path,
    ensemble_idx: int,
    device: torch.device,
    smoke: bool = False,
    M_train_budgets: tuple[int, ...] | None = None,
    layers: tuple[int, ...] | None = None,
) -> dict:
    member_dir = run_dir / 'members' / f'{ensemble_idx:03d}'
    if not (member_dir / 'probe_data' / 'meta.json').exists():
        raise FileNotFoundError(
            f'{member_dir}/probe_data missing; this member must already have '
            f'probe_data materialized by the simple/attn pipeline.'
        )

    seed = BASE_SEED + ensemble_idx
    if M_train_budgets is None:
        M_train_budgets = (64, 128, 256) if smoke else M_TRAIN_BUDGETS

    summary: dict = {
        'ensemble_idx': ensemble_idx,
        'seed': seed,
        'rep': REP_NAME,
        'smoke': smoke,
        'targets': list(PRIMARY_TARGETS),
        'M_train_budgets': list(M_train_budgets),
    }

    # ---- Phase A: init hidden cache (idempotent) ----
    t0 = time.perf_counter()
    cache_meta = build_init_hidden_cache(
        member_dir=member_dir, seed=seed, device=device,
    )
    summary['cache_wall_s'] = time.perf_counter() - t0
    summary['cache_meta'] = cache_meta

    # ---- Phase B: simple probes for init rep ----
    simple_out = member_dir / 'probe_runs' / REP_NAME / 'probe_results.json'
    if simple_out.exists():
        print(f'[init member {ensemble_idx}] simple probes exist, skipping')
    else:
        t0 = time.perf_counter()
        print(f'[init member {ensemble_idx}] simple probes '
              f'({len(PRIMARY_TARGETS)} targets x 9 layers x {len(M_train_budgets)} budgets)')
        out = run_probe_sweep(
            member_dir=member_dir, rep_name=REP_NAME,
            targets=PRIMARY_TARGETS,
            layers=layers,
            M_train_budgets=M_train_budgets,
            device=device, seed=ensemble_idx,
        )
        summary['simple_probes'] = {
            'n_rows': len(out['rows']),
            'wall_s': time.perf_counter() - t0,
        }

    # ---- Phase C: attn probes for init rep ----
    attn_out = member_dir / 'probe_runs_attn' / REP_NAME / 'probe_results.json'
    if attn_out.exists():
        print(f'[init member {ensemble_idx}] attn probes exist, skipping')
    else:
        t0 = time.perf_counter()
        print(f'[init member {ensemble_idx}] attn probes '
              f'({len(PRIMARY_TARGETS)} targets x 9 layers x {len(M_train_budgets)} budgets)')
        out = run_attn_probe_sweep(
            src_member_dir=member_dir, dst_member_dir=member_dir,
            rep_name=REP_NAME,
            targets=PRIMARY_TARGETS,
            layers=layers,
            M_train_budgets=M_train_budgets,
            device=device, seed=ensemble_idx,
        )
        summary['attn_probes'] = {
            'n_rows': len(out['rows']),
            'wall_s': time.perf_counter() - t0,
        }

    (member_dir / f'init_baseline_result.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True,
                   help='Existing unified N=100 run dir.')
    p.add_argument('--ensemble-idx', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='Reduce M_train budgets to (64,128,256) for fast smoke.')
    p.add_argument('--layers', nargs='*', type=int, default=None)
    p.add_argument('--M-train', nargs='*', type=int, default=None)
    a = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    device = (torch.device('cuda:0') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'[run_member_init] member {a.ensemble_idx} on {device}')

    kwargs = dict(
        run_dir=a.run_dir, ensemble_idx=a.ensemble_idx,
        device=device, smoke=a.smoke,
    )
    if a.layers is not None: kwargs['layers'] = tuple(a.layers)
    if a.M_train:           kwargs['M_train_budgets'] = tuple(a.M_train)
    run_member_init(**kwargs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
