"""End-to-end driver for one ensemble member of the preliminary
mechanistic-interpretability run.

Phase 1: train true + perm_glob base models (paired seeds).
Phase 2: materialize probe X/targets, cache hidden states, train probes.

Probe statistics (Phase 3) and figures are computed across members
by ``stats.py`` / ``figures.py`` after this script has run for every i.

Output layout (relative to <run_dir>):
    members/<i>/
      true/{config.json, log.jsonl, best.pt, final.pt, result.json}
      perm_glob/{config.json, log.jsonl, best.pt, final.pt, result.json}
      probe_data/{X_<split>.npy, sigma_<split>.npy, Y_<tgt>_<split>.npy, meta.json}
      hidden_cache/<rep>/{H_layer_<L>_<split>.npy, meta.json}
      probe_runs/<rep>/probe_results.json
      member_result.json

Compute on a single A6000: full-budget (step_count=1.5e5) ≈ 50 min per base
model + ~5 min caching + ~5 min probes = ~110 min total per member. Smoke
(--smoke) drops step_count to 5000 (~3 min per base model).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from interp.prelim_v2.train_paired import train_paired, DEFAULTS as TRAIN_DEFAULTS
from interp.prelim_v2.probe_data import materialize_probe_data, build_hidden_cache
from interp.prelim_v2.probe import run_probe_sweep
from interp.prelim_v2.probe_attn import run_attn_probe_sweep


# Seed schedule. Init seed for the i-th pair is BASE_SEED + i so that the
# true and perm_glob models in pair i share the same initialization and the
# same X stream; only the labels differ.
BASE_SEED = 2_000_000

# Per-spec; the budget sweep ALSO covers the full-budget (1024) cell. Stats
# in Phase 3 read the M_train=1024 entries.
M_TRAIN_BUDGETS = (64, 128, 256, 512, 1024)


def member_dir_for(run_dir: Path, idx: int) -> Path:
    return run_dir / 'members' / f'{idx:03d}'


def run_member(
    *,
    run_dir: Path,
    ensemble_idx: int,
    device: torch.device,
    smoke: bool = False,
    skip_phases: tuple[str, ...] = (),
) -> dict:
    """Run phase 1 + phase 2 for one ensemble member."""
    member_dir = member_dir_for(run_dir, ensemble_idx)
    member_dir.mkdir(parents=True, exist_ok=True)

    seed = BASE_SEED + ensemble_idx
    step_count = 5000 if smoke else None
    val_every = 500 if smoke else None
    summary: dict = {
        'ensemble_idx': ensemble_idx,
        'seed': seed,
        'smoke': smoke,
        'step_count_override': step_count,
    }

    # ---- Phase 1: paired training (true, then perm_glob with same init/stream)
    if 'train' not in skip_phases:
        for label_perm in ('true', 'perm_glob'):
            out_dir = member_dir / label_perm
            if out_dir.exists() and (out_dir / 'best.pt').exists():
                print(f'[member {ensemble_idx}] {label_perm} already trained, skipping')
            else:
                if out_dir.exists():
                    raise FileExistsError(
                        f'{out_dir} exists but has no best.pt; refusing to clobber'
                    )
                t0 = time.perf_counter()
                print(f'[member {ensemble_idx}] training {label_perm} (seed={seed}, '
                      f'step_count={step_count or TRAIN_DEFAULTS["step_count"]})')
                res = train_paired(
                    seed=seed, label_perm=label_perm,
                    output_dir=out_dir, device=device,
                    step_count=step_count, val_every=val_every,
                )
                summary.setdefault('train', {})[label_perm] = res
                print(f'[member {ensemble_idx}] {label_perm} done in '
                      f'{time.perf_counter() - t0:.0f}s, best_val={res["best_val_loss"]:.4f}')

    # ---- Phase 2a: probe X + targets (shared across reps)
    if 'probe_data' not in skip_phases:
        probe_data_dir = member_dir / 'probe_data'
        if probe_data_dir.exists():
            print(f'[member {ensemble_idx}] probe_data already exists, skipping')
        else:
            splits = {'train': 1024, 'val': 512, 'test': 10_000} if not smoke else {
                'train': 256, 'val': 128, 'test': 512,
            }
            print(f'[member {ensemble_idx}] materializing probe data {splits}')
            t0 = time.perf_counter()
            meta = materialize_probe_data(member_dir, ensemble_idx, splits=splits)
            print(f'[member {ensemble_idx}] probe_data done in {time.perf_counter() - t0:.0f}s')
            summary['probe_data'] = meta

    # ---- Phase 2b: hidden cache per rep
    if 'hidden_cache' not in skip_phases:
        for rep in ('true', 'perm_glob'):
            cache_dir = member_dir / 'hidden_cache' / rep
            if cache_dir.exists():
                print(f'[member {ensemble_idx}] hidden_cache/{rep} exists, skipping')
                continue
            ckpt = member_dir / rep / 'best.pt'
            if not ckpt.exists():
                raise FileNotFoundError(f'{ckpt} not found; train phase missing?')
            print(f'[member {ensemble_idx}] caching hidden states for {rep}')
            t0 = time.perf_counter()
            meta = build_hidden_cache(
                ckpt_path=ckpt, member_dir=member_dir,
                rep_name=rep, device=device,
            )
            summary.setdefault('hidden_cache', {})[rep] = meta
            print(f'[member {ensemble_idx}] {rep} cache done in '
                  f'{time.perf_counter() - t0:.0f}s')

    # ---- Phase 2c: probes per rep
    if 'probes' not in skip_phases:
        budgets = M_TRAIN_BUDGETS if not smoke else (64, 128, 256)
        for rep in ('true', 'perm_glob'):
            probe_out = member_dir / 'probe_runs' / rep / 'probe_results.json'
            if probe_out.exists():
                print(f'[member {ensemble_idx}] probes for {rep} already done, skipping')
                continue
            print(f'[member {ensemble_idx}] training probes for {rep} '
                  f'(targets x layers x budgets = 2 x 9 x {len(budgets)})')
            t0 = time.perf_counter()
            out = run_probe_sweep(
                member_dir=member_dir, rep_name=rep,
                M_train_budgets=budgets, device=device, seed=ensemble_idx,
            )
            summary.setdefault('probes', {})[rep] = {
                'n_rows': len(out['rows']),
                'wall_s': time.perf_counter() - t0,
            }

    # ---- Phase 2d: attn probes per rep (timestep-shared attention-pooled)
    if 'attn_probes' not in skip_phases:
        budgets = M_TRAIN_BUDGETS if not smoke else (64, 128, 256)
        for rep in ('true', 'perm_glob'):
            probe_out = member_dir / 'probe_runs_attn' / rep / 'probe_results.json'
            if probe_out.exists():
                print(f'[member {ensemble_idx}] attn probes for {rep} already done, skipping')
                continue
            print(f'[member {ensemble_idx}] training attn probes for {rep} '
                  f'(targets x layers x budgets = 2 x 9 x {len(budgets)})')
            t0 = time.perf_counter()
            out = run_attn_probe_sweep(
                src_member_dir=member_dir, dst_member_dir=member_dir,
                rep_name=rep,
                M_train_budgets=budgets, device=device, seed=ensemble_idx,
            )
            summary.setdefault('probes_attn', {})[rep] = {
                'n_rows': len(out['rows']),
                'wall_s': time.perf_counter() - t0,
            }

    (member_dir / 'member_result.json').write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--ensemble-idx', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='Reduce step counts + probe splits for a quick test.')
    p.add_argument('--skip', nargs='*', default=(),
                   choices=['train', 'probe_data', 'hidden_cache',
                            'probes', 'attn_probes'])
    a = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(a.gpu))
    device = (torch.device('cuda:0') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'[run_member] member {a.ensemble_idx} on {device} '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    run_member(
        run_dir=a.run_dir, ensemble_idx=a.ensemble_idx,
        device=device, smoke=a.smoke, skip_phases=tuple(a.skip),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
