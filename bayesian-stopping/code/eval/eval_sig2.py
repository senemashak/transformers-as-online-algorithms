"""Evaluate the two σ² ablation models (D_disc_cv_sig2, D_logu_cv_sig2)
on the same eval pipeline as run_eval.py, writing sidecar artifacts to
results/phase5/ that DO NOT overwrite the existing σ¹ outputs.

Sidecars produced:
    payoff_matrix_sig2.csv / .json     — R/R* on the three test regimes.
    agreement_tensor_sig2.npz           — 2×3×7 agreement tensor.
    trajectories_sig2.npz               — per-(run, regime) mean/std.
    persigma_results_sig2.json          — same schema as persigma_results.json.

The σ¹ test caches (seed-43) and per-σ caches (seed-44) are reused; oracle
tables and baselines are built identically.
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
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import STATIC_DISTRIBUTIONS
from eval.agreement import per_step_agreement
from eval.payoff import expected_payoff, normalized_payoff
from eval.per_sigma_payoff import per_sigma_breakdown
from eval.policies import model_action
from eval.run_eval import (
    BASELINE_NAMES, TEST_REGIMES,
    _build_baselines_for_regime, _build_random_oracle_table,
    _build_static_oracle_table, _eta, _load_persigma_test, _load_test,
)
from eval.trajectory import trajectory_mean_std
from data.distributions import static_sigma
from train.io import load_checkpoint


RESULTS_PHASE5 = V3_ROOT / 'results' / 'phase5'
SIG2_RUNS = ('D_disc_cv_sig2', 'D_logu_cv_sig2')


def _sig2_distribution(run_name: str) -> str:
    """'D_disc_cv_sig2' -> 'D_disc' (strip _sig2 then drop the supervision tail)."""
    base = run_name.removesuffix('_sig2')
    return '_'.join(base.split('_')[:-1])


def _select_oracle(run_name, baselines):
    dist = _sig2_distribution(run_name)
    if dist in STATIC_DISTRIBUTIONS:
        return baselines['oracle_static']
    return baselines['oracle_disc'] if dist == 'D_disc' else baselines['oracle_logu']


def _select_map(run_name, baselines):
    dist = _sig2_distribution(run_name)
    return baselines['MAP_sigma_logu'] if dist == 'D_logu' else baselines['MAP_sigma_disc']


def main() -> int:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_sig2] device={device}')

    # Reuse the σ¹ pipeline machinery: same test caches, oracle tables, baselines.
    eta = _eta()
    test_caches = {r: _load_test(r) for r in TEST_REGIMES}
    static_tables = {r: _build_static_oracle_table(static_sigma(r)) for r in TEST_REGIMES}
    random_tables = {
        'D_disc': _build_random_oracle_table('D_disc'),
        'D_logu': _build_random_oracle_table('D_logu'),
    }
    baselines_by_regime = {
        regime: _build_baselines_for_regime(
            regime, test_caches[regime][0], eta, static_tables, random_tables,
        )
        for regime in TEST_REGIMES
    }
    R_star_per_regime = {
        regime: expected_payoff(
            baselines_by_regime[regime]['oracle_static']['action'],
            test_caches[regime][0],
        )
        for regime in TEST_REGIMES
    }
    print(f'[eval_sig2] R*: {R_star_per_regime}')

    # Per-cell loop.
    payoff_matrix_R: dict = {}
    payoff_matrix: dict = {}
    agreement_tensor = np.zeros((len(SIG2_RUNS), len(TEST_REGIMES), len(BASELINE_NAMES)))
    trajectories: dict = {}

    for ri, run_name in enumerate(SIG2_RUNS):
        t0 = time.perf_counter()
        model, head, _ = load_checkpoint(run_name, which='best')
        model = model.to(device).eval()
        assert head == 'cv'
        print(f'\n[eval_sig2] === {run_name} (head={head}) ===')

        payoff_matrix[run_name] = {}
        payoff_matrix_R[run_name] = {}
        trajectories[run_name] = {}

        for ci, regime in enumerate(TEST_REGIMES):
            X, _, _ = test_caches[regime]
            bl = baselines_by_regime[regime]
            m_act, m_thr = model_action(model, X, head, device)
            R = expected_payoff(m_act, X)
            R_star = R_star_per_regime[regime]
            R_over = normalized_payoff(R, R_star)
            payoff_matrix[run_name][regime] = R_over
            payoff_matrix_R[run_name][regime] = R

            oracle = _select_oracle(run_name, bl)
            map_ = _select_map(run_name, bl)
            ag = {
                'oracle': per_step_agreement(m_act, oracle['action']),
                'plug_in': per_step_agreement(m_act, bl['plug_in']['action']),
                'prior_only': per_step_agreement(m_act, bl['prior_only']['action']),
                'MAP_sigma': per_step_agreement(m_act, map_['action']),
                'MLE_sigma': per_step_agreement(m_act, bl['MLE_sigma']['action']),
                'secretary': per_step_agreement(m_act, bl['secretary']['action']),
                'myopic': per_step_agreement(m_act, bl['myopic']['action']),
            }
            for bi, bn in enumerate(BASELINE_NAMES):
                agreement_tensor[ri, ci, bi] = ag[bn]

            traj = trajectory_mean_std(m_thr)
            trajectories[run_name][regime] = traj

            print(f'[eval_sig2]   {regime}: R/R*={R_over:.4f}, '
                  f'oracle_agree={ag["oracle"]:.4f}')

        # Per-σ breakdown using the matching random-ADP oracle as denominator.
        dist = _sig2_distribution(run_name)
        X_ps, sigma_i_ps, _ = _load_persigma_test(dist)
        rows = per_sigma_breakdown(
            model, head, X_ps, sigma_i_ps, dist, device, random_tables[dist],
        )
        # Stash in the persigma JSON below.
        persigma_results_local = locals().setdefault('persigma_results', {})
        persigma_results_local[run_name] = rows
        print(f'[eval_sig2]   per-σ: ' +
              ', '.join(f'{r["name"]}={r["R_over_R_star"]:.3f}' for r in rows))

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'[eval_sig2]   done in {time.perf_counter()-t0:.1f}s')

    # Write sidecar artifacts.
    np.savez_compressed(
        RESULTS_PHASE5 / 'agreement_tensor_sig2.npz',
        agreement=agreement_tensor,
        run_names=np.array(list(SIG2_RUNS)),
        test_regimes=np.array(TEST_REGIMES),
        baseline_names=np.array(BASELINE_NAMES),
    )
    print(f'wrote {RESULTS_PHASE5 / "agreement_tensor_sig2.npz"}')

    (RESULTS_PHASE5 / 'payoff_matrix_raw_sig2.json').write_text(json.dumps({
        'R_over_R_star': payoff_matrix,
        'R': payoff_matrix_R,
        'R_star': R_star_per_regime,
    }, indent=2))
    print(f'wrote {RESULTS_PHASE5 / "payoff_matrix_raw_sig2.json"}')

    # CSV in the same shape as payoff_matrix.csv.
    import csv as _csv
    with (RESULTS_PHASE5 / 'payoff_matrix_sig2.csv').open('w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['run'] + list(TEST_REGIMES))
        for run_name in SIG2_RUNS:
            w.writerow([run_name] + [f'{payoff_matrix[run_name][r]:.4f}'
                                     for r in TEST_REGIMES])
    print(f'wrote {RESULTS_PHASE5 / "payoff_matrix_sig2.csv"}')

    # Trajectories: same key schema as the σ¹ npz.
    traj_payload = {}
    for run_name, regimes in trajectories.items():
        for regime, t in regimes.items():
            traj_payload[f'{run_name}__{regime}__mean'] = t['mean']
            traj_payload[f'{run_name}__{regime}__std'] = t['std']
    np.savez_compressed(RESULTS_PHASE5 / 'trajectories_sig2.npz', **traj_payload)
    print(f'wrote {RESULTS_PHASE5 / "trajectories_sig2.npz"}')

    (RESULTS_PHASE5 / 'persigma_results_sig2.json').write_text(
        json.dumps(persigma_results_local, indent=2),
    )
    print(f'wrote {RESULTS_PHASE5 / "persigma_results_sig2.json"}')

    print('\n[eval_sig2] DONE.')
    print('payoff matrix:')
    for run_name in SIG2_RUNS:
        row = payoff_matrix[run_name]
        print(f'  {run_name:20s}  ' + '  '.join(f'{r}={row[r]:.4f}' for r in TEST_REGIMES))
    return 0


if __name__ == '__main__':
    sys.exit(main())
