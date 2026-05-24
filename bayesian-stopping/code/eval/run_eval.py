"""
V3 Step 5 driver. Runs all 30 (trained model × test regime) cells plus the
per-σ payoff figure on the four random-variance runs. Uses:

  * seed-43 per-regime test caches (`v3/data/cache/D_{1,2,3}_test.npz`)
    for the payoff matrix, agreement tensor, threshold trajectories.
  * seed-44 per-σ test caches (`v3/data/cache/D_{disc,logu}_persigma_test.npz`)
    for the per-σ payoff breakdown.
  * Step 2 oracle tables in `v3/oracle/tables/` for per-regime and
    per-training-distribution oracle policies.
  * Step 4 checkpoints in `v3/checkpoints/<run_name>/best.pt`, loaded via
    `train.io.load_checkpoint`.

Per-cell artifacts go to `v3/results/phase5/<run_name>/<test_regime>.json`.
Sweep-level artifacts (payoff matrix CSV/TEX/PNG, agreement tensor,
trajectories, per-σ figures, summary) are produced by `eval.render` after
the cells run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent                          # v3/code/
V3_ROOT = CODE_ROOT.parent                        # v3/
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import (
    ALL_DISTRIBUTIONS,
    MU_0,
    RANDOM_DISTRIBUTIONS,
    SIGMA_REGIMES,
    STATIC_DISTRIBUTIONS,
    TAU0_2,
    static_sigma,
)
from data.streaming import CACHE_DIR_DEFAULT, cache_path, load_cache
from eval.agreement import per_step_agreement
from eval.payoff import expected_payoff, normalized_payoff
from eval.per_sigma_payoff import per_sigma_breakdown
from eval.policies import (
    map_sigma_plugin,
    mle_sigma_plugin,
    model_action,
    myopic,
    plug_in,
    prior_only,
    random_oracle,
    secretary,
    static_oracle,
    threshold_to_action,
)
from eval.trajectory import trajectory_mean_std
from oracle.conjugate import compute_eta
from oracle.random_adp import load_table as load_random_table
from oracle.random_adp_torch import make_sigma_grid as make_logu_grid
from oracle.static_adp import solve_adp
from train.configs import CHECKPOINT_ROOT, ORACLE_TABLES, V3_ROOT as _V3_ROOT
from train.io import load_checkpoint


RESULTS_PHASE5 = V3_ROOT / 'results' / 'phase5'

TEST_REGIMES = ('D_1', 'D_2', 'D_3')
ALL_RUNS = [f'{d}_{s}' for d in ALL_DISTRIBUTIONS for s in ('cv', 'act')]
CV_RUNS = [f'{d}_cv' for d in ALL_DISTRIBUTIONS]

BASELINE_NAMES = (
    'oracle', 'plug_in', 'prior_only', 'MAP_sigma', 'MLE_sigma',
    'secretary', 'myopic',
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _load_test(regime: str):
    X, sigma_i, mu_i = load_cache(regime, 'test')
    return X.astype(np.float64), sigma_i, mu_i


def _load_persigma_test(distribution: str):
    p = CACHE_DIR_DEFAULT / f'{distribution}_persigma_test.npz'
    z = np.load(p, allow_pickle=False)
    return z['X'].astype(np.float64), z['sigma_i'], z['mu_i']


def _build_static_oracle_table(sigma: float) -> dict:
    """Solve static ADP at production resolution (K=2048, J=128)."""
    n = 256
    C, g = solve_adp(n, MU_0, sigma * sigma, TAU0_2, K=2048, J=128)
    return {'C_hat': C, 'grids': g}


def _build_random_oracle_table(distribution: str) -> dict:
    name = (
        'D_disc_K256_J64.npz' if distribution == 'D_disc'
        else 'D_logu_K256_J64_Js64.npz'
    )
    return load_random_table(ORACLE_TABLES / name)


def _eta(n: int = 256) -> np.ndarray:
    return compute_eta(n)


def _is_static(run_name: str) -> bool:
    return run_name.removesuffix('_cv').removesuffix('_act') in STATIC_DISTRIBUTIONS


def _is_random(run_name: str) -> bool:
    return run_name.removesuffix('_cv').removesuffix('_act') in RANDOM_DISTRIBUTIONS


def _model_distribution(run_name: str) -> str:
    return '_'.join(run_name.split('_')[:-1])


# ---------------------------------------------------------------------------
# Precompute all baselines and the trained-model action per test regime
# ---------------------------------------------------------------------------

def _build_baselines_for_regime(regime: str, X: np.ndarray, eta: np.ndarray,
                                static_tables: Dict[str, dict],
                                random_tables: Dict[str, dict]) -> Dict[str, dict]:
    sigma = static_sigma(regime)
    static_table = static_tables[regime]

    baselines: Dict[str, dict] = {}

    act, thr = static_oracle(X, sigma, static_table)
    baselines['oracle_static'] = {'action': act, 'threshold': thr}

    act, thr = plug_in(X, sigma, eta)
    baselines['plug_in'] = {'action': act, 'threshold': thr}

    act, thr = prior_only(X, sigma, eta)
    baselines['prior_only'] = {'action': act, 'threshold': thr}

    act, thr = myopic(X, sigma)
    baselines['myopic'] = {'action': act, 'threshold': thr}

    sg_disc = np.array([1.0, 10.0, 100.0])
    log_omega_disc = np.zeros(3)
    act, thr = map_sigma_plugin(X, sg_disc, log_omega_disc, eta)
    baselines['MAP_sigma_disc'] = {'action': act, 'threshold': thr}

    sg_logu, log_omega_logu = make_logu_grid('logu', J_sigma=64)
    act, thr = map_sigma_plugin(X, sg_logu, log_omega_logu, eta)
    baselines['MAP_sigma_logu'] = {'action': act, 'threshold': thr}

    act, thr = mle_sigma_plugin(X, eta)
    baselines['MLE_sigma'] = {'action': act, 'threshold': thr}

    act, thr = secretary(X)
    baselines['secretary'] = {'action': act, 'threshold': thr}

    # Random oracles (used as the "oracle" reference for random-variance models).
    act, thr = random_oracle(X, random_tables['D_disc'])
    baselines['oracle_disc'] = {'action': act, 'threshold': thr}
    act, thr = random_oracle(X, random_tables['D_logu'])
    baselines['oracle_logu'] = {'action': act, 'threshold': thr}

    return baselines


def _select_oracle_for_run(run_name: str, baselines: Dict[str, dict]) -> dict:
    """Pick the agreement/normalization oracle: per-regime static for
    static models, per-training-distribution random for random models."""
    dist = _model_distribution(run_name)
    if dist in STATIC_DISTRIBUTIONS:
        return baselines['oracle_static']
    if dist == 'D_disc':
        return baselines['oracle_disc']
    return baselines['oracle_logu']


def _select_map_for_run(run_name: str, baselines: Dict[str, dict]) -> dict:
    """Pick the MAP-σ baseline matching the model's training prior."""
    dist = _model_distribution(run_name)
    if dist == 'D_logu':
        return baselines['MAP_sigma_logu']
    return baselines['MAP_sigma_disc']                              # default for static + D_disc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run() -> int:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    eta = _eta()
    print(f'[run_eval] device={device}, eta_1={eta[0]:.4f}, eta_{{n-2}}={eta[-1]:.4f}')

    # ----- Load test caches -----
    print('[run_eval] loading test caches')
    test_caches = {regime: _load_test(regime) for regime in TEST_REGIMES}

    # ----- Build oracle tables -----
    print('[run_eval] building oracle tables')
    static_tables = {r: _build_static_oracle_table(static_sigma(r)) for r in TEST_REGIMES}
    random_tables = {
        'D_disc': _build_random_oracle_table('D_disc'),
        'D_logu': _build_random_oracle_table('D_logu'),
    }

    # ----- Precompute baselines for each test regime -----
    print('[run_eval] precomputing baselines per regime')
    t0 = time.perf_counter()
    baselines_by_regime: Dict[str, Dict[str, dict]] = {}
    for regime in TEST_REGIMES:
        X, _, _ = test_caches[regime]
        baselines_by_regime[regime] = _build_baselines_for_regime(
            regime, X, eta, static_tables, random_tables,
        )
    print(f'[run_eval]   baselines done in {time.perf_counter() - t0:.1f}s')

    # ----- Compute per-regime oracle R* (used as the normalizer) -----
    print('[run_eval] per-regime oracle R*')
    R_star_per_regime = {}
    for regime in TEST_REGIMES:
        X, _, _ = test_caches[regime]
        R_star_per_regime[regime] = expected_payoff(
            baselines_by_regime[regime]['oracle_static']['action'], X,
        )
    print(f'[run_eval]   R*: {R_star_per_regime}')

    # ----- Iterate (run_name, test_regime) cells -----
    payoff_matrix: Dict[str, Dict[str, float]] = {}            # R/R*
    payoff_matrix_R: Dict[str, Dict[str, float]] = {}          # raw R
    agreement_tensor = np.zeros((len(ALL_RUNS), len(TEST_REGIMES), len(BASELINE_NAMES)))
    trajectories: Dict[str, Dict[str, dict]] = {}

    for ri, run_name in enumerate(ALL_RUNS):
        print(f'\n[run_eval] === {run_name} ({ri+1}/{len(ALL_RUNS)}) ===')
        t0 = time.perf_counter()
        model, head, meta = load_checkpoint(run_name, which='best')
        model = model.to(device).eval()
        print(f'[run_eval]   loaded {run_name} (head={head})')

        payoff_matrix[run_name] = {}
        payoff_matrix_R[run_name] = {}
        trajectories[run_name] = {}

        for ci, regime in enumerate(TEST_REGIMES):
            X, _, _ = test_caches[regime]
            bl = baselines_by_regime[regime]

            # Trained model action + threshold.
            t_model = time.perf_counter()
            m_act, m_thr = model_action(model, X, head, device)
            t_model = time.perf_counter() - t_model

            # Payoff.
            R = expected_payoff(m_act, X)
            R_star = R_star_per_regime[regime]
            R_over_R_star = normalized_payoff(R, R_star)
            payoff_matrix[run_name][regime] = R_over_R_star
            payoff_matrix_R[run_name][regime] = R

            # Agreement against the 7 baselines.
            oracle_for_run = _select_oracle_for_run(run_name, bl)
            map_for_run = _select_map_for_run(run_name, bl)
            ag = {
                'oracle': per_step_agreement(m_act, oracle_for_run['action']),
                'plug_in': per_step_agreement(m_act, bl['plug_in']['action']),
                'prior_only': per_step_agreement(m_act, bl['prior_only']['action']),
                'MAP_sigma': per_step_agreement(m_act, map_for_run['action']),
                'MLE_sigma': per_step_agreement(m_act, bl['MLE_sigma']['action']),
                'secretary': per_step_agreement(m_act, bl['secretary']['action']),
                'myopic': per_step_agreement(m_act, bl['myopic']['action']),
            }
            for bi, bn in enumerate(BASELINE_NAMES):
                agreement_tensor[ri, ci, bi] = ag[bn]

            # Threshold trajectory (cv only — act has no interpretable threshold).
            traj = None
            if head == 'cv':
                traj = trajectory_mean_std(m_thr)
                # Cast to lists for JSON; keep the np arrays in `trajectories` for plotting.
                trajectories[run_name][regime] = {
                    'mean': traj['mean'], 'std': traj['std'],
                }

            # Per-cell JSON.
            cell_dir = RESULTS_PHASE5 / run_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / f'{regime}.json').write_text(json.dumps({
                'run_name': run_name,
                'head': head,
                'test_regime': regime,
                'R': R, 'R_star': R_star, 'R_over_R_star': R_over_R_star,
                'agreement': ag,
                'trajectory_mean': traj['mean'].tolist() if traj is not None else None,
                'trajectory_std': traj['std'].tolist() if traj is not None else None,
                'wall_model_forward_s': t_model,
            }, indent=2))
            print(f'[run_eval]   {regime}: R/R*={R_over_R_star:.4f}, '
                  f'oracle_agree={ag["oracle"]:.4f}')

        # Free GPU memory before next model.
        del model
        torch.cuda.empty_cache()
        print(f'[run_eval]   {run_name} all cells done in '
              f'{time.perf_counter() - t0:.1f}s')

    # ----- Per-σ payoff breakdown for the 4 random-variance runs -----
    print('\n[run_eval] per-σ payoff breakdown (random-variance runs)')
    persigma_results: Dict[str, list] = {}
    for run_name in ('D_disc_cv', 'D_disc_act', 'D_logu_cv', 'D_logu_act'):
        dist = _model_distribution(run_name)
        X_ps, sigma_i_ps, _ = _load_persigma_test(dist)
        model, head, _ = load_checkpoint(run_name, which='best')
        model = model.to(device).eval()
        rows = per_sigma_breakdown(
            model, head, X_ps, sigma_i_ps, dist, device, random_tables[dist],
        )
        persigma_results[run_name] = rows
        del model
        torch.cuda.empty_cache()
        print(f'[run_eval]   {run_name}: {[(r["name"], round(r["R_over_R_star"], 3)) for r in rows]}')

    # ----- Save sweep-level outputs -----
    np.savez_compressed(
        RESULTS_PHASE5 / 'agreement_tensor.npz',
        agreement=agreement_tensor,
        run_names=np.array(ALL_RUNS),
        test_regimes=np.array(TEST_REGIMES),
        baseline_names=np.array(BASELINE_NAMES),
    )
    (RESULTS_PHASE5 / 'payoff_matrix_raw.json').write_text(json.dumps({
        'R_over_R_star': payoff_matrix,
        'R': payoff_matrix_R,
        'R_star': R_star_per_regime,
        'baseline_R': _baseline_R_table(baselines_by_regime, test_caches),
    }, indent=2))
    (RESULTS_PHASE5 / 'persigma_results.json').write_text(json.dumps(
        persigma_results, indent=2,
    ))
    # trajectories: save as npz
    traj_payload = {}
    for run_name, regimes in trajectories.items():
        for regime, traj in regimes.items():
            traj_payload[f'{run_name}__{regime}__mean'] = traj['mean']
            traj_payload[f'{run_name}__{regime}__std'] = traj['std']
    # Add baseline trajectories too — needed for the threshold trajectory plot.
    for regime in TEST_REGIMES:
        bl = baselines_by_regime[regime]
        for bname in ('oracle_static', 'plug_in', 'prior_only',
                      'MAP_sigma_disc', 'MAP_sigma_logu', 'MLE_sigma',
                      'myopic', 'secretary'):
            t = bl[bname]['threshold']
            traj_payload[f'baseline__{regime}__{bname}__mean'] = np.nanmean(t, axis=0)
            traj_payload[f'baseline__{regime}__{bname}__std'] = np.nanstd(t, axis=0)
    np.savez_compressed(RESULTS_PHASE5 / 'trajectories.npz', **traj_payload)

    print(f'\n[run_eval] DONE. payoff matrix:')
    for run_name in ALL_RUNS:
        row = payoff_matrix[run_name]
        print(f'  {run_name:14s}  '
              f'D_1={row["D_1"]:.4f}  D_2={row["D_2"]:.4f}  D_3={row["D_3"]:.4f}')
    return 0


def _baseline_R_table(baselines_by_regime, test_caches) -> Dict[str, Dict[str, float]]:
    """Compute baseline R per (baseline_name, test_regime) for
    payoff_matrix_baselines.csv."""
    out: Dict[str, Dict[str, float]] = {}
    for bname in ('plug_in', 'prior_only', 'MAP_sigma_disc',
                  'MAP_sigma_logu', 'MLE_sigma', 'secretary', 'myopic'):
        out[bname] = {}
        for regime in TEST_REGIMES:
            X, _, _ = test_caches[regime]
            act = baselines_by_regime[regime][bname]['action']
            out[bname][regime] = expected_payoff(act, X)
    return out


if __name__ == '__main__':
    sys.exit(run())
