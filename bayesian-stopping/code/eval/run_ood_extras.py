"""Compute extras for the OOD section: per-step action agreement and
per-σ R/R* values across the seven evaluation σ ∈ {0.3, 1, 3, 10, 30, 100, 300}.

For each σ, we use the iid-σ test cache (existing D_{1,2,3}_test for σ ∈
{1, 10, 100}; OOD caches D_ood_<tag>_test for σ ∈ {0.3, 3, 30, 300}).

Outputs:
  results/ood/extras/agreement_oracle_extended.npz
      agreement[N_models, N_regimes] vs per-regime static-σ oracle
  results/ood/extras/per_sigma_extended.json
      For each (run, σ): R, R_star_random_disc, R_star_random_logu;
      derived ratios R/R*_random_disc and R/R*_random_logu.

Models: all 10 (D_1_cv, D_1_act, ..., D_logu_act). σ²-ablation checkpoints
substituted for D_disc_cv / D_logu_cv per the §6 convention.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import ALL_DISTRIBUTIONS, MU_0, TAU0_2
from data.streaming import load_cache as load_iid_cache
from eval.agreement import per_step_agreement
from eval.payoff import expected_payoff
from eval.policies import model_action, random_oracle, static_oracle
from oracle.random_adp import load_table as load_random_table
from oracle.static_adp import solve_adp
from train.configs import ORACLE_TABLES
from train.io import load_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Display order: by σ ascending. Each entry: (cache_regime, σ, label).
DISPLAY_ORDER: List[Tuple[str, float, str]] = [
    ('D_ood_0p3', 0.3,   r'$\sigma{=}0.3$'),
    ('D_1',       1.0,   r'$\sigma{=}1$'),
    ('D_ood_3',   3.0,   r'$\sigma{=}3$'),
    ('D_2',       10.0,  r'$\sigma{=}10$'),
    ('D_ood_30',  30.0,  r'$\sigma{=}30$'),
    ('D_3',       100.0, r'$\sigma{=}100$'),
    ('D_ood_300', 300.0, r'$\sigma{=}300$'),
]
ALL_RUNS = [f'{d}_{s}' for d in ALL_DISTRIBUTIONS for s in ('cv', 'act')]
SIG2_SWAP = {'D_disc_cv': 'D_disc_cv_sig2', 'D_logu_cv': 'D_logu_cv_sig2'}

K_PROD, J_PROD = 2048, 128
N = 256

OOD_ROOT = V3_ROOT / 'results' / 'ood'
EXTRAS_DIR = OOD_ROOT / 'extras'


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_static_table(sigma: float) -> dict:
    """Load static ADP table for σ ∈ {0.3, 1, 3, 10, 30, 100, 300}."""
    if sigma in (1.0, 10.0, 100.0):
        idx = {1.0: 1, 10.0: 2, 100.0: 3}[sigma]
        path = ORACLE_TABLES / f'D{idx}_static_K{K_PROD}_J{J_PROD}.npz'
    else:
        tag = {0.3: '0p3', 3.0: '3', 30.0: '30', 300.0: '300'}[sigma]
        path = ORACLE_TABLES / f'D_ood_{tag}_static_K{K_PROD}_J{J_PROD}.npz'
    z = np.load(path, allow_pickle=False)
    return {'C_hat': z['C_hat'], 'grids': z['grids']}


def _load_random_oracles() -> Dict[str, dict]:
    return {
        'D_disc': load_random_table(ORACLE_TABLES / 'D_disc_K256_J64.npz'),
        'D_logu': load_random_table(ORACLE_TABLES / 'D_logu_K256_J64_Js64.npz'),
    }


def _load_test_cache(regime: str) -> np.ndarray:
    """Load X for either an in-prior regime (D_1/D_2/D_3) or an OOD regime."""
    if regime in ('D_1', 'D_2', 'D_3'):
        X, _, _ = load_iid_cache(regime, 'test')
        return X.astype(np.float64)
    z = np.load(V3_ROOT / 'code' / 'data' / 'cache' / f'{regime}_test.npz',
                allow_pickle=False)
    return z['X'].astype(np.float64)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def main() -> int:
    EXTRAS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[ood-extras] device={device}')

    # Load oracles + caches up front.
    static_tables: Dict[str, dict] = {}
    random_tables = _load_random_oracles()
    caches: Dict[str, np.ndarray] = {}
    print('[ood-extras] loading caches and static tables')
    for regime, sigma, _ in DISPLAY_ORDER:
        caches[regime] = _load_test_cache(regime)
        static_tables[regime] = _load_static_table(sigma)
        print(f'  {regime} (σ={sigma}): cache {caches[regime].shape}')

    # Pre-compute per-regime oracle actions and random-ADP-{disc,logu} R.
    print('\n[ood-extras] computing per-regime oracle actions + random-ADP R')
    oracle_actions: Dict[str, np.ndarray] = {}
    R_random_disc: Dict[str, float] = {}
    R_random_logu: Dict[str, float] = {}
    for regime, sigma, _ in DISPLAY_ORDER:
        X = caches[regime]
        a_oracle, _ = static_oracle(X, sigma, static_tables[regime])
        oracle_actions[regime] = a_oracle
        a_disc, _ = random_oracle(X, random_tables['D_disc'])
        R_random_disc[regime] = expected_payoff(a_disc, X)
        a_logu, _ = random_oracle(X, random_tables['D_logu'])
        R_random_logu[regime] = expected_payoff(a_logu, X)
        print(f'  {regime}: R*_disc={R_random_disc[regime]:.4f}  '
              f'R*_logu={R_random_logu[regime]:.4f}')

    # Per-(model, regime) action agreement and R.
    print('\n[ood-extras] running models')
    n_runs = len(ALL_RUNS)
    n_reg = len(DISPLAY_ORDER)
    agreement = np.zeros((n_runs, n_reg))
    payoff_R: Dict[str, Dict[str, float]] = {run: {} for run in ALL_RUNS}

    for ri, run_name in enumerate(ALL_RUNS):
        print(f'\n[ood-extras] {run_name}')
        actual = SIG2_SWAP.get(run_name, run_name)
        if actual != run_name:
            print(f'  (loading {actual} for {run_name})')
        model, head, _ = load_checkpoint(actual, which='best')
        model = model.to(device).eval()

        for ci, (regime, sigma, _) in enumerate(DISPLAY_ORDER):
            X = caches[regime]
            t0 = time.perf_counter()
            m_act, _ = model_action(model, X, head, device)
            dt = time.perf_counter() - t0
            ag = per_step_agreement(m_act, oracle_actions[regime])
            R = expected_payoff(m_act, X)
            agreement[ri, ci] = ag
            payoff_R[run_name][regime] = R
            print(f'  {regime} (σ={sigma:>5g}): R={R:>9.3f}  agree={ag:.3f}  '
                  f'fwd={dt:.1f}s')

        del model
        torch.cuda.empty_cache()

    # Save agreement_oracle_extended.npz.
    regime_names = np.array([d[0] for d in DISPLAY_ORDER])
    sigmas = np.array([d[1] for d in DISPLAY_ORDER])
    np.savez_compressed(
        EXTRAS_DIR / 'agreement_oracle_extended.npz',
        agreement=agreement,
        run_names=np.array(ALL_RUNS),
        regimes=regime_names,
        sigmas=sigmas,
    )
    print(f'\nwrote {EXTRAS_DIR / "agreement_oracle_extended.npz"}')

    # Save per_sigma_extended.json.
    per_sigma = {
        'order': [{'regime': d[0], 'sigma': d[1]} for d in DISPLAY_ORDER],
        'R_random_disc': R_random_disc,
        'R_random_logu': R_random_logu,
        'R': payoff_R,
    }
    (EXTRAS_DIR / 'per_sigma_extended.json').write_text(
        json.dumps(per_sigma, indent=2))
    print(f'wrote {EXTRAS_DIR / "per_sigma_extended.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
