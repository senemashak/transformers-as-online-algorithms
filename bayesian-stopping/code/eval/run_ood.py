"""Experiment A: OOD evaluation at σ ∈ {0.3, 3, 30, 300}.

Steps:
  1. Build static ADP tables (production K=2048,J=128) at each new σ.
  2. Build reference tables (K=4096,J=256) and run the convergence check;
     abort if max |C_prod - C_ref| ≥ 1e-3 OR action-label disagreement on
     N_test = 10^4 sequences ≥ 1e-3 for any σ.
  3. Build test caches (N=10^4 per σ, disjoint seeds 5001-5004).
  4. Evaluate every model (10) and every baseline (per-regime oracle,
     plug-in/known-σ rules, MAP-σ at both training priors, MLE-σ,
     secretary, offline-hindsight, random-ADP oracles for D_disc/D_logu)
     on each new cache. Save:
       - results/ood/payoff_matrix_raw.json
       - results/ood/trajectories.npz
       - results/ood/oracles/D_ood_<tag>_static_K{K}_J{J}.npz
       - results/ood/caches/D_ood_<tag>_test.npz
       - results/ood/convergence_check.json

The σ ∈ {1, 10, 100} columns are reused from the existing eval (they are
already in results/phase5/payoff_matrix_raw.json). This driver writes only
the four new columns; the renderer concatenates them with the canonical
three.
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

from data.distributions import (
    ALL_DISTRIBUTIONS,
    MU_0,
    TAU0_2,
    TAU_0,
)
from eval.payoff import expected_payoff, normalized_payoff
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
from train.configs import ORACLE_TABLES
from train.io import load_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OOD_SIGMAS: List[float] = [0.3, 3.0, 30.0, 300.0]
OOD_TAGS: List[str] = ['0p3', '3', '30', '300']
OOD_REGIMES: List[str] = [f'D_ood_{t}' for t in OOD_TAGS]

OOD_ROOT = V3_ROOT / 'results' / 'ood'
ORACLE_DIR = CODE_ROOT / 'oracle' / 'tables'                      # same as D{1,2,3}_static_*
CACHE_DIR = CODE_ROOT / 'data' / 'cache'                          # same as D_{1,2,3}_test.npz

K_PROD, J_PROD = 2048, 128
K_REF, J_REF = 4096, 256
N_TEST = 10_000
N = 256                                                          # sequence length
SEED_BASE = 5000                                                 # disjoint from existing 42xx/43xx/44xx

CONVERGENCE_TOL = 1.0e-3                                         # gate: action-label disagreement only

ALL_RUNS = [f'{d}_{s}' for d in ALL_DISTRIBUTIONS for s in ('cv', 'act')]

# Display order for the extended payoff matrix:
DISPLAY_REGIMES_ALL: List[Tuple[str, float]] = [
    ('D_ood_0p3', 0.3),
    ('D_1',       1.0),
    ('D_ood_3',   3.0),
    ('D_2',       10.0),
    ('D_ood_30',  30.0),
    ('D_3',       100.0),
    ('D_ood_300', 300.0),
]


# ---------------------------------------------------------------------------
# Step 1+2: Oracle tables at production and reference resolutions, with
# convergence check.
# ---------------------------------------------------------------------------

def _solve_save(sigma: float, K: int, J: int, path: Path) -> dict:
    print(f'  solving sigma={sigma}, K={K}, J={J} ... ', end='', flush=True)
    t0 = time.perf_counter()
    C, g = solve_adp(N, MU_0, sigma * sigma, TAU0_2, K=K, J=J)
    dt = time.perf_counter() - t0
    print(f'done in {dt:.1f}s')
    np.savez_compressed(
        path,
        C_hat=C,
        grids=g,
        meta=np.array([N, MU_0, TAU0_2, K, J, sigma], dtype=np.float64),
    )
    return {'C_hat': C, 'grids': g}


def _ood_table_path(tag: str, K: int, J: int) -> Path:
    """Naming convention parallels D{1,2,3}_static_K{K}_J{J}.npz."""
    return ORACLE_DIR / f'D_ood_{tag}_static_K{K}_J{J}.npz'


def _build_or_load_oracle(sigma: float, tag: str, K: int, J: int) -> dict:
    path = _ood_table_path(tag, K, J)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        print(f'  loaded sigma={sigma}, K={K}, J={J} from cache')
        return {'C_hat': z['C_hat'], 'grids': z['grids']}
    return _solve_save(sigma, K, J, path)


def _existing_static_table_path(idx: int, K: int, J: int) -> Path:
    """Path for D1/D2/D3 static tables — written by phase2_static.py."""
    return ORACLE_DIR / f'D{idx}_static_K{K}_J{J}.npz'


def _load_existing_static(idx: int, K: int, J: int) -> dict:
    z = np.load(_existing_static_table_path(idx, K, J), allow_pickle=False)
    return {'C_hat': z['C_hat'], 'grids': z['grids']}


def _convergence_one(sigma: float, prod: dict, ref: dict, rng: np.random.Generator) -> dict:
    """Per-σ convergence: report max |ΔC| (diagnostic) and action-label
    disagreement (gate). Returns dict ready for the combined report."""
    from oracle.static_adp import C_hat_lin
    max_C_disagree_per_t = []
    for i in range(N - 1):
        ref_at_prod = C_hat_lin(i, prod['grids'][i], ref['C_hat'], ref['grids'])
        max_C_disagree_per_t.append(float(np.abs(prod['C_hat'][i] - ref_at_prod).max()))
    max_C_disagree = float(np.max(max_C_disagree_per_t))

    mu_i = rng.normal(MU_0, TAU_0, size=N_TEST)
    noise = rng.standard_normal(size=(N_TEST, N))
    X = (mu_i[:, None] + sigma * noise).astype(np.float64)
    a_prod, _ = static_oracle(X, sigma, prod)
    a_ref, _ = static_oracle(X, sigma, ref)
    action_disagree = float(np.mean(a_prod[:, :N - 1] != a_ref[:, :N - 1]))

    passed = action_disagree < CONVERGENCE_TOL
    return {
        'sigma': sigma,
        'max_C_disagree': max_C_disagree,
        'action_label_disagree': action_disagree,
        'passed': passed,
        'K_prod': K_PROD, 'J_prod': J_PROD,
        'K_ref': K_REF, 'J_ref': J_REF,
    }


def step1_2_oracles_and_convergence() -> Tuple[Dict[str, dict], dict]:
    """Build tables at prod and ref resolutions for each OOD σ; run
    convergence check on N_test=10^4 sequences. The gate is action-label
    disagreement (matches the in-distribution convention; max |ΔC| is
    reported as a diagnostic). Aborts only if action-label gate fails.

    Also re-runs the existing σ ∈ {1, 10, 100} check from saved tables so
    the appendix can present a single combined seven-row block.
    """
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)

    print('[ood] step 1+2: oracle tables at prod and ref resolutions (OOD σ)')
    prod_tables: Dict[str, dict] = {}
    ref_tables: Dict[str, dict] = {}
    for sigma, tag, regime in zip(OOD_SIGMAS, OOD_TAGS, OOD_REGIMES):
        prod_tables[regime] = _build_or_load_oracle(sigma, tag, K_PROD, J_PROD)
        ref_tables[regime] = _build_or_load_oracle(sigma, tag, K_REF, J_REF)

    print('[ood] convergence check on all seven σ (gate: action-label < 1e-3)')
    rng_check = np.random.default_rng(seed=99999)
    convergence_report: dict = {
        'tol': CONVERGENCE_TOL,
        'gate': 'action_label_disagree',
        'per_sigma': {},
    }
    all_pass = True

    # Existing in-distribution σ ∈ {1, 10, 100} — reuse saved tables.
    for idx, sigma in [(1, 1.0), (2, 10.0), (3, 100.0)]:
        regime = f'D_{idx}'
        try:
            prod = _load_existing_static(idx, K_PROD, J_PROD)
            ref  = _load_existing_static(idx, K_REF, J_REF)
        except FileNotFoundError as exc:
            print(f'  σ={sigma:>5.1f} [{regime}]: missing table ({exc.filename}); '
                  f'skipping in-distribution row')
            continue
        rep = _convergence_one(sigma, prod, ref, rng_check)
        convergence_report['per_sigma'][regime] = rep
        all_pass = all_pass and rep['passed']
        print(f'  σ={sigma:>5.1f} [{regime}]    max|ΔC|={rep["max_C_disagree"]:.3e}  '
              f'Δaction={rep["action_label_disagree"]:.3e}  '
              f'{"PASS" if rep["passed"] else "FAIL"}')

    # New OOD σ values.
    for sigma, regime in zip(OOD_SIGMAS, OOD_REGIMES):
        rep = _convergence_one(sigma, prod_tables[regime], ref_tables[regime], rng_check)
        convergence_report['per_sigma'][regime] = rep
        all_pass = all_pass and rep['passed']
        print(f'  σ={sigma:>5.1f} [{regime:>10s}]  max|ΔC|={rep["max_C_disagree"]:.3e}  '
              f'Δaction={rep["action_label_disagree"]:.3e}  '
              f'{"PASS" if rep["passed"] else "FAIL"}')

    convergence_report['all_pass'] = all_pass

    OOD_ROOT.mkdir(parents=True, exist_ok=True)
    (OOD_ROOT / 'convergence_check.json').write_text(json.dumps(
        convergence_report, indent=2,
    ))

    if not all_pass:
        print('[ood] CONVERGENCE FAILED on the action-label gate — aborting.')
        sys.exit(1)

    return prod_tables, convergence_report


# ---------------------------------------------------------------------------
# Step 3: Test caches
# ---------------------------------------------------------------------------

def step3_test_caches() -> Dict[str, np.ndarray]:
    """Generate N=10^4 sequences per OOD σ. Seeds 5001-5004.
    Caches go to code/data/cache/ alongside D_{1,2,3}_test.npz so they
    are loadable via data.streaming.load_cache(regime, 'test').
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print('[ood] step 3: test caches')
    caches: Dict[str, np.ndarray] = {}                          # regime -> X
    for i, (sigma, tag, regime) in enumerate(
        zip(OOD_SIGMAS, OOD_TAGS, OOD_REGIMES), start=1,
    ):
        seed = SEED_BASE + i                                    # 5001..5004
        path = CACHE_DIR / f'{regime}_test.npz'
        if path.exists():
            print(f'  loaded {regime} test cache (seed={seed})')
            z = np.load(path, allow_pickle=False)
            caches[regime] = z['X']
            continue
        rng = np.random.default_rng(seed=seed)
        mu_i = rng.normal(MU_0, TAU_0, size=N_TEST)
        noise = rng.standard_normal(size=(N_TEST, N))
        sigma_i = np.full(N_TEST, sigma, dtype=np.float64)
        X = mu_i[:, None] + sigma * noise
        np.savez_compressed(
            path,
            X=X.astype(np.float64),
            sigma_i=sigma_i.astype(np.float64),
            mu_i=mu_i.astype(np.float64),
            seed=np.array([seed], dtype=np.int64),
        )
        caches[regime] = X.astype(np.float64)
        print(f'  wrote {path.name} (seed={seed})')
    return caches


# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------

def _build_baselines(regime: str, sigma: float, X: np.ndarray, eta: np.ndarray,
                     static_table: dict, random_tables: Dict[str, dict]) -> Dict[str, dict]:
    bl: Dict[str, dict] = {}
    a, t = static_oracle(X, sigma, static_table); bl['oracle_static'] = {'action': a, 'threshold': t}
    a, t = plug_in(X, sigma, eta);                bl['plug_in'] = {'action': a, 'threshold': t}
    a, t = prior_only(X, sigma, eta);             bl['prior_only'] = {'action': a, 'threshold': t}
    a, t = myopic(X, sigma);                      bl['myopic'] = {'action': a, 'threshold': t}

    sg_disc = np.array([1.0, 10.0, 100.0])
    log_omega_disc = np.zeros(3)
    a, t = map_sigma_plugin(X, sg_disc, log_omega_disc, eta)
    bl['MAP_sigma_disc'] = {'action': a, 'threshold': t}

    sg_logu, log_omega_logu = make_logu_grid('logu', J_sigma=64)
    a, t = map_sigma_plugin(X, sg_logu, log_omega_logu, eta)
    bl['MAP_sigma_logu'] = {'action': a, 'threshold': t}

    a, t = mle_sigma_plugin(X, eta);              bl['MLE_sigma'] = {'action': a, 'threshold': t}
    a, t = secretary(X);                          bl['secretary'] = {'action': a, 'threshold': t}

    a, t = random_oracle(X, random_tables['D_disc']); bl['random_oracle_disc'] = {'action': a, 'threshold': t}
    a, t = random_oracle(X, random_tables['D_logu']); bl['random_oracle_logu'] = {'action': a, 'threshold': t}
    return bl


def _offline_hindsight_value(sigma: float, n: int) -> float:
    """E[max_t X_t; D] = sigma * E[Z_n] + mu_0 (mu_0 = 0).

    E[Z_n] = ∫ z * n * Φ(z)^(n-1) * φ(z) dz
    """
    from scipy.special import ndtr  # Φ
    from scipy.stats import norm
    from scipy.integrate import quad
    def integrand(z, n=n):
        return z * n * ndtr(z) ** (n - 1) * norm.pdf(z)
    EZn, _ = quad(integrand, -10.0, 10.0, epsabs=1e-10)
    return float(sigma * EZn)                                   # mu_0 = 0


def _build_random_oracle_table(distribution: str) -> dict:
    name = (
        'D_disc_K256_J64.npz' if distribution == 'D_disc'
        else 'D_logu_K256_J64_Js64.npz'
    )
    return load_random_table(ORACLE_TABLES / name)


def step4_evaluate(prod_tables: Dict[str, dict],
                   caches: Dict[str, np.ndarray]) -> dict:
    """Evaluate all 10 models + every baseline on each OOD regime."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[ood] step 4: evaluate (device={device})')
    eta = compute_eta(N)

    # Random oracles (loaded once; reused per regime)
    random_tables = {
        'D_disc': _build_random_oracle_table('D_disc'),
        'D_logu': _build_random_oracle_table('D_logu'),
    }

    # Build per-regime baselines and per-regime R*
    baselines_by_regime: Dict[str, Dict[str, dict]] = {}
    R_star_per_regime: Dict[str, float] = {}
    R_baseline_per_regime: Dict[str, Dict[str, float]] = {}     # raw R per baseline
    R_offline_per_regime: Dict[str, float] = {}

    for sigma, regime in zip(OOD_SIGMAS, OOD_REGIMES):
        X = caches[regime]
        print(f'  build baselines for {regime} (σ={sigma})')
        bl = _build_baselines(regime, sigma, X, eta, prod_tables[regime], random_tables)
        baselines_by_regime[regime] = bl
        R_star_per_regime[regime] = expected_payoff(bl['oracle_static']['action'], X)
        R_offline_per_regime[regime] = _offline_hindsight_value(sigma, N)
        R_baseline_per_regime[regime] = {
            bn: expected_payoff(bl[bn]['action'], X)
            for bn in ('oracle_static', 'plug_in', 'prior_only', 'myopic',
                       'MAP_sigma_disc', 'MAP_sigma_logu', 'MLE_sigma',
                       'secretary', 'random_oracle_disc', 'random_oracle_logu')
        }
        R_baseline_per_regime[regime]['offline'] = R_offline_per_regime[regime]

    # Trained models
    payoff_matrix: Dict[str, Dict[str, float]] = {}
    payoff_matrix_R: Dict[str, Dict[str, float]] = {}
    trajectories: Dict[str, Dict[str, dict]] = {}

    for ri, run_name in enumerate(ALL_RUNS):
        print(f'\n[ood] === {run_name} ({ri+1}/{len(ALL_RUNS)}) ===')
        # Use the σ²-ablation checkpoint for D_disc_cv / D_logu_cv per the
        # main eval convention.
        actual_run = {
            'D_disc_cv': 'D_disc_cv_sig2',
            'D_logu_cv': 'D_logu_cv_sig2',
        }.get(run_name, run_name)
        if actual_run != run_name:
            print(f'  (loading {actual_run} for {run_name})')
        model, head, _ = load_checkpoint(actual_run, which='best')
        model = model.to(device).eval()

        payoff_matrix[run_name] = {}
        payoff_matrix_R[run_name] = {}
        trajectories[run_name] = {}

        for sigma, regime in zip(OOD_SIGMAS, OOD_REGIMES):
            X = caches[regime]
            m_act, m_thr = model_action(model, X, head, device)
            R = expected_payoff(m_act, X)
            R_star = R_star_per_regime[regime]
            R_over_R_star = normalized_payoff(R, R_star)
            payoff_matrix[run_name][regime] = R_over_R_star
            payoff_matrix_R[run_name][regime] = R

            if head == 'cv':
                tr = trajectory_mean_std(m_thr)
                trajectories[run_name][regime] = {
                    'mean': tr['mean'], 'std': tr['std'],
                }
            print(f'  {regime}: R={R:.4f}  R*={R_star:.4f}  R/R*={R_over_R_star:.4f}')

        del model
        torch.cuda.empty_cache()

    # Save payoff_matrix_raw.json for OOD regimes
    OOD_ROOT.mkdir(parents=True, exist_ok=True)
    raw = {
        'R_over_R_star': payoff_matrix,
        'R': payoff_matrix_R,
        'R_star': R_star_per_regime,
        'baseline_R': R_baseline_per_regime,
        'sigmas': dict(zip(OOD_REGIMES, OOD_SIGMAS)),
        'seeds': {regime: SEED_BASE + i for i, regime in enumerate(OOD_REGIMES, start=1)},
    }
    (OOD_ROOT / 'payoff_matrix_raw.json').write_text(json.dumps(raw, indent=2))
    print(f'\n[ood] wrote {OOD_ROOT / "payoff_matrix_raw.json"}')

    # Save trajectories.npz
    payload: Dict[str, np.ndarray] = {}
    for run_name, regimes in trajectories.items():
        for regime, tr in regimes.items():
            payload[f'{run_name}__{regime}__mean'] = tr['mean']
            payload[f'{run_name}__{regime}__std'] = tr['std']
    for regime in OOD_REGIMES:
        bl = baselines_by_regime[regime]
        for bname in ('oracle_static', 'plug_in', 'prior_only', 'myopic',
                      'MAP_sigma_disc', 'MAP_sigma_logu', 'MLE_sigma',
                      'secretary', 'random_oracle_disc', 'random_oracle_logu'):
            t = bl[bname]['threshold']
            payload[f'baseline__{regime}__{bname}__mean'] = np.nanmean(t, axis=0)
            payload[f'baseline__{regime}__{bname}__std'] = np.nanstd(t, axis=0)
        payload[f'baseline__{regime}__offline__value'] = np.array(
            [R_offline_per_regime[regime]], dtype=np.float64,
        )
    np.savez_compressed(OOD_ROOT / 'trajectories.npz', **payload)
    print(f'[ood] wrote {OOD_ROOT / "trajectories.npz"}')

    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print('=' * 60)
    print('Experiment A: OOD on σ (driver)')
    print('=' * 60)

    prod_tables, conv = step1_2_oracles_and_convergence()
    caches = step3_test_caches()
    raw = step4_evaluate(prod_tables, caches)

    print('\n' + '=' * 60)
    print('Experiment A: complete. Summary:')
    print('=' * 60)
    print('R/R* on OOD regimes:')
    for run in ALL_RUNS:
        row = raw['R_over_R_star'][run]
        cells = '  '.join(f'σ={s:>5.1f}: {row[r]:.3f}' for r, s in zip(OOD_REGIMES, OOD_SIGMAS))
        print(f'  {run:14s}  {cells}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
