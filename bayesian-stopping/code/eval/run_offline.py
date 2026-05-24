"""
Hindsight-supervision evaluation: 4 offline-supervised models on the 7
test caches (D_1, D_2, D_3, D_ood_{0.3, 3, 30, 300}).

For each (model, regime):
  * cv-offline models: accept iff X_t >= V_hat_t (model output).
  * act-offline models: accept iff sigmoid(logit_t) > tau, evaluated at
    tau in {0.3, 0.5, 0.7}. The 0.5 row is the headline.

Outputs (under v3/results/offline-supervision/):
    payoff_matrix_raw.json       — R, R*, R/R* per (model, regime).
                                    Includes both 0.5 and the {0.3, 0.7}
                                    sweep for act-offline models.
    payoff_matrix_extended.json  — 14x7 matrix combining the four
                                    offline-supervised models with the
                                    four oracle-supervised counterparts
                                    (sig2 for cv) plus the six static
                                    single-regime models. Convenient for
                                    the figure renderer.

R* per regime is the per-regime static-σ Bayes-optimal oracle's
expected payoff, identical to the convention used by phase5 / ood. We
recompute it here from the same test caches and oracle tables to keep
this driver self-contained.

The driver is read-only with respect to the existing checkpoints — it
only touches the new offline-supervised checkpoints under
checkpoints/D_*_offline/ and the existing test caches under
data/cache/. No re-training; no figure generation (that comes in a
later step after the user signs off on the raw matrix).
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

from data.distributions import MU_0, TAU0_2, TAU_0
from data.streaming import load_cache
from eval.payoff import expected_payoff, normalized_payoff
from eval.policies import static_oracle, threshold_to_action
from eval.trajectory import trajectory_mean_std
from oracle.static_adp import solve_adp
from train.configs import ORACLE_TABLES
from train.io import load_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OFFLINE_RUNS = (
    'D_disc_cv_offline', 'D_disc_act_offline',
    'D_logu_cv_offline', 'D_logu_act_offline',
)
# Comparator oracle runs: cv on random distributions uses _sig2 (matches
# the σ² normalizer that the offline-cv runs were trained with).
ORACLE_COMPARATORS = {
    'D_disc_cv_offline':  'D_disc_cv_sig2',
    'D_disc_act_offline': 'D_disc_act',
    'D_logu_cv_offline':  'D_logu_cv_sig2',
    'D_logu_act_offline': 'D_logu_act',
}
# All 14 runs in the extended matrix (oracle counterparts + offline + the
# six static single-regime models). Order pairs each oracle next to its
# offline counterpart for visual comparison.
EXTENDED_RUNS: Tuple[str, ...] = (
    'D_1_cv', 'D_1_act',
    'D_2_cv', 'D_2_act',
    'D_3_cv', 'D_3_act',
    'D_disc_cv_sig2',  'D_disc_cv_offline',
    'D_disc_act',      'D_disc_act_offline',
    'D_logu_cv_sig2',  'D_logu_cv_offline',
    'D_logu_act',      'D_logu_act_offline',
)

ID_REGIMES: Tuple[Tuple[str, float], ...] = (
    ('D_1', 1.0), ('D_2', 10.0), ('D_3', 100.0),
)
OOD_REGIMES: Tuple[Tuple[str, float], ...] = (
    ('D_ood_0p3', 0.3), ('D_ood_3', 3.0),
    ('D_ood_30', 30.0), ('D_ood_300', 300.0),
)
ALL_REGIMES: Tuple[Tuple[str, float], ...] = (
    # Display order: low σ → high σ (matches the OOD-render convention).
    ('D_ood_0p3', 0.3), ('D_1', 1.0), ('D_ood_3', 3.0),
    ('D_2', 10.0), ('D_ood_30', 30.0),
    ('D_3', 100.0), ('D_ood_300', 300.0),
)

ACT_THRESHOLDS = (0.3, 0.5, 0.7)

OUT_ROOT = V3_ROOT / 'results' / 'offline-supervision'
ORACLE_DIR = ORACLE_TABLES                     # solved tables saved here
N = 256
K_PROD, J_PROD = 2048, 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_X(regime: str) -> np.ndarray:
    X, _, _ = load_cache(regime, 'test')
    return X.astype(np.float64)


def _existing_static_oracle_path(regime: str) -> Path:
    """Production-resolution static-ADP table path for a regime.

    Static training distributions D_1/D_2/D_3 are stored as D{1,2,3}_static_K{K}_J{J}.npz;
    OOD regimes as D_ood_<tag>_static_K{K}_J{J}.npz.
    """
    if regime in ('D_1', 'D_2', 'D_3'):
        idx = int(regime.split('_')[1])
        return ORACLE_DIR / f'D{idx}_static_K{K_PROD}_J{J_PROD}.npz'
    if regime.startswith('D_ood_'):
        tag = regime[len('D_ood_'):]
        return ORACLE_DIR / f'D_ood_{tag}_static_K{K_PROD}_J{J_PROD}.npz'
    raise ValueError(f'unknown regime: {regime!r}')


def _load_or_solve_static_table(regime: str, sigma: float) -> dict:
    """Load the static-ADP table at production resolution; solve+cache
    if missing (keeps the driver self-contained even on a fresh machine
    where the OOD oracles have not been pre-built)."""
    path = _existing_static_oracle_path(regime)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {'C_hat': z['C_hat'], 'grids': z['grids']}
    print(f'  solving static ADP for {regime} (σ={sigma}) — not cached')
    C, g = solve_adp(N, MU_0, sigma * sigma, TAU0_2, K=K_PROD, J=J_PROD)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, C_hat=C, grids=g,
        meta=np.array([N, MU_0, TAU0_2, K_PROD, J_PROD, sigma], dtype=np.float64),
    )
    return {'C_hat': C, 'grids': g}


def _model_threshold_and_logits(
    model, X: np.ndarray, head: str, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward `model` once on X; return (cv_threshold, act_logits).

    For cv-supervised models, act_logits is None.
    For act-supervised models, cv_threshold is None.
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(Xt)
    if head == 'cv':
        return out['cv'].cpu().numpy().astype(np.float64), None
    if head == 'act':
        return None, out['act'].cpu().numpy().astype(np.float64)
    raise ValueError(f'unknown head {head!r}')


def _action_from_logits(logits: np.ndarray, tau: float) -> np.ndarray:
    """Action from sigmoid(logit) > tau. tau=0.5 maps to logit > 0.

    logit > log(tau / (1 - tau)) is the equivalent threshold; we compute
    that once per tau (broadcasts over all positions). Terminal step is
    forced acceptance.
    """
    N_seq, n = logits.shape
    if not (0.0 < tau < 1.0):
        raise ValueError(f'tau must be in (0, 1), got {tau}')
    logit_thr = float(np.log(tau / (1.0 - tau)))
    action = np.zeros((N_seq, n), dtype=bool)
    action[:, : n - 1] = logits[:, : n - 1] > logit_thr
    action[:, n - 1] = True
    return action


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run() -> int:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[offline-eval] device={device}')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ----- Load test caches -----
    print('[offline-eval] loading test caches')
    caches: Dict[str, np.ndarray] = {}
    for regime, _ in ALL_REGIMES:
        caches[regime] = _load_test_X(regime)
        print(f'  {regime}: X.shape={caches[regime].shape}')

    # ----- Load static-ADP oracle tables, compute R* per regime -----
    print('[offline-eval] solving/loading per-regime static-σ oracles → R*')
    R_star: Dict[str, float] = {}
    static_tables: Dict[str, dict] = {}
    for regime, sigma in ALL_REGIMES:
        tbl = _load_or_solve_static_table(regime, sigma)
        static_tables[regime] = tbl
        oracle_act, _ = static_oracle(caches[regime], sigma, tbl)
        R_star[regime] = expected_payoff(oracle_act, caches[regime])
        print(f'  R*[{regime}] (σ={sigma}) = {R_star[regime]:.4f}')

    # ----- Evaluate the 14 runs (oracle + offline + static single-regime) -----
    payoff_R: Dict[str, Dict[str, float]] = {}              # raw R per (run, regime)
    payoff_norm: Dict[str, Dict[str, float]] = {}           # R / R*
    # For act-supervised models we additionally evaluate the {0.3, 0.7}
    # threshold sweep on top of the canonical 0.5. Stored under
    # `act_threshold_sweep[run][regime][tau] = R/R*`.
    act_threshold_sweep: Dict[str, Dict[str, Dict[str, float]]] = {}
    trajectories_cv: Dict[str, Dict[str, dict]] = {}        # cv runs only

    for ri, run_name in enumerate(EXTENDED_RUNS):
        print(f'\n[offline-eval] === {run_name} ({ri+1}/{len(EXTENDED_RUNS)}) ===')
        try:
            model, head, _ = load_checkpoint(run_name, which='best')
        except FileNotFoundError as exc:
            print(f'  SKIP — checkpoint not found: {exc}')
            continue
        model = model.to(device).eval()

        payoff_R[run_name] = {}
        payoff_norm[run_name] = {}
        if head == 'act':
            act_threshold_sweep[run_name] = {}
        if head == 'cv':
            trajectories_cv[run_name] = {}

        for regime, _ in ALL_REGIMES:
            X = caches[regime]
            t0 = time.perf_counter()
            thr, logits = _model_threshold_and_logits(model, X, head, device)
            if head == 'cv':
                m_act = threshold_to_action(X, thr)
                trajectories_cv[run_name][regime] = trajectory_mean_std(thr)
            else:
                m_act = _action_from_logits(logits, tau=0.5)
            R = expected_payoff(m_act, X)
            R_over_R_star = normalized_payoff(R, R_star[regime])
            payoff_R[run_name][regime] = R
            payoff_norm[run_name][regime] = R_over_R_star
            extra = ''
            if head == 'act':
                tau_results: Dict[str, float] = {}
                for tau in ACT_THRESHOLDS:
                    if tau == 0.5:
                        tau_results[f'{tau:.2f}'] = R_over_R_star
                        continue
                    a_t = _action_from_logits(logits, tau=tau)
                    R_t = expected_payoff(a_t, X)
                    tau_results[f'{tau:.2f}'] = normalized_payoff(R_t, R_star[regime])
                act_threshold_sweep[run_name][regime] = tau_results
                extra = '  τ-sweep ' + ' '.join(
                    f'{tau:.1f}={tau_results[f"{tau:.2f}"]:.3f}' for tau in ACT_THRESHOLDS
                )
            dt = time.perf_counter() - t0
            print(f'  {regime:>10s}: R={R:.4f}  R*={R_star[regime]:.4f}  '
                  f'R/R*={R_over_R_star:.4f}  ({dt*1000:.0f} ms){extra}')

        del model
        torch.cuda.empty_cache()

    # ----- Per-distribution mean cv and act gaps -----
    gaps: Dict[str, Dict[str, Dict[str, float]]] = {}        # dist -> kind -> regime -> gap
    for dist, oracle_cv, offline_cv, oracle_act, offline_act in (
        ('D_disc', 'D_disc_cv_sig2', 'D_disc_cv_offline',
         'D_disc_act',  'D_disc_act_offline'),
        ('D_logu', 'D_logu_cv_sig2', 'D_logu_cv_offline',
         'D_logu_act',  'D_logu_act_offline'),
    ):
        per_regime_cv = {}
        per_regime_act = {}
        for regime, _ in ALL_REGIMES:
            if (oracle_cv in payoff_norm) and (offline_cv in payoff_norm):
                per_regime_cv[regime] = (
                    payoff_norm[offline_cv][regime] - payoff_norm[oracle_cv][regime]
                )
            if (oracle_act in payoff_norm) and (offline_act in payoff_norm):
                per_regime_act[regime] = (
                    payoff_norm[offline_act][regime] - payoff_norm[oracle_act][regime]
                )
        gaps[dist] = {
            'cv_per_regime': per_regime_cv,
            'act_per_regime': per_regime_act,
            'cv_mean': float(np.mean(list(per_regime_cv.values()))) if per_regime_cv else None,
            'act_mean': float(np.mean(list(per_regime_act.values()))) if per_regime_act else None,
        }

    # ----- Save outputs -----
    raw = {
        'R_over_R_star': payoff_norm,
        'R': payoff_R,
        'R_star': R_star,
        'act_threshold_sweep': act_threshold_sweep,
        'extended_runs': list(EXTENDED_RUNS),
        'all_regimes_in_display_order': [r for r, _ in ALL_REGIMES],
        'gaps': gaps,
    }
    raw_path = OUT_ROOT / 'payoff_matrix_raw.json'
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f'\n[offline-eval] wrote {raw_path}')

    # Save trajectories for the cv runs (useful for the later figure step).
    traj_payload: Dict[str, np.ndarray] = {}
    for run_name, regimes in trajectories_cv.items():
        for regime, tr in regimes.items():
            traj_payload[f'{run_name}__{regime}__mean'] = tr['mean']
            traj_payload[f'{run_name}__{regime}__std'] = tr['std']
    if traj_payload:
        traj_path = OUT_ROOT / 'trajectories_cv.npz'
        np.savez_compressed(traj_path, **traj_payload)
        print(f'[offline-eval] wrote {traj_path}')

    # ----- Print the human-readable 14×7 matrix -----
    print('\n[offline-eval] 14×7 payoff matrix (R/R*; rows = runs, cols = regimes)')
    header_regimes = [r for r, _ in ALL_REGIMES]
    col_w = 9
    print(f'{"run":<24s}' + ''.join(f'{r:>{col_w}s}' for r in header_regimes))
    for run_name in EXTENDED_RUNS:
        if run_name not in payoff_norm:
            print(f'{run_name:<24s}  [missing checkpoint]')
            continue
        cells = ''.join(f'{payoff_norm[run_name][r]:>{col_w}.4f}' for r in header_regimes)
        print(f'{run_name:<24s}{cells}')

    print('\n[offline-eval] per-distribution gaps (offline − oracle, R/R*)')
    for dist, info in gaps.items():
        print(f'  {dist}:')
        if info['cv_mean'] is not None:
            print(f'    cv_mean  = {info["cv_mean"]:+.4f}')
            for regime, g in info['cv_per_regime'].items():
                print(f'      cv  {regime:>10s} {g:+.4f}')
        if info['act_mean'] is not None:
            print(f'    act_mean = {info["act_mean"]:+.4f}')
            for regime, g in info['act_per_regime'].items():
                print(f'      act {regime:>10s} {g:+.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(run())
