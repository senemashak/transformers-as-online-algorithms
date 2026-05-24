"""
Smoke 2 — random labeler at sigma=10 point mass vs static labeler at sigma=10.

End-to-end test of the LABELING PIPELINE (sample -> label) at point-mass
sigma=10. Step 2's reduction test exercised the ADP query directly; Smoke 2
exercises label_random / label_static through the data pipeline. Catches
state-translation and coordinate bugs that a direct ADP test wouldn't.

Setup:
    - Sample 1000 sequences with sigma_i = 10 (D_2 regime).
    - Solve a random ADP at point-mass sigma=10 (M=1, sigma_max=10).
    - Solve a static ADP at sigma=10 (K=2048, J=128).
    - Run both labelers, compare action labels at t = 2..n-1 (we exclude
      t=1 because the random labeler hardcodes y_act_1 = 0 by design,
      and that's a deliberate design difference, not an ADP-correctness
      indicator). Continuation values won't be pointwise-identical because
      one ADP is bilinear-2D and the other linear-1D; we only gate on
      action labels.

Gate: action-label agreement >= 99.99% on t = 2..n-1 (matching Step 2's
reduction-test number of 8.8e-5 disagreement).

Writes v3/results/phase3/smoke2.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from data.distributions import MU_0, N, TAU0_2
from data.labeling import label_random, label_static
from oracle.random_adp_torch import solve_random_adp_torch
from oracle.static_adp import solve_adp


SIGMA_TEST = 10.0
N_SEQ = 1000
SEED = 67890
GATE_AGREEMENT = 0.9999

RESULTS_DIR = V3_ROOT / 'results' / 'phase3'


def main() -> int:
    rng = np.random.default_rng(SEED)
    mu = rng.normal(MU_0, np.sqrt(TAU0_2), size=N_SEQ)
    X = mu[:, None] + rng.normal(0.0, SIGMA_TEST, size=(N_SEQ, N))

    # Random ADP at point-mass sigma=10 (production resolution).
    table_random = solve_random_adp_torch(
        sigma_grid=np.array([SIGMA_TEST]),
        log_omega=np.array([0.0]),
        sigma_max=SIGMA_TEST,
        n=N, mu_0=MU_0, tau0_2=TAU0_2,
        K1=256, K2=256, J=64,
    )
    # Static ADP at sigma=10 (production resolution).
    sigma2 = SIGMA_TEST ** 2
    C_hat, grids = solve_adp(N, MU_0, sigma2, TAU0_2, K=2048, J=128)
    table_static = {'C_hat': C_hat, 'grids': grids}

    y_cv_r, y_act_r = label_random(X, table_random)
    y_cv_s, y_act_s = label_static(X, SIGMA_TEST, table_static)

    # Compare on t=2..n-1 (indices 1..n-2). Both arrays are (N, n).
    sl = slice(1, N - 1)
    act_disagreement_t_ge_2 = float((y_act_r[:, sl] != y_act_s[:, sl]).mean())
    act_agreement_t_ge_2 = 1.0 - act_disagreement_t_ge_2

    # Diagnostic: t=1 disagreement (random hardcodes 0; static may accept).
    n_act_static_t1 = int(y_act_s[:, 0].sum())
    n_act_random_t1 = int(y_act_r[:, 0].sum())
    cv_disagree_max_t_ge_2 = float(np.abs(y_cv_r[:, sl] - y_cv_s[:, sl]).max())
    cv_disagree_med_t_ge_2 = float(np.median(np.abs(y_cv_r[:, sl] - y_cv_s[:, sl])))

    passed = act_agreement_t_ge_2 >= GATE_AGREEMENT

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append('# Smoke 2 — random labeler at point-mass sigma=10 vs static labeler\n\n')
    md.append(
        f'Setting: {N_SEQ} sequences with sigma_i = {SIGMA_TEST}, n={N}.\n'
        f'Random ADP: K1=K2=256, J=64, point-mass sigma=10. '
        f'Static ADP: K=2048, J=128, sigma=10.\n'
        f'Comparison restricted to t=2..n-1 (random labeler hardcodes '
        f'y_act_1=0 by design).\n\n'
    )
    md.append('| metric | value |\n|---|---|\n')
    md.append(f'| action agreement (t=2..n-1) | {act_agreement_t_ge_2:.6f} |\n')
    md.append(f'| action disagreement (t=2..n-1) | {act_disagreement_t_ge_2:.4e} |\n')
    md.append(f'| max |y_cv_random - y_cv_static| (t=2..n-1) | {cv_disagree_max_t_ge_2:.3e} |\n')
    md.append(f'| median |y_cv_random - y_cv_static| (t=2..n-1) | {cv_disagree_med_t_ge_2:.3e} |\n')
    md.append(f'| static labeler accepts at t=1 (count / {N_SEQ}) | {n_act_static_t1} |\n')
    md.append(f'| random labeler accepts at t=1 (count / {N_SEQ}) | {n_act_random_t1} |\n')
    md.append(f'\n**Gate: action agreement >= {GATE_AGREEMENT}: '
              f'{"PASS" if passed else "FAIL"}**\n')
    (RESULTS_DIR / 'smoke2.md').write_text(''.join(md))
    print(''.join(md))

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
