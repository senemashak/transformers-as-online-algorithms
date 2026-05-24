"""
Smoke 1 — V3 static labeler must agree bit-for-bit with V2's reference.

Generates 1000 sequences from each of D_1, D_2, D_3 (deterministic seed),
runs them through both labelers, compares y_cv and y_act element-wise.

Gates:
    - y_act exact equality on every (sequence, t).
    - y_cv max absolute difference < 1e-12 (float64 round-trip noise).

Writes v3/results/phase3/smoke1.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from data.distributions import MU_0, STATIC_DISTRIBUTIONS, TAU0_2, sample, static_sigma
from data.labeling import label_static
from data.labeling_v2_reference import label_sequences_v2
from oracle.static_adp import solve_adp


N_SEQ = 1000
SEED = 12345
N = 256
K = 2048
J = 128
GATE_CV = 1e-12

RESULTS_DIR = V3_ROOT / 'results' / 'phase3'


def run_one(distribution: str) -> dict:
    sigma = static_sigma(distribution)
    sigma2 = sigma * sigma
    rng = np.random.default_rng(SEED + int(distribution.split('_')[1]))
    X, _, _ = sample(distribution, N_SEQ, rng)

    C_hat, grids = solve_adp(N, MU_0, sigma2, TAU0_2, K=K, J=J)
    table_static = {'C_hat': C_hat, 'grids': grids}

    # V3 static labeler
    y_cv_v3, y_act_v3 = label_static(X, sigma, table_static)
    # V2 reference labeler
    y_cv_v2, y_act_v2 = label_sequences_v2(X, MU_0, TAU0_2, sigma2, C_hat, grids)

    # V3 stores (N, n) with terminal placeholder; V2 stores (N, n-1).
    # Compare on the t=1..n-1 slice.
    y_cv_v3_compare = y_cv_v3[:, : N - 1]
    y_act_v3_compare = y_act_v3[:, : N - 1]

    cv_diff = np.abs(y_cv_v3_compare - y_cv_v2)
    act_match = (y_act_v3_compare == y_act_v2)
    return {
        'distribution': distribution,
        'sigma': sigma,
        'max_cv_diff': float(cv_diff.max()),
        'mean_cv_diff': float(cv_diff.mean()),
        'act_disagreement': int((~act_match).sum()),
        'act_total': int(act_match.size),
    }


def main() -> int:
    rows = [run_one(d) for d in STATIC_DISTRIBUTIONS]
    cv_pass = all(r['max_cv_diff'] < GATE_CV for r in rows)
    act_pass = all(r['act_disagreement'] == 0 for r in rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append('# Smoke 1 — V3 static labeler vs V2 reference\n\n')
    md.append(f'Setting: {N_SEQ} sequences per regime, n={N}, '
              f'static-ADP at K={K}, J={J}.\n')
    md.append(f'Gates: y_act exact equality; max |y_cv_v3 − y_cv_v2| < {GATE_CV:.0e}.\n\n')
    md.append('| regime | sigma | max |Δy_cv| | mean |Δy_cv| | y_act disagreement | gate |\n')
    md.append('|---|---|---|---|---|---|\n')
    for r in rows:
        passed = r['max_cv_diff'] < GATE_CV and r['act_disagreement'] == 0
        md.append(
            f'| {r["distribution"]} | {r["sigma"]:g} | '
            f'{r["max_cv_diff"]:.3e} | {r["mean_cv_diff"]:.3e} | '
            f'{r["act_disagreement"]} / {r["act_total"]} | '
            f'{"PASS" if passed else "FAIL"} |\n'
        )
    overall = 'PASS' if (cv_pass and act_pass) else 'FAIL'
    md.append(f'\n**Overall: {overall}**\n')
    (RESULTS_DIR / 'smoke1.md').write_text(''.join(md))
    print(''.join(md))

    return 0 if (cv_pass and act_pass) else 1


if __name__ == '__main__':
    sys.exit(main())
