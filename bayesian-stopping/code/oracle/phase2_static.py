"""
Phase 2A — static-variance ADP: solve at production and reference resolution
for each of D_1, D_2, D_3; pickle the tables; run the convergence gate on
10^4 freshly-seeded test sequences per regime.

Production: K = 2048, J = 128.
Reference:  K = 4096, J = 256.

Convergence gate:
  (1) Max absolute |C_hat^prod - C_hat^ref| on the production grid, per regime.
  (2) Action-label disagreement on 10^4 test sequences (seed 43), per regime.
Gate: (2) below 1e-3 per regime.

Outputs:
  v3/oracle/tables/D{1,2,3}_static_K2048_J128.npz       (production)
  v3/oracle/tables/D{1,2,3}_static_K4096_J256.npz       (reference)
  v3/results/phase2/static_convergence.md               (report)

Indexing convention: t in {1, ..., n-1} 1-indexed; arrays use i = t-1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from oracle.conjugate import interp_uniform, posterior_path_batch
from oracle.static_adp import C_hat_lin, solve_adp


# v3 spec hyperparameters
N = 256
MU_0 = 0.0
TAU0_2 = 100.0
SIGMA_VALUES = {1: 1.0, 2: 10.0, 3: 100.0}

K_PROD, J_PROD = 2048, 128
K_REF, J_REF = 4096, 256
N_TEST = 10_000
SEED_TEST = 43
GATE_ACTION = 1e-3


TABLES_DIR = HERE / 'tables'
RESULTS_DIR = V3_ROOT / 'results' / 'phase2'


def _save_static_table(path, C_hat, grids, K, J, sigma):
    np.savez_compressed(
        path,
        C_hat=C_hat, grids=grids,
        meta=np.array([N, MU_0, TAU0_2, K, J, sigma], dtype=np.float64),
    )


def _convergence_for_regime(regime_id: int) -> dict:
    sigma = SIGMA_VALUES[regime_id]
    sigma2 = sigma * sigma

    t0 = time.perf_counter()
    C_prod, g_prod = solve_adp(N, MU_0, sigma2, TAU0_2, K=K_PROD, J=J_PROD)
    wall_prod = time.perf_counter() - t0

    t0 = time.perf_counter()
    C_ref, g_ref = solve_adp(N, MU_0, sigma2, TAU0_2, K=K_REF, J=J_REF)
    wall_ref = time.perf_counter() - t0

    # Save tables.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _save_static_table(
        TABLES_DIR / f'D{regime_id}_static_K{K_PROD}_J{J_PROD}.npz',
        C_prod, g_prod, K_PROD, J_PROD, sigma,
    )
    _save_static_table(
        TABLES_DIR / f'D{regime_id}_static_K{K_REF}_J{J_REF}.npz',
        C_ref, g_ref, K_REF, J_REF, sigma,
    )

    # Metric (1): max |C_prod - C_ref_at_prod_grid| over (i, k).
    per_stage_max = np.empty(N - 1)
    for i in range(N - 1):
        v_ref_at_prod = interp_uniform(g_prod[i], g_ref[i], C_ref[i])
        per_stage_max[i] = np.abs(C_prod[i] - v_ref_at_prod).max()
    max_C_diff = float(per_stage_max.max())
    max_C_diff_stage = int(per_stage_max.argmax()) + 1  # 1-indexed t

    # Metric (2): action-label disagreement on 10^4 test sequences.
    rng = np.random.default_rng(SEED_TEST)
    mu = rng.normal(MU_0, np.sqrt(TAU0_2), size=N_TEST)
    X = mu[:, None] + rng.normal(0.0, sigma, size=(N_TEST, N))
    mu_path, _ = posterior_path_batch(X, MU_0, TAU0_2, sigma2)

    a_prod = np.empty((N_TEST, N - 1), dtype=bool)
    a_ref = np.empty((N_TEST, N - 1), dtype=bool)
    for i in range(N - 1):
        C_p_i = C_hat_lin(i, mu_path[:, i], C_prod, g_prod)
        C_r_i = C_hat_lin(i, mu_path[:, i], C_ref, g_ref)
        a_prod[:, i] = X[:, i] >= C_p_i
        a_ref[:, i] = X[:, i] >= C_r_i
    action_disagreement = float((a_prod != a_ref).mean())

    return {
        'regime_id': regime_id, 'sigma': sigma,
        'wall_prod_sec': wall_prod, 'wall_ref_sec': wall_ref,
        'max_C_diff': max_C_diff, 'max_C_diff_stage_1idx': max_C_diff_stage,
        'max_C_diff_rel_to_sigma': max_C_diff / sigma,
        'action_disagreement': action_disagreement,
        'pass_action': action_disagreement < GATE_ACTION,
    }


def _write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall_pass = all(r['pass_action'] for r in rows)
    lines = []
    lines.append('# Phase 2A — Static ADP convergence report\n')
    lines.append(
        f'Spec: Algorithm 1 (Section 3.1 of v3 spec). n={N}, mu_0={MU_0}, '
        f'tau_0^2={TAU0_2}.\n\n'
        f'Production: K={K_PROD}, J={J_PROD}. Reference: K={K_REF}, J={J_REF}.\n'
        f'Test set: {N_TEST} sequences per regime, seed={SEED_TEST}.\n'
        f'Gate: action-label disagreement < {GATE_ACTION:.0e} per regime.\n\n'
    )
    lines.append('| regime | sigma | wall_prod (s) | wall_ref (s) | '
                 'max |Ĉ_prod − Ĉ_ref|  | (rel σ) | argmax t | '
                 'action-label disagreement | pass |\n')
    lines.append('|---|---|---|---|---|---|---|---|---|\n')
    for r in rows:
        lines.append(
            f'| D_{r["regime_id"]} | {r["sigma"]:g} | '
            f'{r["wall_prod_sec"]:.2f} | {r["wall_ref_sec"]:.2f} | '
            f'{r["max_C_diff"]:.3e} | {r["max_C_diff_rel_to_sigma"]:.3e} | '
            f'{r["max_C_diff_stage_1idx"]} | '
            f'{r["action_disagreement"]:.3e} | '
            f'{"PASS" if r["pass_action"] else "FAIL"} |\n'
        )
    lines.append(f'\nOverall gate: **{"PASS" if overall_pass else "FAIL"}**\n')
    path.write_text(''.join(lines))


def main() -> int:
    rows = []
    for regime_id in (1, 2, 3):
        print(f'[static] solving regime D_{regime_id} (sigma={SIGMA_VALUES[regime_id]})')
        row = _convergence_for_regime(regime_id)
        rows.append(row)
        print(
            f'[static]   wall: prod={row["wall_prod_sec"]:.2f}s, ref={row["wall_ref_sec"]:.2f}s; '
            f'max|ΔĈ|={row["max_C_diff"]:.3e}; '
            f'action-disagreement={row["action_disagreement"]:.3e}; '
            f'pass={row["pass_action"]}'
        )

    _write_report(RESULTS_DIR / 'static_convergence.md', rows)
    print(f'\n[static] wrote report to {RESULTS_DIR / "static_convergence.md"}')

    return 0 if all(r['pass_action'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
