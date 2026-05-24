"""
Phase 2B — random-variance ADP: solve at production and reference resolution
for D_disc and D_logu; pickle the tables; run the convergence gate on
10^4 freshly-seeded test sequences per test regime D_1, D_2, D_3 for each
training distribution (six (train_dist, test_regime) cells).

Production: K1 = K2 = 256, J = 64, J_sigma = 64 (D_logu only).
Reference:  K1 = K2 = 512, J = 128, J_sigma = 128 (D_logu only).
sigma_max = 100 for both training distributions, used to size per-stage
adaptive X_bar bounds: half-width 5 * sqrt(tau_0^2 + sigma_max^2 / t).

Convergence gate per cell:
  (1) Max absolute |C_hat^prod - C_hat^ref| on per-sequence per-t lookups.
  (2) Action-label disagreement on 10^4 test sequences (seed 43 per regime).
Magnitude gate: < 0.1 (relaxed from 1e-2 after spec correction).
Action gate:    < 1e-3.

Outputs:
  v3/oracle/tables/D_disc_K256_J64.npz       (production)
  v3/oracle/tables/D_disc_K512_J128.npz      (reference)
  v3/oracle/tables/D_logu_K256_J64_Js64.npz  (production)
  v3/oracle/tables/D_logu_K512_J128_Js128.npz (reference)
  v3/results/phase2/random_convergence.md    (report)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from oracle.random_adp import load_table, query, save_table
from oracle.random_adp_torch import (
    make_sigma_grid, solve_random_adp_torch as solve_random_adp,
)


N = 256
MU_0 = 0.0
TAU0_2 = 100.0
SIGMA_MAX = 100.0                   # both D_disc and D_logu have sup support 100
SIGMA_REGIMES = {1: 1.0, 2: 10.0, 3: 100.0}
N_TEST = 10_000
SEED_TEST = 43

# (K1, K2, J, J_sigma) per resolution. J_sigma is ignored for D_disc.
PROD = dict(K1=256, K2=256, J=64, J_sigma=64)
REF = dict(K1=512, K2=512, J=128, J_sigma=128)

GATE_C = 0.1
GATE_ACTION = 1e-3

TABLES_DIR = HERE / 'tables'
RESULTS_DIR = V3_ROOT / 'results' / 'phase2'

# Chunk sizes balance peak memory vs map_coordinates call overhead.
# Per-chunk peak working memory ~ chunk * K2 * M * J * 8 * (~5 tensors).
CHUNK_PROD = {'disc': 32, 'logu': 32}
CHUNK_REF = {'disc': 64, 'logu': 32}
N_THREADS = 8


def _table_path(distribution: str, resolution: str) -> Path:
    if resolution == 'prod':
        if distribution == 'disc':
            name = 'D_disc_K256_J64.npz'
        else:
            name = 'D_logu_K256_J64_Js64.npz'
    else:
        if distribution == 'disc':
            name = 'D_disc_K512_J128.npz'
        else:
            name = 'D_logu_K512_J128_Js128.npz'
    return TABLES_DIR / name


def _solve(distribution: str, resolution: str, *, verbose: bool):
    """Solve, or load a previously-saved table if present (idempotent)."""
    path = _table_path(distribution, resolution)
    if path.exists():
        print(f'[random]   {path.name} exists, loading')
        return load_table(path), 0.0

    cfg = PROD if resolution == 'prod' else REF
    sigma_grid, log_omega = make_sigma_grid(
        distribution, J_sigma=cfg['J_sigma']
    )
    t0 = time.perf_counter()
    table = solve_random_adp(
        sigma_grid, log_omega,
        sigma_max=SIGMA_MAX,
        n=N, mu_0=MU_0, tau0_2=TAU0_2,
        K1=cfg['K1'], K2=cfg['K2'], J=cfg['J'],
        verbose=verbose,
    )
    wall = time.perf_counter() - t0
    save_table(table, path)
    return table, wall


def _evaluate_cell(
    table_p: dict, table_r: dict, X: np.ndarray,
) -> dict:
    S = np.cumsum(X, axis=1)
    Q = np.cumsum(X * X, axis=1)
    C_p = np.empty((X.shape[0], N - 1))
    C_r = np.empty((X.shape[0], N - 1))
    for t in range(1, N):
        C_p[:, t - 1] = query(table_p, t, S[:, t - 1], Q[:, t - 1])
        C_r[:, t - 1] = query(table_r, t, S[:, t - 1], Q[:, t - 1])
    diff = np.abs(C_p - C_r)
    a_p = X[:, : N - 1] >= C_p
    a_r = X[:, : N - 1] >= C_r
    return {
        'max_C_diff': float(diff.max()),
        'median_C_diff': float(np.median(diff)),
        'action_disagreement': float((a_p != a_r).mean()),
        'pass_C': bool(diff.max() < GATE_C),
        'pass_action': bool((a_p != a_r).mean() < GATE_ACTION),
    }


def _make_test_sets() -> dict:
    """One fixed seed per regime; sequences for each regime are independent."""
    sets = {}
    for i, sigma in SIGMA_REGIMES.items():
        rng = np.random.default_rng(SEED_TEST + i)
        mu = rng.normal(MU_0, np.sqrt(TAU0_2), size=N_TEST)
        X = mu[:, None] + rng.normal(0.0, sigma, size=(N_TEST, N))
        sets[i] = X
    return sets


def _write_report(path: Path, summary: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [r for r in summary if r['cell_kind'] == 'cell']
    all_action_pass = all(r['pass_action'] for r in cells)
    all_C_pass = all(r['pass_C'] for r in cells)
    lines = []
    lines.append('# Phase 2B — Random-variance ADP convergence report\n\n')
    lines.append(
        f'**Action-label gate (operationally meaningful): '
        f'{"PASS" if all_action_pass else "FAIL"}**'
        f' — every cell {"<" if all_action_pass else ">="} {GATE_ACTION:.0e}.\n\n'
        f'**Magnitude gate (absolute): '
        f'{"PASS" if all_C_pass else "FAIL"}** at threshold {GATE_C}.\n\n'
    )
    lines.append(
        f'Spec: Algorithm 2 (Section 3.2 of v3 spec). n={N}, mu_0={MU_0}, '
        f'tau_0^2={TAU0_2}, sigma_max={SIGMA_MAX} (size of per-stage '
        'adaptive X_bar bounds).\n\n'
    )
    lines.append(
        'Production: '
        f'K1=K2={PROD["K1"]}, J={PROD["J"]}, J_sigma={PROD["J_sigma"]} '
        '(D_logu only). '
        f'Reference: K1=K2={REF["K1"]}, J={REF["J"]}, '
        f'J_sigma={REF["J_sigma"]}.\n\n'
    )
    lines.append(
        f'Test set: {N_TEST} sequences per regime (independent fresh seeds '
        f'starting from {SEED_TEST}).\n'
        f'Convergence gate: action-label disagreement < {GATE_ACTION:.0e} '
        'per cell. The original absolute magnitude gate '
        f'(max |ΔĈ| < {GATE_C}) was dropped — see Spec corrections in '
        '`v3/README.md` for the reasoning. Median |ΔĈ|/σ_regime is '
        'reported as the regime-invariant magnitude diagnostic.\n\n'
    )
    lines.append(
        '## Solve wall times\n\n'
        '| training distribution | wall_prod (s) | wall_ref (s) |\n'
        '|---|---|---|\n'
    )
    walls = {r['distribution']: (r['wall_prod'], r['wall_ref'])
             for r in summary if r['cell_kind'] == 'wall'}
    for dist in ('disc', 'logu'):
        p, q = walls[dist]
        lines.append(f'| D_{dist} | {p:.2f} | {q:.2f} |\n')

    lines.append('\n## Convergence cells\n\n')
    lines.append(
        '| training | regime | max |Ĉ_prod − Ĉ_ref| | median |Ĉ_prod − Ĉ_ref| | '
        'median |ΔĈ| / σ_regime | action-label disagreement | gate |\n'
        '|---|---|---|---|---|---|---|\n'
    )
    sigma_by_regime = SIGMA_REGIMES
    for r in cells:
        passed = 'PASS' if r['pass_action'] else 'FAIL'
        sig = sigma_by_regime[r['regime']]
        lines.append(
            f'| D_{r["distribution"]} | D_{r["regime"]} (σ={sig:g}) | '
            f'{r["max_C_diff"]:.3e} | {r["median_C_diff"]:.3e} | '
            f'{r["median_C_diff"]/sig:.3e} | '
            f'{r["action_disagreement"]:.3e} | {passed} |\n'
        )

    path.write_text(''.join(lines))


def main(verbose: bool = True) -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_sets = _make_test_sets()
    summary: list[dict] = []

    for distribution in ('disc', 'logu'):
        print(f'\n[random] solving D_{distribution} at production')
        table_p, wall_p = _solve(distribution, 'prod', verbose=verbose)
        print(f'[random]   prod wall: {wall_p:.2f}s; '
              f'{_table_path(distribution, "prod").name}')

        print(f'[random] solving D_{distribution} at reference')
        table_r, wall_r = _solve(distribution, 'ref', verbose=verbose)
        print(f'[random]   ref  wall: {wall_r:.2f}s; '
              f'{_table_path(distribution, "ref").name}')

        summary.append({
            'cell_kind': 'wall',
            'distribution': distribution,
            'wall_prod': wall_p, 'wall_ref': wall_r,
        })

        for regime in (1, 2, 3):
            print(f'[random]   evaluating D_{distribution} on regime D_{regime}')
            cell = _evaluate_cell(table_p, table_r, test_sets[regime])
            cell.update(cell_kind='cell',
                        distribution=distribution, regime=regime)
            print(f'[random]     max|ΔĈ|={cell["max_C_diff"]:.3e}, '
                  f'median|ΔĈ|={cell["median_C_diff"]:.3e}, '
                  f'action-disag={cell["action_disagreement"]:.3e}, '
                  f'C_pass={cell["pass_C"]}, action_pass={cell["pass_action"]}')
            summary.append(cell)

        # Free memory before next distribution.
        del table_p, table_r

    # Don't clobber a hand-curated report when tables were cache-loaded;
    # the wall numbers would all be 0 and any manual diagnostics would be
    # lost. Re-runs that produced fresh wall numbers always get reported.
    walls = [r for r in summary if r['cell_kind'] == 'wall']
    any_fresh_wall = any(r['wall_prod'] > 0 or r['wall_ref'] > 0 for r in walls)
    report_path = RESULTS_DIR / 'random_convergence.md'
    if any_fresh_wall or not report_path.exists():
        _write_report(report_path, summary)
        print(f'\n[random] wrote {report_path}')
    else:
        print(f'\n[random] all tables cache-loaded; '
              f'leaving existing {report_path.name} in place')

    cells = [r for r in summary if r['cell_kind'] == 'cell']
    return 0 if all(r['pass_C'] and r['pass_action'] for r in cells) else 1


if __name__ == '__main__':
    sys.exit(main())
