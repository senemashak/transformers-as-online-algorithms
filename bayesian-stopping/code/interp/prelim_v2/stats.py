"""Phase 3: paired statistics across ensemble members.

Cells are (target, layer) — one control contrast (perm_glob), one probe
configuration (timestep-shared simple linear). Per cell, with N matched
ensemble members:

    Delta_i = M_permglob_{i, ...} - M_true_{i, ...}
    bar_Delta = mean(Delta_i),  SE = sd(Delta) / sqrt(N)
    95% CI    = bar_Delta +/- 1.96 * SE
    W         = sum(Delta_i > 0)
    p (sign)  = Pr[Z >= W],  Z ~ Binomial(N, 1/2)   (one-sided)

Bonferroni reference (k = 3 targets * 9 layers = 27 cells per contrast):
alpha_cell = 0.05/27 ~ 1.85e-3. At N = 100, Pr[W >= 65] ~ 1.76e-3 < alpha_cell,
so W >= 65 is the conservative sign-test threshold.

Reads M_train = max_budget = 1024 rows from each member's
probe_runs/<rep>/probe_results.json. Writes:
    stats/per_cell_paired.csv
    stats/per_cell_paired.json
    stats/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom


HEADLINE_M_TRAIN = 1024
TARGETS = ('sigma_mle_sqrt', 'C_star', 'loglik_true_params_raw_mse')
BONFERRONI_K = 27
ALPHA = 0.05
ALPHA_CELL_BONF = ALPHA / BONFERRONI_K
def percentile_interval(values, alpha: float = ALPHA) -> tuple[float, float]:
    """Empirical central (1-alpha) interval of the ACTUAL per-member values.

    Returns the (alpha/2, 1-alpha/2) percentiles of `values` itself --- e.g.
    for alpha=0.05 the 2.5th and 97.5th percentiles across the N members. This
    is a descriptive interval for where the middle 95% of per-member errors (or
    paired differences) lie, NOT a confidence interval for the mean (which
    would be ~sqrt(N) times narrower). It makes no distributional assumption.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float('nan'), float('nan')
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _load_member_rows(member_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (true_rows, perm_glob_rows) from one member's probe_runs/.
    Concatenates rows from probe_results.json + probe_results_loglik.json
    (loglik is split into its own file by run_member_loglik to keep existing
    outputs read-only)."""
    def _read_rep(rep: str) -> list[dict]:
        rows = json.loads((member_dir / 'probe_runs' / rep / 'probe_results.json').read_text())
        ll = member_dir / 'probe_runs' / rep / 'probe_results_loglik.json'
        if ll.exists():
            rows = rows + json.loads(ll.read_text())
        return rows
    return _read_rep('true'), _read_rep('perm_glob')


def _pick(rows: list[dict], target: str, layer: int, M_train: int) -> dict | None:
    for r in rows:
        if r['target'] == target and r['layer'] == layer and r['M_train_sequences'] == M_train:
            return r
    return None


def compute_stats(
    run_dir: Path, members: list[int] | None = None,
    M_train: int = HEADLINE_M_TRAIN,
) -> dict:
    """Aggregate paired statistics across all ensemble members in run_dir."""
    members_dir = run_dir / 'members'
    if members is None:
        members = []
        for d in sorted(members_dir.iterdir()):
            if not d.is_dir():
                continue
            ok = ((d / 'probe_runs' / 'true' / 'probe_results.json').exists()
                  and (d / 'probe_runs' / 'perm_glob' / 'probe_results.json').exists())
            if ok:
                members.append(int(d.name))
    if len(members) == 0:
        raise RuntimeError(f'No members with completed probes found under {members_dir}')
    print(f'[stats] using {len(members)} complete members: {members}')

    # Find available layers and targets from the first member.
    true_rows0, _ = _load_member_rows(members_dir / f'{members[0]:03d}')
    layers = sorted({r['layer'] for r in true_rows0
                     if r['M_train_sequences'] == M_train})
    targets_present = sorted({r['target'] for r in true_rows0
                              if r['M_train_sequences'] == M_train})
    # Use auto-detected target list (default to module constant TARGETS for
    # back-compat, but override when the actual data has different targets).
    targets_to_use = targets_present if targets_present else list(TARGETS)
    print(f'[stats] targets: {targets_to_use}')

    out_rows: list[dict] = []
    deltas_by_cell: dict[tuple[str, int], list[float]] = {}
    for tgt in targets_to_use:
        for layer in layers:
            deltas: list[float] = []
            test_true: list[float] = []
            test_perm: list[float] = []
            for i in members:
                t_rows, p_rows = _load_member_rows(members_dir / f'{i:03d}')
                t_cell = _pick(t_rows, tgt, layer, M_train)
                p_cell = _pick(p_rows, tgt, layer, M_train)
                if t_cell is None or p_cell is None:
                    raise RuntimeError(
                        f'Missing cell for member={i} target={tgt} layer={layer} '
                        f'M_train={M_train}')
                d = p_cell['test_loss'] - t_cell['test_loss']
                deltas.append(d)
                test_true.append(t_cell['test_loss'])
                test_perm.append(p_cell['test_loss'])
            deltas_arr = np.asarray(deltas, dtype=np.float64)
            N = deltas_arr.size
            mean_d = float(deltas_arr.mean())
            sd_d = float(deltas_arr.std(ddof=1)) if N > 1 else float('nan')
            se_d = sd_d / math.sqrt(N) if N > 1 else float('nan')
            ci_lo, ci_hi = percentile_interval(deltas_arr)   # 2.5/97.5 pct of members
            W = int((deltas_arr > 0).sum())
            # One-sided sign-test p: Pr[Z >= W] = 1 - Pr[Z <= W-1] = binom.sf(W-1, N, 0.5)
            p_sign = float(binom.sf(W - 1, N, 0.5)) if N > 0 else float('nan')
            out_rows.append(dict(
                target=tgt, layer=layer, M_train=M_train, N=N,
                mean_delta=mean_d, sd_delta=sd_d, se_delta=se_d,
                ci_lo_95=ci_lo, ci_hi_95=ci_hi,
                ci_method='empirical_2.5_97.5_percentile_of_members',
                W=W, p_sign_one_sided=p_sign,
                mean_test_true=float(np.mean(test_true)),
                mean_test_perm_glob=float(np.mean(test_perm)),
            ))
            deltas_by_cell[(tgt, layer)] = deltas

    stats_dir = run_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = stats_dir / 'per_cell_paired.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    # JSON (includes per-member deltas)
    json_path = stats_dir / 'per_cell_paired.json'
    json_path.write_text(json.dumps({
        'members': members,
        'N': len(members),
        'M_train': M_train,
        'cells': out_rows,
        'deltas_by_cell': {f'{t}|{l}': v for (t, l), v in deltas_by_cell.items()},
    }, indent=2, default=str))

    # Summary markdown
    md_path = stats_dir / 'summary.md'
    md = _summary_md(out_rows, members)
    md_path.write_text(md)

    print(f'wrote {csv_path}')
    print(f'wrote {json_path}')
    print(f'wrote {md_path}')
    return {'rows': out_rows, 'members': members,
            'deltas_by_cell': deltas_by_cell}


def _summary_md(rows: list[dict], members: list[int]) -> str:
    N = len(members)
    M_train_used = rows[0]['M_train'] if rows else HEADLINE_M_TRAIN
    lines = [
        '# Paired-ensemble probing statistics',
        '',
        f'- N = {N} ensemble members',
        f'- M_train = {M_train_used} sequences',
        f'- Control contrast: perm_glob vs true',
        f'- Probe: timestep-shared simple linear, σ-normalized MSE',
        f'- Bonferroni reference: k={BONFERRONI_K}, alpha_cell = {ALPHA_CELL_BONF:.2e}',
        '',
        '## Per-cell paired statistics',
        '',
        '| target | layer | mean Δ | 95% CI | W | p (sign) | Bonf? |',
        '|---|---:|---:|---|---:|---:|:---:|',
    ]
    for r in rows:
        ci_str = f"[{r['ci_lo_95']:+.4f}, {r['ci_hi_95']:+.4f}]"
        bonf = '✓' if r['p_sign_one_sided'] <= ALPHA_CELL_BONF else ''
        lines.append(
            f"| {r['target']} | {r['layer']} | {r['mean_delta']:+.4f} | "
            f"{ci_str} | {r['W']}/{r['N']} | {r['p_sign_one_sided']:.2e} | {bonf} |"
        )
    lines.append('')
    lines.append('Δ = M_perm_glob - M_true on test split (σ-normalized MSE).')
    lines.append('Positive Δ means the true representation produces a lower test error.')
    return '\n'.join(lines) + '\n'


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--M-train', type=int, default=HEADLINE_M_TRAIN)
    a = p.parse_args()
    compute_stats(a.run_dir, M_train=a.M_train)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
