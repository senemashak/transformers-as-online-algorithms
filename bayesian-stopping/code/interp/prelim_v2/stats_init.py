"""Phase 3 for the untrained-init baseline: paired statistics for the
init-vs-true contrast, for both probe families.

Computes Delta_i = test_loss_init_i - test_loss_true_i per cell (target, layer,
M_train) across N matched members, identically to stats.py / stats_attn.py
except the control representation is `init` (the seed-deterministic
untrained model) instead of `perm_glob`. Bonferroni multiplicity is per
contrast (k=27 = 3 targets x 9 layers) so the per-cell threshold and
significance reference match the existing perm_glob_vs_true tables.

Outputs (under <run_dir>):
  stats_init/
    per_cell_paired.csv             simple, init - true
    per_cell_paired.json
    summary.md
  stats_init_attn/
    per_cell_paired.csv             attn, init - true
    per_cell_paired.json
    summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

from interp.prelim_v2.stats import (
    _pick, HEADLINE_M_TRAIN, TARGETS,
    BONFERRONI_K, ALPHA, ALPHA_CELL_BONF, percentile_interval,
)


CONTROL_REP = 'init'


def _load_member_rows_for(
    member_dir: Path, control_rep: str, subdir: str = 'probe_runs',
) -> tuple[list[dict], list[dict]]:
    """Concatenates probe_results.json + probe_results_loglik.json (if present)
    for both the true and control rep, so loglik is auto-detected by stats."""
    def _read(rep: str) -> list[dict]:
        rows = json.loads((member_dir / subdir / rep / 'probe_results.json').read_text())
        ll = member_dir / subdir / rep / 'probe_results_loglik.json'
        if ll.exists():
            rows = rows + json.loads(ll.read_text())
        return rows
    return _read('true'), _read(control_rep)


def _find_members(run_dir: Path, subdir: str) -> list[int]:
    """Members complete for the (true, init) pair under <subdir>."""
    members_dir = run_dir / 'members'
    out: list[int] = []
    for d in sorted(members_dir.iterdir()):
        if not d.is_dir() and not d.is_symlink():
            continue
        if ((d / subdir / 'true' / 'probe_results.json').exists() and
            (d / subdir / CONTROL_REP / 'probe_results.json').exists()):
            out.append(int(d.name))
    return out


def _paired_stats(deltas: list[float]) -> dict:
    arr = np.asarray(deltas, dtype=np.float64)
    N = arr.size
    mean_d = float(arr.mean())
    sd_d = float(arr.std(ddof=1)) if N > 1 else float('nan')
    se_d = sd_d / math.sqrt(N) if N > 1 else float('nan')
    ci_lo, ci_hi = percentile_interval(arr)
    W = int((arr > 0).sum())
    p_sign = float(binom.sf(W - 1, N, 0.5)) if N > 0 else float('nan')
    return dict(
        N=N, mean_delta=mean_d, sd_delta=sd_d, se_delta=se_d,
        ci_lo_95=ci_lo, ci_hi_95=ci_hi,
        ci_method='empirical_2.5_97.5_percentile_of_members',
        W=W, p_sign_one_sided=p_sign,
    )


def _summary_md(rows: list[dict], members: list[int], probe_family: str) -> str:
    N = len(members)
    M_train_used = rows[0]['M_train'] if rows else HEADLINE_M_TRAIN
    lines = [
        f'# Paired-ensemble probing statistics (init contrast, {probe_family} probe)',
        '',
        f'- N = {N} ensemble members',
        f'- M_train = {M_train_used} sequences',
        f'- Control contrast: init vs true (init = seed-deterministic untrained model)',
        f'- Probe: {probe_family}',
        f'- Bonferroni reference (per contrast): k={BONFERRONI_K}, alpha_cell = {ALPHA_CELL_BONF:.2e}',
        '',
        '## Per-cell paired statistics',
        '',
        '| target | layer | mean Δ | 95% interval | W | p (sign) | Bonf? |',
        '|---|---:|---:|---|---:|---:|:---:|',
    ]
    for r in rows:
        ci = f"[{r['ci_lo_95']:+.4f}, {r['ci_hi_95']:+.4f}]"
        bonf = '✓' if r['p_sign_one_sided'] <= ALPHA_CELL_BONF else ''
        lines.append(
            f"| {r['target']} | {r['layer']} | {r['mean_delta']:+.4f} | "
            f"{ci} | {r['W']}/{r['N']} | {r['p_sign_one_sided']:.2e} | {bonf} |"
        )
    lines.append('')
    lines.append('Δ = test_loss(init) − test_loss(true) per member; '
                 'positive Δ means the trained representation produces lower test error.')
    return '\n'.join(lines) + '\n'


def compute_init_paired(
    run_dir: Path,
    subdir: str,                 # 'probe_runs' for simple, 'probe_runs_attn' for attn
    probe_family: str,           # human-readable label
    out_subdir: str,             # 'stats_init' or 'stats_init_attn'
    M_train: int = HEADLINE_M_TRAIN,
) -> dict:
    members = _find_members(run_dir, subdir)
    if not members:
        raise RuntimeError(f'No members complete for init contrast under {run_dir}/members/*/{subdir}')
    print(f'[stats_init/{probe_family}] using {len(members)} members')

    members_dir = run_dir / 'members'
    tr0, _ = _load_member_rows_for(
        members_dir / f'{members[0]:03d}', CONTROL_REP, subdir=subdir)
    layers = sorted({r['layer'] for r in tr0 if r['M_train_sequences'] == M_train})
    targets_present = sorted({r['target'] for r in tr0 if r['M_train_sequences'] == M_train})
    targets_to_use = targets_present if targets_present else list(TARGETS)
    print(f'[stats_init/{probe_family}] targets: {targets_to_use}')

    rows: list[dict] = []
    deltas_by_cell: dict[tuple[str, int], list[float]] = {}
    for tgt in targets_to_use:
        for layer in layers:
            deltas: list[float] = []
            test_true: list[float] = []
            test_init: list[float] = []
            for i in members:
                t_rows, i_rows = _load_member_rows_for(
                    members_dir / f'{i:03d}', CONTROL_REP, subdir=subdir)
                t_cell = _pick(t_rows, tgt, layer, M_train)
                i_cell = _pick(i_rows, tgt, layer, M_train)
                if t_cell is None or i_cell is None:
                    raise RuntimeError(
                        f'Missing cell for member={i} target={tgt} layer={layer} '
                        f'M_train={M_train} (subdir={subdir})')
                deltas.append(i_cell['test_loss'] - t_cell['test_loss'])
                test_true.append(t_cell['test_loss'])
                test_init.append(i_cell['test_loss'])
            ps = _paired_stats(deltas)
            rows.append(dict(
                target=tgt, layer=layer, M_train=M_train, **ps,
                mean_test_true=float(np.mean(test_true)),
                mean_test_init=float(np.mean(test_init)),
            ))
            deltas_by_cell[(tgt, layer)] = deltas

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'per_cell_paired.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    (out_dir / 'per_cell_paired.json').write_text(json.dumps({
        'members': members, 'N': len(members), 'M_train': M_train,
        'probe_family': probe_family,
        'control': CONTROL_REP,
        'cells': rows,
        'deltas_by_cell': {f'{t}|{l}': v for (t, l), v in deltas_by_cell.items()},
    }, indent=2, default=str))
    (out_dir / 'summary.md').write_text(_summary_md(rows, members, probe_family))
    print(f'  wrote {csv_path}')
    return {'rows': rows, 'members': members, 'deltas_by_cell': deltas_by_cell}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--M-train', type=int, default=HEADLINE_M_TRAIN)
    a = p.parse_args()
    compute_init_paired(a.run_dir, subdir='probe_runs',
                        probe_family='timestep-shared simple linear',
                        out_subdir='stats_init', M_train=a.M_train)
    compute_init_paired(a.run_dir, subdir='probe_runs_attn',
                        probe_family='timestep-shared attention-pooled',
                        out_subdir='stats_init_attn', M_train=a.M_train)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
