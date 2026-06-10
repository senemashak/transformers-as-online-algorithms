"""Phase 3 for the attention-pooled run: paired statistics + simple-vs-attn
comparison.

Inputs:
  - <src_run_dir> : the existing simple-probe run (read-only).
  - <attn_run_dir>: the attention-pooled run (this run's outputs).

Outputs (under <attn_run_dir>):
  stats/
    per_cell_paired.csv          attn-alone, same schema as the simple-run
    per_cell_paired.json
    summary.md
  comparison/
    per_cell_simple_vs_attn.csv  per (target, layer, rep, M_train): paired
                                 attn improvement over simple across members
    selectivity_change.csv       per (target, layer, M_train): paired change
                                 in the true-vs-control gap when going from
                                 simple to attention
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


def _load_member_rows(member_dir: Path, subdir: str = 'probe_runs',
                      ) -> tuple[list[dict], list[dict]]:
    """Return (true_rows, perm_glob_rows) from a member's probe_runs* subdir.
    Concatenates probe_results.json + probe_results_loglik.json if both exist."""
    def _read_rep(rep: str) -> list[dict]:
        rows = json.loads((member_dir / subdir / rep / 'probe_results.json').read_text())
        ll = member_dir / subdir / rep / 'probe_results_loglik.json'
        if ll.exists():
            rows = rows + json.loads(ll.read_text())
        return rows
    return _read_rep('true'), _read_rep('perm_glob')


# When attn outputs live in the same run dir as simple outputs, attn reads
# from probe_runs_attn/ instead of probe_runs/.
ATTN_SUBDIR = 'probe_runs_attn'


def _paired_stats(deltas: list[float]) -> dict:
    arr = np.asarray(deltas, dtype=np.float64)
    N = arr.size
    mean_d = float(arr.mean())
    sd_d = float(arr.std(ddof=1)) if N > 1 else float('nan')
    se_d = sd_d / math.sqrt(N) if N > 1 else float('nan')
    ci_lo, ci_hi = percentile_interval(arr)   # 2.5/97.5 pct of members
    W = int((arr > 0).sum())
    p_sign = float(binom.sf(W - 1, N, 0.5)) if N > 0 else float('nan')
    return dict(
        N=N, mean_delta=mean_d, sd_delta=sd_d, se_delta=se_d,
        ci_lo_95=ci_lo, ci_hi_95=ci_hi, ci_method='empirical_2.5_97.5_percentile_of_members',
        W=W, p_sign_one_sided=p_sign,
    )


def _find_members(src_run_dir: Path, attn_run_dir: Path) -> list[int]:
    """Members complete in BOTH runs."""
    src_members = src_run_dir / 'members'
    attn_members = attn_run_dir / 'members'
    out: list[int] = []
    for d in sorted(src_members.iterdir()):
        if not d.is_dir() and not d.is_symlink():
            continue
        i = int(d.name)
        ok_src = (
            (d / 'probe_runs' / 'true'      / 'probe_results.json').exists() and
            (d / 'probe_runs' / 'perm_glob' / 'probe_results.json').exists()
        )
        d2 = attn_members / d.name
        ok_attn = (
            (d2 / ATTN_SUBDIR / 'true'      / 'probe_results.json').exists() and
            (d2 / ATTN_SUBDIR / 'perm_glob' / 'probe_results.json').exists()
        )
        if ok_src and ok_attn:
            out.append(i)
    if not out:
        raise RuntimeError(
            f'No members with completed probes in BOTH '
            f'{src_members} and {attn_members}'
        )
    return out


def compute_attn_paired_stats(
    src_run_dir: Path, attn_run_dir: Path,
    M_train: int = HEADLINE_M_TRAIN,
) -> dict:
    """Attn-alone paired stats: Δ_i = test_attn_perm_glob - test_attn_true."""
    members = _find_members(src_run_dir, attn_run_dir)
    attn_members = attn_run_dir / 'members'
    print(f'[stats_attn] attn-alone paired: N={len(members)} members')

    # Layers from first member.
    tr0, _ = _load_member_rows(attn_members / f'{members[0]:03d}', subdir=ATTN_SUBDIR)
    layers = sorted({r['layer'] for r in tr0
                     if r['M_train_sequences'] == M_train})
    targets_present = sorted({r['target'] for r in tr0
                              if r['M_train_sequences'] == M_train})
    targets_to_use = targets_present if targets_present else list(TARGETS)
    print(f'[stats_attn] targets: {targets_to_use}')

    rows: list[dict] = []
    deltas_by_cell: dict[tuple[str, int], list[float]] = {}
    for tgt in targets_to_use:
        for layer in layers:
            deltas: list[float] = []
            test_true: list[float] = []
            test_perm: list[float] = []
            for i in members:
                t_rows, p_rows = _load_member_rows(attn_members / f'{i:03d}', subdir=ATTN_SUBDIR)
                t_cell = _pick(t_rows, tgt, layer, M_train)
                p_cell = _pick(p_rows, tgt, layer, M_train)
                if t_cell is None or p_cell is None:
                    raise RuntimeError(
                        f'Missing attn cell for member={i} target={tgt} '
                        f'layer={layer} M_train={M_train}')
                d = p_cell['test_loss'] - t_cell['test_loss']
                deltas.append(d)
                test_true.append(t_cell['test_loss'])
                test_perm.append(p_cell['test_loss'])
            ps = _paired_stats(deltas)
            rows.append(dict(
                target=tgt, layer=layer, M_train=M_train, **ps,
                mean_test_true=float(np.mean(test_true)),
                mean_test_perm_glob=float(np.mean(test_perm)),
            ))
            deltas_by_cell[(tgt, layer)] = deltas

    # In unified mode the simple-probe stats already occupy 'stats/'; route
    # attn-alone paired stats to 'stats_attn/' to avoid clobbering them.
    stats_dir = attn_run_dir / 'stats_attn'
    stats_dir.mkdir(parents=True, exist_ok=True)

    csv_path = stats_dir / 'per_cell_paired.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (stats_dir / 'per_cell_paired.json').write_text(json.dumps({
        'members': members, 'N': len(members), 'M_train': M_train,
        'probe_type': 'timestep_shared_attention_pooled',
        'cells': rows,
        'deltas_by_cell': {f'{t}|{l}': v for (t, l), v in deltas_by_cell.items()},
    }, indent=2, default=str))
    (stats_dir / 'summary.md').write_text(_paired_summary_md(rows, members))
    print(f'  wrote {csv_path}')
    return {'rows': rows, 'members': members, 'deltas_by_cell': deltas_by_cell}


def _paired_summary_md(rows: list[dict], members: list[int]) -> str:
    N = len(members)
    M_train_used = rows[0]['M_train'] if rows else HEADLINE_M_TRAIN
    lines = [
        '# Paired-ensemble probing statistics (attention-pooled probe)',
        '',
        f'- N = {N} ensemble members',
        f'- M_train = {M_train_used} sequences',
        '- Control contrast: perm_glob vs true',
        '- Probe: timestep-shared attention-pooled (linear FF), σ-normalized MSE',
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
    lines.append('Δ = M_perm_glob - M_true on test split (σ-normalized MSE), '
                 'attention-pooled probe.')
    return '\n'.join(lines) + '\n'


def compute_simple_vs_attn(
    src_run_dir: Path, attn_run_dir: Path,
    M_train: int = HEADLINE_M_TRAIN,
) -> dict:
    """Paired (per-member) attn improvement over simple, and the change in
    the true-vs-control selectivity gap.
    """
    members = _find_members(src_run_dir, attn_run_dir)
    src_members  = src_run_dir  / 'members'
    attn_members = attn_run_dir / 'members'
    print(f'[stats_attn] simple-vs-attn: N={len(members)} members')

    # Layers from first member.
    tr0, _ = _load_member_rows(src_members / f'{members[0]:03d}')
    layers = sorted({r['layer'] for r in tr0
                     if r['M_train_sequences'] == M_train})
    targets_present = sorted({r['target'] for r in tr0
                              if r['M_train_sequences'] == M_train})
    targets_to_use = targets_present if targets_present else list(TARGETS)

    per_rep_rows: list[dict] = []
    sel_rows: list[dict] = []

    for tgt in targets_to_use:
        for layer in layers:
            # Collect paired test losses across members.
            simple_true: list[float] = []
            simple_perm: list[float] = []
            attn_true:   list[float] = []
            attn_perm:   list[float] = []
            for i in members:
                st_rows, sp_rows = _load_member_rows(src_members  / f'{i:03d}')
                at_rows, ap_rows = _load_member_rows(attn_members / f'{i:03d}', subdir=ATTN_SUBDIR)
                cells = [
                    ('simple', 'true',      _pick(st_rows, tgt, layer, M_train)),
                    ('simple', 'perm_glob', _pick(sp_rows, tgt, layer, M_train)),
                    ('attn',   'true',      _pick(at_rows, tgt, layer, M_train)),
                    ('attn',   'perm_glob', _pick(ap_rows, tgt, layer, M_train)),
                ]
                for tag, rep, c in cells:
                    if c is None:
                        raise RuntimeError(
                            f'Missing {tag}/{rep} for member={i} target={tgt} '
                            f'layer={layer} M_train={M_train}')
                simple_true.append(cells[0][2]['test_loss'])
                simple_perm.append(cells[1][2]['test_loss'])
                attn_true.append(  cells[2][2]['test_loss'])
                attn_perm.append(  cells[3][2]['test_loss'])

            sT = np.asarray(simple_true)
            sP = np.asarray(simple_perm)
            aT = np.asarray(attn_true)
            aP = np.asarray(attn_perm)

            # Per-rep attn improvement Δ_i = simple - attn (positive = attn better)
            for rep, simple_vec, attn_vec in (('true', sT, aT),
                                              ('perm_glob', sP, aP)):
                imp = (simple_vec - attn_vec).tolist()
                ps = _paired_stats(imp)
                per_rep_rows.append(dict(
                    target=tgt, layer=layer, base_rep=rep, M_train=M_train,
                    mean_test_simple=float(simple_vec.mean()),
                    mean_test_attn=float(attn_vec.mean()),
                    **{f'attn_improvement_{k}': v for k, v in ps.items()},
                ))

            # Selectivity change: per-member, gap_attn - gap_simple where
            # gap = test_perm_glob - test_true. Positive => attn widens the
            # true-vs-control gap (more selectivity).
            gap_simple = (sP - sT).tolist()
            gap_attn   = (aP - aT).tolist()
            sel_delta  = [a - s for a, s in zip(gap_attn, gap_simple)]
            ps_sel = _paired_stats(sel_delta)
            sel_rows.append(dict(
                target=tgt, layer=layer, M_train=M_train,
                mean_gap_simple=float(np.mean(gap_simple)),
                mean_gap_attn=float(np.mean(gap_attn)),
                **{f'sel_change_{k}': v for k, v in ps_sel.items()},
            ))

    comp_dir = attn_run_dir / 'comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = comp_dir / 'per_cell_simple_vs_attn.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_rep_rows[0].keys()))
        w.writeheader()
        for r in per_rep_rows:
            w.writerow(r)
    print(f'  wrote {csv_path}')

    sel_csv = comp_dir / 'selectivity_change.csv'
    with sel_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys()))
        w.writeheader()
        for r in sel_rows:
            w.writerow(r)
    print(f'  wrote {sel_csv}')

    md_path = comp_dir / 'summary.md'
    md_path.write_text(_comp_summary_md(per_rep_rows, sel_rows, members))
    print(f'  wrote {md_path}')

    return {'per_rep_rows': per_rep_rows, 'sel_rows': sel_rows, 'members': members}


def _comp_summary_md(per_rep_rows: list[dict], sel_rows: list[dict],
                     members: list[int]) -> str:
    N = len(members)
    lines = [
        '# Simple vs attention-pooled comparison (paired across members)',
        '',
        f'- N = {N} ensemble members',
        f'- M_train = {per_rep_rows[0]["M_train"]} sequences',
        '- Probe families: timestep-shared simple linear vs '
        'timestep-shared attention-pooled (linear FF)',
        '',
        '## Per-rep attention improvement (positive = attn lowers test sMSE)',
        '',
        '| target | layer | rep | mean(simple) | mean(attn) | mean Δ | 95% CI | W/N | p (sign) |',
        '|---|---:|---|---:|---:|---:|---|---:|---:|',
    ]
    for r in per_rep_rows:
        ci_str = (f"[{r['attn_improvement_ci_lo_95']:+.4f}, "
                  f"{r['attn_improvement_ci_hi_95']:+.4f}]")
        lines.append(
            f"| {r['target']} | {r['layer']} | {r['base_rep']} | "
            f"{r['mean_test_simple']:.4f} | {r['mean_test_attn']:.4f} | "
            f"{r['attn_improvement_mean_delta']:+.4f} | {ci_str} | "
            f"{r['attn_improvement_W']}/{r['attn_improvement_N']} | "
            f"{r['attn_improvement_p_sign_one_sided']:.2e} |"
        )
    lines += [
        '',
        '## Selectivity change: attn widens the true-vs-control gap?',
        '',
        '`gap = test_perm_glob - test_true` per member; '
        '`Δ_sel = gap_attn - gap_simple`. Positive => attention pooling '
        'separates the trained representation from the control further '
        'than the simple probe does.',
        '',
        '| target | layer | mean(gap_simple) | mean(gap_attn) | mean Δ_sel | 95% CI | W/N | p (sign) |',
        '|---|---:|---:|---:|---:|---|---:|---:|',
    ]
    for r in sel_rows:
        ci_str = (f"[{r['sel_change_ci_lo_95']:+.4f}, "
                  f"{r['sel_change_ci_hi_95']:+.4f}]")
        lines.append(
            f"| {r['target']} | {r['layer']} | "
            f"{r['mean_gap_simple']:.4f} | {r['mean_gap_attn']:.4f} | "
            f"{r['sel_change_mean_delta']:+.4f} | {ci_str} | "
            f"{r['sel_change_W']}/{r['sel_change_N']} | "
            f"{r['sel_change_p_sign_one_sided']:.2e} |"
        )
    return '\n'.join(lines) + '\n'


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=None,
                   help='Unified run dir with both probe families in '
                        'probe_runs/ and probe_runs_attn/.')
    p.add_argument('--src-run-dir', type=Path, default=None,
                   help='(legacy split-dir layout) simple-probe run dir.')
    p.add_argument('--attn-run-dir', type=Path, default=None,
                   help='(legacy split-dir layout) attn-probe run dir.')
    p.add_argument('--M-train', type=int, default=HEADLINE_M_TRAIN)
    a = p.parse_args()
    if a.run_dir is not None:
        src_root = a.run_dir
        attn_root = a.run_dir
    else:
        if a.src_run_dir is None or a.attn_run_dir is None:
            p.error('either --run-dir or both --src-run-dir and --attn-run-dir')
        src_root = a.src_run_dir
        attn_root = a.attn_run_dir
    compute_attn_paired_stats(src_root, attn_root, M_train=a.M_train)
    compute_simple_vs_attn(src_root, attn_root, M_train=a.M_train)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
