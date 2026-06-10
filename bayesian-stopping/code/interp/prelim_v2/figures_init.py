"""Paired-improvement figures for the init contrast (Δ = init − true).

Reads stats_init/per_cell_paired.csv (simple probe) and
stats_init_attn/per_cell_paired.csv (attention-pooled probe) — written by
stats_init.py — and renders one paired-improvement plot per (target,
probe family). Same styling as fig_paired_improvement: dark-green CI
band (central 95%), darker ±1 std band, mean line, dark-blue sign-test
W=x/N annotations with headroom.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from interp.prelim_v2.figures import (
    TARGET_LABELS, TARGET_YLABEL, PAIRED_COLOR,
)


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ('layer', 'M_train', 'N', 'W'):
                r[k] = int(r[k])
            for k in ('mean_delta', 'sd_delta', 'se_delta',
                      'ci_lo_95', 'ci_hi_95', 'p_sign_one_sided',
                      'mean_test_true', 'mean_test_init'):
                if k in r:
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def _fig_paired(rows: list[dict], out_dir: Path, suffix: str,
                title_prefix: str, label_prefix: str) -> None:
    targets = sorted({r['target'] for r in rows})
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        tr = sorted([r for r in rows if r['target'] == tgt],
                    key=lambda r: r['layer'])
        layers = np.asarray([r['layer'] for r in tr])
        means = np.asarray([r['mean_delta'] for r in tr])
        ci_lo = np.asarray([r['ci_lo_95'] for r in tr])
        ci_hi = np.asarray([r['ci_hi_95'] for r in tr])
        sd = np.asarray([r['sd_delta'] for r in tr])
        Ws = [r['W'] for r in tr]
        N = tr[0]['N']
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        ax.fill_between(layers, ci_lo, ci_hi,
                        color='#e6a92c', alpha=0.12, linewidth=0,
                        label='central 95% (2.5--97.5 pct)')
        ax.fill_between(layers, means - sd, means + sd,
                        color='#e6a92c', alpha=0.28, linewidth=0,
                        label=r'$\pm 1$ std (members)')
        ax.plot(layers, means, marker='o', lw=2, color='#e6a92c',
                label=r'mean $\bar\Delta$')
        ax.axhline(0.0, color='black', lw=0.8, ls='-')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
        for l, hi, W in zip(layers, ci_hi, Ws):
            ax.annotate(f'W={W}/{N}', xy=(l, hi), xytext=(0, 6),
                        textcoords='offset points', ha='center',
                        fontsize=8, color=PAIRED_COLOR)
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(r'$\bar\Delta$ = mean(init − true) test error')
        ax.set_title(f'{title_prefix}: {TARGET_LABELS.get(tgt, tgt)} (N={N})')
        ax.set_xticks(layers); ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best')
        fig.tight_layout()
        p = out_dir / f'{label_prefix}_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


def render_init_paired(run_dir: Path) -> None:
    """Render init paired plots for both probe families if stats files exist."""
    fig_dir = run_dir / 'figures'
    simple_csv = run_dir / 'stats_init' / 'per_cell_paired.csv'
    attn_csv   = run_dir / 'stats_init_attn' / 'per_cell_paired.csv'
    if simple_csv.exists():
        _fig_paired(_read_csv(simple_csv), fig_dir, 'simple',
                    'Paired improvement (init vs true), simple probe',
                    'init_paired_improvement')
    else:
        print(f'  [init paired] {simple_csv} missing, skipping simple')
    if attn_csv.exists():
        _fig_paired(_read_csv(attn_csv), fig_dir, 'attn',
                    'Paired improvement (init vs true), attention-pooled probe',
                    'init_paired_improvement_attn')
    else:
        print(f'  [init paired] {attn_csv} missing, skipping attn')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    render_init_paired(a.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
