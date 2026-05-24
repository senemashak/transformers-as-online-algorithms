"""Renderers for Experiment B (regime-shift).

Produces (per cv model unless noted):
  results/regime-shift/trajectories-shift-<run>.png  — 2×2 trajectory figure
  results/regime-shift/rho-shift-<run>.png            — 4-pair overlay of ρ_t
  results/regime-shift/t_half-hist-D_disc_cv.png      — 4-pair histogram
  results/regime-shift/summary_table.{csv,md}         — combined metrics

Color convention in the trajectory plots:
  - Pre-shift oracle reference (σ_a):  blue
  - Post-shift oracle reference (σ_b): red
  - Model trajectory:                  purple
  - Vertical dashed line at t=128:     gray
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

RS_ROOT = V3_ROOT / 'results' / 'regime-shift'
OVERLEAF = V3_ROOT / 'report' / 'figures'

PAIRS: List[Tuple[float, float]] = [
    (1.0, 100.0), (100.0, 1.0), (10.0, 100.0), (100.0, 10.0),
]
SHIFT_T = 128
N = 256
RHO_TRUNC_T = 240                    # last point on the ρ-overlay (avoid terminal collapse)


PAIR_COLORS = {
    (1.0, 100.0):   '#1f77b4',       # sharp upward    — blue
    (100.0, 1.0):   '#d62728',       # sharp downward  — red
    (10.0, 100.0):  '#2ca02c',       # moderate upward — green
    (100.0, 10.0):  '#9467bd',       # moderate down   — purple
}
PAIR_LABEL = {
    (1.0, 100.0):   r'$\sigma{=}1 \to 100$',
    (100.0, 1.0):   r'$\sigma{=}100 \to 1$',
    (10.0, 100.0):  r'$\sigma{=}10 \to 100$',
    (100.0, 10.0):  r'$\sigma{=}100 \to 10$',
}

PAIR_TAG = lambda a, b: f'{int(a)}_{int(b)}'

MODEL_LABEL = {
    'D_disc_cv': r'$\mathcal{D}_{\mathrm{disc}}$\_cv',
    'D_logu_cv': r'$\mathcal{D}_{\mathrm{logu}}$\_cv',
}
MODEL_COLOR = {
    'D_disc_cv': '#1f77b4',
    'D_logu_cv': '#2ca02c',
}

# Baselines overlaid on every shift trajectory panel (in addition to the
# piecewise σ_a/σ_b iid oracle references). Skips known-σ baselines
# (plug-in / prior-only / myopic) — those need a single σ.
#
# Tuple: (key, label, color, linestyle, emphasis).
# emphasis=True -> thick line, full alpha, high zorder, distinctive long-dash.
# Random-ADP oracles are the primary comparison target -> emphasised.
SHIFT_TRAJ_BASELINES = [
    ('random_oracle_disc',  r'random-ADP oracle ($\mathcal{D}_{\mathrm{disc}}$)',  '#0b3d91', (0, (10, 2)),       True),
    ('random_oracle_logu',  r'random-ADP oracle ($\mathcal{D}_{\mathrm{logu}}$)',  '#0a6e0a', (0, (10, 2, 2, 2)), True),
    ('MAP_sigma_disc',      r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{disc}}$ prior)', '#9467bd', '-.',                False),
    ('MAP_sigma_logu',      r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{logu}}$ prior)', '#e377c2', '-.',                False),
    ('MLE_sigma',           r'MLE-$\sigma$',                                       '#bcbd22', '--',                False),
    ('secretary',           'secretary (running max)',                             '#17becf', (0, (3, 1, 1, 1)),  False),
]


# ---------------------------------------------------------------------------
# 2×2 trajectory figure per model
# ---------------------------------------------------------------------------

def render_trajectory_figure(run_name: str) -> None:
    traj = dict(np.load(RS_ROOT / 'trajectories.npz', allow_pickle=False))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=130)
    t_axis = np.arange(1, N + 1)

    for ai, (sa, sb) in enumerate(PAIRS):
        ax = axes[ai // 2, ai % 2]
        tag = PAIR_TAG(sa, sb)

        ref_a = traj[f'shift_{tag}__ref_a_iid']                  # σ_a iid mean ref
        ref_b = traj[f'shift_{tag}__ref_b_iid']                  # σ_b iid mean ref
        thr_mean = traj[f'{run_name}__shift_{tag}__thr_mean']
        thr_std = traj[f'{run_name}__shift_{tag}__thr_std']

        # Piecewise oracle reference: σ_a on t ∈ [1, 128], σ_b on t ∈ [129, 256].
        # Truncate the σ_b reference at t = 250 to avoid the terminal collapse to ~0.
        pre_x = t_axis[:SHIFT_T]
        pre_y = ref_a[:SHIFT_T]
        post_x = t_axis[SHIFT_T:250]                             # t = 129..250
        post_y = ref_b[SHIFT_T:250]

        # Reference lines: pre-shift in blue, post-shift in red.
        ax.plot(pre_x, pre_y, color='#1f77b4', lw=1.4, ls='--',
                label=fr'static-$\sigma{{=}}{int(sa)}$ oracle (pre)')
        ax.plot(post_x, post_y, color='#d62728', lw=1.4, ls='--',
                label=fr'static-$\sigma{{=}}{int(sb)}$ oracle (post)')

        # All non-known-σ baselines (random-ADP, MAP-σ, MLE-σ, secretary).
        # Emphasised baselines (random-ADP oracles) get thick, full-alpha
        # lines on top of the rest; others stay thin and translucent.
        for bkey, label, color, ls, emphasis in SHIFT_TRAJ_BASELINES:
            k_mean = f'baseline__shift_{tag}__{bkey}__mean'
            k_std = f'baseline__shift_{tag}__{bkey}__std'
            if k_mean not in traj:
                continue
            bm = np.where(np.isfinite(traj[k_mean]), traj[k_mean], np.nan)
            bs = (np.where(np.isfinite(traj[k_std]), traj[k_std], np.nan)
                  if k_std in traj else None)
            has_band = bs is not None and np.nanmax(bs) > 1e-9
            if emphasis:
                line_lw, line_alpha, line_zorder = 2.4, 1.00, 15
                band_alpha = 0.18
            else:
                line_lw, line_alpha, line_zorder = 0.9, 0.85, 5
                band_alpha = 0.10
            if has_band:
                ax.fill_between(t_axis, bm - bs, bm + bs,
                                color=color, alpha=band_alpha, linewidth=0)
            ax.plot(t_axis, bm, color=color, lw=line_lw, ls=ls,
                    label=label, alpha=line_alpha, zorder=line_zorder)

        # Model trajectory (mean ± 1 std).
        # Clip lower band to a tiny positive value for log-scale plotting.
        lower = np.clip(thr_mean - thr_std, 1e-2, None)
        upper = np.clip(thr_mean + thr_std, 1e-2, None)
        ax.fill_between(t_axis, lower, upper, color='#9467bd',
                        alpha=0.20, linewidth=0)
        ax.plot(t_axis, np.clip(thr_mean, 1e-2, None),
                color='#9467bd', lw=2.0, label=f'{MODEL_LABEL[run_name]} (focal)')

        # Shift marker.
        ax.axvline(SHIFT_T, color='gray', lw=1.0, ls=':', alpha=0.7,
                   label=r'shift ($t{=}128$)')

        ax.set_yscale('log')
        ax.set_title(PAIR_LABEL[(sa, sb)], fontsize=11)
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel('threshold value (log)', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='best')

    fig.suptitle(f'Threshold trajectories under regime shift: {MODEL_LABEL[run_name]}',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RS_ROOT / f'trajectories-shift-{run_name}.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    out_overleaf = OVERLEAF / out.name
    out_overleaf.parent.mkdir(parents=True, exist_ok=True)
    out_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {out_overleaf}')


# ---------------------------------------------------------------------------
# ρ_t overlay figure per model
# ---------------------------------------------------------------------------

def render_rho_overlay(run_name: str) -> None:
    traj = dict(np.load(RS_ROOT / 'trajectories.npz', allow_pickle=False))

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=130)
    t_axis = np.arange(1, N + 1)

    for (sa, sb) in PAIRS:
        tag = PAIR_TAG(sa, sb)
        rho_mean = traj[f'{run_name}__shift_{tag}__rho_mean']
        rho_std = traj[f'{run_name}__shift_{tag}__rho_std']
        # Plot t ∈ [128, RHO_TRUNC_T] (skip terminal).
        slc = slice(SHIFT_T - 1, RHO_TRUNC_T)
        x = t_axis[slc]
        y = rho_mean[slc]
        ystd = rho_std[slc]
        color = PAIR_COLORS[(sa, sb)]
        ax.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=color, lw=1.8, label=PAIR_LABEL[(sa, sb)])

    ax.axhline(0.0, color='black', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(0.5, color='black', lw=0.8, ls='--', alpha=0.4,
               label=r'$\rho{=}0.5$ (half-adapted)')
    ax.axvline(SHIFT_T, color='gray', lw=1.0, ls=':', alpha=0.7,
               label=r'shift ($t{=}128$)')
    ax.set_xlim(SHIFT_T - 5, RHO_TRUNC_T + 2)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel(r'$\rho_t$  ($0$: still at $\sigma_a$ scale; $1$: fully at $\sigma_b$ scale)',
                  fontsize=10)
    ax.set_title(fr'Adaptation-progress $\rho_t$ for {MODEL_LABEL[run_name]} '
                 r'(across-sequence mean $\pm$ 1 std)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    out = RS_ROOT / f'rho-shift-{run_name}.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    out_overleaf = OVERLEAF / out.name
    out_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {out_overleaf}')


# ---------------------------------------------------------------------------
# t_{1/2} histograms for D_disc_cv
# ---------------------------------------------------------------------------

def render_t_half_histograms(run_name: str = 'D_disc_cv') -> None:
    z = np.load(RS_ROOT / 't_half_per_seq.npz', allow_pickle=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=130)
    n_post = N - SHIFT_T                                         # 128
    bins = np.arange(1, n_post + 2)                              # 1..128 + 1 = right edges

    for ai, (sa, sb) in enumerate(PAIRS):
        ax = axes[ai // 2, ai % 2]
        tag = PAIR_TAG(sa, sb)
        t_half = z[f'{run_name}__shift_{tag}__t_half']
        color = PAIR_COLORS[(sa, sb)]

        # Histogram (linear x for upward, log x for downward could be useful;
        # keep linear for visual parity).
        n_capped = int((t_half == n_post).sum())
        ax.hist(t_half, bins=bins, color=color, alpha=0.85, edgecolor='white',
                linewidth=0.3)
        # Add a marker for the cap-bin count.
        ax.text(0.97, 0.95, f'capped at $t_{{1/2}}={n_post}$:\n{n_capped} / {len(t_half)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                          edgecolor='none'))

        ax.set_yscale('log')
        ax.set_xlabel(r'$t_{1/2}$ (post-shift adaptation time)', fontsize=9)
        ax.set_ylabel('count (log)', fontsize=9)
        ax.set_title(PAIR_LABEL[(sa, sb)], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    fig.suptitle(fr'Per-sequence $t_{{1/2}}$ distribution for {MODEL_LABEL[run_name]}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RS_ROOT / f't_half-hist-{run_name}.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    out_overleaf = OVERLEAF / out.name
    out_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {out_overleaf}')


# ---------------------------------------------------------------------------
# Combined results table
# ---------------------------------------------------------------------------

def write_summary_table() -> None:
    raw = json.loads((RS_ROOT / 'raw.json').read_text())
    metrics = raw['metrics']

    columns = [
        ('mean t_{1/2}', 't_half_per_seq_mean', '{:.1f}'),
        ('median', 't_half_per_seq_median', '{:.0f}'),
        ('p10', 't_half_per_seq_p10', '{:.0f}'),
        ('p90', 't_half_per_seq_p90', '{:.0f}'),
        ('pre err', 'pre_shift_tracking_error', '{:.3f}'),
        ('post err', 'post_shift_tracking_error', '{:.2f}'),
        ('rho_settled (pop)', 'rho_settled_population', '{:+.3f}'),
    ]
    csv_path = RS_ROOT / 'summary_table.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'sigma_a', 'sigma_b'] + [c[0] for c in columns])
        for run in metrics:
            for tag, vals in metrics[run].items():
                row = [run, vals['sigma_a'], vals['sigma_b']]
                row += [c[2].format(vals[c[1]]) for c in columns]
                w.writerow(row)
    print(f'wrote {csv_path}')

    md_path = RS_ROOT / 'summary_table.md'
    lines = []
    lines.append('| model | shift | mean $t_{1/2}$ | median | $p_{10}$ | $p_{90}$ | pre err | post err | $\\rho_{200}$ (pop) |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for run in metrics:
        for tag, vals in metrics[run].items():
            sa, sb = vals['sigma_a'], vals['sigma_b']
            shift = f'$\\sigma{{=}}{int(sa)} \\to {int(sb)}$'
            lines.append(
                f'| {run} | {shift} | '
                f'{vals["t_half_per_seq_mean"]:.1f} | '
                f'{vals["t_half_per_seq_median"]:.0f} | '
                f'{vals["t_half_per_seq_p10"]:.0f} | '
                f'{vals["t_half_per_seq_p90"]:.0f} | '
                f'{vals["pre_shift_tracking_error"]:.3f} | '
                f'{vals["post_shift_tracking_error"]:.2f} | '
                f'{vals["rho_settled_population"]:+.3f} |'
            )
    md_path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {md_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    for run in ('D_disc_cv', 'D_logu_cv'):
        render_trajectory_figure(run)
        render_rho_overlay(run)
    render_t_half_histograms('D_disc_cv')
    write_summary_table()
    return 0


if __name__ == '__main__':
    sys.exit(main())
