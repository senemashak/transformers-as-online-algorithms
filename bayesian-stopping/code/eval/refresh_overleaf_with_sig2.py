"""Refresh the report/figures/ artifacts so that cv_disc and
cv_logu data come from the σ² ablation runs (D_disc_cv_sig2,
D_logu_cv_sig2) instead of the σ¹ counterparts.

Filenames in report/figures/ are unchanged — only the
underlying data is swapped. Static-σ trajectory PNGs are not touched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

SOFT_TURQUOISE_CMAP = LinearSegmentedColormap.from_list(
    'SoftTurquoise', ['#f4fbfa', '#cfeae6', '#9bd4cb'])
SOFT_PIYG_CMAP = LinearSegmentedColormap.from_list(
    'SoftPiYG', ['#e6a8c2', '#fbe6ee', '#ffffff', '#e9f3df', '#b5d8a8'])

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

RESULTS = V3_ROOT / 'results' / 'phase5'
EXP = V3_ROOT / 'report' / 'figures'
TEST_REGIMES = ('D_1', 'D_2', 'D_3')

# σ¹ run name -> σ² replacement. Display label is the σ¹ name (drop-in).
SWAP = {'D_disc_cv': 'D_disc_cv_sig2', 'D_logu_cv': 'D_logu_cv_sig2'}


# ----- helpers -----

def _load_payoff_with_swap():
    rows = []
    with (RESULTS / 'payoff_matrix.csv').open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            rows.append([row[0]] + [float(x) for x in row[1:]])
    sig2 = {}
    with (RESULTS / 'payoff_matrix_sig2.csv').open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            sig2[row[0]] = [float(x) for x in row[1:]]
    out = []
    for row in rows:
        if row[0] in SWAP:
            out.append([row[0]] + sig2[SWAP[row[0]]])           # keep label
        else:
            out.append(row)
    return out                                                  # [[name, d1, d2, d3], ...]


def render_payoff_matrix() -> None:
    rows = _load_payoff_with_swap()
    names = [r[0] for r in rows]
    M = np.array([r[1:] for r in rows]).T                       # (3 regimes, N models)
    fig, ax = plt.subplots(figsize=(11.0, 3.2), dpi=130)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap=SOFT_PIYG_CMAP, aspect='auto')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(3)); ax.set_yticklabels(TEST_REGIMES, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='black', fontsize=9)
    ax.set_title(r'$R/R^\star$  (per-regime static-σ oracle)', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=r'$R/R^\star$')
    fig.tight_layout()
    fig.savefig(EXP / 'payoff-matrix.png', dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / "payoff-matrix.png"}')


def render_agreement_oracle() -> None:
    z1 = np.load(RESULTS / 'agreement_tensor.npz', allow_pickle=False)
    z2 = np.load(RESULTS / 'agreement_tensor_sig2.npz', allow_pickle=False)
    bn = list(z1['baseline_names']); oi = bn.index('oracle')
    A1 = z1['agreement'][:, :, oi]                                  # (10, 3)
    A2 = z2['agreement'][:, :, oi]                                  # (2, 3)
    runs1 = list(z1['run_names'])
    runs2 = list(z2['run_names'])
    sig2_lookup = dict(zip(runs2, A2))                              # by sig2 run name
    A = []
    names = []
    for i, run in enumerate(runs1):
        if run in SWAP:
            A.append(sig2_lookup[SWAP[run]])
        else:
            A.append(A1[i])
        names.append(run)
    A = np.array(A).T                                           # (3 regimes, N models)
    fig, ax = plt.subplots(figsize=(11.0, 3.2), dpi=130)
    im = ax.imshow(A, vmin=0.0, vmax=1.0, cmap=SOFT_TURQUOISE_CMAP, aspect='auto')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(3)); ax.set_yticklabels(TEST_REGIMES, fontsize=10)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='black', fontsize=9)
    ax.set_title('Per-step action agreement vs. per-regime oracle', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='agreement')
    fig.tight_layout()
    fig.savefig(EXP / 'agreement-oracle.png', dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / "agreement-oracle.png"}')


def render_per_sigma_dist(dist: str) -> None:
    sig1 = json.loads((RESULTS / 'persigma_results.json').read_text())
    sig2 = json.loads((RESULTS / 'persigma_results_sig2.json').read_text())
    rows_cv = sig2[f'{dist}_cv_sig2']
    rows_act = sig1[f'{dist}_act']
    names = [r['name'] for r in rows_cv]
    cv = [r['R_over_R_star'] for r in rows_cv]
    act = [r['R_over_R_star'] for r in rows_act]
    counts = [r['count'] for r in rows_cv]
    head_color = {'cv': '#1f77b4', 'act': '#2ca02c'}            # trajectory blue / green
    head_alpha = 0.55                                            # vibrant but still transparent
    x = np.arange(len(names)); width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
    bars_cv = ax.bar(x - width/2, cv, width, color=head_color['cv'],
                     alpha=head_alpha, label=f'{dist}_cv')
    bars_act = ax.bar(x + width/2, act, width, color=head_color['act'],
                      alpha=head_alpha, label=f'{dist}_act')
    ax.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6,
               label='random-ADP oracle (R/R* = 1)')
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$R/R^\star$  ($R^\star$ = per-distribution random-ADP oracle)',
                  fontsize=10)
    ax.set_xlabel('σ bin', fontsize=10)
    ax.set_title(f'Per-σ payoff breakdown: {dist}_cv vs {dist}_act', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    all_vals = cv + act
    ax.set_ylim(min(min(all_vals)-0.12, -0.1), max(max(all_vals)+0.18, 1.25))
    for bars, vals in ((bars_cv, cv), (bars_act, act)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2.0, v+(0.025 if v>=0 else -0.05),
                    f'{v:.3f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8)
    for xi, c in zip(x, counts):
        ax.text(xi, ax.get_ylim()[0]+0.02, f'n={c}', ha='center', va='bottom',
                fontsize=7, color='gray')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fname = f'per-sigma-d-{dist.split("_")[1]}.png'
    fig.savefig(EXP / fname, dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / fname}')


# ----- trajectory plots: load both npz files, full §4 portfolio -----

def _load_traj_merged():
    main = np.load(RESULTS / 'trajectories.npz')
    sig2 = np.load(RESULTS / 'trajectories_sig2.npz')
    extras_path = RESULTS / 'trajectories_extras.npz'
    extras = (np.load(extras_path) if extras_path.exists() else None)
    traj = {k: main[k] for k in main.files}
    for k in sig2.files:
        traj[k] = sig2[k]
    if extras is not None:
        for k in extras.files:
            traj[k] = extras[k]
    return traj


BASELINE_KEYS = [
    ('oracle_static', 'per-regime oracle', 'black', '--'),
    ('random_oracle_disc', r'random-ADP oracle ($\mathcal{D}_{\mathrm{disc}}$)',
     '#1f4d8a', '-'),
    ('random_oracle_logu', r'random-ADP oracle ($\mathcal{D}_{\mathrm{logu}}$)',
     '#1f6e1f', '-'),
    ('plug_in', 'plug-in (known σ)', '#ff7f0e', '-'),
    ('prior_only', 'prior-only (known σ)', '#7f7f7f', ':'),
    ('myopic', 'myopic (known σ)', '#8c564b', '-'),
    ('MAP_sigma_disc', r'MAP-σ ($\mathcal{D}_{\mathrm{disc}}$ prior)',
     '#9467bd', '-.'),
    ('MAP_sigma_logu', r'MAP-σ ($\mathcal{D}_{\mathrm{logu}}$ prior)',
     '#e377c2', '-.'),
    ('MLE_sigma', 'MLE-σ', '#bcbd22', '--'),
    ('secretary', 'secretary (running max)', '#17becf', (0, (3, 1, 1, 1))),
]


def render_trajectory_for_run(focal_run_actual: str, focal_label: str,
                              focal_color: str, out_filename: str,
                              yscale: str = 'linear',
                              clip_negatives: bool = True) -> None:
    traj = _load_traj_merged()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ci, regime in enumerate(TEST_REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)

        # Focal model.
        m = traj[f'{focal_run_actual}__{regime}__mean']
        s = traj[f'{focal_run_actual}__{regime}__std']
        ax.fill_between(t_axis, m-s, m+s, color=focal_color, alpha=0.20, linewidth=0)
        ax.plot(t_axis, m, color=focal_color, lw=2.2, label=f'{focal_label} (focal)')

        for bkey, label, color, ls in BASELINE_KEYS:
            k_mean = f'baseline__{regime}__{bkey}__mean'
            k_std = f'baseline__{regime}__{bkey}__std'
            if k_mean not in traj:
                continue
            bm = traj[k_mean]; bs = traj[k_std] if k_std in traj else None
            bm = np.where(np.isfinite(bm), bm, np.nan)
            if bs is not None:
                bs = np.where(np.isfinite(bs), bs, np.nan)
            has_band = bs is not None and np.nanmax(bs) > 1e-9
            if has_band:
                ax.fill_between(t_axis, bm-bs, bm+bs, color=color, alpha=0.10, linewidth=0)
            ax.plot(t_axis, bm, color=color, lw=0.9, ls=ls, label=label, alpha=0.85)

        off_key = f'baseline__{regime}__offline__value'
        if off_key in traj:
            ax.axhline(float(traj[off_key][0]), color='black', lw=0.8,
                       ls=(0, (1, 1)), alpha=0.7,
                       label=r'offline (hindsight) $\mathbb{E}[\max_t X_t]$')

        ax.set_title(f'{regime} (test)', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.1)
        elif clip_negatives:
            ax.set_ylim(bottom=0)
        if ci == 0:
            label_y = 'threshold value (mean ± 1 std' + (', log' if yscale == 'log' else '') + ')'
            ax.set_ylabel(label_y, fontsize=9)
        ax.grid(True, alpha=0.3, which='both' if yscale == 'log' else 'major')
        if ci == 0:
            ax.legend(fontsize=7, loc='lower left' if yscale == 'log' else 'upper left')
    suptitle = f'Threshold trajectories: {focal_label}'
    if yscale == 'log':
        suptitle += '  (log y)'
    elif not clip_negatives:
        suptitle += '  (full y range)'
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EXP / out_filename, dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / out_filename}')


def render_trajectories_disc_logu_zoom() -> None:
    traj = _load_traj_merged()
    REGIMES = ('D_1', 'D_2', 'D_3')
    ZOOM_Y = {'D_1': (0, 10), 'D_2': (0, 40), 'D_3': (150, 300)}
    runs = [
        ('D_disc_cv_sig2', 'D_disc_cv', '#1f77b4'),
        ('D_logu_cv_sig2', 'D_logu_cv', '#2ca02c'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ci, regime in enumerate(REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)
        ok = f'baseline__{regime}__oracle_static__mean'
        om = traj[ok]; os = traj[ok.replace('__mean', '__std')]
        ax.fill_between(t_axis, om-os, om+os, color='black', alpha=0.10, linewidth=0)
        ax.plot(t_axis, om, color='black', lw=1.4, ls='--',
                label='per-regime oracle', zorder=20)
        for actual, label, color in runs:
            m = traj[f'{actual}__{regime}__mean']
            s = traj[f'{actual}__{regime}__std']
            ax.fill_between(t_axis, m-s, m+s, color=color, alpha=0.15, linewidth=0)
            ax.plot(t_axis, m, color=color, lw=2.0, label=label, alpha=0.95)
        lo, hi = ZOOM_Y[regime]
        ax.set_ylim(lo, hi)
        ax.set_title(f'{regime} (test) — y∈[{lo}, {hi}]', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if ci == 0:
            ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if ci == 2:
            ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Threshold trajectories (zoom) — D_disc_cv vs D_logu_cv', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EXP / 'trajectories-disc-logu-zoom.png', dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / "trajectories-disc-logu-zoom.png"}')


def render_trajectories_all_cv(yscale: str = 'linear',
                               out_filename: str = 'trajectories-all-cv.png',
                               clip_negatives: bool = True) -> None:
    """All-cv overlay; D_disc_cv and D_logu_cv lines now come from sig2 runs."""
    traj = _load_traj_merged()
    # Display label -> actual key prefix in npz.
    ALL_CV = [
        ('D_1_cv',    'D_1_cv',    '#d62728'),
        ('D_2_cv',    'D_2_cv',    '#ff7f0e'),
        ('D_3_cv',    'D_3_cv',    '#9467bd'),
        ('D_disc_cv', 'D_disc_cv_sig2', '#1f77b4'),         # σ² substituted in
        ('D_logu_cv', 'D_logu_cv_sig2', '#2ca02c'),         # σ² substituted in
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130, sharey=True)
    for ci, regime in enumerate(TEST_REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)
        ok = f'baseline__{regime}__oracle_static__mean'
        if ok in traj:
            om = traj[ok]; os = traj[ok.replace('__mean', '__std')]
            ax.fill_between(t_axis, om-os, om+os, color='black', alpha=0.10, linewidth=0)
            ax.plot(t_axis, om, color='black', lw=1.4, ls='--',
                    label='oracle (per-regime)', zorder=20)
        # Draw D_disc_cv last with thicker line so it isn't hidden by D_logu_cv / oracle.
        draw_order = [r for r in ALL_CV if r[0] != 'D_disc_cv'] + \
                     [r for r in ALL_CV if r[0] == 'D_disc_cv']
        for label, actual, color in draw_order:
            mk = f'{actual}__{regime}__mean'; sk = f'{actual}__{regime}__std'
            if mk not in traj:
                continue
            m = traj[mk]; s = traj[sk]
            is_disc = label == 'D_disc_cv'
            ax.fill_between(t_axis, m-s, m+s, color=color,
                            alpha=0.12 if is_disc else 0.08, linewidth=0)
            ax.plot(t_axis, m, color=color, lw=2.6 if is_disc else 1.5,
                    label=label, alpha=0.95, zorder=15 if is_disc else 5)
        ax.set_title(f'{regime} (test)', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.1)
        elif clip_negatives:
            ax.set_ylim(bottom=0)
        if ci == 0:
            label_y = 'threshold value (mean ± 1 std' + (', log' if yscale == 'log' else '') + ')'
            ax.set_ylabel(label_y, fontsize=9)
        ax.grid(True, alpha=0.3, which='both' if yscale == 'log' else 'major')
        if ci == 2:
            ax.legend(fontsize=7, loc='lower right' if yscale == 'log' else 'upper right')
    suptitle = 'Threshold trajectories — all 5 cv-trained models on each test regime'
    if yscale == 'log':
        suptitle += '  (log y)'
    elif not clip_negatives:
        suptitle += '  (full y range)'
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EXP / out_filename, dpi=300)
    plt.close(fig)
    print(f'wrote {EXP / out_filename}')


def main() -> int:
    render_payoff_matrix()
    render_agreement_oracle()
    render_per_sigma_dist('D_disc')
    render_per_sigma_dist('D_logu')
    render_trajectory_for_run('D_disc_cv_sig2', 'D_disc_cv', '#1f77b4',
                              'trajectories-d-disc-cv.png')
    render_trajectory_for_run('D_logu_cv_sig2', 'D_logu_cv', '#2ca02c',
                              'trajectories-d-logu-cv.png')
    render_trajectory_for_run('D_1_cv', 'D_1_cv', '#d62728',
                              'trajectories-d-1-cv.png', clip_negatives=False)
    render_trajectory_for_run('D_1_cv', 'D_1_cv', '#d62728',
                              'trajectories-d-1-cv-log.png', yscale='log')
    render_trajectories_disc_logu_zoom()
    render_trajectories_all_cv(yscale='linear',
                               out_filename='trajectories-all-cv.png')
    render_trajectories_all_cv(yscale='log',
                               out_filename='trajectories-all-cv-log.png')
    render_trajectories_all_cv(yscale='linear',
                               out_filename='trajectories-all-cv-full.png',
                               clip_negatives=False)

    # Per-cv-run zoomed trajectories (per-panel y-limits per ZOOM_YLIMS_USER
    # in eval.render). render_trajectories_zoomed writes phase5 PNGs; mirror
    # the random-σ variants here so overleaf stays in sync on refresh.
    from eval.render import render_trajectories_zoomed
    render_trajectories_zoomed()
    PHASE5 = V3_ROOT / 'results' / 'phase5'
    for src_name, dst_name in [
        ('trajectories_D_disc_cv_zoomed.png', 'trajectories-d-disc-cv-zoomed.png'),
        ('trajectories_D_logu_cv_zoomed.png', 'trajectories-d-logu-cv-zoomed.png'),
    ]:
        (EXP / dst_name).write_bytes((PHASE5 / src_name).read_bytes())
        print(f'wrote {EXP / dst_name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
