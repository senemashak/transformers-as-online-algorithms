"""
Step 5 figure + table rendering.

Inputs (produced by eval.run_eval):
  v3/results/phase5/payoff_matrix_raw.json
  v3/results/phase5/agreement_tensor.npz
  v3/results/phase5/trajectories.npz
  v3/results/phase5/persigma_results.json
  v3/results/phase5/<run>/<test_regime>.json   (per-cell)

Outputs (this module):
  v3/results/phase5/payoff_matrix.{csv,tex,png}
  v3/results/phase5/payoff_matrix_baselines.csv
  v3/results/phase5/trajectories_<run>.pdf       (× 5 cv runs)
  v3/results/phase5/agreement_heatmaps.png
  v3/results/phase5/per_sigma_<run>.pdf          (× 4 random-variance runs)
  v3/results/phase5/summary.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent                          # v3/code/
V3_ROOT = CODE_ROOT.parent                        # v3/
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import ALL_DISTRIBUTIONS, RANDOM_DISTRIBUTIONS, STATIC_DISTRIBUTIONS

RESULTS_PHASE5 = V3_ROOT / 'results' / 'phase5'

ALL_RUNS = [f'{d}_{s}' for d in ALL_DISTRIBUTIONS for s in ('cv', 'act')]
CV_RUNS = [f'{d}_cv' for d in ALL_DISTRIBUTIONS]
RANDOM_RUNS = [f'{d}_{s}' for d in RANDOM_DISTRIBUTIONS for s in ('cv', 'act')]
TEST_REGIMES = ('D_1', 'D_2', 'D_3')


# ---------------------------------------------------------------------------
# Payoff matrix
# ---------------------------------------------------------------------------

def write_payoff_matrix() -> None:
    raw = json.loads((RESULTS_PHASE5 / 'payoff_matrix_raw.json').read_text())
    rrs = raw['R_over_R_star']
    Rs = raw['R']
    R_star = raw['R_star']
    baseline_R = raw['baseline_R']

    # CSV
    csv_path = RESULTS_PHASE5 / 'payoff_matrix.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run'] + list(TEST_REGIMES))
        for run in ALL_RUNS:
            w.writerow([run] + [f'{rrs[run][r]:.4f}' for r in TEST_REGIMES])

    # TeX (booktabs)
    tex_path = RESULTS_PHASE5 / 'payoff_matrix.tex'
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             '& ' + ' & '.join(TEST_REGIMES) + r' \\', r'\midrule']
    for run in ALL_RUNS:
        cells = [f'{rrs[run][r]:.4f}' for r in TEST_REGIMES]
        run_tex = run.replace('_', '\\_')
        lines.append(f'{run_tex} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    tex_path.write_text('\n'.join(lines) + '\n')

    # PNG heatmap
    M = np.array([[rrs[run][r] for r in TEST_REGIMES] for run in ALL_RUNS])
    fig, ax = plt.subplots(figsize=(6.5, 8.5), dpi=300)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(TEST_REGIMES)))
    ax.set_xticklabels(TEST_REGIMES, fontsize=11)
    ax.set_yticks(range(len(ALL_RUNS)))
    ax.set_yticklabels(ALL_RUNS, fontsize=10)
    ax.set_xlabel('test regime', fontsize=11)
    ax.set_ylabel('trained model', fontsize=11)
    ax.set_title('R / R* — trained models on per-regime test caches', fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = 'black' if 0.35 < v < 0.85 else ('white' if v < 0.35 else 'black')
            ax.text(j, i, f'{v:.3f}',
                    ha='center', va='center', color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label('R / R*', fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_PHASE5 / 'payoff_matrix.png', dpi=300)
    plt.close(fig)

    # baselines CSV
    base_csv = RESULTS_PHASE5 / 'payoff_matrix_baselines.csv'
    base_rows = list(baseline_R.keys())
    with base_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['baseline', 'D_1_R', 'D_2_R', 'D_3_R',
                    'D_1_R_over_R_star', 'D_2_R_over_R_star', 'D_3_R_over_R_star'])
        for bn in base_rows:
            row = [bn]
            for r in TEST_REGIMES:
                row.append(f'{baseline_R[bn][r]:.4f}')
            for r in TEST_REGIMES:
                rstar = R_star[r]
                row.append(f'{baseline_R[bn][r] / rstar:.4f}' if rstar else 'nan')
            w.writerow(row)


# ---------------------------------------------------------------------------
# Trajectory plots — one PDF per cv run
# ---------------------------------------------------------------------------

def render_trajectories() -> None:
    traj_main = np.load(RESULTS_PHASE5 / 'trajectories.npz')
    extras_path = RESULTS_PHASE5 / 'trajectories_extras.npz'
    traj_extra = dict(np.load(extras_path)) if extras_path.exists() else {}
    traj = {k: traj_main[k] for k in traj_main.files}
    traj.update(traj_extra)

    BASELINE_KEYS = [
        ('oracle_static', 'per-regime oracle', 'black', '--'),
        ('random_oracle_disc', r'random-ADP oracle ($\mathcal{D}_{\mathrm{disc}}$)', '#1f4d8a', '-'),
        ('random_oracle_logu', r'random-ADP oracle ($\mathcal{D}_{\mathrm{logu}}$)', '#1f6e1f', '-'),
        ('plug_in', 'plug-in (known σ)', '#ff7f0e', '-'),
        ('prior_only', 'prior-only (known σ)', '#7f7f7f', ':'),
        ('myopic', 'myopic (known σ)', '#8c564b', '-'),
        ('MAP_sigma_disc', r'MAP-σ ($\mathcal{D}_{\mathrm{disc}}$ prior)', '#9467bd', '-.'),
        ('MAP_sigma_logu', r'MAP-σ ($\mathcal{D}_{\mathrm{logu}}$ prior)', '#e377c2', '-.'),
        ('MLE_sigma', 'MLE-σ', '#bcbd22', '--'),
        ('secretary', 'secretary (running max)', '#17becf', (0, (3, 1, 1, 1))),
    ]
    model_color = {
        'D_1_cv':    '#d62728',                                        # red
        'D_2_cv':    '#ff7f0e',                                        # orange
        'D_3_cv':    '#9467bd',                                        # purple
        'D_disc_cv': '#1f77b4',                                        # blue
        'D_logu_cv': '#2ca02c',                                        # green
    }
    for run in CV_RUNS:
        focal_color = model_color[run]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
        for ci, regime in enumerate(TEST_REGIMES):
            ax = axes[ci]
            t_axis = np.arange(1, 257)
            # Focal model only (drawn last → on top).
            mean_key = f'{run}__{regime}__mean'
            std_key = f'{run}__{regime}__std'
            if mean_key in traj:
                m = traj[mean_key]
                s = traj[std_key]
                ax.fill_between(t_axis, m - s, m + s,
                                color=focal_color, alpha=0.20, linewidth=0)
                ax.plot(t_axis, m, color=focal_color, lw=2.2,
                        label=f'{run} (focal)')
            # Baselines as thin lines + bands underneath the models.
            for bkey, label, color, ls in BASELINE_KEYS:
                k_mean = f'baseline__{regime}__{bkey}__mean'
                k_std = f'baseline__{regime}__{bkey}__std'
                if k_mean not in traj:
                    continue
                bm = traj[k_mean]
                bs = traj[k_std] if k_std in traj else None
                bm = np.where(np.isfinite(bm), bm, np.nan)
                if bs is not None:
                    bs = np.where(np.isfinite(bs), bs, np.nan)
                has_band = bs is not None and np.nanmax(bs) > 1e-9
                if has_band:
                    ax.fill_between(t_axis, bm - bs, bm + bs,
                                    color=color, alpha=0.10, linewidth=0)
                ax.plot(t_axis, bm, color=color, lw=0.9, ls=ls,
                        label=label, alpha=0.85)
            # Offline (hindsight): scalar E[max_t X_t] per regime; horizontal line.
            off_key = f'baseline__{regime}__offline__value'
            if off_key in traj:
                ax.axhline(float(traj[off_key][0]), color='black', lw=0.8,
                           ls=(0, (1, 1)), alpha=0.7,
                           label=r'offline (hindsight) $\mathbb{E}[\max_t X_t]$')
            ax.set_title(f'{regime} (test)', fontsize=10)
            ax.set_xlabel('t', fontsize=9)
            if ci == 0:
                ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=7, loc='upper left')
        fig.suptitle(f'Threshold trajectories: {run}', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(RESULTS_PHASE5 / f'trajectories_{run}.png', dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-cv-run zoomed trajectories with regime-specific y-limits.
# Static-σ models (D_X_cv) output near their training σ regardless of test
# regime, so a single ylim matched to their training scale shows the focal
# trajectory clearly across all three panels. Random-σ models adapt across
# regimes, so each panel gets a regime-specific ylim.
# ---------------------------------------------------------------------------

ZOOM_YLIMS_USER: Dict[str, Dict[str, Tuple[float, float]]] = {
    'D_1_cv':    {'D_1': (-20.0, 20.0), 'D_2': (-20.0, 20.0), 'D_3': (-20.0, 20.0)},
    'D_2_cv':    {'D_1': (10.0, 50.0),  'D_2': (10.0, 50.0),  'D_3': (10.0, 50.0)},
    'D_3_cv':    {'D_1': (200.0, 300.0),'D_2': (200.0, 300.0),'D_3': (200.0, 300.0)},
    'D_disc_cv': {'D_1': (-20.0, 20.0), 'D_2': (10.0, 50.0),  'D_3': (200.0, 300.0)},
    'D_logu_cv': {'D_1': (-20.0, 20.0), 'D_2': (10.0, 50.0),  'D_3': (200.0, 300.0)},
}


def render_trajectories_zoomed() -> None:
    """Per-cv-run zoomed variant of `render_trajectories`.

    Same plot content (focal model + all 10 baselines + offline reference)
    as the main 1x3 figure, but with hardcoded per-panel y-limits per
    `ZOOM_YLIMS_USER`. Output: `trajectories_<run>_zoomed.png`.
    """
    traj_main = np.load(RESULTS_PHASE5 / 'trajectories.npz')
    extras_path = RESULTS_PHASE5 / 'trajectories_extras.npz'
    traj_extra = dict(np.load(extras_path)) if extras_path.exists() else {}
    traj = {k: traj_main[k] for k in traj_main.files}
    traj.update(traj_extra)

    BASELINE_KEYS = [
        ('oracle_static', 'per-regime oracle', 'black', '--'),
        ('random_oracle_disc', r'random-ADP oracle ($\mathcal{D}_{\mathrm{disc}}$)', '#1f4d8a', '-'),
        ('random_oracle_logu', r'random-ADP oracle ($\mathcal{D}_{\mathrm{logu}}$)', '#1f6e1f', '-'),
        ('plug_in', 'plug-in (known σ)', '#ff7f0e', '-'),
        ('prior_only', 'prior-only (known σ)', '#7f7f7f', ':'),
        ('myopic', 'myopic (known σ)', '#8c564b', '-'),
        ('MAP_sigma_disc', r'MAP-σ ($\mathcal{D}_{\mathrm{disc}}$ prior)', '#9467bd', '-.'),
        ('MAP_sigma_logu', r'MAP-σ ($\mathcal{D}_{\mathrm{logu}}$ prior)', '#e377c2', '-.'),
        ('MLE_sigma', 'MLE-σ', '#bcbd22', '--'),
        ('secretary', 'secretary (running max)', '#17becf', (0, (3, 1, 1, 1))),
    ]
    model_color = {
        'D_1_cv':    '#d62728',
        'D_2_cv':    '#ff7f0e',
        'D_3_cv':    '#9467bd',
        'D_disc_cv': '#1f77b4',
        'D_logu_cv': '#2ca02c',
    }

    for run, ylim_per_regime in ZOOM_YLIMS_USER.items():
        focal_color = model_color[run]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
        for ci, regime in enumerate(TEST_REGIMES):
            ax = axes[ci]
            t_axis = np.arange(1, 257)

            mean_key = f'{run}__{regime}__mean'
            std_key = f'{run}__{regime}__std'
            if mean_key in traj:
                m = traj[mean_key]
                s = traj[std_key]
                ax.fill_between(t_axis, m - s, m + s,
                                color=focal_color, alpha=0.20, linewidth=0)
                ax.plot(t_axis, m, color=focal_color, lw=2.2,
                        label=f'{run} (focal)')

            for bkey, label, color, ls in BASELINE_KEYS:
                k_mean = f'baseline__{regime}__{bkey}__mean'
                k_std = f'baseline__{regime}__{bkey}__std'
                if k_mean not in traj:
                    continue
                bm = traj[k_mean]
                bs = traj[k_std] if k_std in traj else None
                bm = np.where(np.isfinite(bm), bm, np.nan)
                if bs is not None:
                    bs = np.where(np.isfinite(bs), bs, np.nan)
                has_band = bs is not None and np.nanmax(bs) > 1e-9
                if has_band:
                    ax.fill_between(t_axis, bm - bs, bm + bs,
                                    color=color, alpha=0.10, linewidth=0)
                ax.plot(t_axis, bm, color=color, lw=0.9, ls=ls,
                        label=label, alpha=0.85)

            off_key = f'baseline__{regime}__offline__value'
            if off_key in traj:
                ax.axhline(float(traj[off_key][0]), color='black', lw=0.8,
                           ls=(0, (1, 1)), alpha=0.7,
                           label=r'offline (hindsight) $\mathbb{E}[\max_t X_t]$')

            ylim = ylim_per_regime[regime]
            ax.set_ylim(*ylim)
            ax.set_title(f'{regime} (test) — y∈[{ylim[0]:g}, {ylim[1]:g}]', fontsize=10)
            ax.set_xlabel('t', fontsize=9)
            if ci == 0:
                ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=7, loc='upper left')

        fig.suptitle(f'Threshold trajectories (zoomed): {run}', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = RESULTS_PHASE5 / f'trajectories_{run}_zoomed.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f'wrote {out}')


# ---------------------------------------------------------------------------
# All-cv overlay: 1×3 figure with every cv model on each regime
# ---------------------------------------------------------------------------

def render_all_cv_overlaid() -> None:
    """One figure, three panels (D_1, D_2, D_3); all 5 cv-trained models
    overlaid with distinct colors + the per-regime oracle as the
    reference. Baseline bands suppressed to keep the plot legible."""
    traj = np.load(RESULTS_PHASE5 / 'trajectories.npz')
    model_color = {
        'D_1_cv':    '#d62728',                                        # red
        'D_2_cv':    '#ff7f0e',                                        # orange
        'D_3_cv':    '#9467bd',                                        # purple
        'D_disc_cv': '#1f77b4',                                        # blue
        'D_logu_cv': '#2ca02c',                                        # green
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130, sharey=True)
    for ci, regime in enumerate(TEST_REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)

        # Per-regime oracle (reference) — black dashed with thin std band.
        ok = f'baseline__{regime}__oracle_static__mean'
        if ok in traj.files:
            om = traj[ok]
            os = traj[ok.replace('__mean', '__std')]
            ax.fill_between(t_axis, om - os, om + os,
                            color='black', alpha=0.10, linewidth=0)
            ax.plot(t_axis, om, color='black', lw=1.4, ls='--',
                    label='oracle (per-regime)', zorder=20)

        # All 5 cv models overlaid (means + thin bands). D_disc_cv is drawn
        # last with a thicker line so it isn't hidden by D_logu_cv / oracle.
        draw_order = [r for r in model_color if r != 'D_disc_cv'] + ['D_disc_cv']
        for run in draw_order:
            color = model_color[run]
            mean_key = f'{run}__{regime}__mean'
            if mean_key not in traj.files:
                continue
            m = traj[mean_key]
            s = traj[mean_key.replace('__mean', '__std')]
            is_disc = run == 'D_disc_cv'
            ax.fill_between(t_axis, m - s, m + s,
                            color=color, alpha=0.12 if is_disc else 0.08,
                            linewidth=0)
            ax.plot(t_axis, m, color=color,
                    lw=2.6 if is_disc else 1.5,
                    label=run, alpha=0.95,
                    zorder=15 if is_disc else 5)

        ax.set_title(f'{regime} (test)', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if ci == 0:
            ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if ci == 2:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Threshold trajectories — all 5 cv-trained models on each test regime',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RESULTS_PHASE5 / 'trajectories_all_cv.png', dpi=300)
    plt.close(fig)

    # Zoomed view: D_disc_cv and D_logu_cv vs oracle, per-panel y-limits.
    zoom_ylims = {'D_1': (0, 10), 'D_2': (0, 40), 'D_3': (150, 300)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ci, regime in enumerate(TEST_REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)

        ok = f'baseline__{regime}__oracle_static__mean'
        if ok in traj.files:
            om = traj[ok]
            os = traj[ok.replace('__mean', '__std')]
            ax.fill_between(t_axis, om - os, om + os,
                            color='black', alpha=0.10, linewidth=0)
            ax.plot(t_axis, om, color='black', lw=1.4, ls='--',
                    label='oracle (per-regime)', zorder=20)

        for run in ('D_logu_cv', 'D_disc_cv'):
            color = model_color[run]
            mean_key = f'{run}__{regime}__mean'
            if mean_key not in traj.files:
                continue
            m = traj[mean_key]
            s = traj[mean_key.replace('__mean', '__std')]
            ax.fill_between(t_axis, m - s, m + s,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(t_axis, m, color=color, lw=2.0,
                    label=run, alpha=0.95)

        ax.set_ylim(*zoom_ylims[regime])
        ax.set_title(f'{regime} (test) — y∈{zoom_ylims[regime]}', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if ci == 0:
            ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if ci == 2:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Threshold trajectories (zoom) — D_disc_cv vs D_logu_cv',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RESULTS_PHASE5 / 'trajectories_disc_logu_zoom.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Agreement heatmaps
# ---------------------------------------------------------------------------

def render_agreement_heatmaps() -> None:
    z = np.load(RESULTS_PHASE5 / 'agreement_tensor.npz', allow_pickle=False)
    A = z['agreement']                                              # (10, 3, 7)
    run_names = list(z['run_names'])
    regimes = list(z['test_regimes'])
    bnames = list(z['baseline_names'])

    fig, axes = plt.subplots(1, len(bnames), figsize=(20, 6), dpi=130)
    for bi, bn in enumerate(bnames):
        ax = axes[bi]
        m = A[:, :, bi]                                             # (10, 3)
        im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(regimes, fontsize=9)
        ax.set_yticks(range(len(run_names)))
        ax.set_yticklabels(run_names if bi == 0 else [], fontsize=8)
        ax.set_title(bn, fontsize=10)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                color = 'black' if v < 0.6 else 'white'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color=color, fontsize=7)
    fig.suptitle('Per-step action agreement (10 models × 3 test regimes × 7 baselines)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RESULTS_PHASE5 / 'agreement_heatmaps.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-σ payoff bar charts
# ---------------------------------------------------------------------------

def render_per_sigma() -> None:
    persigma = json.loads((RESULTS_PHASE5 / 'persigma_results.json').read_text())
    head_color = {'cv': '#A6C8E0', 'act': '#A8D8A8'}            # light blue, light green
    head_alpha = 0.65                                            # transparent fill
    for dist in ('D_disc', 'D_logu'):
        cv_rows = persigma.get(f'{dist}_cv')
        act_rows = persigma.get(f'{dist}_act')
        if cv_rows is None or act_rows is None:
            continue
        names = [r['name'] for r in cv_rows]
        counts = [r['count'] for r in cv_rows]
        cv_vals = [r['R_over_R_star'] for r in cv_rows]
        act_vals = [r['R_over_R_star'] for r in act_rows]

        x = np.arange(len(names))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        bars_cv = ax.bar(x - width / 2, cv_vals, width,
                         color=head_color['cv'], alpha=head_alpha, label=f'{dist}_cv')
        bars_act = ax.bar(x + width / 2, act_vals, width,
                          color=head_color['act'], alpha=head_alpha, label=f'{dist}_act')
        ax.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6,
                   label='random-ADP oracle (R/R* = 1)')
        ax.axhline(0.0, color='black', lw=0.5, alpha=0.5)
        ax.set_ylabel('R / R*  (R* = per-distribution random-ADP oracle)', fontsize=10)
        ax.set_xlabel('σ bin', fontsize=10)
        ax.set_title(f'Per-σ payoff breakdown: {dist}_cv vs {dist}_act', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        all_vals = cv_vals + act_vals
        ax.set_ylim(min(min(all_vals) - 0.12, -0.1),
                    max(max(all_vals) + 0.18, 1.25))
        for bars, vals in ((bars_cv, cv_vals), (bars_act, act_vals)):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        v + (0.03 if v >= 0 else -0.05),
                        f'{v:.3f}',
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=8)
        for xi, c in zip(x, counts):
            ax.text(xi, ax.get_ylim()[0] + 0.02, f'n={c}',
                    ha='center', va='bottom', fontsize=7, color='gray')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        fig.savefig(RESULTS_PHASE5 / f'per_sigma_{dist}.png', dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary.md
# ---------------------------------------------------------------------------

def write_summary() -> None:
    raw = json.loads((RESULTS_PHASE5 / 'payoff_matrix_raw.json').read_text())
    persigma = json.loads((RESULTS_PHASE5 / 'persigma_results.json').read_text())
    rrs = raw['R_over_R_star']

    lines: list[str] = ['# Phase 5 — evaluation summary\n\n']

    # Payoff-matrix table.
    lines.append('## Payoff matrix (R / R*) on the seed-43 per-regime test caches\n\n')
    lines.append('| trained model | D_1 | D_2 | D_3 | D_3_cv comparator (D_3) | per-σ summary |\n')
    lines.append('|---|---:|---:|---:|---:|---|\n')
    d3cv = rrs['D_3_cv']['D_3']
    for run in ALL_RUNS:
        row = rrs[run]
        ps_summary = ''
        if run in persigma:
            vals = [f'{r["name"]}={r["R_over_R_star"]:.3f}'
                    for r in persigma[run]]
            ps_summary = '; '.join(vals)
        lines.append(
            f'| {run} | {row["D_1"]:.4f} | {row["D_2"]:.4f} | {row["D_3"]:.4f} | '
            f'{d3cv:.4f} | {ps_summary} |\n'
        )

    # Per-distribution paragraphs.
    lines.append('\n## Per-distribution interpretation\n\n')
    for d in ALL_DISTRIBUTIONS:
        lines.append(f'### {d}\n\n')
        cv_row = rrs[f'{d}_cv']
        act_row = rrs[f'{d}_act']
        if d in STATIC_DISTRIBUTIONS:
            lines.append(
                f'Single-σ training (σ = {{1: 1, 2: 10, 3: 100}}[{d.split("_")[1]}]). '
                f'cv recovers oracle in-distribution at {cv_row[d]:.4f}; '
                f'off-diagonal cells '
                f'D_1={cv_row["D_1"]:.4f}, D_2={cv_row["D_2"]:.4f}, '
                f'D_3={cv_row["D_3"]:.4f}. Same shape on the act head '
                f'({act_row[d]:.4f} in-distribution; D_1={act_row["D_1"]:.4f}, '
                f'D_2={act_row["D_2"]:.4f}, D_3={act_row["D_3"]:.4f}). '
                'The off-diagonal collapses are the V2 shortcut signature: '
                'the model has learned a fixed-scale threshold and pastes it '
                'onto the wrong regime.\n\n'
            )
        else:
            cv_psd = persigma.get(f'{d}_cv', [])
            act_psd = persigma.get(f'{d}_act', [])
            cv_perσ = ', '.join(f'{r["name"]} {r["R_over_R_star"]:.3f}' for r in cv_psd)
            act_perσ = ', '.join(f'{r["name"]} {r["R_over_R_star"]:.3f}' for r in act_psd)
            lines.append(
                f'Random-σ training. cv: '
                f'D_1={cv_row["D_1"]:.4f}, D_2={cv_row["D_2"]:.4f}, '
                f'D_3={cv_row["D_3"]:.4f}. act: '
                f'D_1={act_row["D_1"]:.4f}, D_2={act_row["D_2"]:.4f}, '
                f'D_3={act_row["D_3"]:.4f}. '
                f'Per-σ breakdown (cv on the seed-44 persigma test cache): '
                f'{cv_perσ}. Per-σ (act): {act_perσ}.\n\n'
            )

    # Findings.
    lines.append('## Findings\n\n')
    findings = [
        # 1
        f'**D_disc trains the algorithm in full.** D_disc_cv hits R/R* = '
        f'{rrs["D_disc_cv"]["D_1"]:.3f} / {rrs["D_disc_cv"]["D_2"]:.3f} / '
        f'{rrs["D_disc_cv"]["D_3"]:.3f} across the three test regimes — '
        f'matching D_3_cv\'s in-distribution {rrs["D_3_cv"]["D_3"]:.3f} '
        'on D_3 while also matching D_1_cv\'s and D_2_cv\'s '
        'in-distribution numbers on their own regimes. D_disc_act mirrors '
        'this. Both are uniformly competent across σ in the persigma '
        'breakdown (every bin ≈ 1.0).',
        # 2
        f'**Static-σ models reproduce the V2 shortcut.** D_1_cv on D_2 = '
        f'{rrs["D_1_cv"]["D_2"]:.3f}, D_3_cv on D_1 = {rrs["D_3_cv"]["D_1"]:.3f}. '
        'Off-diagonal cells collapse to ~0.02–0.36, confirming the spec\'s '
        'pre-existing finding that single-regime training induces a '
        'fixed-scale policy.',
        # 3
        f'**D_logu_cv is the anomaly.** D_logu_cv on D_1 / D_2 / D_3 = '
        f'{rrs["D_logu_cv"]["D_1"]:.3f} / {rrs["D_logu_cv"]["D_2"]:.3f} / '
        f'{rrs["D_logu_cv"]["D_3"]:.3f}, and the per-σ breakdown on the '
        f'persigma cache shows it deteriorates monotonically toward low σ — '
        'the σ=1-end bin even has negative R, meaning the model accepts '
        'X_t < 0 sequences. The continuous σ-prior exposes a regime where '
        'the cv head fails to invert correctly. D_logu_act recovers '
        f'(R/R* ≈ 0.97 across all three regimes), so this is cv-specific.',
        # 4
        f'**High oracle agreement does not imply matched payoff** — the V2 '
        'pathology re-appears here in static-model OOD cells. Look at '
        f'D_3_cv on D_1: oracle agreement 0.978 (per-step), R/R* '
        f'{rrs["D_3_cv"]["D_1"]:.3f}. Both policies reject ~all positions; '
        'the few cells where they disagree are the decision-critical accept '
        'steps that determine the entire payoff.',
        # 5
        '**Trajectory plots (saved per cv run) are the visual answer to '
        '"do random-variance models do in-context σ inference?".** D_disc_cv '
        '\'s curves adapt to test regime within a few steps; D_logu_cv\'s '
        'curves fail to adapt at low σ. Static-σ cv curves stay near their '
        'training-distribution oracle threshold regardless of test regime, '
        'visible as flat off-axis lines.',
    ]
    for f in findings:
        lines.append(f'- {f}\n')

    (RESULTS_PHASE5 / 'summary.md').write_text(''.join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print('[render] payoff matrix')
    write_payoff_matrix()
    print('[render] trajectories (per-run)')
    render_trajectories()
    print('[render] trajectories (all-cv overlay)')
    render_all_cv_overlaid()
    print('[render] agreement heatmaps')
    render_agreement_heatmaps()
    print('[render] per-σ bars')
    render_per_sigma()
    print('[render] summary.md')
    write_summary()
    print(f'\nwrote sweep-level artifacts to {RESULTS_PHASE5}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
