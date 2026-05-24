"""Render σ¹ vs σ² ablation figures into results/phase5/.

Outputs:
    payoff_matrix_with_sig2.png         — full 12-row R/R* heatmap (σ¹ rows + σ² rows).
    agreement_oracle_with_sig2.png       — 12-row × 3-regime oracle-agreement heatmap.
    per_sigma_D_disc_sig2_compare.png    — grouped bars: σ¹ cv vs σ² cv vs σ¹ act per σ-bin.
    per_sigma_D_logu_sig2_compare.png    — same for D_logu.
    trajectories_D_disc_cv_sig2.png      — per-model trajectory, full §4 portfolio.
    trajectories_D_logu_cv_sig2.png      — same for D_logu.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

RESULTS_PHASE5 = V3_ROOT / 'results' / 'phase5'
TEST_REGIMES = ('D_1', 'D_2', 'D_3')
SIG2_RUNS = ('D_disc_cv_sig2', 'D_logu_cv_sig2')


def render_payoff_matrix_with_sig2() -> None:
    """Heatmap of R/R* combining σ¹ rows + σ² rows."""
    rows: list = []
    with (RESULTS_PHASE5 / 'payoff_matrix.csv').open() as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append((row[0], [float(x) for x in row[1:]]))
    with (RESULTS_PHASE5 / 'payoff_matrix_sig2.csv').open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows.append((row[0], [float(x) for x in row[1:]]))
    names = [r[0] for r in rows]
    M = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 1 + 0.45 * len(names)), dpi=130)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(TEST_REGIMES)))
    ax.set_xticklabels(TEST_REGIMES, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = 'black' if v < 0.6 else 'white'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color=color, fontsize=9)
    ax.set_title(r'$R/R^\star$  (per-regime static-σ oracle denominator)',
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=r'$R/R^\star$')
    fig.tight_layout()
    out = RESULTS_PHASE5 / 'payoff_matrix_with_sig2.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')


def render_agreement_oracle_with_sig2() -> None:
    """Oracle-column heatmap, 12 rows."""
    z1 = np.load(RESULTS_PHASE5 / 'agreement_tensor.npz', allow_pickle=False)
    z2 = np.load(RESULTS_PHASE5 / 'agreement_tensor_sig2.npz', allow_pickle=False)
    bn = list(z1['baseline_names'])
    oi = bn.index('oracle')
    runs = list(z1['run_names']) + list(z2['run_names'])
    A = np.concatenate([z1['agreement'][:, :, oi], z2['agreement'][:, :, oi]], axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 1 + 0.4 * len(runs)), dpi=130)
    im = ax.imshow(A, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(TEST_REGIMES)))
    ax.set_xticklabels(TEST_REGIMES, fontsize=10)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=9)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            color = 'black' if v < 0.6 else 'white'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color=color, fontsize=9)
    ax.set_title('Per-step action agreement vs. per-regime oracle', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='agreement')
    fig.tight_layout()
    out = RESULTS_PHASE5 / 'agreement_oracle_with_sig2.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')


def render_per_sigma_compare() -> None:
    """Grouped bars: σ¹ cv vs σ² cv vs σ¹ act per σ-bin, for each distribution."""
    sig1 = json.loads((RESULTS_PHASE5 / 'persigma_results.json').read_text())
    sig2 = json.loads((RESULTS_PHASE5 / 'persigma_results_sig2.json').read_text())

    color = {
        'sig1_cv':  '#4C72B0',     # blue
        'sig2_cv':  '#9467bd',     # purple
        'sig1_act': '#DD8452',     # warm orange
    }

    for dist in ('D_disc', 'D_logu'):
        rows_sig1_cv = sig1.get(f'{dist}_cv', [])
        rows_sig2_cv = sig2.get(f'{dist}_cv_sig2', [])
        rows_sig1_act = sig1.get(f'{dist}_act', [])
        if not (rows_sig1_cv and rows_sig2_cv and rows_sig1_act):
            print(f'  skipping {dist} — missing rows')
            continue

        names = [r['name'] for r in rows_sig1_cv]
        cv1 = [r['R_over_R_star'] for r in rows_sig1_cv]
        cv2 = [r['R_over_R_star'] for r in rows_sig2_cv]
        act = [r['R_over_R_star'] for r in rows_sig1_act]
        counts = [r['count'] for r in rows_sig1_cv]

        x = np.arange(len(names))
        width = 0.27
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=130)
        bars_cv1 = ax.bar(x - width, cv1, width, color=color['sig1_cv'],
                          alpha=0.9, label=f'{dist}_cv (σ¹)')
        bars_cv2 = ax.bar(x, cv2, width, color=color['sig2_cv'],
                          alpha=0.9, label=f'{dist}_cv_sig2 (σ²)')
        bars_act = ax.bar(x + width, act, width, color=color['sig1_act'],
                          alpha=0.9, label=f'{dist}_act (σ¹)')

        ax.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6,
                   label='random-ADP oracle (R/R* = 1)')
        ax.axhline(0.0, color='black', lw=0.5, alpha=0.5)
        ax.set_ylabel(r'$R/R^\star$  (per-distribution random-ADP oracle)',
                      fontsize=10)
        ax.set_xlabel('σ bin', fontsize=10)
        ax.set_title(f'Per-σ payoff breakdown — σ¹ vs σ² ablation: {dist}',
                     fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        all_vals = cv1 + cv2 + act
        ax.set_ylim(min(min(all_vals) - 0.12, -0.1),
                    max(max(all_vals) + 0.18, 1.25))
        for bars, vals in ((bars_cv1, cv1), (bars_cv2, cv2), (bars_act, act)):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        v + (0.025 if v >= 0 else -0.05),
                        f'{v:.3f}',
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=8)
        for xi, c in zip(x, counts):
            ax.text(xi, ax.get_ylim()[0] + 0.02, f'n={c}',
                    ha='center', va='bottom', fontsize=7, color='gray')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        out = RESULTS_PHASE5 / f'per_sigma_{dist}_sig2_compare.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f'wrote {out}')


def render_trajectories_sig2() -> None:
    """Per-model trajectory plots for the sig2 runs, full §4 portfolio."""
    traj_main = np.load(RESULTS_PHASE5 / 'trajectories.npz')
    traj_sig2 = np.load(RESULTS_PHASE5 / 'trajectories_sig2.npz')
    extras = (np.load(RESULTS_PHASE5 / 'trajectories_extras.npz')
              if (RESULTS_PHASE5 / 'trajectories_extras.npz').exists() else {})

    traj = {k: traj_main[k] for k in traj_main.files}
    for k in traj_sig2.files:
        traj[k] = traj_sig2[k]
    if hasattr(extras, 'files'):
        for k in extras.files:
            traj[k] = extras[k]

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

    focal_color = {'D_disc_cv_sig2': '#9467bd', 'D_logu_cv_sig2': '#2ca02c'}

    for run in SIG2_RUNS:
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
                                color=focal_color[run], alpha=0.20, linewidth=0)
                ax.plot(t_axis, m, color=focal_color[run], lw=2.2,
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

            ax.set_title(f'{regime} (test)', fontsize=10)
            ax.set_xlabel('t', fontsize=9)
            if ci == 0:
                ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
            ax.grid(True, alpha=0.3)
            if ci == 2:
                ax.legend(fontsize=7, loc='upper right')
        fig.suptitle(f'Threshold trajectories: {run}  (σ² ablation)', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = RESULTS_PHASE5 / f'trajectories_{run}.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f'wrote {out}')


def main() -> int:
    render_payoff_matrix_with_sig2()
    render_agreement_oracle_with_sig2()
    render_per_sigma_compare()
    render_trajectories_sig2()
    return 0


if __name__ == '__main__':
    sys.exit(main())
