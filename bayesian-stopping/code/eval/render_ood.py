"""Renderers for Experiment A (OOD on σ).

Produces:
  results/ood/payoff-matrix-extended.png          — 7 rows × 10 cols
  results/ood/trajectories-d-disc-cv-ood.png       — 1×4 OOD trajectories
  results/ood/trajectories-d-logu-cv-ood.png       — 1×4 OOD trajectories
  results/ood/summary_table.csv                    — focused per-σ comparison

Also copies the figures into report/figures/ with names matching the
naming convention used by the rest of the paper (-extended suffix).

Reads from:
  results/ood/payoff_matrix_raw.json               (OOD R/R*, R, R*, baseline R)
  results/ood/trajectories.npz                     (OOD focal + baseline traj)
  results/phase5/payoff_matrix.csv                 (σ¹ runs σ=1,10,100)
  results/phase5/payoff_matrix_sig2.csv            (σ² runs σ=1,10,100)
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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

OOD_ROOT = V3_ROOT / 'results' / 'ood'
PHASE5 = V3_ROOT / 'results' / 'phase5'
OVERLEAF = V3_ROOT / 'report' / 'figures'

# Display order: by σ, with OOD interleaved.
DISPLAY_REGIMES: List[Tuple[str, float, str]] = [
    ('D_ood_0p3', 0.3,   r'$\sigma{=}0.3$'),
    ('D_1',       1.0,   r'$\sigma{=}1$'),
    ('D_ood_3',   3.0,   r'$\sigma{=}3$'),
    ('D_2',       10.0,  r'$\sigma{=}10$'),
    ('D_ood_30',  30.0,  r'$\sigma{=}30$'),
    ('D_3',       100.0, r'$\sigma{=}100$'),
    ('D_ood_300', 300.0, r'$\sigma{=}300$'),
]
ALL_RUNS = [
    'D_1_cv','D_1_act','D_2_cv','D_2_act','D_3_cv','D_3_act',
    'D_disc_cv','D_disc_act','D_logu_cv','D_logu_act',
]

# Diverging colormap whose pink/green endpoints match the level of pastel
# transparency of the threshold-trajectory shading (the trajectory shading is
# color=#2ca02c (and #1f77b4) at alpha=0.20, visually equivalent to ≈#d5ecd5).
# Green endpoint here is the trajectory green at a slightly higher alpha; pink
# endpoint is set to a complementary pale rose at the same lightness level.
SOFT_PIYG_CMAP = LinearSegmentedColormap.from_list(
    'SoftPiYG', ['#e08fa8', '#f4d4dd', '#ffffff', '#d8ecd2', '#90c890'])

# Display labels for run names in figures (cv → cv, act → act).
RUN_LABELS = {r: r for r in ALL_RUNS}


# Baseline lines to overlay on OOD trajectory plots; full canonical set
# (matches the BASELINE_KEYS palette used in eval.render).
#
# Tuple: (key, label, color, linestyle, emphasis).
# emphasis=True -> thick line, full alpha, high zorder, distinctive long-dash.
# Random-ADP oracles are the primary comparison target -> emphasised.
OOD_TRAJ_BASELINES = [
    ('oracle_static',       'per-regime oracle',                                     'black',   '--',                False),
    ('random_oracle_disc',  r'random-ADP oracle ($\mathcal{D}_{\mathrm{disc}}$)',     '#0b3d91', (0, (10, 2)),       True),
    ('random_oracle_logu',  r'random-ADP oracle ($\mathcal{D}_{\mathrm{logu}}$)',     '#0a6e0a', (0, (10, 2, 2, 2)), True),
    ('plug_in',             r'plug-in (known $\sigma$)',                              '#ff7f0e', '-',                 False),
    ('prior_only',          r'prior-only (known $\sigma$)',                           '#7f7f7f', ':',                 False),
    ('myopic',              r'myopic (known $\sigma$)',                               '#8c564b', '-',                 False),
    ('MAP_sigma_disc',      r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{disc}}$ prior)',    '#9467bd', '-.',                False),
    ('MAP_sigma_logu',      r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{logu}}$ prior)',    '#e377c2', '-.',                False),
    ('MLE_sigma',           r'MLE-$\sigma$',                                          '#bcbd22', '--',                False),
    ('secretary',           'secretary (running max)',                                '#17becf', (0, (3,1,1,1)),     False),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_existing_R_over_R_star() -> Dict[str, Dict[str, float]]:
    """For σ=1,10,100 cells, use the σ²-swapped CSV view (matches the
    canonical payoff-matrix figure and §6 numbers).
    Returns: {run_name: {regime: R/R*}}.
    """
    sig1 = PHASE5 / 'payoff_matrix.csv'
    sig2 = PHASE5 / 'payoff_matrix_sig2.csv'
    swap = {'D_disc_cv': 'D_disc_cv_sig2', 'D_logu_cv': 'D_logu_cv_sig2'}

    def _read(p: Path) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        with p.open() as f:
            r = csv.reader(f); header = next(r)
            for row in r:
                out[row[0]] = [float(x) for x in row[1:]]
        return out

    sig1_rows = _read(sig1)                                     # keys = ALL_RUNS
    sig2_rows = _read(sig2)                                     # keys = D_disc_cv_sig2, D_logu_cv_sig2

    out: Dict[str, Dict[str, float]] = {}
    regimes_existing = ('D_1', 'D_2', 'D_3')
    for run in ALL_RUNS:
        src = sig2_rows[swap[run]] if run in swap else sig1_rows[run]
        out[run] = dict(zip(regimes_existing, src))
    return out


def _load_ood_R_over_R_star() -> Dict[str, Dict[str, float]]:
    raw = json.loads((OOD_ROOT / 'payoff_matrix_raw.json').read_text())
    return raw['R_over_R_star']                                 # {run: {regime: R/R*}}


def _load_ood_baseline_R() -> Dict[str, Dict[str, float]]:
    raw = json.loads((OOD_ROOT / 'payoff_matrix_raw.json').read_text())
    return raw['baseline_R']                                    # {regime: {bn: R}}


def _load_ood_R_star() -> Dict[str, float]:
    raw = json.loads((OOD_ROOT / 'payoff_matrix_raw.json').read_text())
    return raw['R_star']                                        # {regime: R*}


# ---------------------------------------------------------------------------
# Extended payoff matrix
# ---------------------------------------------------------------------------

def render_extended_payoff_matrix() -> None:
    existing = _load_existing_R_over_R_star()
    ood = _load_ood_R_over_R_star()

    # M shape: (n_regimes, n_runs).
    n_reg = len(DISPLAY_REGIMES)
    n_run = len(ALL_RUNS)
    M = np.zeros((n_reg, n_run))
    for i, (regime, _, _) in enumerate(DISPLAY_REGIMES):
        for j, run in enumerate(ALL_RUNS):
            if regime in existing[run]:
                M[i, j] = existing[run][regime]
            else:
                M[i, j] = ood[run][regime]

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=130)
    im = ax.imshow(M, vmin=-0.25, vmax=1.0, cmap=SOFT_PIYG_CMAP, aspect='auto')
    ax.set_xticks(range(n_run))
    ax.set_xticklabels([RUN_LABELS[r] for r in ALL_RUNS], fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels([d[2] for d in DISPLAY_REGIMES], fontsize=10)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='black', fontsize=8)

    ax.set_title(r'$R/R^\star$  (per-regime static-$\sigma$ oracle; OOD $\sigma$ rows in italics)',
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r'$R/R^\star$')

    # Italicize OOD row labels by overriding the y-tick labels with formatted text.
    yticklabels = []
    for regime, _, lbl in DISPLAY_REGIMES:
        if regime.startswith('D_ood'):
            yticklabels.append(lbl + ' (OOD)')
        else:
            yticklabels.append(lbl)
    ax.set_yticklabels(yticklabels, fontsize=10)

    fig.tight_layout()
    out = OOD_ROOT / 'payoff-matrix-extended.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    # Also copy to report.
    fig_overleaf = OVERLEAF / 'payoff-matrix-extended.png'
    fig_overleaf.parent.mkdir(parents=True, exist_ok=True)
    fig_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {fig_overleaf}')


# ---------------------------------------------------------------------------
# OOD trajectory plots
# ---------------------------------------------------------------------------

def _render_ood_trajectories_one(focal_run: str, focal_label: str, focal_color: str,
                                 out_filename: str) -> None:
    """2×2 trajectory figure on OOD σ ∈ {0.3, 3, 30, 300} for one focal model.
    The in-distribution trajectories are reported separately via
    `refresh_overleaf_with_sig2.render_trajectory_for_run`."""
    traj_ood = dict(np.load(OOD_ROOT / 'trajectories.npz', allow_pickle=False))

    OOD_REGIMES = [r for r, _, _ in DISPLAY_REGIMES if r.startswith('D_ood')]
    OOD_LABELS  = {r: lbl for r, _, lbl in DISPLAY_REGIMES if r.startswith('D_ood')}

    fig, axes_grid = plt.subplots(2, 2, figsize=(13, 7.5), dpi=130)
    axes = axes_grid.flatten()
    for ci, regime in enumerate(OOD_REGIMES):
        ax = axes[ci]
        t_axis = np.arange(1, 257)

        m = traj_ood[f'{focal_run}__{regime}__mean']
        s = traj_ood[f'{focal_run}__{regime}__std']
        ax.fill_between(t_axis, m - s, m + s, color=focal_color, alpha=0.20, linewidth=0)
        ax.plot(t_axis, m, color=focal_color, lw=2.2, label=f'{focal_label} (focal)')

        data_baselines = traj_ood
        for bkey, label, color, ls, emphasis in OOD_TRAJ_BASELINES:
            k_mean = f'baseline__{regime}__{bkey}__mean'
            k_std = f'baseline__{regime}__{bkey}__std'
            if k_mean not in data_baselines:
                continue
            bm = data_baselines[k_mean]; bs = data_baselines.get(k_std, None)
            bm = np.where(np.isfinite(bm), bm, np.nan)
            if bs is not None:
                bs = np.where(np.isfinite(bs), bs, np.nan)
            has_band = bs is not None and np.nanmax(bs) > 1e-9
            if emphasis:
                line_lw, line_alpha, line_zorder = 2.4, 1.00, 15
                band_alpha = 0.18
            else:
                line_lw, line_alpha, line_zorder = 0.9, 0.85, 5
                band_alpha = 0.10
            if has_band:
                ax.fill_between(t_axis, bm - bs, bm + bs, color=color,
                                alpha=band_alpha, linewidth=0)
            ax.plot(t_axis, bm, color=color, lw=line_lw, ls=ls,
                    label=label, alpha=line_alpha, zorder=line_zorder)

        off_key = f'baseline__{regime}__offline__value'
        if off_key in data_baselines:
            ax.axhline(float(data_baselines[off_key][0]), color='black', lw=0.8,
                       ls=(0, (1, 1)), alpha=0.7,
                       label=r'offline (hindsight) $\mathbb{E}[\max_t X_t]$')

        ax.set_title(f'{OOD_LABELS[regime]} (OOD)', fontsize=10)
        ax.set_xlabel('t', fontsize=9)
        if ci % 2 == 0:                                             # leftmost in each row
            ax.set_ylabel('threshold value (mean ± 1 std)', fontsize=9)
        if ci == 0:
            ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Threshold trajectories: {focal_label}  (OOD $\\sigma$ regimes)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OOD_ROOT / out_filename
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out}')

    out_overleaf = OVERLEAF / out_filename
    out_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {out_overleaf}')


def render_ood_trajectories() -> None:
    _render_ood_trajectories_one('D_disc_cv', r'$\mathcal{D}_{\mathrm{disc}}$\_cv',
                                 '#1f77b4', 'trajectories-d-disc-cv-ood.png')
    _render_ood_trajectories_one('D_logu_cv', r'$\mathcal{D}_{\mathrm{logu}}$\_cv',
                                 '#2ca02c', 'trajectories-d-logu-cv-ood.png')


# ---------------------------------------------------------------------------
# Per-σ summary table (focused: 4 random models + 5 baselines)
# ---------------------------------------------------------------------------

def write_summary_table() -> None:
    raw = json.loads((OOD_ROOT / 'payoff_matrix_raw.json').read_text())
    R_star = raw['R_star']
    bR = raw['baseline_R']
    mR = raw['R']

    OOD = ('D_ood_0p3', 'D_ood_3', 'D_ood_30', 'D_ood_300')
    SIGMAS = (0.3, 3.0, 30.0, 300.0)
    ROWS_MODEL = ('D_disc_cv', 'D_disc_act', 'D_logu_cv', 'D_logu_act')
    ROWS_BASE = (
        'random_oracle_disc',
        'random_oracle_logu',
        'MAP_sigma_disc',
        'MAP_sigma_logu',
        'MLE_sigma',
    )

    out_csv = OOD_ROOT / 'summary_table.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy'] + [f'sigma={s}' for s in SIGMAS])

        for run in ROWS_MODEL:
            cells = [f'{mR[run][r]/R_star[r]:.4f}' for r in OOD]
            w.writerow([run] + cells)

        for bn in ROWS_BASE:
            cells = [f'{bR[r][bn]/R_star[r]:.4f}' for r in OOD]
            w.writerow([bn] + cells)
    print(f'wrote {out_csv}')

    # Also a markdown rendering for the writeup.
    out_md = OOD_ROOT / 'summary_table.md'
    lines = []
    lines.append('| policy | $\\sigma{=}0.3$ | $\\sigma{=}3$ | $\\sigma{=}30$ | $\\sigma{=}300$ |')
    lines.append('|---|---:|---:|---:|---:|')

    def _fmt(v: float) -> str:
        return f'{v:+.3f}'

    for run in ROWS_MODEL:
        cells = ' | '.join(_fmt(mR[run][r]/R_star[r]) for r in OOD)
        lines.append(f'| **{run}** | {cells} |')
    lines.append('| | | | | |')
    for bn in ROWS_BASE:
        cells = ' | '.join(_fmt(bR[r][bn]/R_star[r]) for r in OOD)
        lines.append(f'| {bn} | {cells} |')

    out_md.write_text('\n'.join(lines) + '\n')
    print(f'wrote {out_md}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OOD_ROOT.mkdir(parents=True, exist_ok=True)
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    render_extended_payoff_matrix()
    render_ood_trajectories()
    write_summary_table()
    return 0


if __name__ == '__main__':
    sys.exit(main())
