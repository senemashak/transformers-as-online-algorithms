"""
Figure + table rendering for the offline-supervision experiment.

Inputs:
    v3/results/offline-supervision/payoff_matrix_raw.json
    v3/results/offline-supervision/trajectories_cv.npz
    v3/results/phase5/trajectories.npz           (per-regime static oracle on D_1/D_2/D_3)
    v3/results/ood/trajectories.npz              (per-regime static oracle on OOD regimes)

Outputs (all under v3/results/offline-supervision/):
    payoff-matrix-extended-offline.png   14×7 heatmap; oracle and offline rows paired.
    trajectories-D_disc_cv_offline.png   cv-offline threshold per regime (7 panels),
    trajectories-D_logu_cv_offline.png      overlaying oracle-sig2 and the static
                                            per-regime oracle C_t*.
    summary-offline-disc.{csv,md}        2x2-ish summary table for D_disc.
    summary-offline-logu.{csv,md}        2x2-ish summary table for D_logu.
    bars-oracle-vs-offline.png           Two-panel grouped bar chart.

The numeric summary appended to README.md is updated separately in
`_update_readme_summary`.
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


# Pastel pink (R/R* = 0) → soft cream (≈ 0.5) → pastel green (R/R* = 1).
# The three-stop ramp keeps the midrange readable (not muddy brown).
PASTEL_PINK_GREEN = LinearSegmentedColormap.from_list(
    'pastel_pink_green',
    [
        (0.00, '#f4b6c2'),   # pastel pink
        (0.50, '#fdf4d9'),   # warm cream midpoint
        (1.00, '#b9e3c6'),   # pastel green
    ],
    N=256,
)

# Bar palette — four pastels, pairing cv (blue family) with act (green
# family); within each family the offline run is the lighter shade.
PASTEL_BAR_COLORS = {
    'oracle_cv':  '#9ec5e8',     # pastel blue
    'offline_cv': '#d6e8f5',     # very pale blue
    'oracle_act': '#9ed8a8',     # pastel green
    'offline_act':'#dcf0dc',     # very pale green
}

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

OFFLINE_ROOT = V3_ROOT / 'results' / 'offline-supervision'
PHASE5 = V3_ROOT / 'results' / 'phase5'
OOD = V3_ROOT / 'results' / 'ood'

# ALL_REGIMES in low-σ → high-σ display order
ALL_REGIMES: List[Tuple[str, float]] = [
    ('D_ood_0p3', 0.3),
    ('D_1', 1.0),
    ('D_ood_3', 3.0),
    ('D_2', 10.0),
    ('D_ood_30', 30.0),
    ('D_3', 100.0),
    ('D_ood_300', 300.0),
]
REGIME_NAMES = [r for r, _ in ALL_REGIMES]
REGIME_SIGMAS = {r: s for r, s in ALL_REGIMES}

# Display labels for the heatmap rows (LaTeX-ish; matplotlib mathtext).
RUN_LABELS = {
    'D_1_cv':              r'$\mathcal{D}_1$\_cv',
    'D_1_act':             r'$\mathcal{D}_1$\_act',
    'D_2_cv':              r'$\mathcal{D}_2$\_cv',
    'D_2_act':             r'$\mathcal{D}_2$\_act',
    'D_3_cv':              r'$\mathcal{D}_3$\_cv',
    'D_3_act':             r'$\mathcal{D}_3$\_act',
    'D_disc_cv_sig2':      r'$\mathcal{D}_\mathrm{disc}$\_cv (oracle)',
    'D_disc_cv_offline':   r'$\mathcal{D}_\mathrm{disc}$\_cv (offline)',
    'D_disc_act':          r'$\mathcal{D}_\mathrm{disc}$\_act (oracle)',
    'D_disc_act_offline':  r'$\mathcal{D}_\mathrm{disc}$\_act (offline)',
    'D_logu_cv_sig2':      r'$\mathcal{D}_\mathrm{logu}$\_cv (oracle)',
    'D_logu_cv_offline':   r'$\mathcal{D}_\mathrm{logu}$\_cv (offline)',
    'D_logu_act':          r'$\mathcal{D}_\mathrm{logu}$\_act (oracle)',
    'D_logu_act_offline':  r'$\mathcal{D}_\mathrm{logu}$\_act (offline)',
}

EXTENDED_RUNS = [
    'D_1_cv', 'D_1_act', 'D_2_cv', 'D_2_act', 'D_3_cv', 'D_3_act',
    'D_disc_cv_sig2', 'D_disc_cv_offline',
    'D_disc_act',     'D_disc_act_offline',
    'D_logu_cv_sig2', 'D_logu_cv_offline',
    'D_logu_act',     'D_logu_act_offline',
]

# Pairs of (oracle, offline) rows that should be visually grouped.
PAIRS = [
    ('D_disc_cv_sig2', 'D_disc_cv_offline'),
    ('D_disc_act',     'D_disc_act_offline'),
    ('D_logu_cv_sig2', 'D_logu_cv_offline'),
    ('D_logu_act',     'D_logu_act_offline'),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_raw() -> dict:
    return json.loads((OFFLINE_ROOT / 'payoff_matrix_raw.json').read_text())


def _load_trajectories() -> Dict[str, np.ndarray]:
    """Union of the three trajectory caches we need.

    offline-supervision/trajectories_cv.npz has the focal-model means/stds.
    phase5/trajectories.npz has baseline__<regime>__oracle_static__mean
    for regime in {D_1, D_2, D_3}. ood/trajectories.npz has the OOD
    regime keys. We merge into a single dict.
    """
    out: Dict[str, np.ndarray] = {}
    for p in (OFFLINE_ROOT / 'trajectories_cv.npz',
              PHASE5 / 'trajectories.npz',
              OOD / 'trajectories.npz'):
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            # First write wins; later sources don't clobber the focal-model
            # data from offline-supervision.
            if k not in out:
                out[k] = z[k]
    return out


# ---------------------------------------------------------------------------
# 14×7 heatmap
# ---------------------------------------------------------------------------

def render_payoff_matrix_extended() -> None:
    raw = _load_raw()
    rrs = raw['R_over_R_star']
    M = np.array([[rrs[run][r] for r in REGIME_NAMES] for run in EXTENDED_RUNS])

    fig, ax = plt.subplots(figsize=(9.0, 9.2), dpi=300)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap=PASTEL_PINK_GREEN, aspect='auto')

    ax.set_xticks(range(len(REGIME_NAMES)))
    ax.set_xticklabels(REGIME_NAMES, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(EXTENDED_RUNS)))
    ax.set_yticklabels([RUN_LABELS[r] for r in EXTENDED_RUNS], fontsize=10)
    ax.set_xlabel('test regime', fontsize=11)
    ax.set_title('R / R* — 14 trained models × 7 test regimes\n'
                 'each oracle row is followed by its offline-supervised counterpart',
                 fontsize=11)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            # Pastel palette is uniformly light; black text is legible on every cell.
            ax.text(j, i, f'{v:.3f}',
                    ha='center', va='center', color='black', fontsize=9)

    # Light horizontal separator under each (oracle, offline) pair row.
    for pair_idx, (oracle, offline) in enumerate(PAIRS):
        i_off = EXTENDED_RUNS.index(offline)
        ax.axhline(i_off + 0.5, color='black', lw=0.8, alpha=0.6)
    # Heavy separator after the six static rows and before the random-variance block.
    ax.axhline(5.5, color='black', lw=1.5, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label('R / R*', fontsize=10)
    fig.tight_layout()
    fig.savefig(OFFLINE_ROOT / 'payoff-matrix-extended-offline.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory plots (cv-offline focal; overlay oracle-cv + static C_t*)
# ---------------------------------------------------------------------------

def _y_limits_for_regime(sigma: float) -> Tuple[float, float]:
    """Per-panel y-limits chosen so the focal/oracle bands are visible.

    Static-σ-oracle C_t* sits around ~2.5·σ early and drops to near 0
    at t = n. We use roughly [-σ/2, 3.5·σ].
    """
    return (-0.5 * sigma, 3.5 * sigma)


def render_trajectories_for_offline_cv() -> None:
    traj = _load_trajectories()

    pairs_cv = [
        ('D_disc_cv_offline', 'D_disc_cv_sig2', '#1f77b4', '#69b8e0'),
        ('D_logu_cv_offline', 'D_logu_cv_sig2', '#2ca02c', '#8ed089'),
    ]

    for offline_run, oracle_run, focal_color, oracle_color in pairs_cv:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=130, sharex=False)
        axes = axes.flatten()
        for ci, (regime, sigma) in enumerate(ALL_REGIMES):
            ax = axes[ci]
            t_axis = np.arange(1, 257)

            # Focal — offline cv (focal_color, thick).
            m_key = f'{offline_run}__{regime}__mean'
            s_key = f'{offline_run}__{regime}__std'
            if m_key in traj:
                m = traj[m_key]
                s = traj[s_key]
                ax.fill_between(t_axis, m - s, m + s,
                                color=focal_color, alpha=0.20, linewidth=0)
                ax.plot(t_axis, m, color=focal_color, lw=2.2,
                        label=f'{offline_run} (focal)')

            # Oracle-supervised counterpart (different color, thinner).
            m_key = f'{oracle_run}__{regime}__mean'
            s_key = f'{oracle_run}__{regime}__std'
            if m_key in traj:
                m = traj[m_key]
                s = traj[s_key]
                ax.fill_between(t_axis, m - s, m + s,
                                color=oracle_color, alpha=0.18, linewidth=0)
                ax.plot(t_axis, m, color=oracle_color, lw=1.6,
                        label=f'{oracle_run} (oracle)')

            # Per-regime static-σ oracle C_t* (dashed black).
            base_key = f'baseline__{regime}__oracle_static__mean'
            if base_key in traj:
                bm = np.where(np.isfinite(traj[base_key]), traj[base_key], np.nan)
                ax.plot(t_axis, bm, color='black', lw=1.0, ls='--',
                        label=r'static-$\sigma$ oracle $C^\star_t$')

            ax.set_title(f'{regime}  (σ={sigma:g})', fontsize=10)
            ax.set_xlabel('t', fontsize=9)
            if ci % 4 == 0:
                ax.set_ylabel('threshold (mean ± 1 std)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*_y_limits_for_regime(sigma))
            if ci == 0:
                ax.legend(fontsize=8, loc='upper right')

        # Hide the unused 8th subplot (we have 7 regimes, grid is 2x4).
        for ci in range(len(ALL_REGIMES), len(axes)):
            axes[ci].set_visible(False)

        fig.suptitle(
            f'Threshold trajectories: {offline_run} vs {oracle_run}\n'
            'the offline-cv threshold sits above the oracle-cv threshold by '
            'construction (prophet ≥ online)',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = OFFLINE_ROOT / f'trajectories-{offline_run}.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-distribution summary tables (CSV + Markdown)
# ---------------------------------------------------------------------------

def _summary_rows_for_distribution(rrs: dict, dist: str) -> List[Dict[str, float]]:
    """Return 4 rows × 7 regimes for the {oracle-cv, offline-cv, oracle-act, offline-act}
    quartet at training distribution `dist`."""
    if dist == 'D_disc':
        runs = [
            ('oracle-cv',  'D_disc_cv_sig2'),
            ('offline-cv', 'D_disc_cv_offline'),
            ('oracle-act', 'D_disc_act'),
            ('offline-act','D_disc_act_offline'),
        ]
    elif dist == 'D_logu':
        runs = [
            ('oracle-cv',  'D_logu_cv_sig2'),
            ('offline-cv', 'D_logu_cv_offline'),
            ('oracle-act', 'D_logu_act'),
            ('offline-act','D_logu_act_offline'),
        ]
    else:
        raise ValueError(dist)

    rows = []
    for label, run in runs:
        row = {'kind': label, 'run': run}
        for regime in REGIME_NAMES:
            row[regime] = rrs[run][regime]
        rows.append(row)
    return rows


def write_summary_tables() -> None:
    raw = _load_raw()
    rrs = raw['R_over_R_star']
    gaps = raw['gaps']

    for dist in ('D_disc', 'D_logu'):
        rows = _summary_rows_for_distribution(rrs, dist)

        # CSV
        csv_path = OFFLINE_ROOT / f'summary-offline-{dist[2:]}.csv'
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['kind', 'run'] + REGIME_NAMES)
            for row in rows:
                w.writerow([row['kind'], row['run']]
                           + [f'{row[r]:.4f}' for r in REGIME_NAMES])
            # Append per-regime cv and act gaps.
            cv_gaps = gaps[dist]['cv_per_regime']
            act_gaps = gaps[dist]['act_per_regime']
            w.writerow([])
            w.writerow(['gap (offline - oracle)', 'kind'] + REGIME_NAMES)
            w.writerow(['cv gap', '—'] + [f'{cv_gaps[r]:+.4f}' for r in REGIME_NAMES])
            w.writerow(['act gap', '—'] + [f'{act_gaps[r]:+.4f}' for r in REGIME_NAMES])

        # Markdown
        md_path = OFFLINE_ROOT / f'summary-offline-{dist[2:]}.md'
        lines = [
            f'# Offline-supervision summary on {dist}',
            '',
            f'R / R* on each of the seven test regimes for the four runs '
            f'(oracle-cv = `{rows[0]["run"]}`, offline-cv = `{rows[1]["run"]}`, '
            f'oracle-act = `{rows[2]["run"]}`, offline-act = `{rows[3]["run"]}`).',
            '',
            '| kind | run | ' + ' | '.join(REGIME_NAMES) + ' |',
            '|------|-----|' + '|'.join(['---'] * len(REGIME_NAMES)) + '|',
        ]
        for row in rows:
            cells = ' | '.join(f'{row[r]:.4f}' for r in REGIME_NAMES)
            lines.append(f'| {row["kind"]} | `{row["run"]}` | {cells} |')
        lines += [
            '',
            '## Per-regime gap (offline − oracle, R/R*)',
            '',
            '| head | ' + ' | '.join(REGIME_NAMES) + ' | mean |',
            '|------|' + '|'.join(['---'] * (len(REGIME_NAMES) + 1)) + '|',
        ]
        cv_gaps = gaps[dist]['cv_per_regime']
        act_gaps = gaps[dist]['act_per_regime']
        lines.append('| cv  | ' + ' | '.join(f'{cv_gaps[r]:+.4f}' for r in REGIME_NAMES)
                     + f' | {gaps[dist]["cv_mean"]:+.4f} |')
        lines.append('| act | ' + ' | '.join(f'{act_gaps[r]:+.4f}' for r in REGIME_NAMES)
                     + f' | {gaps[dist]["act_mean"]:+.4f} |')
        md_path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Side-by-side bar chart
# ---------------------------------------------------------------------------

def render_oracle_vs_offline_bars() -> None:
    raw = _load_raw()
    rrs = raw['R_over_R_star']

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), dpi=140, sharex=True)

    panels = [
        ('D_disc', 'D_disc_cv_sig2', 'D_disc_cv_offline',
         'D_disc_act',  'D_disc_act_offline', axes[0]),
        ('D_logu', 'D_logu_cv_sig2', 'D_logu_cv_offline',
         'D_logu_act',  'D_logu_act_offline', axes[1]),
    ]
    x = np.arange(len(REGIME_NAMES))
    bar_w = 0.20

    for dist, ocv, offcv, oact, offact, ax in panels:
        y_ocv  = [rrs[ocv][r]   for r in REGIME_NAMES]
        y_offcv= [rrs[offcv][r] for r in REGIME_NAMES]
        y_oact = [rrs[oact][r]  for r in REGIME_NAMES]
        y_offact=[rrs[offact][r]for r in REGIME_NAMES]

        ax.bar(x - 1.5 * bar_w, y_ocv,   bar_w, label='oracle cv',
               color=PASTEL_BAR_COLORS['oracle_cv'],
               edgecolor='#3b3b3b', linewidth=0.5)
        ax.bar(x - 0.5 * bar_w, y_offcv, bar_w, label='offline cv',
               color=PASTEL_BAR_COLORS['offline_cv'],
               edgecolor='#3b3b3b', linewidth=0.5)
        ax.bar(x + 0.5 * bar_w, y_oact,  bar_w, label='oracle act',
               color=PASTEL_BAR_COLORS['oracle_act'],
               edgecolor='#3b3b3b', linewidth=0.5)
        ax.bar(x + 1.5 * bar_w, y_offact,bar_w, label='offline act',
               color=PASTEL_BAR_COLORS['offline_act'],
               edgecolor='#3b3b3b', linewidth=0.5)

        ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(0.0, color='black', lw=0.6, alpha=0.4)
        ax.set_ylabel('R / R*', fontsize=10)
        ax.set_title(f'Training distribution: {dist}', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(-0.30, 1.10)
        ax.legend(fontsize=8, loc='lower right', ncol=4)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f'{r}\n(σ={REGIME_SIGMAS[r]:g})' for r in REGIME_NAMES],
        fontsize=9,
    )
    fig.suptitle('Oracle- vs offline-supervision payoff per test regime', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OFFLINE_ROOT / 'bars-oracle-vs-offline.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README numeric summary
# ---------------------------------------------------------------------------

def update_readme_with_summary() -> None:
    """Append (or replace) a '## Headline numbers' section in README.md."""
    raw = _load_raw()
    gaps = raw['gaps']
    readme = OFFLINE_ROOT / 'README.md'
    text = readme.read_text() if readme.exists() else ''
    marker = '## Headline numbers (offline − oracle R/R*)'

    block_lines = [marker, '']
    block_lines.append('| training dist | cv mean gap | act mean gap |')
    block_lines.append('|---------------|-------------|--------------|')
    for dist in ('D_disc', 'D_logu'):
        cv = gaps[dist]['cv_mean']
        act = gaps[dist]['act_mean']
        block_lines.append(f'| `{dist}` | {cv:+.4f} | {act:+.4f} |')
    block_lines.append('')
    block_lines.append('### Per-regime breakdown')
    block_lines.append('')
    block_lines.append(
        '| training dist | head | '
        + ' | '.join(REGIME_NAMES) + ' |'
    )
    block_lines.append(
        '|---------------|------|'
        + '|'.join(['---'] * len(REGIME_NAMES)) + '|'
    )
    for dist in ('D_disc', 'D_logu'):
        for head in ('cv', 'act'):
            row = gaps[dist][f'{head}_per_regime']
            block_lines.append(
                f'| `{dist}` | {head} | '
                + ' | '.join(f'{row[r]:+.4f}' for r in REGIME_NAMES)
                + ' |'
            )
    block_lines.append('')
    interpretive = (
        'On `D_logu` both gaps are within 0.02 of zero — the hindsight-prophet '
        'labels carry essentially the same information as the ADP oracle for '
        'this continuous prior. On `D_disc` the cv-offline run lags the oracle '
        'by 0.059 on average; the damage is concentrated almost entirely at '
        '`D_ood_3` (cv gap −0.31, act gap −0.13), the interpolation regime '
        'between the discrete training σ values. On all three in-prior regimes '
        '(σ ∈ {1, 10, 100}) the cv-offline gap is under 0.03 cell-for-cell.'
    )
    block_lines.append(interpretive)
    block_lines.append('')

    new_section = '\n'.join(block_lines)

    if marker in text:
        # Replace existing section: everything from marker to next `## ` or EOF.
        head_idx = text.find(marker)
        rest = text[head_idx:]
        next_hdr = rest.find('\n## ', 1)
        if next_hdr == -1:
            text = text[:head_idx] + new_section
        else:
            text = text[:head_idx] + new_section + rest[next_hdr + 1:]
    else:
        if not text.endswith('\n'):
            text = text + '\n'
        text = text + '\n' + new_section
    readme.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    render_payoff_matrix_extended()
    print('[render-offline] payoff-matrix-extended-offline.png')
    render_trajectories_for_offline_cv()
    print('[render-offline] trajectories-D_disc_cv_offline.png + D_logu')
    write_summary_tables()
    print('[render-offline] summary-offline-{disc,logu}.{csv,md}')
    render_oracle_vs_offline_bars()
    print('[render-offline] bars-oracle-vs-offline.png')
    update_readme_with_summary()
    print('[render-offline] README.md updated')
    return 0


if __name__ == '__main__':
    sys.exit(main())
