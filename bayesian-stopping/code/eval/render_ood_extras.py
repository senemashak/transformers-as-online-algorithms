"""Render extended versions of:
  - agreement-oracle.png   (10 models × 7 regimes; old 3-regime version is replaced)
  - per-sigma-d-disc.png   (D_disc_cv vs D_disc_act, 7 σ bars)
  - per-sigma-d-logu.png   (D_logu_cv vs D_logu_act, 7 σ bars)

Uses results/ood/extras/{agreement_oracle_extended.npz, per_sigma_extended.json}
produced by run_ood_extras.py. Outputs go to report/figures/ with the
same canonical names; previous 3-regime / 5-bin versions are overwritten.
"""

from __future__ import annotations

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

OOD_ROOT = V3_ROOT / 'results' / 'ood'
EXTRAS_DIR = OOD_ROOT / 'extras'
OVERLEAF = V3_ROOT / 'report' / 'figures'

ALL_RUNS = [
    'D_1_cv','D_1_act','D_2_cv','D_2_act','D_3_cv','D_3_act',
    'D_disc_cv','D_disc_act','D_logu_cv','D_logu_act',
]
SIGMA_LABELS = {
    0.3:   r'$\sigma{=}0.3$',
    1.0:   r'$\sigma{=}1$',
    3.0:   r'$\sigma{=}3$',
    10.0:  r'$\sigma{=}10$',
    30.0:  r'$\sigma{=}30$',
    100.0: r'$\sigma{=}100$',
    300.0: r'$\sigma{=}300$',
}
OOD_SIGMAS = {0.3, 3.0, 30.0, 300.0}

SOFT_TURQUOISE_CMAP = LinearSegmentedColormap.from_list(
    'SoftTurquoise', ['#f4fbfa', '#cfeae6', '#9bd4cb'])


# ---------------------------------------------------------------------------
# Extended agreement-oracle heatmap
# ---------------------------------------------------------------------------

def render_agreement_oracle_extended() -> None:
    z = np.load(EXTRAS_DIR / 'agreement_oracle_extended.npz', allow_pickle=False)
    agreement = z['agreement']                                   # (n_runs, n_reg)
    run_names = list(z['run_names'])
    sigmas = list(z['sigmas'])

    # Transpose to match the canonical horizontal layout: regimes on y, runs on x.
    A = agreement.T                                              # (n_reg, n_runs)
    n_reg, n_run = A.shape

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=130)
    im = ax.imshow(A, vmin=0.0, vmax=1.0, cmap=SOFT_TURQUOISE_CMAP, aspect='auto')
    ax.set_xticks(range(n_run))
    ax.set_xticklabels(run_names, fontsize=9, rotation=30, ha='right')

    yticklabels = []
    for s in sigmas:
        lbl = SIGMA_LABELS[float(s)]
        if float(s) in OOD_SIGMAS:
            lbl = lbl + ' (OOD)'
        yticklabels.append(lbl)
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels(yticklabels, fontsize=10)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color='black' if v < 0.6 else 'white', fontsize=8)
    ax.set_title('Per-step action agreement vs. per-regime static-σ oracle',
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='agreement')
    fig.tight_layout()

    out = OVERLEAF / 'agreement-oracle.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    # Also save under results/ood/.
    out_results = OOD_ROOT / 'agreement-oracle-extended.png'
    out_results.write_bytes(out.read_bytes())
    print(f'wrote {out_results}')


# ---------------------------------------------------------------------------
# Extended per-σ figure (one per random-variance dist)
# ---------------------------------------------------------------------------

PHASE5_DIR = V3_ROOT / 'results' / 'phase5'

# Baselines overlaid on every panel. Tuple: (key, label, color, marker).
BASELINE_OVERLAYS = [
    ('random_oracle_disc', r'random-ADP ($\mathcal{D}_{\mathrm{disc}}$)', '#0b3d91', '^'),
    ('random_oracle_logu', r'random-ADP ($\mathcal{D}_{\mathrm{logu}}$)', '#0a6e0a', 'v'),
    ('MAP_sigma_disc',     r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{disc}}$)', '#9467bd', 'D'),
    ('MAP_sigma_logu',     r'MAP-$\sigma$ ($\mathcal{D}_{\mathrm{logu}}$)', '#e377c2', 'd'),
    ('MLE_sigma',          r'MLE-$\sigma$ plug-in',                          '#7f7f7f', 'x'),
]


def _load_per_regime_R_star(regimes):
    """Return np.array of per-regime static-σ oracle R⋆ for each regime
    (the ceiling used as the headline normalizer throughout the report)."""
    phase5 = json.loads((PHASE5_DIR / 'payoff_matrix_raw.json').read_text())
    ood    = json.loads((OOD_ROOT  / 'payoff_matrix_raw.json').read_text())
    Rstar_phase5 = phase5.get('R_star', {})
    Rstar_ood    = ood.get('R_star', {})
    out = []
    for r in regimes:
        if r in Rstar_phase5:
            out.append(Rstar_phase5[r])
        elif r in Rstar_ood:
            out.append(Rstar_ood[r])
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)


def _load_baseline_R_per_regime(regimes):
    """Return {baseline_key: np.array of raw baseline R per regime (NaN where missing)}.

    Pulls from two sources:
      - results/phase5/payoff_matrix_raw.json    : in-prior σ ∈ {1, 10, 100}
      - results/ood/payoff_matrix_raw.json       : OOD σ ∈ {0.3, 3, 30, 300}
    """
    phase5 = json.loads((PHASE5_DIR / 'payoff_matrix_raw.json').read_text())
    ood    = json.loads((OOD_ROOT  / 'payoff_matrix_raw.json').read_text())
    bR_phase5 = phase5.get('baseline_R', {})
    bR_ood    = ood.get('baseline_R', {})
    Rstar_phase5 = phase5.get('R_star', {})
    Rstar_ood    = ood.get('R_star', {})

    out = {}
    for key, *_ in BASELINE_OVERLAYS:
        vals = []
        for regime in regimes:
            v = np.nan
            if regime in Rstar_phase5:
                try:
                    raw = bR_phase5[key][regime]
                    if raw is not None:
                        v = raw
                except KeyError:
                    pass
            elif regime in Rstar_ood:
                try:
                    raw = bR_ood[regime][key]
                    if raw is not None:
                        v = raw
                except KeyError:
                    pass
            vals.append(v)
        out[key] = np.array(vals, dtype=float)
    return out


def render_per_sigma_extended(dist: str) -> None:
    """dist ∈ {'D_disc', 'D_logu'}. Bars: 7 σ values, two heads side-by-side.
    Five baselines (random-ADP disc/logu, MAP-σ disc/logu, MLE-σ) are overlaid
    as dashed lines + markers; lines drop through NaN where data is unavailable.
    """
    js = json.loads((EXTRAS_DIR / 'per_sigma_extended.json').read_text())

    sigmas = [item['sigma'] for item in js['order']]
    regimes = [item['regime'] for item in js['order']]
    cv_run = f'{dist}_cv'
    act_run = f'{dist}_act'
    other = 'D_logu' if dist == 'D_disc' else 'D_disc'
    other_cv_run = f'{other}_cv'
    other_act_run = f'{other}_act'

    cv_R = np.array([js['R'][cv_run][r] for r in regimes])
    act_R = np.array([js['R'][act_run][r] for r in regimes])
    other_cv_R = np.array([js['R'][other_cv_run][r] for r in regimes])
    other_act_R = np.array([js['R'][other_act_run][r] for r in regimes])
    # Use per-regime static-σ oracle as R⋆ throughout, matching the rest of the
    # report and letting baselines be plotted on the same axis as the bars.
    R_star = _load_per_regime_R_star(regimes)

    cv_norm = cv_R / R_star
    act_norm = act_R / R_star
    other_cv_norm = other_cv_R / R_star
    other_act_norm = other_act_R / R_star

    baseline_R = _load_baseline_R_per_regime(regimes)
    baseline_vals = {k: v / R_star for k, v in baseline_R.items()}

    head_color = {'cv': '#1f77b4', 'act': '#2ca02c'}
    head_alpha = 0.55
    x = np.arange(len(sigmas), dtype=np.float64)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=130)
    bars_cv = ax.bar(x - width / 2, cv_norm, width,
                     color=head_color['cv'], alpha=head_alpha, label=f'{dist}_cv')
    bars_act = ax.bar(x + width / 2, act_norm, width,
                      color=head_color['act'], alpha=head_alpha, label=f'{dist}_act')

    # Cross-distribution overlays: the OTHER trained models on the same axes
    # (e.g. D_logu_cv/_act when this panel is D_disc). Plotted with the
    # opposite-distribution trajectory hue so the comparison is visual.
    other_hue = '#2ca02c' if dist == 'D_disc' else '#1f77b4'
    ax.plot(x, other_cv_norm, color=other_hue, ls='-', marker='o', lw=1.6,
            markersize=6, alpha=0.85, label=f'{other}\\_cv', zorder=14)
    ax.plot(x, other_act_norm, color=other_hue, ls='-', marker='s', lw=1.6,
            markersize=6, alpha=0.55, label=f'{other}\\_act', zorder=14)

    # Data-only / prior-aware baseline overlays: dashed line + marker per baseline.
    for key, label, color, marker in BASELINE_OVERLAYS:
        y = baseline_vals[key]
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, color=color, ls='--', marker=marker, lw=1.2,
                markersize=6, alpha=0.85, label=label, zorder=12)

    ax.axhline(1.0, color='gray', lw=1.0, ls=':', alpha=0.6,
               label=r'per-regime oracle ($R/R^\star{=}1$)')
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.5)

    for i, sig in enumerate(sigmas):
        if float(sig) in OOD_SIGMAS:
            ax.axvspan(i - 0.5, i + 0.5, color='#20c997', alpha=0.18,
                       linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels([SIGMA_LABELS[float(s)] for s in sigmas], fontsize=10)
    ax.set_xlabel('σ', fontsize=10)
    ax.set_ylabel(r'$R/R^\star$  ($R^\star$ = per-regime static-$\sigma$ oracle)',
                  fontsize=10)
    ax.set_title(rf'Per-$\sigma$ payoff: ${{\mathcal{{D}}_{{\mathrm{{{dist[2:]}}}}}}}$\_cv vs.\ '
                 rf'${{\mathcal{{D}}_{{\mathrm{{{dist[2:]}}}}}}}$\_act, baselines overlaid '
                 rf'(in-prior + OOD; shading marks OOD $\sigma$)',
                 fontsize=11)
    all_vals = (list(cv_norm) + list(act_norm)
                + list(other_cv_norm) + list(other_act_norm))
    for y in baseline_vals.values():
        all_vals.extend(v for v in y if np.isfinite(v))
    ax.set_ylim(min(min(all_vals) - 0.12, -0.1),
                max(max(all_vals) + 0.18, 1.25))
    for bars, vals in ((bars_cv, cv_norm), (bars_act, act_norm)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    v + (0.03 if v >= 0 else -0.05),
                    f'{v:.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2, framealpha=0.92)
    fig.tight_layout()

    # Use -extended suffix in overleaf to avoid clashing with the
    # in-distribution-only `per-sigma-d-{disc,logu}.png` produced by
    # refresh_overleaf_with_sig2.py.
    name = f'per-sigma-d-{dist[2:]}-extended.png'
    out = OVERLEAF / name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

    out_results = OOD_ROOT / name
    out_results.write_bytes(out.read_bytes())
    print(f'wrote {out_results}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    render_agreement_oracle_extended()
    render_per_sigma_extended('D_disc')
    render_per_sigma_extended('D_logu')
    return 0


if __name__ == '__main__':
    sys.exit(main())
