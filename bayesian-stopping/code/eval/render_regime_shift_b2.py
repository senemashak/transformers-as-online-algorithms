"""Renderers for Experiment B.2 (OOD regime-shift).

Produces (per cv model unless noted):
  results/regime-shift/trajectories-shift-b2-<run>.png   2x2: four large-step OOD pairs
  results/regime-shift/rho-shift-combined-<run>.png      10-pair (B.1 + B.2) overlay
  results/regime-shift/sidebyside-1-100-vs-3-300-<run>.png    D_disc_cv comparison
  results/regime-shift/t_half-hist-b2-<run>.png          (3,300) and (300,3) histograms
  results/regime-shift/summary_table_combined.{csv,md}   10-pair x 2-model metrics
  results/regime-shift/README_b2.md                      provenance + rho_255 note

Combined rho_t figure encodes:
  - Upward shifts (sigma_b > sigma_a) in cool colors; downward in warm.
  - Line thickness ~ scale ratio: thick (lw=2.4) for 100x pairs, thin (lw=1.2) for 10x.
  - x in [128, 256], y in [0, 1.2], horizontal references at rho=0 and rho=1.

Combined summary table reports rho_200 (population mean) instead of rho_255.
The B.2 spec asked for rho_255, but on all twelve B.2 cells the iid-sigma
mean references collapse near horizon and rho_255 falls outside [0, 1] (see
README_b2.md). rho_255 is preserved verbatim in raw_b2.json for completeness.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

PAIRS_B1: List[Tuple[float, float]] = [
    (1.0, 100.0), (100.0, 1.0), (10.0, 100.0), (100.0, 10.0),
]
PAIRS_B2: List[Tuple[float, float]] = [
    (3.0, 30.0), (30.0, 3.0), (3.0, 300.0), (300.0, 3.0),
    (30.0, 300.0), (300.0, 30.0),
]
PAIRS_B2_LARGE_STEP: List[Tuple[float, float]] = [
    (3.0, 300.0), (300.0, 3.0), (30.0, 300.0), (300.0, 30.0),
]
PAIRS_ALL: List[Tuple[float, float]] = PAIRS_B1 + PAIRS_B2

SHIFT_T = 128
N = 256
RHO_TRUNC_T = 240                                                  # last point on rho overlay
PRE_REF_TRUNC = SHIFT_T                                            # plot ref_a on t = 1..128
POST_REF_TRUNC = 250                                               # plot ref_b on t = 129..250

PAIR_TAG = lambda a, b: f'{int(a)}_{int(b)}'

MODEL_LABEL = {
    'D_disc_cv': r'$\mathcal{D}_{\mathrm{disc}}$\_cv',
    'D_logu_cv': r'$\mathcal{D}_{\mathrm{logu}}$\_cv',
}

# Baseline lines overlaid on every shift trajectory panel (in addition to the
# piecewise σ_a/σ_b iid oracle references already drawn). Skips known-σ
# baselines (plug-in / prior-only / myopic) — those require a single σ
# and are misleading on a switching cache.
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

# Cool = upward, warm = downward. Within each direction, hue keys to scale ratio.
PAIR_COLORS: Dict[Tuple[float, float], str] = {
    # Upward (cool)
    (1.0, 100.0):   '#08306b',                                     # 100x  very dark blue
    (3.0, 300.0):   '#2171b5',                                     # 100x  medium blue
    (10.0, 100.0):  '#6baed6',                                     # 10x   light blue
    (3.0, 30.0):    '#41ab5d',                                     # 10x   medium green
    (30.0, 300.0):  '#74c476',                                     # 10x   light green
    # Downward (warm)
    (100.0, 1.0):   '#67000d',                                     # 100x  very dark red
    (300.0, 3.0):   '#a50f15',                                     # 100x  dark red
    (100.0, 10.0):  '#fb6a4a',                                     # 10x   light red
    (30.0, 3.0):    '#fd8d3c',                                     # 10x   orange
    (300.0, 30.0):  '#fc4e2a',                                     # 10x   red-orange
}
PAIR_LABEL: Dict[Tuple[float, float], str] = {
    (1.0, 100.0):   r'$1 \to 100$',
    (3.0, 300.0):   r'$3 \to 300$',
    (10.0, 100.0):  r'$10 \to 100$',
    (3.0, 30.0):    r'$3 \to 30$',
    (30.0, 300.0):  r'$30 \to 300$',
    (100.0, 1.0):   r'$100 \to 1$',
    (300.0, 3.0):   r'$300 \to 3$',
    (100.0, 10.0):  r'$100 \to 10$',
    (30.0, 3.0):    r'$30 \to 3$',
    (300.0, 30.0):  r'$300 \to 30$',
}


def _scale_ratio(sa: float, sb: float) -> float:
    return max(sa, sb) / min(sa, sb)


def _line_width(sa: float, sb: float) -> float:
    return 2.4 if _scale_ratio(sa, sb) >= 50 else 1.2


def _legend_order(pairs: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Upward first (cool), then downward (warm); within each, by ratio descending."""
    up = sorted([p for p in pairs if p[1] > p[0]],
                key=lambda p: (-_scale_ratio(*p), p[0]))
    down = sorted([p for p in pairs if p[1] < p[0]],
                  key=lambda p: (-_scale_ratio(*p), p[0]))
    return up + down


def _save(fig, name: str) -> None:
    out = RS_ROOT / name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    out_overleaf = OVERLEAF / out.name
    out_overleaf.write_bytes(out.read_bytes())
    print(f'wrote {out_overleaf}')


# ---------------------------------------------------------------------------
# Trajectory four-panel figure (B.2 large-step pairs)
# ---------------------------------------------------------------------------

def _draw_trajectory_panel(ax, traj: dict, run_name: str, sa: float, sb: float,
                           title: str) -> None:
    tag = PAIR_TAG(sa, sb)
    ref_a = traj[f'shift_{tag}__ref_a_iid']
    ref_b = traj[f'shift_{tag}__ref_b_iid']
    thr_mean = traj[f'{run_name}__shift_{tag}__thr_mean']
    thr_std = traj[f'{run_name}__shift_{tag}__thr_std']

    t_axis = np.arange(1, N + 1)

    # Pre-shift oracle ref (sigma_a) -- blue dashed -- on t = 1..128.
    ax.plot(t_axis[:PRE_REF_TRUNC], ref_a[:PRE_REF_TRUNC],
            color='#1f77b4', lw=1.4, ls='--',
            label=fr'static-$\sigma{{=}}{int(sa)}$ oracle (pre)')
    # Post-shift oracle ref (sigma_b) -- red dashed -- on t = 129..250 (avoid horizon).
    ax.plot(t_axis[SHIFT_T:POST_REF_TRUNC], ref_b[SHIFT_T:POST_REF_TRUNC],
            color='#d62728', lw=1.4, ls='--',
            label=fr'static-$\sigma{{=}}{int(sb)}$ oracle (post)')

    # All non-known-σ baselines over the full horizon (mean + low-alpha std band).
    # Emphasised baselines (random-ADP oracles) get thick, full-alpha lines on
    # top of the rest; non-emphasised baselines stay thin and translucent.
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

    # Model trajectory: clip to a small positive value for the log-y axis.
    lower = np.clip(thr_mean - thr_std, 1e-2, None)
    upper = np.clip(thr_mean + thr_std, 1e-2, None)
    ax.fill_between(t_axis, lower, upper, color='#9467bd',
                    alpha=0.20, linewidth=0)
    ax.plot(t_axis, np.clip(thr_mean, 1e-2, None),
            color='#9467bd', lw=2.0,
            label=f'{MODEL_LABEL[run_name]} (focal)')

    # Vertical dashed line at the shift -- per spec.
    ax.axvline(SHIFT_T, color='gray', lw=1.0, ls='--', alpha=0.7,
               label=r'shift ($t{=}128$)')

    ax.set_yscale('log')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('t', fontsize=9)
    ax.set_ylabel('threshold (log)', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=6, loc='best')


def render_trajectory_figure_b2(run_name: str) -> None:
    traj = dict(np.load(RS_ROOT / 'trajectories_b2.npz', allow_pickle=False))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=130)
    for ai, (sa, sb) in enumerate(PAIRS_B2_LARGE_STEP):
        ax = axes[ai // 2, ai % 2]
        title = PAIR_LABEL[(sa, sb)] + '  (OOD endpoints)'
        _draw_trajectory_panel(ax, traj, run_name, sa, sb, title)
    fig.suptitle('B.2 trajectories (OOD large-step pairs): '
                 f'{MODEL_LABEL[run_name]}',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f'trajectories-shift-b2-{run_name}.png')


# ---------------------------------------------------------------------------
# Combined rho_t overlay (B.1 + B.2 = 10 pairs)
# ---------------------------------------------------------------------------

def render_rho_overlay_combined(run_name: str) -> None:
    traj_b1 = dict(np.load(RS_ROOT / 'trajectories.npz', allow_pickle=False))
    traj_b2 = dict(np.load(RS_ROOT / 'trajectories_b2.npz', allow_pickle=False))
    traj = {**traj_b1, **traj_b2}

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=130)
    t_axis = np.arange(1, N + 1)

    for (sa, sb) in _legend_order(PAIRS_ALL):
        tag = PAIR_TAG(sa, sb)
        rho_mean = traj[f'{run_name}__shift_{tag}__rho_mean']
        rho_std = traj[f'{run_name}__shift_{tag}__rho_std']
        slc = slice(SHIFT_T - 1, RHO_TRUNC_T)
        x = t_axis[slc]
        y = rho_mean[slc]
        ystd = rho_std[slc]
        color = PAIR_COLORS[(sa, sb)]
        # Per-sequence ρ std band. Alpha kept low because 10 pairs overlap.
        ax.fill_between(x, y - ystd, y + ystd, color=color,
                        alpha=0.08, linewidth=0)
        ax.plot(x, y, color=color, lw=_line_width(sa, sb),
                label=PAIR_LABEL[(sa, sb)])

    # Reference lines.
    ax.axhline(0.0, color='black', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(0.5, color='gray', lw=0.6, ls='--', alpha=0.4)
    ax.axvline(SHIFT_T, color='gray', lw=1.0, ls='--', alpha=0.7,
               label=r'shift ($t{=}128$)')

    ax.set_xlim(SHIFT_T - 5, N + 2)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel(r'$\rho_t$', fontsize=11)
    ax.set_title(f'Combined adaptation $\\rho_t$ for {MODEL_LABEL[run_name]} '
                 r'(B.1 + B.2; 10 pairs)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2,
              title=r'cool $=$ upward; warm $=$ downward; thick $=100\times$, thin $=10\times$')
    fig.tight_layout()
    _save(fig, f'rho-shift-combined-{run_name}.png')


# ---------------------------------------------------------------------------
# Side-by-side: D_disc_cv on (1, 100) [B.1] vs (3, 300) [B.2]
# ---------------------------------------------------------------------------

def render_side_by_side_comparison(run_name: str = 'D_disc_cv') -> None:
    traj_b1 = dict(np.load(RS_ROOT / 'trajectories.npz', allow_pickle=False))
    traj_b2 = dict(np.load(RS_ROOT / 'trajectories_b2.npz', allow_pickle=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
    _draw_trajectory_panel(
        axes[0], traj_b1, run_name, 1.0, 100.0,
        title='B.1: $1 \\to 100$ (training-prior endpoints)',
    )
    _draw_trajectory_panel(
        axes[1], traj_b2, run_name, 3.0, 300.0,
        title='B.2: $3 \\to 300$ (OOD endpoints, same 100$\\times$ ratio)',
    )
    fig.suptitle(f'{MODEL_LABEL[run_name]}: memorized vs interpolated '
                 r'$\sigma$-conditional rule (100$\times$ upward shift)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f'sidebyside-1-100-vs-3-300-{run_name}.png')


# ---------------------------------------------------------------------------
# Per-sequence t_{1/2} histograms: (3, 300) and (300, 3) for D_disc_cv
# ---------------------------------------------------------------------------

def render_t_half_histograms_b2(run_name: str = 'D_disc_cv') -> None:
    z = np.load(RS_ROOT / 't_half_per_seq_b2.npz', allow_pickle=False)
    pairs = [(3.0, 300.0), (300.0, 3.0)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=130)
    n_post = N - SHIFT_T                                           # = 128
    bins = np.arange(1, n_post + 2)

    for ax, (sa, sb) in zip(axes, pairs):
        tag = PAIR_TAG(sa, sb)
        t_half = z[f'{run_name}__shift_{tag}__t_half']
        n_capped = int((t_half == n_post).sum())
        ax.hist(t_half, bins=bins,
                color=PAIR_COLORS[(sa, sb)], alpha=0.85,
                edgecolor='white', linewidth=0.3)
        ax.text(0.97, 0.95,
                f'never crossed ($t_{{1/2}}={n_post}$):\n{n_capped} / {len(t_half)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                          edgecolor='none'))
        ax.set_yscale('log')
        ax.set_xlabel(r'$t_{1/2}$ (post-shift adaptation time)', fontsize=9)
        ax.set_ylabel('count (log)', fontsize=9)
        ax.set_title(PAIR_LABEL[(sa, sb)] + '  (OOD)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    fig.suptitle(f'B.2 per-sequence $t_{{1/2}}$ for {MODEL_LABEL[run_name]} '
                 r'(100$\times$ shifts at OOD endpoints)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f't_half-hist-b2-{run_name}.png')


# ---------------------------------------------------------------------------
# Combined results table (B.1 + B.2 -> CSV + Markdown)
# ---------------------------------------------------------------------------

def write_combined_summary_table() -> None:
    raw_b1 = json.loads((RS_ROOT / 'raw.json').read_text())['metrics']
    raw_b2 = json.loads((RS_ROOT / 'raw_b2.json').read_text())['metrics']

    rows: List[Dict] = []
    for (sa, sb) in PAIRS_ALL:
        tag = PAIR_TAG(sa, sb)
        pair_set = 'B.1' if (sa, sb) in PAIRS_B1 else 'B.2'
        for run in ('D_disc_cv', 'D_logu_cv'):
            v = (raw_b1.get(run, {}).get(tag)
                 if pair_set == 'B.1' else raw_b2.get(run, {}).get(tag))
            if v is None:
                continue
            rows.append({
                'pair_set':       pair_set,
                'model':          run,
                'sigma_a':        v['sigma_a'],
                'sigma_b':        v['sigma_b'],
                'mean_t_half':    v['t_half_per_seq_mean'],
                'p10_t_half':     v['t_half_per_seq_p10'],
                'p50_t_half':     v['t_half_per_seq_median'],
                'p90_t_half':     v['t_half_per_seq_p90'],
                'never_crossed':  v['t_half_never_crossed'],
                'pre_shift_err':  v['pre_shift_tracking_error'],
                'post_shift_err': v['post_shift_tracking_error'],
                'rho_200_pop':    v['rho_settled_population'],
            })

    csv_path = RS_ROOT / 'summary_table_combined.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_set', 'model', 'sigma_a', 'sigma_b',
                    'mean_t_half', 'p10_t_half', 'p50_t_half', 'p90_t_half',
                    'never_crossed', 'pre_shift_err', 'post_shift_err',
                    'rho_200_pop'])
        for r in rows:
            w.writerow([
                r['pair_set'], r['model'],
                f'{int(r["sigma_a"])}', f'{int(r["sigma_b"])}',
                f'{r["mean_t_half"]:.1f}',
                f'{r["p10_t_half"]:.0f}',
                f'{r["p50_t_half"]:.0f}',
                f'{r["p90_t_half"]:.0f}',
                r['never_crossed'],
                f'{r["pre_shift_err"]:.3f}',
                f'{r["post_shift_err"]:.3f}',
                f'{r["rho_200_pop"]:+.3f}',
            ])
    print(f'wrote {csv_path}')

    md_lines = [
        '| set | model | shift | mean $t_{1/2}$ | $p_{10}$ | $p_{50}$ | $p_{90}$ | never | pre err | post err | $\\rho_{200}$ |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        sa, sb = int(r['sigma_a']), int(r['sigma_b'])
        shift = f'$\\sigma{{=}}{sa} \\to {sb}$'
        md_lines.append(
            f'| {r["pair_set"]} | {r["model"]} | {shift} | '
            f'{r["mean_t_half"]:.1f} | {r["p10_t_half"]:.0f} | '
            f'{r["p50_t_half"]:.0f} | {r["p90_t_half"]:.0f} | '
            f'{r["never_crossed"]} | '
            f'{r["pre_shift_err"]:.3f} | {r["post_shift_err"]:.3f} | '
            f'{r["rho_200_pop"]:+.3f} |'
        )
    md_path = RS_ROOT / 'summary_table_combined.md'
    md_path.write_text('\n'.join(md_lines) + '\n')
    print(f'wrote {md_path}')


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

README_NOTE = """\
# Experiment B.2 -- OOD regime-shift artifacts

Generated by:
  code/eval/run_regime_shift.py --pairs b2     (raw artifacts)
  code/eval/render_regime_shift_b2.py          (figures + combined table)

## Caches (six new pairs from sigma in {3, 30, 300})
  caches/D_shift_3_30_test.npz       seed 7001
  caches/D_shift_30_3_test.npz       seed 7002
  caches/D_shift_3_300_test.npz      seed 7003
  caches/D_shift_300_3_test.npz      seed 7004
  caches/D_shift_30_300_test.npz     seed 7005
  caches/D_shift_300_30_test.npz     seed 7006

## Artifacts
  raw_b2.json                 metrics per (model, pair); B.2-only
  trajectories_b2.npz         focal mean/std + rho mean/std + iid refs
  t_half_per_seq_b2.npz       per-sequence Delta-t arrays for histograms
  iid_refs/iid_ref_sigma_<s>.npz   cached iid-sigma mean references for
                                   sigma in {3, 30, 300}, built once per run

## Numerical-stability note (rho_t)

The B.2 spec asks for rho_255 with a |log <C*_b> - log <C*_a>| < 0.05
clip. On all twelve B.2 cells the log-difference at t=255 is well above
0.05 (1-2 units), so the clip never fires -- but both iid-sigma references
collapse toward zero in absolute terms near horizon (the static oracle's
terminal threshold is the myopic posterior mean, and that population-
averages to ~0 under mu_i ~ N(0, tau_0^2)). With both refs near-zero in
absolute terms, rho_255 ratios become ill-conditioned despite the
log-difference being well-defined: all 12 rho_255 values land outside
[0, 1], between -3.2 and +5.0. rho_200 is the clean "settled" probe
(well-behaved in [-0.16, +0.80] for B.2) and is what appears in
`summary_table_combined.{csv,md}` and in the rho_t overlay /
side-by-side figures. rho_255 is preserved verbatim in raw_b2.json for
completeness (fields rho_final_population, rho_final_per_seq_*).
"""


def write_readme_note() -> None:
    out = RS_ROOT / 'README_b2.md'
    out.write_text(README_NOTE)
    print(f'wrote {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OVERLEAF.mkdir(parents=True, exist_ok=True)
    for run in ('D_disc_cv', 'D_logu_cv'):
        render_trajectory_figure_b2(run)
        render_rho_overlay_combined(run)
    render_side_by_side_comparison('D_disc_cv')
    render_t_half_histograms_b2('D_disc_cv')
    write_combined_summary_table()
    write_readme_note()
    return 0


if __name__ == '__main__':
    sys.exit(main())
