"""
Presentation-quality re-renders of the Stage B training curves.

Two outputs:
  v3/results/phase4/training_curves.png            (1600 x 900 @ 150 dpi)
    2 rows (cv top, act bottom) x 5 cols (D_1, D_2, D_3, D_disc, D_logu)
    Log y on cv, linear y on act, best-step marked with red vertical line + label

  v3/results/phase4/training_curves_per_sigma.png  (1200 x 800 @ 150 dpi)
    2 x 2 panels for the four random-variance runs, log y, all per-σ subgroup
    curves overlaid on aggregate val
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from train.configs import RESULTS_PHASE4


DISTRIBUTIONS = ['D_1', 'D_2', 'D_3', 'D_disc', 'D_logu']
SUPS = ['cv', 'act']


def _curves_path(run: str) -> Path:
    return RESULTS_PHASE4 / run / 'curves.npz'


def _read_end_event(run: str) -> dict:
    log_path = RESULTS_PHASE4 / run / 'log.jsonl'
    if not log_path.exists():
        return {}
    with log_path.open('r') as f:
        for line in f:
            ev = json.loads(line)
            if ev['kind'] == 'end':
                return ev
    return {}


def _global_y_limits_log(runs: list[str]) -> tuple[float, float]:
    lo = float('inf')
    hi = -float('inf')
    for r in runs:
        p = _curves_path(r)
        if not p.exists():
            continue
        z = np.load(p)
        for arr_name in ('train_loss', 'val_loss'):
            arr = z[arr_name]
            if arr.size:
                pos = arr[arr > 0]
                if pos.size:
                    lo = min(lo, float(pos.min()))
                    hi = max(hi, float(pos.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (1e-6, 1e2)
    return (lo * 0.5, hi * 2.0)


def _global_y_limits_linear(runs: list[str]) -> tuple[float, float]:
    hi = -float('inf')
    for r in runs:
        p = _curves_path(r)
        if not p.exists():
            continue
        z = np.load(p)
        # For linear y on act, focus on val (train is noisier).
        if z['val_loss'].size:
            hi = max(hi, float(z['val_loss'].max()))
    if not np.isfinite(hi):
        return (0.0, 1.0)
    return (0.0, hi * 1.05)


def render_grid(out: Path) -> None:
    cv_runs = [f'{d}_cv' for d in DISTRIBUTIONS]
    act_runs = [f'{d}_act' for d in DISTRIBUTIONS]
    cv_y = _global_y_limits_log(cv_runs)
    act_y = _global_y_limits_linear(act_runs)

    fig, axes = plt.subplots(
        2, 5,
        figsize=(1600 / 150, 900 / 150),
        sharex=False,
    )
    for col, dist in enumerate(DISTRIBUTIONS):
        for row, sup in enumerate(SUPS):
            ax = axes[row, col]
            run = f'{dist}_{sup}'
            p = _curves_path(run)
            if not p.exists():
                ax.set_title(f'{run} (missing)', fontsize=10)
                continue
            z = np.load(p)
            ax.plot(z['train_step'], z['train_loss'], color='#cccccc', lw=0.6, label='train')
            ax.plot(z['val_step'], z['val_loss'], color='C0', lw=1.4, label='val')

            end_ev = _read_end_event(run)
            best_step = end_ev.get('best_step')
            if best_step is not None and best_step >= 0:
                ax.axvline(best_step, color='red', lw=0.8, alpha=0.8, ls='--')
                ax.text(
                    best_step, 1.0,
                    f'best @ {best_step}',
                    transform=ax.get_xaxis_transform(),
                    color='red', fontsize=7,
                    rotation=90, va='top', ha='right',
                )

            if sup == 'cv':
                ax.set_yscale('log')
                ax.set_ylim(cv_y)
            else:
                ax.set_yscale('linear')
                ax.set_ylim(act_y)
            ax.set_title(run, fontsize=10)
            if row == 1:
                ax.set_xlabel('step', fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    'val loss (log)' if sup == 'cv' else 'val loss (linear)',
                    fontsize=9,
                )
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Stage B training curves — train (light) + val (dark), best step (red dashed)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def render_per_sigma(out: Path) -> None:
    runs = ('D_disc_cv', 'D_disc_act', 'D_logu_cv', 'D_logu_act')
    fig, axes = plt.subplots(
        2, 2, figsize=(1200 / 150, 800 / 150), sharex=False,
    )
    cmap = plt.get_cmap('viridis')
    for ax, run in zip(axes.flatten(), runs):
        p = _curves_path(run)
        if not p.exists():
            ax.set_title(f'{run} (missing)', fontsize=11)
            continue
        z = np.load(p)
        ax.plot(z['val_step'], z['val_loss'],
                color='black', lw=1.6, label='aggregate val')
        sigma_keys = sorted([k for k in z.files if k.startswith('val_loss_sigma_')])
        n = max(1, len(sigma_keys) - 1)
        for k_idx, k in enumerate(sigma_keys):
            label = k.replace('val_loss_', '')
            ax.plot(
                z['val_step'], z[k],
                color=cmap(0.15 + 0.7 * (k_idx / n)),
                lw=1.2, alpha=0.9, label=label,
            )

        end_ev = _read_end_event(run)
        best_step = end_ev.get('best_step')
        if best_step is not None:
            ax.axvline(best_step, color='red', lw=0.8, alpha=0.8, ls='--')
            ax.text(
                best_step, 1.0, f'best @ {best_step}',
                transform=ax.get_xaxis_transform(),
                color='red', fontsize=7,
                rotation=90, va='top', ha='right',
            )

        ax.set_yscale('log')
        ax.set_xlabel('step', fontsize=9)
        ax.set_ylabel('val loss (log)', fontsize=9)
        ax.set_title(run, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Per-σ-group val loss on the four random-variance runs',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> int:
    out_grid = RESULTS_PHASE4 / 'training_curves.png'
    out_sigma = RESULTS_PHASE4 / 'training_curves_per_sigma.png'
    render_grid(out_grid)
    render_per_sigma(out_sigma)
    print(f'wrote {out_grid}')
    print(f'wrote {out_sigma}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
