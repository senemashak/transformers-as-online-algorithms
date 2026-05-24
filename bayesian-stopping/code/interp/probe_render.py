"""
Render probe-sweep results.

Inputs:
    results/probing/logu-act/runs/<run>/probe_results.json
    (optional) results/probing/logu-act/runs/<control_run>/probe_results.json

Outputs (under results/probing/logu-act/):
    probe-curves-logu-act.png       7-panel grid, one panel per target,
                                    normalized MSE vs layer for linear (solid)
                                    and MLP (dashed). Control noise floor
                                    drawn as a dotted horizontal band if the
                                    control sweep is available; otherwise
                                    annotated "control: pending".
    probe-results-logu-act.csv      per-(target, layer, variant) row.
    probe-summary-logu-act.md       per-target summary table.

CLI:
    python -m interp.probe_render
    python -m interp.probe_render --control D_logu_act_control
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from interp.probe_paths import RESULTS_ROOT, RUNS_ROOT, PROBE_VARIANT  # noqa: E402

TARGETS_DISPLAY = [
    ('S_t',                r'$S_t$  (cumulative sum)'),
    ('Q_t',                r'$Q_t$  (cumulative sum-of-squares)'),
    ('X_bar',              r'$\bar X_t$'),
    ('sigma_hat2',         r'$\hat\sigma_t^2$'),
    ('sigma_MAP',          r'$\hat\sigma_t^{\mathrm{MAP}}$'),
    ('sigma_MLE',          r'$\hat\sigma_t^{\mathrm{MLE}}$'),
    ('C_star',             r'$\widehat C^\star_t$  (oracle threshold)'),
    ('C_plugin_MAP',       r'$\widehat C_t^{\mathrm{plug\text{-}in, MAP}}$'),
    ('C_plugin_MLE',       r'$\widehat C_t^{\mathrm{plug\text{-}in, MLE}}$'),
    ('log_lik_sigma_1',    r'$\log p(X_{1:t}\mid\sigma=1)$'),
    ('log_lik_sigma_10',   r'$\log p(X_{1:t}\mid\sigma=10)$'),
    ('log_lik_sigma_100',  r'$\log p(X_{1:t}\mid\sigma=100)$'),
    ('log_lik_true_sigma', r'$\log p(X_{1:t}\mid\sigma_i)$ (true)'),
]


def _load_results(run: str, probe_kind: str = 'attn') -> Dict[tuple, dict]:
    suffix = '' if probe_kind == 'attn' else f'_{probe_kind}'
    path = RUNS_ROOT / run / f'probe_results{suffix}.json'
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {(c['target'], c['layer'], c['variant']): c for c in raw['cells']}


def _series(results: Dict[tuple, dict], target: str, variant: str
            ) -> Optional[np.ndarray]:
    """Return (layers, test_norm_mse) for the target/variant pair, or None."""
    rows = [(c['layer'], c['test_norm_mse']) for k, c in results.items()
            if c['target'] == target and c['variant'] == variant]
    if not rows:
        return None
    rows.sort()
    return np.array(rows)


def _first_decoded_layer(
    series: np.ndarray, control: Optional[np.ndarray], ratio: float = 2.0
) -> Optional[int]:
    """Smallest ℓ at which probe norm_MSE ≤ control norm_MSE / ratio."""
    if control is None or series is None:
        return None
    ctrl_by_layer = {int(l): m for l, m in control}
    for layer, mse in series:
        layer = int(layer)
        if layer not in ctrl_by_layer:
            continue
        if mse * ratio <= ctrl_by_layer[layer]:
            return layer
    return None


def render(
    run: str = 'D_logu_act',
    control_run: Optional[str] = None,
    control_run_2: Optional[str] = None,
    probe_kind: str = 'attn',
) -> int:
    """Render probe curves + CSV + summary.

    Args:
        run:            trained-model run name.
        control_run:    primary noise-floor baseline (random-init).
        control_run_2:  secondary noise-floor baseline (constant-label-trained).
        probe_kind:     'attn' (default) reads + writes the existing filenames;
                        'noattn' reads the per-timestep-probe results and writes
                        sibling output files with a `-noattn` suffix.
    """
    trained = _load_results(run, probe_kind)
    ctrl1 = _load_results(control_run, probe_kind) if control_run else {}
    ctrl2 = _load_results(control_run_2, probe_kind) if control_run_2 else {}
    targets_present = sorted({c['target'] for c in trained.values()})
    if not targets_present:
        print(f'[probe-render] no results found for {run}')
        return 1

    ordered_targets = [t for t, _ in TARGETS_DISPLAY if t in targets_present]
    for t in targets_present:
        if t not in ordered_targets:
            ordered_targets.append(t)

    n_panels = len(ordered_targets)
    cols = min(n_panels, 4)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), dpi=140)
    axes = np.atleast_1d(axes).flatten()

    pretty = {k: lab for k, lab in TARGETS_DISPLAY}

    for ai, target in enumerate(ordered_targets):
        ax = axes[ai]
        lin = _series(trained, target, 'linear')
        mlp = _series(trained, target, 'mlp')
        if lin is not None:
            ax.plot(lin[:, 0], lin[:, 1], 'o-', color='#1f77b4', lw=1.8, label='linear (trained)')
        if mlp is not None:
            ax.plot(mlp[:, 0], mlp[:, 1], 's--', color='#d62728', lw=1.8, label='MLP (trained)')
        # Random-init noise floor (pale blue / pale red dotted, thinner).
        if ctrl1:
            c1_lin = _series(ctrl1, target, 'linear')
            c1_mlp = _series(ctrl1, target, 'mlp')
            if c1_lin is not None:
                ax.plot(c1_lin[:, 0], c1_lin[:, 1], 'o:', color='#7fb3e6',
                        lw=1.1, label='linear (random-init)')
            if c1_mlp is not None:
                ax.plot(c1_mlp[:, 0], c1_mlp[:, 1], 's:', color='#e6928f',
                        lw=1.1, label='MLP (random-init)')
        # Constant-label-trained noise floor (very pale, dot-dashed).
        if ctrl2:
            c2_lin = _series(ctrl2, target, 'linear')
            c2_mlp = _series(ctrl2, target, 'mlp')
            if c2_lin is not None:
                ax.plot(c2_lin[:, 0], c2_lin[:, 1], 'o-.', color='#a8c8e0',
                        lw=0.9, label='linear (const-label)', alpha=0.85)
            if c2_mlp is not None:
                ax.plot(c2_mlp[:, 0], c2_mlp[:, 1], 's-.', color='#e0a8a8',
                        lw=0.9, label='MLP (const-label)', alpha=0.85)
        ax.axhline(1.0, color='black', lw=0.6, ls=':', alpha=0.6)
        ax.set_xlabel(r'layer $\ell$', fontsize=10)
        ax.set_ylabel('normalized MSE', fontsize=10)
        ax.set_title(pretty.get(target, target), fontsize=10)
        ax.set_xticks(range(9))
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ymin = 1e-5 if target == 'X_bar' else 1e-3
        ax.set_ylim(ymin, 2.5)
        # If neither control loaded, annotate.
        if not ctrl1 and not ctrl2:
            ax.text(0.5, 0.95, 'control: pending', transform=ax.transAxes,
                    ha='center', va='top', fontsize=9, color='#666',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='#aaa'))

    for ai in range(n_panels, len(axes)):
        axes[ai].set_visible(False)

    # Single shared legend in the bottom-right empty subplot slot (since the
    # 2x4 grid has 7 panels and 1 free cell).
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower right',
                   bbox_to_anchor=(0.99, 0.04), fontsize=10,
                   frameon=True, ncol=1)

    title_kind = '' if probe_kind == 'attn' else '  [per-timestep probe, no α]'
    fig.suptitle(
        f'Probing analysis: hidden-state decodability per layer  ({run}){title_kind}',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    suffix = '' if probe_kind == 'attn' else f'-{probe_kind}'
    out_path = RESULTS_ROOT / f'probe-curves-{PROBE_VARIANT}{suffix}.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[probe-render] wrote {out_path}')

    # CSV with up to two control columns + ratios.
    csv_path = RESULTS_ROOT / f'probe-results-{PROBE_VARIANT}{suffix}.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'target', 'layer', 'variant',
            'trained_test_norm_mse',
            'random_init_test_norm_mse', 'ratio_random_init_over_trained',
            'const_label_test_norm_mse', 'ratio_const_label_over_trained',
            'trained_test_mse', 'trained_val_norm_mse',
            'best_epoch', 'attn_arg_mean', 'wall_s',
        ])
        for (target, layer, variant), c in sorted(trained.items()):
            c1 = ctrl1.get((target, layer, variant))
            c2 = ctrl2.get((target, layer, variant))
            c1nm = c1['test_norm_mse'] if c1 else None
            c2nm = c2['test_norm_mse'] if c2 else None
            r1 = (c1nm / c['test_norm_mse']) if (c1nm is not None and c['test_norm_mse'] > 0) else None
            r2 = (c2nm / c['test_norm_mse']) if (c2nm is not None and c['test_norm_mse'] > 0) else None
            w.writerow([
                target, layer, variant,
                f'{c["test_norm_mse"]:.6g}',
                f'{c1nm:.6g}' if c1nm is not None else 'TBD',
                f'{r1:.3f}' if r1 is not None else 'TBD',
                f'{c2nm:.6g}' if c2nm is not None else 'TBD',
                f'{r2:.3f}' if r2 is not None else 'TBD',
                f'{c["test_mse"]:.6g}',
                f'{c["val_norm_mse"]:.6g}',
                c['best_epoch'],
                f'{c["attention_argmax_per_t_mean"]:.2f}',
                f'{c["wall_s"]:.1f}',
            ])
    print(f'[probe-render] wrote {csv_path}')

    # Summary table: per-target final-layer nMSE + first-decoded-layer vs each control.
    md_lines = [
        f'# Probing summary: {run}',
        '',
        '`trained_nMSE@ℓ=*` is the test-set normalized MSE at the layer that '
        'minimizes it. `first-decoded vs <control>` is the smallest ℓ at which '
        'the trained probe beats that control by ≥ 2×.',
        '',
    ]
    header_cells = ['target', 'best ℓ (lin)', 'lin trained nMSE',
                    'best ℓ (mlp)', 'mlp trained nMSE']
    if ctrl1:
        header_cells += ['fd vs random-init (lin)', 'fd vs random-init (mlp)']
    if ctrl2:
        header_cells += ['fd vs const-label (lin)', 'fd vs const-label (mlp)']
    md_lines.append('| ' + ' | '.join(header_cells) + ' |')
    md_lines.append('|' + '|'.join(['---'] * len(header_cells)) + '|')
    for target, lab in TARGETS_DISPLAY:
        if target not in ordered_targets:
            continue
        lin = _series(trained, target, 'linear')
        mlp = _series(trained, target, 'mlp')
        def _argmin(s):
            if s is None: return None, None
            i = int(np.argmin(s[:, 1]))
            return int(s[i, 0]), float(s[i, 1])
        bl_lin, vl_lin = _argmin(lin)
        bl_mlp, vl_mlp = _argmin(mlp)
        c1_lin = _series(ctrl1, target, 'linear') if ctrl1 else None
        c1_mlp = _series(ctrl1, target, 'mlp')    if ctrl1 else None
        c2_lin = _series(ctrl2, target, 'linear') if ctrl2 else None
        c2_mlp = _series(ctrl2, target, 'mlp')    if ctrl2 else None
        row = [
            lab,
            (f'ℓ={bl_lin}' if bl_lin is not None else '—'),
            (f'{vl_lin:.4f}' if vl_lin is not None else '—'),
            (f'ℓ={bl_mlp}' if bl_mlp is not None else '—'),
            (f'{vl_mlp:.4f}' if vl_mlp is not None else '—'),
        ]
        if ctrl1:
            fd1l = _first_decoded_layer(lin, c1_lin)
            fd1m = _first_decoded_layer(mlp, c1_mlp)
            row += [(f'ℓ={fd1l}' if fd1l is not None else 'never'),
                    (f'ℓ={fd1m}' if fd1m is not None else 'never')]
        if ctrl2:
            fd2l = _first_decoded_layer(lin, c2_lin)
            fd2m = _first_decoded_layer(mlp, c2_mlp)
            row += [(f'ℓ={fd2l}' if fd2l is not None else 'never'),
                    (f'ℓ={fd2m}' if fd2m is not None else 'never')]
        md_lines.append('| ' + ' | '.join(row) + ' |')
    md_lines.append('')
    if not ctrl1 and not ctrl2:
        md_lines.append('_Both noise floors pending._')
    elif not ctrl1:
        md_lines.append('_random-init noise floor pending._')
    elif not ctrl2:
        md_lines.append('_constant-label noise floor pending._')
    md_path = RESULTS_ROOT / f'probe-summary-{PROBE_VARIANT}{suffix}.md'
    md_path.write_text('\n'.join(md_lines) + '\n')
    print(f'[probe-render] wrote {md_path}')
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='D_logu_act')
    p.add_argument('--control', default=None,
                   help='primary control run (random-init)')
    p.add_argument('--control-2', default=None,
                   help='secondary control run (constant-label-trained)')
    p.add_argument('--probe-kind', choices=['attn', 'noattn'], default='attn',
                   help='which probe-result family to read + which suffix to write')
    args = p.parse_args()
    return render(args.run, args.control, args.control_2, args.probe_kind)


if __name__ == '__main__':
    sys.exit(main())
