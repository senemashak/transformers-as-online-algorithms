"""
Cross-distribution probe-curve comparison.

Reads probe results from both `logu-act` and `disc-act` variant directories
and renders a single figure with both distributions overlaid per panel:
    - linear / MLP curves for each (distribution, variant) pair
    - 4 lines per panel total (2 distributions × 2 probe types)

Output: report/figures/probe-curves-disc-vs-logu.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

OVERLEAF_FIGS = V3_ROOT / 'report' / 'figures'
RESULTS_ROOT = V3_ROOT / 'results' / 'probing'


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


def _load(variant: str, run: str, probe_kind: str = 'attn') -> Dict[tuple, dict]:
    suffix = '' if probe_kind == 'attn' else f'_{probe_kind}'
    path = RESULTS_ROOT / variant / 'runs' / run / f'probe_results{suffix}.json'
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {(c['target'], c['layer'], c['variant']): c for c in raw['cells']}


def _series(results: Dict[tuple, dict], target: str, variant: str
            ) -> Optional[np.ndarray]:
    rows = [(c['layer'], c['test_norm_mse']) for k, c in results.items()
            if c['target'] == target and c['variant'] == variant]
    if not rows:
        return None
    rows.sort()
    return np.array(rows)


def render(probe_kind: str = 'attn') -> int:
    disc = _load('disc-act', 'D_disc_act', probe_kind)
    logu = _load('logu-act', 'D_logu_act', probe_kind)
    if not disc or not logu:
        print('[render-cross] missing results; expected disc-act and logu-act runs')
        return 1

    targets = [t for t, _ in TARGETS_DISPLAY]
    cols = 4
    rows = int(np.ceil(len(targets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), dpi=140)
    axes = np.atleast_1d(axes).flatten()

    pretty = {k: lab for k, lab in TARGETS_DISPLAY}

    # Color scheme:
    #   Linear curves: red (D_disc) and blue (D_logu).
    #   MLP curves:    orange (D_disc) and purple (D_logu) — distinct, saturated.
    style = {
        ('disc', 'linear'): dict(color='#d62728', ls='-',  lw=1.8, marker='o', label=r'linear ($\mathcal{D}_\mathrm{disc}$)'),
        ('disc', 'mlp'):    dict(color='#ff7f0e', ls='--', lw=1.8, marker='s', label=r'MLP ($\mathcal{D}_\mathrm{disc}$)'),
        ('logu', 'linear'): dict(color='#1f77b4', ls='-',  lw=1.8, marker='o', label=r'linear ($\mathcal{D}_\mathrm{logu}$)'),
        ('logu', 'mlp'):    dict(color='#9467bd', ls='--', lw=1.8, marker='s', label=r'MLP ($\mathcal{D}_\mathrm{logu}$)'),
    }

    for ai, target in enumerate(targets):
        ax = axes[ai]
        for dist_key, data in [('disc', disc), ('logu', logu)]:
            for variant in ('linear', 'mlp'):
                s = _series(data, target, variant)
                if s is None:
                    continue
                ax.plot(s[:, 0], s[:, 1], **style[(dist_key, variant)])
        ax.axhline(1.0, color='black', lw=0.6, ls=':', alpha=0.6)
        ax.set_xlabel(r'layer $\ell$', fontsize=10)
        ax.set_ylabel('normalized MSE', fontsize=10)
        ax.set_title(pretty.get(target, target), fontsize=10)
        ax.set_xticks(range(9))
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ymin = 1e-5 if target == 'X_bar' else 1e-3
        ax.set_ylim(ymin, 2.5)

    for ai in range(len(targets), len(axes)):
        axes[ai].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower right',
                   bbox_to_anchor=(0.99, 0.04), fontsize=11,
                   frameon=True, ncol=1)

    title_kind = '' if probe_kind == 'attn' else '  [per-timestep probe, no α]'
    fig.suptitle(
        r'Cross-distribution probing: $\mathcal{D}_{\mathrm{disc}}$\_act vs.\ '
        r'$\mathcal{D}_{\mathrm{logu}}$\_act' + title_kind,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OVERLEAF_FIGS.mkdir(parents=True, exist_ok=True)
    suffix = '' if probe_kind == 'attn' else f'-{probe_kind}'
    out_path = OVERLEAF_FIGS / f'probe-curves-disc-vs-logu{suffix}.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[render-cross] wrote {out_path}')

    # Also write a copy under results/probing/ for symmetry.
    local_out = RESULTS_ROOT / f'probe-curves-disc-vs-logu{suffix}.png'
    import shutil
    shutil.copy2(out_path, local_out)
    print(f'[render-cross] copied → {local_out}')
    return 0


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--probe-kind', choices=['attn', 'noattn'], default='attn')
    args = p.parse_args()
    sys.exit(render(args.probe_kind))
