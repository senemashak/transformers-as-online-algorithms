"""Render the four figure families for the prelim mech-interp run.

(a) Layerwise true vs perm_glob test-error per target.
(b) Paired-improvement bar chart (mean Δ + 95% CI) per target.
(c) Probe-label budget curve: validation loss vs M_train, per target/layer/rep.

All read from <run_dir>/members/*/probe_runs/*/probe_results.json and
<run_dir>/stats/per_cell_paired.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from interp.prelim_v2.stats import percentile_interval

PAIRED_COLOR = '#1f3b73'                                # dark blue for paired-Δ plots


TARGET_LABELS = {
    'sigma_mle_sqrt':              r'$\sqrt{\hat\sigma_t^2}$',
    'C_star':                      r'$\widehat{C}^\star_t$',
    'sigma_hat_exp_relmse':        r'$\hat\sigma_t$ (exp-output, rel-MSE)',
    'eta_t_raw_mse':               r'$\eta_t$',
    'loglik_true_params_raw_mse':  r'$\log p(X_{1:t}\mid\mu_i,\sigma_i)$',
}
# Y-axis label per target (the primary loss this target was trained against).
TARGET_YLABEL = {
    'sigma_mle_sqrt':              r'$\sigma$-normalized MSE (test)',
    'C_star':                      r'$\sigma$-normalized MSE (test)',
    'sigma_hat_exp_relmse':        r'relative $\sigma$-MSE (test)',
    'eta_t_raw_mse':               r'$\eta_t$ MSE (test)',
    'loglik_true_params_raw_mse':  r'log-lik MSE (test)',
}
TARGET_YLABEL_VAL = {
    'sigma_mle_sqrt':              r'best val loss ($\sigma$-norm MSE)',
    'C_star':                      r'best val loss ($\sigma$-norm MSE)',
    'sigma_hat_exp_relmse':        r'best val loss (rel $\sigma$-MSE)',
    'eta_t_raw_mse':               r'best val loss ($\eta_t$ MSE)',
    'loglik_true_params_raw_mse':  r'best val loss (log-lik MSE)',
}
REP_COLORS = {
    'true':      '#1f77b4',
    'perm_glob': '#d62728',
    'init':      '#e6a92c',                              # orange-amber for the untrained-init baseline
}
REP_ORDER = ('init', 'perm_glob', 'true')   # draw init first (back), true last (top)
HEADLINE_M_TRAIN = 1024


def _read_member_rows(member_dir: Path) -> dict:
    """Return {rep_name: list of rows} for every rep this member has under
    probe_runs/<rep>/probe_results.json. Concatenates probe_results_loglik.json
    if present so the loglik target is auto-detected by downstream collectors."""
    out = {}
    for rep in REP_ORDER:
        p = member_dir / 'probe_runs' / rep / 'probe_results.json'
        if p.exists():
            rows = json.loads(p.read_text())
            ll = member_dir / 'probe_runs' / rep / 'probe_results_loglik.json'
            if ll.exists():
                rows = rows + json.loads(ll.read_text())
            out[rep] = rows
    if 'true' not in out or 'perm_glob' not in out:
        raise FileNotFoundError(f'{member_dir} missing true or perm_glob probe outputs')
    return out


def _collect_full_budget(run_dir: Path, M_train: int = HEADLINE_M_TRAIN) -> dict:
    """Return {(rep, target, layer): list of per-member test_loss}."""
    members_dir = run_dir / 'members'
    out = defaultdict(list)
    for d in sorted(members_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            reps = _read_member_rows(d)
        except FileNotFoundError:
            continue
        for rep_name, rows in reps.items():
            for r in rows:
                if r['M_train_sequences'] != M_train:
                    continue
                out[(rep_name, r['target'], r['layer'])].append(r['test_loss'])
    return out


def _collect_budget_curves(run_dir: Path) -> dict:
    """Return {(rep, target, layer): {M_train: list of val_loss_best}}."""
    members_dir = run_dir / 'members'
    out: dict = defaultdict(lambda: defaultdict(list))
    for d in sorted(members_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            reps = _read_member_rows(d)
        except FileNotFoundError:
            continue
        for rep_name, rows in reps.items():
            for r in rows:
                key = (rep_name, r['target'], r['layer'])
                out[key][r['M_train_sequences']].append(r['val_loss_best'])
    return out


def _read_per_cell_csv(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / 'stats' / 'per_cell_paired.csv').open() as f:
        for r in csv.DictReader(f):
            for k in ('layer', 'M_train', 'N', 'W'):
                r[k] = int(r[k])
            for k in ('mean_delta', 'sd_delta', 'se_delta',
                      'ci_lo_95', 'ci_hi_95', 'p_sign_one_sided',
                      'mean_test_true', 'mean_test_perm_glob'):
                r[k] = float(r[k])
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# (a) Layerwise true vs perm_glob test error
# ---------------------------------------------------------------------------

def fig_layerwise(run_dir: Path) -> None:
    full = _collect_full_budget(run_dir)
    targets = sorted({t for (_, t, _) in full})
    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        layers = sorted({l for (rep, t, l) in full if t == tgt})
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        N_ref = 0
        for rep in REP_ORDER:
            means = np.full(len(layers), np.nan)
            ci_lo = np.full(len(layers), np.nan)
            ci_hi = np.full(len(layers), np.nan)
            for i, l in enumerate(layers):
                v = np.asarray(full.get((rep, tgt, l), []), dtype=float)
                if v.size == 0:
                    continue
                means[i] = v.mean()
                if v.size > 1:
                    ci_lo[i], ci_hi[i] = percentile_interval(v)   # 2.5/97.5 pct of members
                    N_ref = max(N_ref, v.size)
                else:
                    ci_lo[i] = ci_hi[i] = means[i]
            xs = np.asarray(layers)
            ax.plot(xs, means, marker='o', lw=2,
                    color=REP_COLORS[rep], label=rep)
            ax.fill_between(xs, ci_lo, ci_hi,
                            color=REP_COLORS[rep], alpha=0.20, linewidth=0)
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(TARGET_YLABEL.get(tgt, r'$\sigma$-normalized MSE (test)'))
        suffix = f' (N={N_ref})' if N_ref else ''
        ax.set_title(f'Layerwise probe test error: {TARGET_LABELS.get(tgt, tgt)}{suffix}')
        ax.set_xticks(layers)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        path = out_dir / f'layerwise_{tgt}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path}')


# ---------------------------------------------------------------------------
# (b) Paired improvement bar chart
# ---------------------------------------------------------------------------

def fig_paired_improvement(run_dir: Path) -> None:
    rows = _read_per_cell_csv(run_dir)
    targets = sorted({r['target'] for r in rows})
    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        tr = [r for r in rows if r['target'] == tgt]
        tr.sort(key=lambda r: r['layer'])
        layers = np.asarray([r['layer'] for r in tr])
        means = np.asarray([r['mean_delta'] for r in tr])
        ci_lo = np.asarray([r['ci_lo_95'] for r in tr])
        ci_hi = np.asarray([r['ci_hi_95'] for r in tr])
        sd = np.asarray([r['sd_delta'] for r in tr])
        Ws = [r['W'] for r in tr]
        N = tr[0]['N']
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        ax.fill_between(layers, ci_lo, ci_hi,
                        color='#2ca02c', alpha=0.12, linewidth=0,
                        label='central 95% (2.5--97.5 pct)')
        ax.fill_between(layers, means - sd, means + sd,
                        color='#2ca02c', alpha=0.28, linewidth=0,
                        label=r'$\pm 1$ std (members)')
        ax.plot(layers, means, marker='o', lw=2, color='#2ca02c',
                label=r'mean $\bar\Delta$')
        ax.axhline(0.0, color='black', lw=0.8, ls='-')
        # Headroom so the W labels don't collide with the top axis border.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
        for l, hi, W in zip(layers, ci_hi, Ws):
            ax.annotate(f'W={W}/{N}', xy=(l, hi), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=PAIRED_COLOR)
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(r'$\bar\Delta$ = mean(perm_glob − true) test error')
        ax.set_title(f'Paired improvement (true vs perm_glob): '
                     f'{TARGET_LABELS.get(tgt, tgt)} (N={N})')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best')
        fig.tight_layout()
        path = out_dir / f'paired_improvement_{tgt}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path}')


# ---------------------------------------------------------------------------
# (c) Probe-label budget curves
# ---------------------------------------------------------------------------

def fig_budget_curves(run_dir: Path) -> None:
    curves = _collect_budget_curves(run_dir)
    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({t for (_, t, _) in curves})
    layers = sorted({l for (_, _, l) in curves})
    for tgt in targets:
        # |T_v| = 255 for both targets (sigma_mle_sqrt: t in [2..256]; C_star: t in [1..255]).
        T_v = 255
        n_cols = 3
        n_rows = (len(layers) + n_cols - 1) // n_cols
        # Pre-compute per-panel data so we can set a shared y-range BEFORE
        # rendering. ymin/ymax are over the (mean ± SE) bands of every panel.
        panel_data = {}                                  # layer -> {rep: (labels, means, stds)}
        y_lo, y_hi = float('inf'), float('-inf')
        for layer in layers:
            d = {}
            for rep in REP_ORDER:
                bd = curves.get((rep, tgt, layer), {})
                if not bd:
                    continue
                budgets = sorted(bd)
                means = np.asarray([np.mean(bd[b]) for b in budgets])
                stds = np.asarray([
                    np.std(bd[b], ddof=1) if len(bd[b]) > 1 else 0.0
                    for b in budgets
                ])
                labels = np.asarray([b * T_v for b in budgets])
                d[rep] = (labels, means, stds)
                y_lo = min(y_lo, float(np.nanmin(means - stds)))
                y_hi = max(y_hi, float(np.nanmax(means + stds)))
            panel_data[layer] = d
        # Pad the shared range slightly so error bars aren't clipped.
        if y_lo > 0:                                     # log-safe
            y_lo *= 0.85
        else:
            y_lo -= 0.05 * max(abs(y_hi), 1.0)
        y_hi *= 1.15

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), dpi=130)
        axes = np.atleast_1d(axes).flatten()
        for ax, layer in zip(axes, layers):
            for rep in REP_ORDER:
                if rep not in panel_data[layer]:
                    continue
                labels, means, stds = panel_data[layer][rep]
                ax.plot(labels, means, marker='o', lw=1.8,
                        color=REP_COLORS[rep], label=rep)
                ax.fill_between(labels, means - stds, means + stds,
                                color=REP_COLORS[rep], alpha=0.20, linewidth=0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(y_lo, y_hi)                      # SHARED y-axis
            ax.set_title(f'layer {layer}')
            ax.set_xlabel(r'probe labels  $B_v = M_{\mathrm{train}} \cdot |T_v|$')
            ax.set_ylabel(TARGET_YLABEL_VAL.get(tgt, r'best val loss ($\sigma$-norm MSE)'))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        for ax in axes[len(layers):]:
            ax.set_visible(False)
        fig.suptitle(f'Probe-label budget sweep: {TARGET_LABELS.get(tgt, tgt)}  '
                     r'(mean $\pm$ std across members; shared y-axis)',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = out_dir / f'budget_curves_{tgt}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path}')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--skip', nargs='*', default=(),
                   choices=['layerwise', 'paired', 'budget'])
    a = p.parse_args()
    if 'layerwise' not in a.skip:
        fig_layerwise(a.run_dir)
    if 'paired' not in a.skip:
        fig_paired_improvement(a.run_dir)
    if 'budget' not in a.skip:
        fig_budget_curves(a.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
