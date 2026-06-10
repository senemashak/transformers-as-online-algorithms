"""Figures for the attention-pooled run + simple-vs-attn comparison.

Produces:
  <attn_run_dir>/figures/
    layerwise_<target>.png            attn-alone, true vs perm_glob (95% CI)
    paired_improvement_<target>.png   attn-alone, paired Δ across layers
    budget_curves_<target>.png        attn-alone, panel per layer (±std)
    simple_vs_attn_<target>.png       overlay of both probe families
    attn_heatmap_<target>_<rep>.png   probe-position attention at t in {25,125,250}
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

from interp.prelim_v2.probe import t_start_for
from interp.prelim_v2.stats import percentile_interval

PAIRED_COLOR = '#1f3b73'                                # dark blue for paired-Δ plots


TARGET_LABELS = {
    'sigma_mle_sqrt':              r'$\sqrt{\hat\sigma_t^2}$',
    'C_star':                      r'$\widehat{C}^\star_t$',
    'sigma_hat_exp_relmse':        r'$\hat\sigma_t$ (exp-output, rel-MSE)',
    'eta_t_raw_mse':               r'$\eta_t$',
    'loglik_true_params_raw_mse':  r'$\log p(X_{1:t}\mid\mu_i,\sigma_i)$',
}
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
    'init':      '#e6a92c',                              # orange-amber for untrained-init baseline
}
REP_ORDER = ('init', 'perm_glob', 'true')   # draw init first (back), true last (top)
PROBE_LINESTYLE = {
    'simple': '-',
    'attn':   '--',
}
HEADLINE_M_TRAIN = 1024

# Attn outputs live in probe_runs_attn/ in the unified N=100 layout.
ATTN_SUBDIR = 'probe_runs_attn'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _members_complete(src_dir: Path, attn_dir: Path,
                      attn_subdir: str = 'probe_runs') -> list[int]:
    out = []
    for d in sorted((src_dir / 'members').iterdir()):
        if not d.is_dir() and not d.is_symlink():
            continue
        i = int(d.name)
        if not all([
            (d / 'probe_runs' / 'true'      / 'probe_results.json').exists(),
            (d / 'probe_runs' / 'perm_glob' / 'probe_results.json').exists(),
            (attn_dir / 'members' / d.name / attn_subdir / 'true'      / 'probe_results.json').exists(),
            (attn_dir / 'members' / d.name / attn_subdir / 'perm_glob' / 'probe_results.json').exists(),
        ]):
            continue
        out.append(i)
    return out


def _load_rows(run_dir: Path, member: int, rep: str,
               subdir: str = 'probe_runs') -> list[dict]:
    """Load probe_results.json + probe_results_loglik.json (if present) and
    concatenate, so the loglik target is auto-detected by collectors."""
    base = run_dir / 'members' / f'{member:03d}' / subdir / rep
    rows = json.loads((base / 'probe_results.json').read_text())
    ll = base / 'probe_results_loglik.json'
    if ll.exists():
        rows = rows + json.loads(ll.read_text())
    return rows


def _reps_present(run_dir: Path, members: list[int], subdir: str) -> list[str]:
    """Reps with probe_results.json in every member dir under <subdir>/<rep>/."""
    out = []
    for rep in REP_ORDER:
        if all((run_dir / 'members' / f'{m:03d}' / subdir / rep / 'probe_results.json').exists()
               for m in members):
            out.append(rep)
    return out


def _collect_full_budget(run_dir: Path, members: list[int],
                         M_train: int = HEADLINE_M_TRAIN,
                         subdir: str = 'probe_runs') -> dict:
    reps = _reps_present(run_dir, members, subdir)
    out = defaultdict(list)
    for i in members:
        for rep in reps:
            for r in _load_rows(run_dir, i, rep, subdir=subdir):
                if r['M_train_sequences'] != M_train:
                    continue
                out[(rep, r['target'], r['layer'])].append(r['test_loss'])
    return out


def _collect_budget_curves(run_dir: Path, members: list[int],
                           subdir: str = 'probe_runs') -> dict:
    reps = _reps_present(run_dir, members, subdir)
    out: dict = defaultdict(lambda: defaultdict(list))
    for i in members:
        for rep in reps:
            for r in _load_rows(run_dir, i, rep, subdir=subdir):
                key = (rep, r['target'], r['layer'])
                out[key][r['M_train_sequences']].append(r['val_loss_best'])
    return out


def _read_attn_paired_csv(attn_run_dir: Path) -> list[dict]:
    rows = []
    # Try the unified-layout location first, fall back to legacy.
    path = attn_run_dir / 'stats_attn' / 'per_cell_paired.csv'
    if not path.exists():
        path = attn_run_dir / 'stats' / 'per_cell_paired.csv'
    with path.open() as f:
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
# (a) Layerwise test sMSE: attn-alone
# ---------------------------------------------------------------------------

def fig_attn_layerwise(attn_dir: Path, members: list[int],
                       attn_subdir: str = 'probe_runs') -> None:
    full = _collect_full_budget(attn_dir, members, subdir=attn_subdir)
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({t for (_, t, _) in full})
    for tgt in targets:
        layers = sorted({l for (rep, t, l) in full if t == tgt})
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        N_ref = 0
        for rep in REP_ORDER:
            means = np.full(len(layers), np.nan)
            ci_lo = np.full(len(layers), np.nan); ci_hi = np.full(len(layers), np.nan)
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
            ax.plot(xs, means, marker='o', lw=2, color=REP_COLORS[rep], label=rep)
            ax.fill_between(xs, ci_lo, ci_hi, color=REP_COLORS[rep],
                            alpha=0.20, linewidth=0)
        suffix = f' (N={N_ref})' if N_ref else ''
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(TARGET_YLABEL.get(tgt, r'$\sigma$-normalized MSE (test)'))
        ax.set_title(f'Attention-pooled probe layerwise error: '
                     f'{TARGET_LABELS.get(tgt, tgt)}{suffix}')
        ax.set_xticks(layers); ax.set_yscale('log')
        ax.grid(True, alpha=0.3); ax.legend(loc='best')
        fig.tight_layout()
        p = out_dir / f'attn_layerwise_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (b) Paired improvement: attn-alone
# ---------------------------------------------------------------------------

def fig_attn_paired(attn_dir: Path) -> None:
    rows = _read_attn_paired_csv(attn_dir)
    targets = sorted({r['target'] for r in rows})
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        tr = sorted([r for r in rows if r['target'] == tgt],
                    key=lambda r: r['layer'])
        layers = np.asarray([r['layer'] for r in tr])
        means = np.asarray([r['mean_delta'] for r in tr])
        ci_lo = np.asarray([r['ci_lo_95'] for r in tr])
        ci_hi = np.asarray([r['ci_hi_95'] for r in tr])
        sd = np.asarray([r['sd_delta'] for r in tr])
        Ws = [r['W'] for r in tr]
        N = tr[0]['N']
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        ax.fill_between(layers, ci_lo, ci_hi, color='#2ca02c', alpha=0.12,
                        linewidth=0, label='central 95% (2.5--97.5 pct)')
        ax.fill_between(layers, means - sd, means + sd, color='#2ca02c', alpha=0.28,
                        linewidth=0, label=r'$\pm 1$ std (members)')
        ax.plot(layers, means, marker='o', lw=2, color='#2ca02c',
                label=r'mean $\bar\Delta$')
        ax.axhline(0.0, color='black', lw=0.8)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
        for l, hi, W in zip(layers, ci_hi, Ws):
            ax.annotate(f'W={W}/{N}', xy=(l, hi), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=PAIRED_COLOR)
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(r'$\bar\Delta$ = mean(perm_glob − true) test error')
        ax.set_title(f'Attention-pooled paired improvement: '
                     f'{TARGET_LABELS.get(tgt, tgt)} (N={N})')
        ax.set_xticks(layers); ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best')
        fig.tight_layout()
        p = out_dir / f'attn_paired_improvement_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (c) Budget curves: attn-alone
# ---------------------------------------------------------------------------

def fig_attn_budget(attn_dir: Path, members: list[int],
                    attn_subdir: str = 'probe_runs') -> None:
    curves = _collect_budget_curves(attn_dir, members, subdir=attn_subdir)
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({t for (_, t, _) in curves})
    layers = sorted({l for (_, _, l) in curves})
    for tgt in targets:
        T_v = 255 if tgt == 'C_star' else 254
        n_cols = 3
        n_rows = (len(layers) + n_cols - 1) // n_cols
        panel_data: dict = {}
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
        if y_lo > 0: y_lo *= 0.85
        else:        y_lo -= 0.05 * max(abs(y_hi), 1.0)
        y_hi *= 1.15

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3 * n_rows), dpi=130)
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
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(y_lo, y_hi)
            ax.set_title(f'layer {layer}')
            ax.set_xlabel(r'probe labels  $B_v = M_{\mathrm{train}}\cdot|\mathcal{T}_v|$')
            ax.set_ylabel(TARGET_YLABEL_VAL.get(tgt, r'best val loss ($\sigma$-norm MSE)'))
            ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8)
        for ax in axes[len(layers):]:
            ax.set_visible(False)
        fig.suptitle(f'Attention-pooled budget sweep: '
                     f'{TARGET_LABELS.get(tgt, tgt)}  '
                     r'(mean $\pm$ std across members; shared y-axis)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = out_dir / f'attn_budget_curves_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (d) Simple vs attention overlay
# ---------------------------------------------------------------------------

def fig_simple_vs_attn(src_dir: Path, attn_dir: Path, members: list[int],
                       attn_subdir: str = 'probe_runs') -> None:
    full_src  = _collect_full_budget(src_dir,  members)
    full_attn = _collect_full_budget(attn_dir, members, subdir=attn_subdir)
    targets = sorted({t for (_, t, _) in full_src})
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        layers = sorted({l for (rep, t, l) in full_src if t == tgt})
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=130)
        for probe, store in (('simple', full_src), ('attn', full_attn)):
            reps_in_store = sorted({rep for (rep, t, _) in store.keys() if t == tgt},
                                   key=lambda r: REP_ORDER.index(r) if r in REP_ORDER else 99)
            for rep in reps_in_store:
                means = np.full(len(layers), np.nan)
                ci_lo = np.full(len(layers), np.nan); ci_hi = np.full(len(layers), np.nan)
                sd = np.full(len(layers), np.nan)
                for i, l in enumerate(layers):
                    v = np.asarray(store.get((rep, tgt, l), []), dtype=float)
                    if v.size == 0: continue
                    means[i] = v.mean()
                    if v.size > 1:
                        ci_lo[i], ci_hi[i] = percentile_interval(v)   # 2.5/97.5 pct of members
                        sd[i] = v.std(ddof=1)
                    else:
                        ci_lo[i] = ci_hi[i] = means[i]
                xs = np.asarray(layers)
                ax.plot(xs, means, marker='o', lw=2,
                        color=REP_COLORS[rep], linestyle=PROBE_LINESTYLE[probe],
                        label=f'{probe} / {rep}')
                # central-95% percentile band (lighter, wider) under the ±1 std band
                ax.fill_between(xs, ci_lo, ci_hi,
                                color=REP_COLORS[rep], alpha=0.06, linewidth=0)
                ax.fill_between(xs, means - sd, means + sd,
                                color=REP_COLORS[rep], alpha=0.14, linewidth=0)
        ax.set_xlabel(r'layer $\ell$')
        ax.set_ylabel(TARGET_YLABEL.get(tgt, r'$\sigma$-normalized MSE (test)'))
        ax.set_title(f'Simple vs attention-pooled probe: '
                     f'{TARGET_LABELS.get(tgt, tgt)} (N={len(members)})')
        ax.set_xticks(layers); ax.set_yscale('log')
        ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        p = out_dir / f'simple_vs_attn_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (d2) Budget curves: simple vs attention overlay (per target, panel per layer)
# ---------------------------------------------------------------------------

def fig_budget_simple_vs_attn(src_dir: Path, attn_dir: Path,
                              members: list[int],
                              attn_subdir: str = 'probe_runs') -> None:
    """Overlay simple- and attention-pooled budget curves, one panel per layer.

    Four lines per panel: {simple, attn} x {true, perm_glob}.
    Bands are mean +/- std across members.
    """
    curves_simple = _collect_budget_curves(src_dir,  members)
    curves_attn   = _collect_budget_curves(attn_dir, members, subdir=attn_subdir)
    targets = sorted({t for (_, t, _) in curves_simple})
    layers  = sorted({l for (_, _, l) in curves_simple})
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in targets:
        T_v = 255 if tgt == 'C_star' else 254
        n_cols = 3
        n_rows = (len(layers) + n_cols - 1) // n_cols

        # Per-panel data: layer -> {(probe, rep): (labels, means, stds)}.
        panel_data: dict = {}
        y_lo, y_hi = float('inf'), float('-inf')
        for layer in layers:
            d: dict = {}
            for probe_tag, store in (('simple', curves_simple),
                                     ('attn',   curves_attn)):
                reps_in_store = sorted({rep for (rep, _, _) in store.keys()},
                                       key=lambda r: REP_ORDER.index(r) if r in REP_ORDER else 99)
                for rep in reps_in_store:
                    bd = store.get((rep, tgt, layer), {})
                    if not bd:
                        continue
                    budgets = sorted(bd)
                    means = np.asarray([np.mean(bd[b]) for b in budgets])
                    stds = np.asarray([
                        np.std(bd[b], ddof=1) if len(bd[b]) > 1 else 0.0
                        for b in budgets
                    ])
                    labels = np.asarray([b * T_v for b in budgets])
                    d[(probe_tag, rep)] = (labels, means, stds)
                    y_lo = min(y_lo, float(np.nanmin(means - stds)))
                    y_hi = max(y_hi, float(np.nanmax(means + stds)))
            panel_data[layer] = d
        if y_lo > 0: y_lo *= 0.85
        else:        y_lo -= 0.05 * max(abs(y_hi), 1.0)
        y_hi *= 1.15

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3 * n_rows), dpi=130)
        axes = np.atleast_1d(axes).flatten()
        for ax, layer in zip(axes, layers):
            for (probe_tag, rep), (labels, means, stds) in panel_data[layer].items():
                ax.plot(labels, means, marker='o', lw=1.8,
                        color=REP_COLORS[rep],
                        linestyle=PROBE_LINESTYLE[probe_tag],
                        label=f'{probe_tag} / {rep}')
                ax.fill_between(labels, means - stds, means + stds,
                                color=REP_COLORS[rep], alpha=0.12, linewidth=0)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(y_lo, y_hi)
            ax.set_title(f'layer {layer}')
            ax.set_xlabel(r'probe labels  $B_v = M_{\mathrm{train}}\cdot|\mathcal{T}_v|$')
            ax.set_ylabel(TARGET_YLABEL_VAL.get(tgt, r'best val loss ($\sigma$-norm MSE)'))
            ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=7)
        for ax in axes[len(layers):]:
            ax.set_visible(False)
        fig.suptitle(f'Budget sweep, simple vs attention-pooled: '
                     f'{TARGET_LABELS.get(tgt, tgt)}  '
                     r'(mean $\pm$ std across members; shared y-axis)',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = out_dir / f'budget_curves_simple_vs_attn_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (e) Probe attention heatmaps at representative prefixes
# ---------------------------------------------------------------------------

def _alpha_at_prefix(s_scores: np.ndarray, c: int) -> np.ndarray:
    """Softmax over s with positions > c masked. Returns array of size len(s)."""
    n = s_scores.size
    out = np.full(n, -np.inf)
    out[:c + 1] = s_scores[:c + 1]
    out = out - out.max()                                    # numeric stability
    exp = np.exp(out); exp[c + 1:] = 0.0
    return exp / exp.sum() if exp.sum() > 0 else exp


def _collect_s_vectors(
    run_dir: Path, members: list[int], M_train: int,
    subdir: str = 'probe_runs',
) -> dict:
    """{(target, rep, layer): list of (n,) np arrays} of learned s vectors."""
    bucket: dict = defaultdict(list)
    for i in members:
        for rep in REP_ORDER:
            for r in _load_rows(run_dir, i, rep, subdir=subdir):
                if r['M_train_sequences'] != M_train:
                    continue
                s = np.asarray(r['attn_summary']['s_position_scores'],
                               dtype=np.float32)
                bucket[(r['target'], rep, r['layer'])].append(s)
    return bucket


def fig_attn_heatmaps(attn_dir: Path, members: list[int],
                      M_train: int = HEADLINE_M_TRAIN,
                      prefixes: tuple[int, ...] = (25, 125, 250),
                      attn_subdir: str = 'probe_runs') -> None:
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = _collect_s_vectors(attn_dir, members, M_train, subdir=attn_subdir)
    layers_sorted = sorted({k[2] for k in bucket})
    targets = sorted({k[0] for k in bucket})

    for tgt in targets:
        for rep in REP_ORDER:
            fig, axes = plt.subplots(1, len(prefixes),
                                     figsize=(4.5 * len(prefixes), 3.5),
                                     dpi=130, sharey=True)
            axes = np.atleast_1d(axes).flatten()
            for ax, c in zip(axes, prefixes):
                grid = np.zeros((len(layers_sorted), c + 1), dtype=np.float32)
                for li, layer in enumerate(layers_sorted):
                    s_list = bucket.get((tgt, rep, layer), [])
                    if not s_list:
                        continue
                    alpha_acc = np.zeros_like(s_list[0])
                    for s in s_list:
                        alpha_acc += _alpha_at_prefix(s, c)
                    alpha_acc /= len(s_list)
                    grid[li, :] = alpha_acc[:c + 1]
                im = ax.imshow(grid, aspect='auto', cmap='magma',
                               origin='lower', interpolation='nearest')
                ax.set_title(f'prefix t={c+1}')
                ax.set_xlabel('position i (0..t)')
                ax.set_yticks(range(len(layers_sorted)))
                ax.set_yticklabels(layers_sorted)
                ax.set_ylabel(r'layer $\ell$')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=r'$\bar\alpha_t[i]$')
            fig.suptitle(f'Probe-position attention (averaged across {len(members)} '
                         f'members): {TARGET_LABELS.get(tgt, tgt)} / {rep}',
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            p = out_dir / f'attn_heatmap_{tgt}_{rep}.png'
            fig.savefig(p); plt.close(fig)
            print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (f) Probe attention heatmaps: uplift-over-uniform + true-minus-perm_glob
#     (cleaner reading of position bias across prefixes)
# ---------------------------------------------------------------------------

def _mean_uplift(s_list: list[np.ndarray], c: int) -> np.ndarray:
    """Per-position multiplicative uplift over uniform, averaged across members.

    uplift[i] = (c+1) * alpha[i]; uniform is 1.0, >1 means upweighted vs uniform.
    Positions > c are NaN.
    """
    if not s_list:
        return np.full(s_list[0].size if s_list else 1, np.nan, dtype=np.float32)
    acc = np.zeros_like(s_list[0])
    for s in s_list:
        acc += _alpha_at_prefix(s, c)
    acc /= len(s_list)
    uplift = acc * (c + 1)
    uplift[c + 1:] = np.nan
    return uplift


def fig_attn_heatmaps_uplift(attn_dir: Path, members: list[int],
                             M_train: int = HEADLINE_M_TRAIN,
                             prefixes: tuple[int, ...] = (25, 125, 250),
                             attn_subdir: str = 'probe_runs') -> None:
    """One figure per target. 3 rows x len(prefixes) cols:
       row 1: uplift_true                  (log2 scale, diverging at 0)
       row 2: uplift_perm_glob             (log2 scale)
       row 3: uplift_true - uplift_perm_glob (linear, diverging at 0)
    """
    import matplotlib.colors as mcolors
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = _collect_s_vectors(attn_dir, members, M_train, subdir=attn_subdir)
    layers_sorted = sorted({k[2] for k in bucket})
    targets = sorted({k[0] for k in bucket})
    nL = len(layers_sorted)

    for tgt in targets:
        # Pre-compute three grids per prefix.
        log_true: list[np.ndarray]  = []
        log_pglob: list[np.ndarray] = []
        diff_grids: list[np.ndarray] = []
        for c in prefixes:
            ut = np.full((nL, c + 1), np.nan, dtype=np.float32)
            up = np.full((nL, c + 1), np.nan, dtype=np.float32)
            for li, layer in enumerate(layers_sorted):
                v_t = _mean_uplift(bucket.get((tgt, 'true',      layer), []), c)
                v_p = _mean_uplift(bucket.get((tgt, 'perm_glob', layer), []), c)
                ut[li, :] = v_t[:c + 1]
                up[li, :] = v_p[:c + 1]
            log_true.append(np.log2(np.maximum(ut, 1e-6)))
            log_pglob.append(np.log2(np.maximum(up, 1e-6)))
            diff_grids.append(ut - up)

        # Symmetric ranges shared per row.
        vmax_log = max(np.nanmax(np.abs(g)) for g in (*log_true, *log_pglob))
        vmax_diff = max(np.nanmax(np.abs(g)) for g in diff_grids)
        vmax_log = max(vmax_log, 0.2)                          # at least ±0.2
        vmax_diff = max(vmax_diff, 0.05)

        fig, axes = plt.subplots(3, len(prefixes),
                                 figsize=(4.7 * len(prefixes), 9), dpi=130)
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('lightgray')                              # mask color
        for row, (label, grids, vmax) in enumerate([
            ('true: $\\log_2$ uplift',      log_true,   vmax_log),
            ('perm_glob: $\\log_2$ uplift', log_pglob,  vmax_log),
            ('true $-$ perm_glob (uplift)', diff_grids, vmax_diff),
        ]):
            for col, (c, g) in enumerate(zip(prefixes, grids)):
                ax = axes[row, col]
                norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                im = ax.imshow(g, aspect='auto', cmap=cmap, norm=norm,
                               origin='lower', interpolation='nearest')
                ax.set_title(f'{label}, prefix t={c+1}' if row == 0
                             else f'prefix t={c+1}')
                ax.set_yticks(range(nL)); ax.set_yticklabels(layers_sorted)
                if col == 0:
                    ax.set_ylabel(f'{label.split(":")[0]}\nlayer $\\ell$'
                                  if row < 2 else 'true − pglob\nlayer $\\ell$')
                ax.set_xlabel('position i')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if row < 2:
                    cbar.set_label(r'$\log_2(\alpha\cdot(t+1))$')
                else:
                    cbar.set_label(r'$\Delta$ uplift')
        fig.suptitle(
            f'Probe-position attention, uplift over uniform '
            f'(avg across {len(members)} members): '
            f'{TARGET_LABELS.get(tgt, tgt)}\n'
            r'positive values $\Rightarrow$ probe attends more than a '
            r'flat $\frac{1}{t+1}$ baseline',
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out_dir / f'attn_heatmap_uplift_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# (g) 1D learned position scores: one curve per layer, per (target, rep)
# ---------------------------------------------------------------------------

def fig_attn_s_curves(attn_dir: Path, members: list[int],
                      M_train: int = HEADLINE_M_TRAIN,
                      attn_subdir: str = 'probe_runs') -> None:
    """For each target, two side-by-side panels (true, perm_glob) with one
    curve per layer showing the learned position-score vector s, averaged
    across members. The learned s is what determines every alpha at every t.
    """
    out_dir = attn_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = _collect_s_vectors(attn_dir, members, M_train, subdir=attn_subdir)
    layers_sorted = sorted({k[2] for k in bucket})
    targets = sorted({k[0] for k in bucket})

    for tgt in targets:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=130,
                                 sharey=True)
        cmap = plt.get_cmap('viridis')
        for ax, rep in zip(axes, ('true', 'perm_glob')):
            for li, layer in enumerate(layers_sorted):
                s_list = bucket.get((tgt, rep, layer), [])
                if not s_list:
                    continue
                s_mean = np.mean(np.stack(s_list, axis=0), axis=0)
                s_mean = s_mean - s_mean.mean()                # center per-layer
                color = cmap(li / max(1, len(layers_sorted) - 1))
                ax.plot(np.arange(s_mean.size), s_mean, lw=1.4,
                        color=color, label=f'L{layer}')
            ax.axhline(0.0, color='black', lw=0.6, ls=':')
            ax.set_xlabel('position i')
            ax.set_title(rep)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=7, ncol=3)
        axes[0].set_ylabel(r'centered learned position score $s[i] - \bar s$')
        fig.suptitle(f'Learned probe-position scores (avg across '
                     f'{len(members)} members): {TARGET_LABELS.get(tgt, tgt)}',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = out_dir / f'attn_s_curves_{tgt}.png'
        fig.savefig(p); plt.close(fig)
        print(f'  wrote {p}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=None,
                   help='Unified run dir; attn at probe_runs_attn/.')
    p.add_argument('--src-run-dir', type=Path, default=None,
                   help='(legacy) simple-probe run dir.')
    p.add_argument('--attn-run-dir', type=Path, default=None,
                   help='(legacy) attn-probe run dir.')
    p.add_argument('--skip', nargs='*', default=(),
                   choices=['layerwise', 'paired', 'budget',
                            'simple_vs_attn', 'budget_simple_vs_attn',
                            'heatmaps', 'heatmaps_uplift', 's_curves'])
    a = p.parse_args()
    if a.run_dir is not None:
        src_root, attn_root, attn_subdir = a.run_dir, a.run_dir, ATTN_SUBDIR
    else:
        if a.src_run_dir is None or a.attn_run_dir is None:
            p.error('either --run-dir or both --src-run-dir and --attn-run-dir')
        src_root, attn_root, attn_subdir = a.src_run_dir, a.attn_run_dir, 'probe_runs'

    members = _members_complete(src_root, attn_root, attn_subdir=attn_subdir)
    print(f'[figures_attn] using {len(members)} complete members')
    if 'layerwise'             not in a.skip: fig_attn_layerwise(attn_root, members, attn_subdir=attn_subdir)
    if 'paired'                not in a.skip: fig_attn_paired(attn_root)
    if 'budget'                not in a.skip: fig_attn_budget(attn_root, members, attn_subdir=attn_subdir)
    if 'simple_vs_attn'        not in a.skip: fig_simple_vs_attn(src_root, attn_root, members, attn_subdir=attn_subdir)
    if 'budget_simple_vs_attn' not in a.skip: fig_budget_simple_vs_attn(src_root, attn_root, members, attn_subdir=attn_subdir)
    if 'heatmaps'              not in a.skip: fig_attn_heatmaps(attn_root, members, attn_subdir=attn_subdir)
    if 'heatmaps_uplift'       not in a.skip: fig_attn_heatmaps_uplift(attn_root, members, attn_subdir=attn_subdir)
    if 's_curves'              not in a.skip: fig_attn_s_curves(attn_root, members, attn_subdir=attn_subdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
