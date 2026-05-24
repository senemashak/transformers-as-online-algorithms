"""
Sweep-level summary writers and plotters, written for use after Stage B
finishes. Reads each run's log.jsonl + curves.npz, produces:

  v3/results/phase4/training_curves.png
  v3/results/phase4/training_curves_per_sigma.png
  v3/results/phase4/training_curves_per_distribution.pdf
  v3/results/phase4/sweep_summary.md
  v3/results/phase4/sweep_summary.csv
  v3/results/phase4/reproducibility.md

Run:  python3 -m train.sweep_summary
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from train.configs import (
    CHECKPOINT_ROOT,
    RESULTS_PHASE4,
    make_run_configs,
)


DISTRIBUTIONS = ['D_1', 'D_2', 'D_3', 'D_disc', 'D_logu']
SUPS = ['cv', 'act']

# Display run name -> actual data source. D_disc_cv / D_logu_cv panels read from
# the σ² ablation runs since those are the ones the §6 results use.
RUN_OVERRIDE = {'D_disc_cv': 'D_disc_cv_sig2', 'D_logu_cv': 'D_logu_cv_sig2'}


def _read_log(path: Path):
    train_evs, val_evs, ckpt_evs = [], [], []
    end_ev = {}
    if not path.exists():
        return train_evs, val_evs, ckpt_evs, end_ev
    with path.open('r') as f:
        for line in f:
            ev = json.loads(line)
            kind = ev['kind']
            if kind == 'train':
                train_evs.append(ev)
            elif kind == 'val':
                val_evs.append(ev)
            elif kind == 'checkpoint':
                ckpt_evs.append(ev)
            elif kind == 'end':
                end_ev = ev
    return train_evs, val_evs, ckpt_evs, end_ev


def _per_run_summary(cfg) -> dict:
    log_path = cfg.log_dir / 'log.jsonl'
    train_evs, val_evs, ckpt_evs, end_ev = _read_log(log_path)
    final_per_sigma = {k: v for k, v in (val_evs[-1].items() if val_evs else {})
                       if k.startswith('val_loss_sigma_')}
    return {
        'run_name': cfg.run_name,
        'distribution': cfg.distribution,
        'supervision': cfg.supervision,
        'present': bool(end_ev),
        'wall_s': end_ev.get('total_wall_s'),
        'total_steps': end_ev.get('total_steps'),
        'final_train_loss': end_ev.get('final_train_loss'),
        'final_val_loss': end_ev.get('final_val_loss'),
        'best_val_loss': end_ev.get('best_val_loss'),
        'best_step': end_ev.get('best_step'),
        'best_path': str(cfg.checkpoint_dir / 'best.pt'),
        'final_per_sigma': final_per_sigma,
    }


def _global_y_range(rows: list[dict], supervision: str) -> tuple[float, float]:
    """Loose log-y range covering both train and val for one supervision."""
    lo = float('inf')
    hi = -float('inf')
    for r in rows:
        if r['supervision'] != supervision:
            continue
        actual_run = RUN_OVERRIDE.get(r['run_name'], r['run_name'])
        curves = _load_curves(actual_run)
        if curves is None:
            continue
        for arr_name in ('train_loss', 'val_loss'):
            arr = curves[arr_name]
            if arr.size:
                pos = arr[arr > 0]
                if pos.size:
                    lo = min(lo, float(pos.min()))
                    hi = max(hi, float(pos.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (1e-4, 1e2)
    # Pad by a decade on each side.
    return (lo * 0.5, hi * 2)


def _curves_path_for_run(run_name: str) -> Path:
    return RESULTS_PHASE4 / run_name / 'curves.npz'


def _load_curves(run_name: str) -> dict | None:
    """Load (train_step, train_loss, val_step, val_loss) for a run.

    Prefers curves.npz; falls back to reading log.jsonl when the run was
    truncated and never wrote curves.npz. Returns None if neither source
    is available.
    """
    npz_path = _curves_path_for_run(run_name)
    if npz_path.exists():
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}
    log_path = RESULTS_PHASE4 / run_name / 'log.jsonl'
    if not log_path.exists():
        return None
    train_step, train_loss, val_step, val_loss = [], [], [], []
    val_per_sigma: dict[str, list[float]] = {}
    val_per_sigma_step: list[int] = []
    with log_path.open('r') as f:
        for line in f:
            ev = json.loads(line)
            kind = ev.get('kind')
            if kind == 'train':
                train_step.append(ev['step'])
                train_loss.append(ev['train_loss'])
            elif kind == 'val':
                val_step.append(ev['step'])
                val_loss.append(ev['val_loss'])
                val_per_sigma_step.append(ev['step'])
                for k, v in ev.items():
                    if k.startswith('val_loss_sigma_'):
                        val_per_sigma.setdefault(k, []).append(v)
    out = {
        'train_step': np.asarray(train_step, dtype=np.int64),
        'train_loss': np.asarray(train_loss, dtype=np.float64),
        'val_step':   np.asarray(val_step,   dtype=np.int64),
        'val_loss':   np.asarray(val_loss,   dtype=np.float64),
    }
    for k, v in val_per_sigma.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

def plot_training_curves_grid(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=False)
    cv_y = _global_y_range(rows, 'cv')
    act_y = _global_y_range(rows, 'act')
    for i, dist in enumerate(DISTRIBUTIONS):
        for j, sup in enumerate(SUPS):
            ax = axes[i, j]
            run = f'{dist}_{sup}'                                # display label
            actual_run = RUN_OVERRIDE.get(run, run)              # data source
            z = _load_curves(actual_run)
            if z is None:
                ax.set_title(f'{run} (missing)', fontsize=10)
                continue
            ax.plot(z['train_step'], z['train_loss'], color='#cccccc', lw=0.7, label='train')
            ax.plot(z['val_step'], z['val_loss'], color='C0', lw=1.4, label='val')
            # Mark best step (val-loss minimum)
            row = next((r for r in rows if r['run_name'] == actual_run), None)
            best_step = row.get('best_step') if row else None
            if best_step is None and z['val_loss'].size:
                best_step = int(z['val_step'][int(np.argmin(z['val_loss']))])
            if best_step:
                ax.axvline(best_step, color='red', lw=0.8, alpha=0.5,
                           linestyle='--')
            ax.set_yscale('log')
            if sup == 'cv':
                ax.set_ylim(cv_y)
            else:
                ax.set_ylim(act_y[0], min(act_y[1], 0.04))
            ax.set_title(run, fontsize=10)
            if i == 4:
                ax.set_xlabel('step')
            if j == 0:
                ax.set_ylabel('loss (log)')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Stage B training curves — train (light) + val (dark), best step (red)')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_sigma_curves(rows: list[dict], out: Path) -> None:
    """4-panel grid: D_disc_cv, D_disc_act, D_logu_cv, D_logu_act with
    per-σ-group val curves overlaid."""
    runs = ('D_disc_cv', 'D_disc_act', 'D_logu_cv', 'D_logu_act')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for ax, run in zip(axes.flatten(), runs):
        actual_run = RUN_OVERRIDE.get(run, run)
        z = _load_curves(actual_run)
        if z is None:
            ax.set_title(f'{run} (missing)', fontsize=11)
            continue
        ax.plot(z['val_step'], z['val_loss'],
                color='black', lw=1.6, label='aggregate val')
        cmap = plt.get_cmap('viridis')
        sigma_keys = sorted([k for k in z if k.startswith('val_loss_sigma_')])
        for k_idx, k in enumerate(sigma_keys):
            label = k.replace('val_loss_', '')
            ax.plot(z['val_step'], z[k],
                    color=cmap(0.15 + 0.7 * k_idx / max(1, len(sigma_keys) - 1)),
                    lw=1.2, alpha=0.85, label=label)
        ax.set_yscale('log')
        ax.set_xlabel('step')
        ax.set_ylabel('val loss (log)')
        ax.set_title(run, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Per-σ-group val loss curves on the four random-variance runs')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_distribution_pdf(rows: list[dict], out: Path) -> None:
    """One panel per distribution; cv (left axis) and act (right axis)
    with separate y-scales. Vector PDF."""
    fig, axes = plt.subplots(5, 1, figsize=(9, 16), sharex=False)
    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        cv_curves = _curves_path_for_run(f'{dist}_cv')
        act_curves = _curves_path_for_run(f'{dist}_act')
        if cv_curves.exists():
            z = np.load(cv_curves)
            ax.plot(z['val_step'], z['val_loss'], color='C0', lw=1.4, label='cv val')
        ax2 = ax.twinx()
        if act_curves.exists():
            z = np.load(act_curves)
            ax2.plot(z['val_step'], z['val_loss'], color='C3', lw=1.4, label='act val')
        ax.set_yscale('log')
        ax2.set_yscale('log')
        ax.set_xlabel('step')
        ax.set_ylabel('cv val loss (log)', color='C0')
        ax2.set_ylabel('act val loss (log)', color='C3')
        ax.set_title(dist, fontsize=12)
        ax.grid(True, alpha=0.3)
        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    fig.suptitle('Per-distribution training curves (cv + act, separate y-axes)')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabular writers
# ---------------------------------------------------------------------------

def write_summary_md(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ['# Stage B sweep summary\n\n']
    lines.append(f'Generated: {time.strftime("%Y-%m-%dT%H:%M:%S")}\n\n')
    lines.append('## Per-run\n\n')
    lines.append('| run | wall (s) | total steps | final train | final val | best val | best step | best path |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|\n')
    for r in rows:
        if not r['present']:
            lines.append(f'| {r["run_name"]} | — | — | — | — | — | — | (missing) |\n')
            continue
        lines.append(
            f'| {r["run_name"]} | {r["wall_s"]:.0f} | {r["total_steps"]} | '
            f'{r["final_train_loss"]:.4e} | {r["final_val_loss"]:.4e} | '
            f'{r["best_val_loss"]:.4e} | {r["best_step"]} | '
            f'`{r["best_path"]}` |\n'
        )

    # Per-σ breakdown for random-variance runs.
    lines.append('\n## Final per-σ-group val loss (random-variance runs)\n\n')
    lines.append('val_loss is the loss-unit value (per-sequence MSE / σ for cv;\n')
    lines.append('per-sequence BCE for act). For D_logu, σ-bins are log_10 σ ∈\n')
    lines.append('[0, 2/3) low, [2/3, 4/3) mid, [4/3, 2] high.\n\n')
    rand_rows = [r for r in rows if r['distribution'] in ('D_disc', 'D_logu')]
    if rand_rows:
        # Determine column union
        keys = sorted({k for r in rand_rows for k in r['final_per_sigma'].keys()})
        header = ['run'] + keys
        lines.append('| ' + ' | '.join(header) + ' |\n')
        lines.append('|' + '|'.join(['---'] * len(header)) + '|\n')
        for r in rand_rows:
            cells = [r['run_name']]
            for k in keys:
                v = r['final_per_sigma'].get(k)
                cells.append(f'{v:.4e}' if v is not None else '—')
            lines.append('| ' + ' | '.join(cells) + ' |\n')
    out.write_text(''.join(lines))


def write_summary_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r['final_per_sigma'].keys()})
    fieldnames = [
        'run_name', 'distribution', 'supervision', 'wall_s', 'total_steps',
        'final_train_loss', 'final_val_loss',
        'best_val_loss', 'best_step', 'best_path',
    ] + keys
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames if k not in keys}
            for k in keys:
                row[k] = r['final_per_sigma'].get(k)
            w.writerow(row)


def write_reproducibility_md(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = """# Reproducibility — Stage B trained models

How to reload any of the 10 trained models for downstream analysis.

## Reload contract

```python
from train.io import load_checkpoint

model, head_name, metadata = load_checkpoint('D_disc_cv', which='best')
# model: GPTStopper instance, both heads loaded, on CPU, in eval() mode
# head_name: 'cv' or 'act' — query only this head
# metadata: dict with config, environment, val_loss, step, etc.

model = model.cuda()      # move to GPU
out = model(X)            # X: (B, n) torch tensor of raw observations
                          # out: {'cv': (B, n) Ĉ_t, 'act': (B, n) logits}
predictions = out[head_name]
```

`which` accepts `'best'`, `'final'`, or `'step_50k'`/`'step_100k'`/etc.
For Stage B's 200k-step static cv runs, periodic checkpoints at
50k/100k/150k/200k are available. For 300k random cv runs, also at 250k
and 300k. Act runs save periodics every 50k up through their step count
(100k for static, 150k for random).

## Canonical fields

The `trained_head` field in each `.pt` is authoritative — filenames are
hints. Step 5+ should never assume the head from the filename.

`metadata['config']` contains the full RunConfig dump:
distribution, supervision, step_count, val_every, periodic_every, lr,
batch_size, warmup_frac, seed, n, d_emb, n_layers, n_heads.

`metadata['environment']` contains: torch_version, numpy_version,
cuda_version, gpu_name, hostname, python_version, training_start,
training_end, total_wall_s.

## Per-σ-group val loss fields (random-variance runs)

Each `val` event in `log.jsonl` carries:

  - `val_loss` — aggregate (per-sequence loss averaged over the val set)
  - `val_loss_sigma_1`, `val_loss_sigma_10`, `val_loss_sigma_100` — for
    D_disc; mean per-sequence loss within each σ group
  - `val_loss_sigma_low`, `val_loss_sigma_mid`, `val_loss_sigma_high` —
    for D_logu; bins are log_10 σ ∈ [0, 2/3), [2/3, 4/3), [4/3, 2]

For static-σ runs, `val_loss_sigma_{σ}` duplicates the aggregate.

These are also exposed in `curves.npz` keyed by the same names.

## Loss form

cv loss is **per-sequence MSE divided by σ_i** (NOT σ_i²) — see
"Spec corrections" entry of 2026-05-06 in v3/README.md. Validation
val_loss is in the same units. For random-variance runs, val_loss is
dominated by σ=100 sequences (since MSE scales as σ² and we divide by
σ — leaving a σ factor in the absolute scale).

act loss is per-sequence mean BCE-with-logits. Naturally regime-invariant.

## Per-run files

Each `v3/checkpoints/<run_name>/` contains:
  - `best.pt`, `final.pt`, `step_*k.pt`
  - `config.json`  (RunConfig dump)
  - `metadata.json`  (env + timing)

Each `v3/results/phase4/<run_name>/` contains:
  - `log.jsonl` (one event per line; kinds: start, train, val, checkpoint, end)
  - `curves.npz` (dense step / loss arrays)
"""
    out.write_text(text)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    cfgs = make_run_configs(stage='full')
    rows = [_per_run_summary(cfg) for cfg in cfgs]
    out_root = RESULTS_PHASE4
    out_root.mkdir(parents=True, exist_ok=True)

    write_summary_md(rows, out_root / 'sweep_summary.md')
    write_summary_csv(rows, out_root / 'sweep_summary.csv')
    write_reproducibility_md(rows, out_root / 'reproducibility.md')
    plot_training_curves_grid(rows, out_root / 'training_curves.png')
    plot_per_sigma_curves(rows, out_root / 'training_curves_per_sigma.png')
    plot_per_distribution_pdf(rows, out_root / 'training_curves_per_distribution.pdf')
    print(f'wrote sweep-level artifacts to {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
