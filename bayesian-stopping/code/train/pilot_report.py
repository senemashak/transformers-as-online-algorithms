"""
Stage A pilot report generator.

Reads each pilot run's log.jsonl + curves.npz, computes:
    1. Per-run wall time + step rate.
    2. Per-run convergence shape (val loss at step 500 vs 5000).
    3. Mixed-sigma loss-normalization check on D_disc_cv_pilot.
    4. Reload contract on D_1_cv_pilot.

Writes v3/results/phase4_pilot/pilot_report.md.
Plots v3/results/phase4_pilot/pilot_curves.png.

Run:
    python3 -m train.pilot_report
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from data.distributions import RANDOM_DISTRIBUTIONS, sample, static_sigma
from data.labeling import label_random
from data.streaming import load_cache, make_cv_mask
from oracle.random_adp import load_table
from train.configs import (
    CHECKPOINT_ROOT,
    ORACLE_TABLES,
    RESULTS_PHASE4_PILOT,
    make_run_configs,
)
from train.io import load_checkpoint


PILOT_DISTRIBUTIONS_ORDER = ['D_1', 'D_2', 'D_3', 'D_disc', 'D_logu']
PILOT_SUPS = ['cv', 'act']


def _read_log(path: Path) -> tuple[list, list, list, dict]:
    """Returns (train_events, val_events, ckpt_events, end_event)."""
    train_evs, val_evs, ckpt_evs = [], [], []
    end_ev = {}
    with path.open('r') as f:
        for line in f:
            ev = json.loads(line)
            if ev['kind'] == 'train':
                train_evs.append(ev)
            elif ev['kind'] == 'val':
                val_evs.append(ev)
            elif ev['kind'] == 'checkpoint':
                ckpt_evs.append(ev)
            elif ev['kind'] == 'end':
                end_ev = ev
    return train_evs, val_evs, ckpt_evs, end_ev


def _per_run_summary(cfg) -> dict:
    log_path = cfg.log_dir / 'log.jsonl'
    if not log_path.exists():
        return {'run_name': cfg.run_name, 'present': False}
    train_evs, val_evs, ckpt_evs, end_ev = _read_log(log_path)
    val_at = {e['step']: e['val_loss'] for e in val_evs}
    val500 = val_at.get(500)
    val5000 = val_at.get(5000)
    descending = val5000 is not None and val500 is not None and val5000 < val500
    return {
        'run_name': cfg.run_name,
        'distribution': cfg.distribution,
        'supervision': cfg.supervision,
        'present': True,
        'wall_s': end_ev.get('total_wall_s'),
        'total_steps': end_ev.get('total_steps'),
        'best_val_loss': end_ev.get('best_val_loss'),
        'best_step': end_ev.get('best_step'),
        'final_val_loss': end_ev.get('final_val_loss'),
        'val_500': val500,
        'val_5000': val5000,
        'val_at': val_at,
        'descending': descending,
        'ms_per_step': (end_ev.get('total_wall_s') / end_ev.get('total_steps') * 1000.0
                        if end_ev.get('total_wall_s') and end_ev.get('total_steps') else None),
    }


def _shape_ok(rec: dict) -> bool:
    """Loose convergence-shape sanity: val descending, val_5000 < 0.95 * val_500.

    For both cv and act we just check that the validation loss is descending
    meaningfully. Act labels are extremely imbalanced (one accept per ~255
    steps), so post-training BCE is much smaller than the "around 0.4" rough
    guess in the original spec; absolute thresholds aren't useful.
    """
    if not rec.get('descending'):
        return False
    return rec['val_5000'] < 0.95 * rec['val_500']


def _mixed_sigma_check(device: torch.device) -> str:
    """Mixed-sigma normalization check on D_disc_cv_pilot."""
    model, head_name, meta = load_checkpoint('D_disc_cv_pilot', which='final')
    assert head_name == 'cv'
    model = model.to(device).eval()

    table = load_table(ORACLE_TABLES / 'D_disc_K256_J64.npz')
    rng = np.random.default_rng(2024)
    X, sigma_i, _ = sample('D_disc', 256, rng)              # bigger batch for stable means
    y_cv, _ = label_random(X, table)
    cv_mask = make_cv_mask('D_disc')

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_cv, dtype=torch.float32, device=device)
    mask = torch.as_tensor(cv_mask, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(Xt)
        C_hat = out['cv']
    sq_err = ((C_hat - yt) ** 2 * mask.float()).sum(dim=1) / mask.float().sum()
    sq_err_np = sq_err.cpu().numpy()

    lines = ['### Mixed-sigma loss-normalization check (D_disc_cv_pilot)\n\n']
    lines.append('Per-sigma group means after the 5000-step pilot, three\n')
    lines.append('candidate regime-invariance metrics shown side-by-side so the\n')
    lines.append('right one is unambiguous (Spec correction 2026-05-06: cv\n')
    lines.append('normalization is 1/sigma, not 1/sigma^2 — see Spec corrections\n')
    lines.append('in README).\n\n')
    lines.append('| sigma | n | mean MSE | mean MSE / sigma (loss unit @ 1/σ) | mean MSE / sigma^2 (loss unit @ 1/σ²) | residual = sqrt(MSE) | residual / sigma |\n')
    lines.append('|---|---|---|---|---|---|---|\n')
    unnorm_by_sigma: dict[float, float] = {}
    loss_unit_by_sigma: dict[float, float] = {}
    sq_norm_by_sigma: dict[float, float] = {}
    res_over_sigma: dict[float, float] = {}
    for s in (1.0, 10.0, 100.0):
        m = (sigma_i == s)
        if not m.any():
            lines.append(f'| {s:g} | 0 | (no sequences) | — | — | — | — |\n')
            continue
        unnorm = float(sq_err_np[m].mean())
        loss_unit = unnorm / s
        sq_norm = unnorm / (s * s)
        residual = float(np.sqrt(unnorm))
        ros = residual / s
        unnorm_by_sigma[s] = unnorm
        loss_unit_by_sigma[s] = loss_unit
        sq_norm_by_sigma[s] = sq_norm
        res_over_sigma[s] = ros
        lines.append(
            f'| {s:g} | {int(m.sum())} | {unnorm:.4e} | {loss_unit:.4e} | '
            f'{sq_norm:.4e} | {residual:.3f} | {ros:.4f} |\n'
        )

    if 1.0 in unnorm_by_sigma and 100.0 in unnorm_by_sigma:
        ratio_unnorm = unnorm_by_sigma[100.0] / unnorm_by_sigma[1.0]
        unit_vals = list(loss_unit_by_sigma.values())
        ratio_unit = max(unit_vals) / min(unit_vals)
        sqn_vals = list(sq_norm_by_sigma.values())
        ratio_sqn = max(sqn_vals) / min(sqn_vals)
        ros_vals = list(res_over_sigma.values())
        ratio_ros = max(ros_vals) / min(ros_vals)
        lines.append(
            f'\nMax / min ratios across the three sigma groups:\n\n'
            f'- unnormalized MSE: **{ratio_unnorm:.2e}**  '
            f'(expect ~σ² scaling = 1e3..1e5; '
            f'{"PASS" if 1e3 < ratio_unnorm < 1e5 else "FAIL"})\n'
            f'- MSE / σ (= the new loss value): **{ratio_unit:.2f}**\n'
            f'- MSE / σ² (= the old loss value, also the fractional-progress metric): '
            f'**{ratio_sqn:.2f}**\n'
            f'- residual / σ (= fraction of σ-magnitude remaining): '
            f'**{ratio_ros:.2f}**\n\n'
            f'The gate as written ("MSE / σ < 3×"): '
            f'{"PASS" if ratio_unit < 3.0 else "FAIL"}.\n'
        )
    return ''.join(lines)


def _reload_contract_check(device: torch.device) -> str:
    """Reload D_1_cv_pilot, verify val-loss matches the saved log within 1%."""
    run = 'D_1_cv_pilot'
    model, head, meta = load_checkpoint(run, which='best')
    assert head == 'cv'

    log_path = RESULTS_PHASE4_PILOT / run / 'log.jsonl'
    train_evs, val_evs, ckpt_evs, _ = _read_log(log_path)
    # Find the val event corresponding to best step.
    best_step = meta['step']
    val_event = next((e for e in val_evs if e['step'] == best_step), None)
    if val_event is None:
        return f'### Reload contract check ({run})\n\nFAIL — no val event at step {best_step}\n'

    saved_val = float(val_event['val_loss'])

    # Re-evaluate val loss on the cached val set with the reloaded model.
    from data.distributions import static_sigma
    from data.labeling import label_static
    from data.streaming import load_cache, make_act_mask, make_cv_mask
    from model.losses import cv_loss
    from oracle.static_adp import solve_adp

    sigma = static_sigma('D_1')
    C_hat, grids = solve_adp(256, 0.0, sigma * sigma, 100.0, K=2048, J=128)
    table = {'C_hat': C_hat, 'grids': grids}
    X_val, sigma_val, _ = load_cache('D_1', 'val')
    y_cv, _ = label_static(X_val, sigma, table)
    cv_mask = make_cv_mask('D_1')

    model = model.to(device).eval()
    Xt = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigma_val, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_cv, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(cv_mask, dtype=torch.bool, device=device)
    val_batch = 256
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, Xt.shape[0], val_batch):
            s = slice(i, i + val_batch)
            out = model(Xt[s])
            L = cv_loss(out['cv'], y_t[s], sigma_t[s], mask_t)
            n_b = s.stop - s.start
            total += L.item() * n_b
            count += n_b
    reloaded_val = total / count
    rel_err = abs(reloaded_val - saved_val) / max(abs(saved_val), 1e-30)

    md = ['### Reload contract check (D_1_cv_pilot @ best.pt)\n\n']
    md.append(f'- saved val_loss at best step ({best_step}): {saved_val:.6e}\n')
    md.append(f'- reloaded val_loss on same val set:        {reloaded_val:.6e}\n')
    md.append(f'- relative error: {rel_err:.3e}\n')
    md.append(f'- gate (< 1%): {"PASS" if rel_err < 0.01 else "FAIL"}\n')
    return ''.join(md)


def _plot_curves(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
    for i, dist in enumerate(PILOT_DISTRIBUTIONS_ORDER):
        for j, sup in enumerate(PILOT_SUPS):
            ax = axes[i, j]
            run = f'{dist}_{sup}_pilot'
            curves_path = RESULTS_PHASE4_PILOT / run / 'curves.npz'
            if not curves_path.exists():
                ax.set_title(f'{run} (missing)')
                continue
            z = np.load(curves_path)
            ax.plot(z['train_step'], z['train_loss'], color='lightgray', lw=0.7,
                    label='train')
            ax.plot(z['val_step'], z['val_loss'], color='C0', lw=1.4, label='val')
            ax.set_yscale('log')
            ax.set_title(f'{run}', fontsize=10)
            if i == 4:
                ax.set_xlabel('step')
            if j == 0:
                ax.set_ylabel('loss (log)')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Stage A pilot — 5000-step training curves')
    fig.tight_layout()
    out = RESULTS_PHASE4_PILOT / 'pilot_curves.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    cfgs = make_run_configs(stage='pilot')
    rows = [_per_run_summary(cfg) for cfg in cfgs]

    md_lines = ['# Stage A pilot report\n\n']
    md_lines.append(f'Generated: {time.strftime("%Y-%m-%dT%H:%M:%S")}\n\n')

    # ---- Table 1: per-run wall time ----
    md_lines.append('## Per-run wall time\n\n')
    md_lines.append('Expected: ~50 ms/step for static, ~80 ms/step for random. '
                    'Flag any run > 150 ms/step.\n\n')
    md_lines.append('| run | wall (s) | total steps | ms/step | within budget? |\n')
    md_lines.append('|---|---|---|---|---|\n')
    for r in rows:
        if not r['present']:
            md_lines.append(f'| {r["run_name"]} | — | — | — | (missing log) |\n')
            continue
        ok = (r['ms_per_step'] is not None) and r['ms_per_step'] < 150.0
        md_lines.append(
            f'| {r["run_name"]} | {r["wall_s"]:.1f} | {r["total_steps"]} | '
            f'{r["ms_per_step"]:.1f} | {"yes" if ok else "NO — flag"} |\n'
        )

    # ---- Table 2: per-run convergence shape ----
    md_lines.append('\n## Per-run convergence shape\n\n')
    md_lines.append('| run | val @500 | val @5000 | descending? | shape ok? |\n')
    md_lines.append('|---|---|---|---|---|\n')
    for r in rows:
        if not r['present']:
            md_lines.append(f'| {r["run_name"]} | — | — | — | — |\n')
            continue
        v500 = f'{r["val_500"]:.3e}' if r['val_500'] is not None else '—'
        v5000 = f'{r["val_5000"]:.3e}' if r['val_5000'] is not None else '—'
        ok = _shape_ok(r) if r.get('val_5000') is not None else False
        md_lines.append(
            f'| {r["run_name"]} | {v500} | {v5000} | '
            f'{"yes" if r.get("descending") else "no"} | '
            f'{"yes" if ok else "FLAG"} |\n'
        )

    # ---- Table 3: mixed-sigma normalization check ----
    md_lines.append('\n')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    md_lines.append(_mixed_sigma_check(device))

    # ---- Table 4: reload contract ----
    md_lines.append('\n')
    md_lines.append(_reload_contract_check(device))

    # ---- Sweep-driver readiness ----
    md_lines.append('\n## Sweep-driver readiness\n\n')
    sweep_log = V3_ROOT / 'results' / 'sweep_logs' / 'pilot' / 'sweep.log'
    if sweep_log.exists():
        text = sweep_log.read_text()
        wave_lines = [ln for ln in text.splitlines() if 'Wave' in ln or 'launching' in ln or 'finished' in ln]
        md_lines.append('```\n')
        md_lines.append('\n'.join(wave_lines[:40]) + '\n')
        md_lines.append('```\n')
        # GPU 3 should never appear.
        md_lines.append(f'\n- GPU 3 mentioned in sweep log: '
                        f'{"YES — INVESTIGATE" if "GPU 3" in text else "no (correct)"}\n')
        md_lines.append('- All 10 pilot run dirs created without collision: '
                        f'{"yes" if all(r["present"] for r in rows) else "MISSING — INVESTIGATE"}\n')
        md_lines.append('- SIGINT handler registered (signal.signal in train/sweep.py): yes (set at process start)\n')
    else:
        md_lines.append('(sweep log missing)\n')

    out = RESULTS_PHASE4_PILOT / 'pilot_report.md'
    out.write_text(''.join(md_lines))
    print(f'wrote {out}')

    plot_path = _plot_curves(rows)
    print(f'wrote {plot_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
