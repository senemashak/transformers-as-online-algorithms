"""D_logu_cv diagnostic — disambiguate gradient-imbalance (H1) vs.
cv-head linear-bottleneck (H2) hypotheses for the low-σ failure.

Both hypotheses predict the same monotonic per-σ payoff curve. They
differ on the cv-head's loss-unit error |Ĉ_t - C*_t| / σ_i:
    H1 → grows monotonically toward low σ (mis-scaled threshold).
    H2 → roughly flat (σ-invariant representation noise).

Run on the seed-44 D_logu per-σ test cache; bin into 5 log_10 σ bins;
solve static ADP once at each bin's geometric-mean σ; compare the
trained model's cv-head output Ĉ_t to that bin's oracle C*_t at the
same posterior-mean state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.distributions import MU_0, TAU0_2
from eval.per_sigma_payoff import D_LOGU_BIN_EDGES
from eval.run_eval import _load_persigma_test
from oracle.conjugate import posterior_path_batch
from oracle.static_adp import C_hat_lin, solve_adp
from train.configs import V3_ROOT as _V3_ROOT
from train.io import load_checkpoint


V3_ROOT = Path(_V3_ROOT)
EXP_FIGS = V3_ROOT / 'report' / 'figures'


def main() -> int:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[diag] device={device}')

    model, head, _ = load_checkpoint('D_logu_cv', which='best')
    model = model.to(device).eval()
    assert head == 'cv'

    X_np, sigma_i, _ = _load_persigma_test('D_logu')
    N, n = X_np.shape

    # Trained-model Ĉ_t (cv head) on the whole cache.
    with torch.no_grad():
        X_t = torch.from_numpy(X_np.astype(np.float32)).to(device)
        out = model(X_t)
        C_hat_model = out['cv'].cpu().numpy().astype(np.float64)     # (N, n)

    # Bin by log_10 σ_i; bin reference σ = 10^(midpoint of decade).
    log10 = np.log10(sigma_i)
    bins = []
    for lo, hi in zip(D_LOGU_BIN_EDGES[:-1], D_LOGU_BIN_EDGES[1:]):
        last = (hi == D_LOGU_BIN_EDGES[-1])
        mask = (log10 >= lo) & (log10 <= hi) if last else (log10 >= lo) & (log10 < hi)
        bins.append({
            'name': f'[{lo:.1f},{hi:.1f}{")" if not last else "]"}',
            'mask': mask,
            'sigma_ref': float(10 ** ((lo + hi) / 2.0)),
            'count': int(mask.sum()),
        })

    # Solve static ADP once per bin reference σ.
    print('[diag] solving static ADP at 5 reference σ values')
    tables = {}
    for b in bins:
        s = b['sigma_ref']
        C_h, g = solve_adp(n, MU_0, s * s, TAU0_2, K=2048, J=128)
        tables[s] = {'C_hat': C_h, 'grids': g}

    abs_err = []
    loss_unit_err = []
    for b in bins:
        idx = np.where(b['mask'])[0]
        if idx.size == 0:
            abs_err.append(np.nan)
            loss_unit_err.append(np.nan)
            continue
        s_ref = b['sigma_ref']
        tbl = tables[s_ref]
        X_b = X_np[idx]
        sigma_b = sigma_i[idx]
        # Posterior path computed with the bin's reference σ (matches the
        # ADP table's grid; same posterior the oracle would use under
        # known-σ = bin reference σ).
        mu_path, _ = posterior_path_batch(X_b, MU_0, TAU0_2, s_ref * s_ref)
        C_star = np.zeros_like(X_b)
        for i in range(n - 1):
            C_star[:, i] = C_hat_lin(i, mu_path[:, i],
                                     tbl['C_hat'], tbl['grids'])
        # cv-mask range: t-index 1..n-2 (1-indexed t ∈ {2, ..., n-1}).
        delta = np.abs(C_hat_model[idx] - C_star)[:, 1:n - 1]
        abs_err.append(float(delta.mean()))
        loss_unit_err.append(float((delta / sigma_b[:, None]).mean()))

    names = [b['name'] for b in bins]
    counts = [b['count'] for b in bins]

    print(f'[diag] abs_err:        {abs_err}')
    print(f'[diag] loss_unit_err:  {loss_unit_err}')

    EXP_FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=130)
    bars = ax.bar(names, abs_err, color='#4C72B0', alpha=0.9)
    ax.set_ylabel(r'mean $|\hat{C}_t - C^*_t|$ (raw units)', fontsize=10)
    ax.set_xlabel(r'$\log_{10}\sigma$ bin', fontsize=10)
    ax.set_title('D_logu_cv diagnostic — absolute error of cv-head output vs. bin-σ oracle',
                 fontsize=11)
    ymax = max(abs_err) * 1.18
    ax.set_ylim(0, ymax)
    for bar, v, c in zip(bars, abs_err, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.01,
                f'{v:.3f}\n(n={c})', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_FIGS / 'diagnostic-logu-abs-error.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=130)
    bars = ax.bar(names, loss_unit_err, color='#DD8452', alpha=0.9)
    ax.set_ylabel(r'mean $|\hat{C}_t - C^*_t|\, /\, \sigma_i$  (loss units)', fontsize=10)
    ax.set_xlabel(r'$\log_{10}\sigma$ bin', fontsize=10)
    ax.set_title('D_logu_cv diagnostic — loss-unit error (cv loss is MSE / σ_i)',
                 fontsize=11)
    ymax = max(loss_unit_err) * 1.18
    ax.set_ylim(0, ymax)
    for bar, v, c in zip(bars, loss_unit_err, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.01,
                f'{v:.3f}\n(n={c})', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_FIGS / 'diagnostic-logu-loss-unit-error.png', dpi=300)
    plt.close(fig)

    print(f'[diag] wrote {EXP_FIGS / "diagnostic-logu-abs-error.png"}')
    print(f'[diag] wrote {EXP_FIGS / "diagnostic-logu-loss-unit-error.png"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
