"""Per-σ payoff breakdown for the four random-variance trained models.

For each random-variance model (D_disc_*, D_logu_*) on its own training
distribution, run on the seed-44 per-σ test cache. Bin sequences by σ_i.
For each bin, report R / R* where R* is the *per-distribution random-ADP
oracle* — same Bayes-optimal policy used as the regime baseline in the
main payoff matrix. This is the "honest gap": both the model and the
oracle only know the prior over σ, not the per-sequence σ.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from eval.policies import model_action, payoff, random_oracle


D_DISC_BINS = [1.0, 10.0, 100.0]                                 # 3 bins
D_LOGU_BIN_EDGES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]                # 5 bins on log_10 σ


def _bin_indices(sigma_i: np.ndarray, distribution: str) -> List[Dict]:
    """Returns a list of bin descriptors with mask + representative σ."""
    bins: List[Dict] = []
    if distribution == 'D_disc':
        for s in D_DISC_BINS:
            mask = sigma_i == s
            if mask.any():
                bins.append({
                    'name': f'σ={int(s)}',
                    'mask': mask,
                    'sigma_repr': float(s),
                    'count': int(mask.sum()),
                })
    elif distribution == 'D_logu':
        log10 = np.log10(sigma_i)
        for lo, hi in zip(D_LOGU_BIN_EDGES[:-1], D_LOGU_BIN_EDGES[1:]):
            if hi == D_LOGU_BIN_EDGES[-1]:
                mask = (log10 >= lo) & (log10 <= hi)
            else:
                mask = (log10 >= lo) & (log10 < hi)
            if mask.any():
                # Bin representative σ: 10**(midpoint of the log decade).
                sigma_repr = float(10 ** ((lo + hi) / 2.0))
                bins.append({
                    'name': f'log₁₀σ∈[{lo:.1f},{hi:.1f})',
                    'mask': mask,
                    'sigma_repr': sigma_repr,
                    'count': int(mask.sum()),
                })
    else:
        raise ValueError(f'no per-σ binning for {distribution!r}')
    return bins


def per_sigma_breakdown(
    model, head: str, X: np.ndarray, sigma_i: np.ndarray,
    distribution: str, device, random_table: dict,
) -> List[Dict]:
    """Run the model on the persigma test cache; per-bin compute R/R*
    where R* is the per-distribution random-ADP oracle (the same regime
    oracle used in the main payoff matrix).

    Returns a list of dicts (one per bin):
        name, sigma_repr, count, R, R_star, R_over_R_star.
    """
    if distribution not in ('D_disc', 'D_logu'):
        raise ValueError(distribution)

    model_act, _ = model_action(model, X, head, device)
    oracle_act, _ = random_oracle(X, random_table)

    model_payoffs = payoff(model_act, X)
    oracle_payoffs = payoff(oracle_act, X)

    bins = _bin_indices(sigma_i, distribution)
    results: List[Dict] = []
    for b in bins:
        idx = np.where(b['mask'])[0]
        R = float(model_payoffs[idx].mean())
        R_star = float(oracle_payoffs[idx].mean())
        results.append({
            'name': b['name'],
            'sigma_repr': b['sigma_repr'],
            'count': b['count'],
            'R': R,
            'R_star': R_star,
            'R_over_R_star': float(R / R_star) if R_star else float('nan'),
        })
    return results
