"""Payoff metric: R = E[X_τ], R/R*."""

from __future__ import annotations

import numpy as np

from eval.policies import payoff as _payoff


def expected_payoff(action: np.ndarray, X: np.ndarray) -> float:
    """R = E[X_τ] estimated as mean over the test set."""
    return float(_payoff(action, X).mean())


def normalized_payoff(R: float, R_oracle: float) -> float:
    """R / R*. Bounded above by 1 (when policy = oracle)."""
    if R_oracle == 0:
        return float('nan')
    return R / R_oracle
