"""Per-step action agreement between two policies on the same test set.

Restricted to the decision steps t = 1..n-1 (terminal t = n is forced
acceptance under both policies, contributes 1.0 trivially).
"""

from __future__ import annotations

import numpy as np


def per_step_agreement(action_a: np.ndarray, action_b: np.ndarray) -> float:
    """action_a, action_b: (N_seq, n) bool. Agreement over (seq, t in 1..n-1)."""
    if action_a.shape != action_b.shape:
        raise ValueError(
            f'shape mismatch: {action_a.shape} vs {action_b.shape}'
        )
    n = action_a.shape[1]
    return float((action_a[:, : n - 1] == action_b[:, : n - 1]).mean())
