"""
V2's `label_sequences`, ported verbatim into v3 for the Smoke 1 diff only.

Source: v2/code/dataset.py:label_sequences (lines 79-110 in V2). Imports
go through V3's `oracle/` modules where the conjugate helpers now live;
the math is identical.

This module exists so Smoke 1 can compare the V3 static labeler bit-for-
bit against V2's reference. Once Step 3 lands, this file can be deleted —
the V3 static labeler is the canonical implementation going forward.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from oracle.conjugate import interp_uniform, posterior_path_batch


def label_sequences_v2(
    X: np.ndarray, mu_0: float, tau0_2: float, sigma2: float,
    C_hat: np.ndarray, grids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Verbatim from v2/code/dataset.py:label_sequences (line 79).

    Args:
        X:     (N, n) observations.
        mu_0, tau0_2, sigma2: prior + noise parameters at known sigma.
        C_hat, grids: precomputed static-ADP threshold table for that sigma.

    Returns:
        y_cv:  (N, n-1) float64.
        y_act: (N, n-1) float64.
    """
    N, n = X.shape
    mu_path, _ = posterior_path_batch(X, mu_0, tau0_2, sigma2)
    y_cv = np.empty((N, n - 1), dtype=np.float64)
    for i in range(n - 1):
        y_cv[:, i] = interp_uniform(mu_path[:, i], grids[i], C_hat[i])
    y_act = (X[:, : n - 1] >= y_cv).astype(np.float64)
    return y_cv, y_act
