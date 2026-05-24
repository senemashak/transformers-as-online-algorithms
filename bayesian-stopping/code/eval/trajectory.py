"""Threshold trajectory: mean and std across test sequences as a function of t."""

from __future__ import annotations

import numpy as np


def trajectory_mean_std(threshold: np.ndarray) -> dict:
    """threshold: (N_seq, n) per-(seq, t) threshold value. Returns dict
    with `mean`, `std` (length n) — both over the sequence axis. NaN
    entries are ignored.
    """
    if np.isnan(threshold).all():
        n = threshold.shape[1]
        return {'mean': np.full(n, np.nan), 'std': np.full(n, np.nan)}
    mean = np.nanmean(threshold, axis=0)
    std = np.nanstd(threshold, axis=0)
    return {'mean': mean, 'std': std}
