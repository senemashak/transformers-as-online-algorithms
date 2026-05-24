"""
Hindsight (offline-prophet) labelers.

Drop-in replacements for `data.labeling.label_random` /
`data.labeling.label_static` that compute their labels purely from the
sampled sequence — no oracle ADP table involved. Used to train the
"offline-supervised" variants of the four random-variance models.

Conventions match the oracle labelers (`data/labeling.py`):

    y_cv:  (N_seq, n) continuation-value targets. Index n-1 (t=n) is a
           placeholder of 0 (terminal step has no continuation value);
           cv_mask in `data/streaming.py` is False there. For the random
           distributions (D_disc, D_logu) cv_mask is also False at index
           0 (t=1) — we still emit a value there for shape parity but it
           is never seen by the loss.
    y_act: (N_seq, n) action labels in {0, 1}. Index n-1 (t=n) is set to
           1 (forced acceptance, matching the existing act loss
           convention).

Definitions (1-indexed t throughout, n-indexed array entries are t-1):
    y_t^cv  = max(X_{t+1}, ..., X_n)              for t = 1..n-1
    y_t^act = 1[X_t >= max(X_{t+1}, ..., X_n)]    for t = 1..n-1
              (note: ties resolve to "accept", same as the oracle labeler)
    y_n^act = 1                                   (forced acceptance)

The strict suffix max is computed via a single reverse-cummax pass —
O(n) per sequence, fully vectorized. Both numpy and torch
implementations are provided so the trainer can label random batches on
GPU and the static-supervised path (which already labels on CPU) can
also use this module.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Strict suffix max
# ---------------------------------------------------------------------------

def _suffix_max_strict_np(X: np.ndarray) -> np.ndarray:
    """suffix_max_strict[:, t-1] = max(X[:, t..n-1]) for t = 0..n-2.
    Last column (t = n-1, i.e. strict suffix of the terminal step) is set
    to -inf because the suffix is empty.
    """
    N, n = X.shape
    suf = np.full_like(X, -np.inf)
    # Suffix max including X[t]: cummax from right to left.
    flipped = X[:, ::-1]
    cum_flipped = np.maximum.accumulate(flipped, axis=1)
    inclusive = cum_flipped[:, ::-1]                       # max(X[t..n-1])
    # Strict (exclude X[t]): shift left by one.
    suf[:, : n - 1] = inclusive[:, 1:]
    return suf


def _suffix_max_strict_torch(X: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of `_suffix_max_strict_np`."""
    N, n = X.shape
    flipped = X.flip(dims=(1,))
    cum_flipped = torch.cummax(flipped, dim=1).values
    inclusive = cum_flipped.flip(dims=(1,))               # max(X[t..n-1])
    suf = torch.full_like(X, float('-inf'))
    suf[:, : n - 1] = inclusive[:, 1:]
    return suf


# ---------------------------------------------------------------------------
# Public labelers (numpy)
# ---------------------------------------------------------------------------

def label_offline(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hindsight cv + act labels.

    Args:
        X: (N_seq, n) sampled sequences.

    Returns:
        y_cv: (N_seq, n) float64. y_cv[:, t-1] = max(X[:, t..n-1]) for
              t = 1..n-1. y_cv[:, n-1] = 0 (placeholder, mask False).
              For the random distributions the t=1 entry is also masked
              by `cv_mask` in `data/streaming.py`; we still fill it in
              for shape parity.
        y_act: (N_seq, n) float64 in {0, 1}. y_act[:, t-1] = 1 if
              X[:, t-1] >= max(X[:, t..n-1]) else 0, for t = 1..n-1.
              y_act[:, n-1] = 1 (forced acceptance).
    """
    N, n = X.shape
    suf = _suffix_max_strict_np(X.astype(np.float64))
    y_cv = np.zeros((N, n), dtype=np.float64)
    y_cv[:, : n - 1] = suf[:, : n - 1]
    y_act = np.zeros((N, n), dtype=np.float64)
    y_act[:, : n - 1] = (X[:, : n - 1] >= suf[:, : n - 1]).astype(np.float64)
    y_act[:, n - 1] = 1.0
    return y_cv, y_act


# ---------------------------------------------------------------------------
# Public labeler (torch, GPU-friendly)
# ---------------------------------------------------------------------------

def label_offline_torch(
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch counterpart of `label_offline`. Returns (y_cv, y_act) on
    the same device/dtype as X (cv promoted to X's dtype; act in same
    dtype as X for direct use by F.binary_cross_entropy_with_logits).
    """
    N, n = X.shape
    Xf = X.to(torch.float64) if X.dtype != torch.float64 else X
    suf = _suffix_max_strict_torch(Xf)
    y_cv = torch.zeros_like(Xf)
    y_cv[:, : n - 1] = suf[:, : n - 1]
    y_act = torch.zeros_like(Xf)
    y_act[:, : n - 1] = (Xf[:, : n - 1] >= suf[:, : n - 1]).to(Xf.dtype)
    y_act[:, n - 1] = 1.0
    return y_cv, y_act
