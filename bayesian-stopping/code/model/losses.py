"""
V3 loss functions.

Two functions:
    cv_loss(C_hat, y_cv, sigma_i, cv_mask): per-sequence MSE divided by
        sigma_i (not sigma_i^2), masked by cv_mask, then averaged over
        the batch.
    act_loss(logits, y_act, act_mask): per-sequence mean BCE-with-logits,
        masked by act_mask, then averaged over the batch.

sigma_i is loss-side only — the model's forward pass never sees it. The
trainer's data pipeline (`v3/data/streaming.py`) returns sigma_i alongside
each batch; the loss takes it here, the model does not.

cv normalization: 1/sigma_i, not 1/sigma_i^2. The original spec called for
1/sigma_i^2 to make loss values comparable across regimes; that turns out
to give per-sequence gradient magnitudes ~ 1/sigma (residual ~ sigma,
divided by sigma^2), so sigma=100 sequences receive 100x less effective
gradient per step than sigma=1. 1/sigma instead gives per-sequence
gradient magnitudes ~ 1 (residual sigma, divided by sigma), regime-
invariant under Adam updates. The tradeoff is loss values are no longer
regime-comparable — they scale as sigma — but loss-value comparability
across regimes was already documented as not held in V2 (each model's
descent shape is the meaningful comparison). See "Spec corrections" in
v3/README.md (entry of 2026-05-06).

Per-sequence (not per-batch) normalization stays. A per-batch mean of
sigma_i would re-introduce coupling between sigma values within a batch.

Mask conventions (set in the streamer):
    cv_mask:  (n,) bool. False at t=n always; for random distributions
              (D_disc, D_logu) also False at t=1 (sample variance
              undefined). True elsewhere.
    act_mask: (n,) bool. False at t=n only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cv_loss(
    C_hat: torch.Tensor,
    y_cv: torch.Tensor,
    sigma_i: torch.Tensor,
    cv_mask: torch.Tensor,
    sigma_power: int = 1,
) -> torch.Tensor:
    """Per-sequence MSE divided by sigma_i ** sigma_power, masked,
    batch-averaged.

    Args:
        C_hat:       (B, n) model output, raw oracle units.
        y_cv:        (B, n) target continuation values, raw oracle units.
                     Terminal entry (t=n) is a placeholder; cv_mask is False.
        sigma_i:     (B,) per-sequence true noise scale.
        cv_mask:     (n,) bool — see module docstring.
        sigma_power: denominator exponent. Default 1 per the V3 spec
                     correction: residual ~ σ, so squared err / σ ~ σ
                     (loss values scale as σ but per-step gradient is
                     σ-invariant). sigma_power=2 is the original spec
                     (loss-value invariance, gradient ~ 1/σ); used for
                     the §7.4 σ vs σ² ablation only.

    Returns:
        scalar loss.
    """
    if C_hat.shape != y_cv.shape:
        raise ValueError(f"shape mismatch: C_hat {C_hat.shape} vs y_cv {y_cv.shape}")
    if cv_mask.shape != (C_hat.shape[1],):
        raise ValueError(
            f"cv_mask shape {cv_mask.shape} must equal (n,) = ({C_hat.shape[1]},)"
        )
    if sigma_i.shape != (C_hat.shape[0],):
        raise ValueError(
            f"sigma_i shape {sigma_i.shape} must equal (B,) = ({C_hat.shape[0]},)"
        )
    if sigma_power not in (1, 2):
        raise ValueError(f"sigma_power must be 1 or 2, got {sigma_power!r}")

    sq_err = (C_hat - y_cv).pow(2)                                       # (B, n)
    mask_f = cv_mask.to(sq_err.dtype)                                     # (n,)
    n_valid = mask_f.sum()                                                # scalar
    sigma_i_f = sigma_i.to(sq_err.dtype)
    sigma_pow = sigma_i_f if sigma_power == 1 else sigma_i_f * sigma_i_f
    per_seq = (sq_err * mask_f).sum(dim=-1) / (n_valid * sigma_pow)
    return per_seq.mean()


def act_loss(
    logits: torch.Tensor,
    y_act: torch.Tensor,
    act_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence mean BCE-with-logits, masked, batch-averaged.

    Args:
        logits:   (B, n) model output (raw logits, not probabilities).
        y_act:    (B, n) target {0, 1} action labels.
        act_mask: (n,) bool — see module docstring.

    Returns:
        scalar loss.
    """
    if logits.shape != y_act.shape:
        raise ValueError(
            f"shape mismatch: logits {logits.shape} vs y_act {y_act.shape}"
        )
    if act_mask.shape != (logits.shape[1],):
        raise ValueError(
            f"act_mask shape {act_mask.shape} must equal (n,) = ({logits.shape[1]},)"
        )
    bce = F.binary_cross_entropy_with_logits(logits, y_act, reduction='none')  # (B, n)
    mask_f = act_mask.to(bce.dtype)
    n_valid = mask_f.sum()
    per_seq = (bce * mask_f).sum(dim=-1) / n_valid
    return per_seq.mean()
