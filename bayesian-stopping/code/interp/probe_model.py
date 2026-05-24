"""
Probe networks.

Per the spec in research-notes_v3.tex §6.X (Probing analysis):

    α_t = softmax(s_{≤ t})     (causally-masked position attention; t-relative)
    v̂_t = FF(α_t^⊤ W_v H^{(ℓ)})

where H^{(ℓ)} ∈ R^{n × d_emb} is the residual-stream snapshot at layer ℓ.

We use a single shared learned position-logit vector `s ∈ R^n` and apply
the causal mask at every t. The projection W_v is a d_emb × d_emb linear
map applied before the attention pool (this is the most efficient
placement and is equivalent to placing it elsewhere up to FF capacity).

Two FF variants:
    linear:  R^{d_emb} → R
    mlp:     R^{d_emb} → R^{512} → GELU → R^{512} → GELU → R
             (the spec says "2-layer MLP with hidden width 512 and GeLU"
              — we read that as one hidden layer; the implementation uses
              one hidden layer of width 512 followed by output, matching
              the spec exactly.)

Outputs (B, n) — one prediction per (sequence, timestep).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooledProbe(nn.Module):
    """Single-scalar regression probe with causal position-attention pooling.

    Args:
        n:       sequence length (256).
        d_emb:   transformer hidden width (128).
        variant: 'linear' or 'mlp'.
        mlp_hidden: hidden width of the MLP variant (ignored otherwise).
    """

    def __init__(self, n: int, d_emb: int, variant: str, mlp_hidden: int = 512):
        super().__init__()
        if variant not in ('linear', 'mlp'):
            raise ValueError(f'variant must be linear or mlp, got {variant!r}')
        self.n = n
        self.d_emb = d_emb
        self.variant = variant

        # Position-attention logits. Single shared vector; causal mask is
        # applied at every t inside `forward`.
        self.s = nn.Parameter(torch.zeros(n))
        # Projection W_v ∈ R^{d_emb × d_emb} applied per position before pooling.
        self.Wv = nn.Linear(d_emb, d_emb, bias=True)

        if variant == 'linear':
            self.ff = nn.Linear(d_emb, 1)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_emb, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )

        # Pre-compute the lower-triangular causal mask once on the right device.
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(n, n, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, n, d_emb) residual-stream snapshot at one layer.
        Returns:
            v_hat: (B, n) per-timestep predictions.
        """
        B, n, d = H.shape
        if n != self.n or d != self.d_emb:
            raise ValueError(
                f'expected (B, {self.n}, {self.d_emb}); got (B, {n}, {d})'
            )

        Z = self.Wv(H)                                                    # (B, n, d_emb)

        # Logits for each (t, s) pair. We use the same per-position logit
        # vector `self.s` for every t; the causal mask sets s > t to -inf.
        s_logits = self.s.unsqueeze(0).expand(n, n).clone()              # (n, n) — row = t
        s_logits = s_logits.masked_fill(~self.causal_mask, float('-inf'))
        alpha = F.softmax(s_logits, dim=-1)                              # (n, n)

        # pooled[b, t, d] = sum_s alpha[t, s] * Z[b, s, d]
        pooled = torch.einsum('ts,bsd->btd', alpha, Z)                   # (B, n, d_emb)

        out = self.ff(pooled).squeeze(-1)                                # (B, n)
        return out

    @torch.no_grad()
    def attention_weights(self) -> torch.Tensor:
        """Return the (n, n) attention matrix used in `forward`."""
        s_logits = self.s.unsqueeze(0).expand(self.n, self.n).clone()
        s_logits = s_logits.masked_fill(~self.causal_mask, float('-inf'))
        return F.softmax(s_logits, dim=-1)


class PerTimestepProbe(nn.Module):
    """Standard per-timestep linear / MLP probe (no attention, no pooling).

    At each timestep t, the probe reads only the residual-stream snapshot
    at position t and predicts v_t directly:

        v_hat_t = FF(H^{(\ell)}_t)

    The same `FF` is shared across all (sequence, timestep) pairs. Used as
    the simpler, more standard probe form alongside `AttentionPooledProbe`.

    Args:
        n:       sequence length (kept for API parity with AttentionPooledProbe).
        d_emb:   transformer hidden width (128 in our setup).
        variant: 'linear' or 'mlp'.
        mlp_hidden: hidden width of the MLP variant.
    """

    def __init__(self, n: int, d_emb: int, variant: str, mlp_hidden: int = 512):
        super().__init__()
        if variant not in ('linear', 'mlp'):
            raise ValueError(f'variant must be linear or mlp, got {variant!r}')
        self.n = n
        self.d_emb = d_emb
        self.variant = variant

        if variant == 'linear':
            self.ff = nn.Linear(d_emb, 1)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_emb, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, n, d_emb) residual-stream snapshot at one layer.
        Returns:
            v_hat: (B, n) per-timestep predictions.
        """
        # FF applied broadcast-style over (B, n) — output (B, n, 1) → squeeze.
        return self.ff(H).squeeze(-1)

    @torch.no_grad()
    def attention_weights(self) -> torch.Tensor:
        """No attention; return an identity-like matrix for API parity (each
        row is the canonical basis vector for that timestep, indicating
        ``the probe always reads from position t at time t'')."""
        return torch.eye(self.n, dtype=torch.float32)
