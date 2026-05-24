"""
V3 model: GPT-2 decoder with both cv and act heads, scale-blind to sigma.

Surgery vs v2/code/model.py (logged in v3/README.md):
  - Removed `input_scale` and `output_scale` buffers (V2 lines 158-165 and
    194-201). The model now sees raw X and emits raw Ĉ in oracle units.
  - Removed the `sigma` constructor argument and the `supervision` flag.
    Both heads coexist; the trainer picks which loss to apply.
  - The cv head emits scalar Ĉ_t, the act head emits a logit.

Architecture: L=8 layers, d_emb=128, M=4 heads, causal mask, no dropout.
Pre-norm GPT-2 block (LN -> attn / LN -> MLP-4x with GELU; residual).
Affine input projection: scalar X_t -> R^d_emb. Learned absolute position
embeddings.

Forward signature:
    model(X, return_attn=False)
        X:                (B, n) float, raw scale.
        returns dict:     {'cv': (B, n) Ĉ_t in raw units,
                            'act': (B, n) logits (sigmoid for accept-prob)}
        if return_attn:   also includes 'attn' (B, L, M, n, n).

The training objective uses positions 0..n-2 (= 1-indexed t = 1..n-1); the
output at position n-1 is unused. Loss masks (cv_mask, act_mask) are
applied by the loss functions in `v3/model/losses.py`, not in the model.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (verbatim port from v2/code/model.py)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, n_max: int):
        super().__init__()
        if d_emb % n_heads != 0:
            raise ValueError("d_emb must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_emb // n_heads
        self.qkv = nn.Linear(d_emb, 3 * d_emb)
        self.out = nn.Linear(d_emb, d_emb)
        mask = torch.tril(torch.ones(n_max, n_max, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, n_max, n_max), persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        if return_attn:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~self.mask[:, :, :n, :n], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, n, d)
            return self.out(y), attn
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, n, d)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.fc1 = nn.Linear(d_emb, 4 * d_emb)
        self.fc2 = nn.Linear(4 * d_emb, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, n_max: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_emb)
        self.attn = CausalSelfAttention(d_emb, n_heads, n_max)
        self.ln2 = nn.LayerNorm(d_emb)
        self.mlp = MLP(d_emb)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, attn
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# V3 top-level model
# ---------------------------------------------------------------------------

class GPTStopper(nn.Module):
    """V3 GPT-2 decoder with two scalar output heads (cv, act).

    The forward pass operates at raw scale: input X is fed as-is, Ĉ_t and
    logits come out as-is. There is no sigma-dependent constant anywhere
    in the architecture — the whole point of V3 is that the model has to
    recover sigma in-context.
    """

    def __init__(
        self,
        n: int = 256,
        d_emb: int = 128,
        n_layers: int = 8,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n = n
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.input_proj = nn.Linear(1, d_emb)
        self.pos_emb = nn.Embedding(n, d_emb)
        self.blocks = nn.ModuleList(
            [Block(d_emb, n_heads, n) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_emb)
        self.cv_head = nn.Linear(d_emb, 1)
        self.act_head = nn.Linear(d_emb, 1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, X: torch.Tensor, return_attn: bool = False,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, n = X.shape
        pos = torch.arange(n, device=X.device)
        x = self.input_proj(X.unsqueeze(-1)) + self.pos_emb(pos)        # (B, n, d_emb)

        if return_attn:
            attns = []
            hiddens = [x] if return_hidden else None
            for block in self.blocks:
                x, a = block(x, return_attn=True)
                attns.append(a)
                if return_hidden:
                    hiddens.append(x)
            x = self.ln_f(x)
            cv = self.cv_head(x).squeeze(-1)                             # (B, n)
            act = self.act_head(x).squeeze(-1)                           # (B, n)
            out = {'cv': cv, 'act': act, 'attn': torch.stack(attns, dim=1)}
            if return_hidden:
                out['hidden'] = torch.stack(hiddens, dim=1)              # (B, L+1, n, d_emb)
            return out

        hiddens = [x] if return_hidden else None
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hiddens.append(x)
        x = self.ln_f(x)
        cv = self.cv_head(x).squeeze(-1)
        act = self.act_head(x).squeeze(-1)
        out = {'cv': cv, 'act': act}
        if return_hidden:
            # Pre-block-0 state (post-positional-embedding) is hiddens[0];
            # post-block-k state is hiddens[k+1]. Layer index ℓ = k for the
            # spec's convention ℓ ∈ {0, ..., L}: ℓ=0 is post-positional-
            # embedding, ℓ=k is post-block-(k-1).
            out['hidden'] = torch.stack(hiddens, dim=1)                  # (B, L+1, n, d_emb)
        return out


# ---------------------------------------------------------------------------
# Inference-time policy extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_policy_batch(
    model: GPTStopper, X: torch.Tensor, head: str,
):
    """Vectorized policy. `head` is 'cv' or 'act'.

    cv:  accept iff X[b, t] >= predicted Ĉ[b, t]
    act: accept iff sigmoid(logit[b, t]) > 0.5  (equivalently logit > 0)

    Args:
        X: (B, n) float tensor on the model's device.
    Returns: (stop_idx (B,), payoff (B,)).
    """
    if head not in ('cv', 'act'):
        raise ValueError(f"head must be 'cv' or 'act', got {head!r}")
    B, n = X.shape
    out = model(X)
    if head == 'act':
        accept = out['act'][:, : n - 1] > 0.0
    else:
        accept = X[:, : n - 1] >= out['cv'][:, : n - 1]
    any_accept = accept.any(dim=1)
    first_idx = accept.float().argmax(dim=1)
    last = torch.full_like(first_idx, n - 1)
    stop_idx = torch.where(any_accept, first_idx, last)
    payoff = X.gather(1, stop_idx.unsqueeze(1)).squeeze(1)
    return stop_idx, payoff
