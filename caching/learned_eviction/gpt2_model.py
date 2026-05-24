"""GPT-2-based cache eviction model.

Follows the "Standard" configuration from Garg et al. 2022 (What Can
Transformers Learn In-Context?): 12 layers, 8 heads, n_embd=256, ~9.5M params,
dropout=0, GELU activation. HuggingFace implementation.

Differences from the paper:
  - Input layout: cache block (k tokens) followed by request block (L tokens),
    rather than interleaved (x, y) pairs. Causal attention means each request
    position sees the full cache + all earlier requests.
  - Separate learned position embeddings for the cache segment and the request
    segment; GPT-2's internal wpe is zero-ed and frozen so it does not double
    count.
  - Output head: per-request-position projection to the item vocabulary
    (513 = U + 1). At each request position j the model predicts which item
    Belady would evict "right after" seeing r_j.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import GPT2Config, GPT2Model


class CacheEvictionGPT2(nn.Module):
    """GPT-2 backbone for cache eviction.

    Args:
        vocab_size:     items + 1 (index 0 reserved for padding / empty slot).
        cache_size:     k = number of cache slots.
        max_request_len: L_max for the request block; position embeddings are
                        allocated up to this length.
        n_embd, n_layer, n_head, n_inner, dropout: GPT-2 hyperparameters.
    """

    def __init__(
        self,
        vocab_size: int = 513,
        cache_size: int = 32,
        max_request_len: int = 1024,
        n_embd: int = 256,
        n_layer: int = 12,
        n_head: int = 8,
        n_inner: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cache_size = cache_size
        self.max_request_len = max_request_len
        self.n_embd = n_embd

        # Item embedding shared between cache and request segments.
        self.item_embed = nn.Embedding(vocab_size, n_embd, padding_idx=0)

        # Separate position embeddings for the two segments.
        self.cache_pos_embed = nn.Embedding(cache_size, n_embd)
        self.request_pos_embed = nn.Embedding(max_request_len, n_embd)

        # Segment embedding: 0 = cache token, 1 = request token.
        self.segment_embed = nn.Embedding(2, n_embd)

        # GPT-2 backbone with internal position embeddings neutralised so
        # they do not fight with our segment-specific pos embeds.
        config = GPT2Config(
            vocab_size=1,  # unused — we pass inputs_embeds
            n_positions=cache_size + max_request_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            activation_function="gelu_new",
            use_cache=False,
        )
        self.gpt2 = GPT2Model(config)
        with torch.no_grad():
            self.gpt2.wpe.weight.zero_()
        self.gpt2.wpe.weight.requires_grad_(False)

        # Item prediction head on request-position outputs.
        self.evict_head = nn.Linear(n_embd, vocab_size)

        self._init_weights_custom()

    def _init_weights_custom(self):
        # HF's GPT2Model initializes its own weights; we init our add-ons.
        for m in (self.cache_pos_embed, self.request_pos_embed,
                  self.segment_embed, self.item_embed):
            nn.init.normal_(m.weight, std=0.02)
        # padding_idx stays zero
        with torch.no_grad():
            self.item_embed.weight[0].zero_()
        nn.init.normal_(self.evict_head.weight, std=0.02)
        nn.init.zeros_(self.evict_head.bias)

    def forward(self, cache: torch.Tensor, requests: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            cache:    (B, k) long  — cache tokens (item ids shifted +1; 0 = empty).
            requests: (B, L) long  — request sequence (shifted +1).

        Returns:
            logits: (B, L, vocab_size) — item-prediction logits at each request
                    position. Softmax over the last dim is the evict-item
                    distribution.
        """
        B, k = cache.shape
        _, L = requests.shape
        device = cache.device
        assert k == self.cache_size, f"cache_size mismatch: {k} vs {self.cache_size}"
        assert L <= self.max_request_len, (
            f"request length {L} exceeds max_request_len {self.max_request_len}"
        )

        # Cache segment embeddings.
        cache_pos = torch.arange(k, device=device).unsqueeze(0).expand(B, -1)
        cache_segment = torch.zeros(B, k, dtype=torch.long, device=device)
        cache_emb = (
            self.item_embed(cache)
            + self.cache_pos_embed(cache_pos)
            + self.segment_embed(cache_segment)
        )

        # Request segment embeddings.
        req_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        req_segment = torch.ones(B, L, dtype=torch.long, device=device)
        req_emb = (
            self.item_embed(requests)
            + self.request_pos_embed(req_pos)
            + self.segment_embed(req_segment)
        )

        # Concatenated sequence: [cache | requests] → causal attention means
        # request_j sees all cache + requests[:j+1].
        inputs_embeds = torch.cat([cache_emb, req_emb], dim=1)

        # GPT-2 still adds wpe(position_ids) internally, but wpe is zero-valued
        # so it contributes nothing.
        out = self.gpt2(inputs_embeds=inputs_embeds, use_cache=False)
        hidden = out.last_hidden_state  # (B, k+L, n_embd)

        # Read out request-position hidden states only.
        req_hidden = hidden[:, k:, :]  # (B, L, n_embd)

        # Per-position item logits.
        logits = self.evict_head(req_hidden)  # (B, L, vocab_size)
        return logits


if __name__ == "__main__":
    # Smoke test.
    model = CacheEvictionGPT2(
        vocab_size=513, cache_size=32, max_request_len=128,
        n_embd=256, n_layer=12, n_head=8, n_inner=1024, dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_params:,}  frozen (wpe): {n_frozen:,}")

    B, k, L = 4, 32, 128
    cache = torch.randint(0, 513, (B, k))
    requests = torch.randint(1, 513, (B, L))
    logits = model(cache, requests)
    print(f"logits: {tuple(logits.shape)}  dtype={logits.dtype}")
