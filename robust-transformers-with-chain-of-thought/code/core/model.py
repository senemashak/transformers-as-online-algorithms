"""
Decoder-style causal transformer for online decision-making.

Shared architecture for both optimal stopping and ski rental.

2D chain mode (sole forward path):
  For each decision step t = 0..T-2, a sub-chain computes:
      V(t, T-2), V(t, T-3), ..., V(t, t)
  using only observations x_0,...,x_{t-1} as context (the posterior prefix).

  V(t) := chain_V(t, t) = chain2d_V[:, decision_pos[t]] is the sole output.
  Action rule:
    Stopping: stop if x_t >= V(t)*M  (value exceeds learned continuation threshold)
    Ski:      buy  if V(t) >= 1       (cost-to-go exceeds buy cost)
  There are no separate V_hat or a_hat heads.

  Sequence layout:
      [x_0,...,x_{T-1} | sub(t=0) | sub(t=1) | ... | sub(t=T-2)]
  Total chain length: T*(T-1)//2

  Attention rules:
    - Obs positions: standard causal
    - Sub-chain t, step j (predicting V(t, T-2-j)):
        attends to obs x_0,...,x_t  (posterior includes x_t; x_{t+1} onwards blocked)
        attends causally within sub-chain t only
"""

import numpy as np
import torch
import torch.nn as nn


class OnlineDecisionTransformer(nn.Module):
    """
    Causal transformer with 2D chain-of-thought for online decision problems.

    V(t) = chain_V(t, t) is the sole prediction per decision step.
    Action: stopping — stop if x_t >= V(t)*M; ski — buy if V(t) >= 1.
    """

    def __init__(
        self,
        M: int,
        d_model: int | None = None,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int | None = None,
        max_n: int = 501,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = M

        import math
        B_max_default = 100
        n_max_train = 200
        eps_target = 0.1

        if d_model is None:
            d_model = ((M + 2 + n_heads - 1) // n_heads) * n_heads
        if d_ff is None:
            d_ff_stopping = M
            d_ff_ski = int(math.ceil(
                (B_max_default + 1) * math.sqrt(n_max_train) / (2 * math.sqrt(eps_target))
            ))
            d_ff = max(d_ff_stopping, d_ff_ski, d_model)
        self.d_model = d_model
        self.max_n = max_n

        # === Observation embeddings ===
        self.value_embed = nn.Embedding(M + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_n, d_model)
        self.horizon_embed = nn.Embedding(max_n + 1, d_model)
        self.B_embed = nn.Embedding(201, d_model)
        self.r_embed = nn.Embedding(11, d_model)
        self.task_embed = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)

        # === Transformer backbone ===
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="relu",
            )
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.final_norm = nn.LayerNorm(d_model)

        # === Chain-of-thought components ===
        self.chain_value_proj = nn.Linear(1, d_model)
        self.start_chain = nn.Parameter(torch.zeros(d_model))
        self.chain_head = nn.Linear(d_model, 1)

        # === 2D chain positional embeddings ===
        self.chain2d_t_embed = nn.Embedding(max_n, d_model)
        self.chain2d_j_embed = nn.Embedding(max_n, d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── 2D chain helpers ────────────────────────────────────────────────────

    @staticmethod
    def _chain2d_offset(t: int, n: int) -> int:
        """Start index within the flat chain for sub-chain at decision step t.

        Flat layout: [sub(t=0), sub(t=1), ..., sub(t=n-2)]
        sub(t) length = n-1-t  (computes V(t,n-2) down to V(t,t))
        offset(t) = sum_{i=0}^{t-1}(n-1-i) = t*(n-1) - t*(t-1)//2
        """
        return t * (n - 1) - t * (t - 1) // 2

    def _get_chain2d_info(self, T: int, device):
        """Return index tensors describing the flat 2D chain of length T*(T-1)//2.

        Returns:
            t_idx        : (L,) decision step t for each flat position
            j_idx        : (L,) within-sub-chain step j (j=0 first computed)
            decision_pos : (T-1,) flat index of V(t,t) for t=0..T-2
                           V(t,t) is the last element of sub-chain t (j = T-2-t)
        """
        L = T * (T - 1) // 2
        t_idx = torch.zeros(L, dtype=torch.long, device=device)
        j_idx = torch.zeros(L, dtype=torch.long, device=device)

        for t in range(T - 1):
            sub_len = T - 1 - t
            off = self._chain2d_offset(t, T)
            t_idx[off:off + sub_len] = t
            j_idx[off:off + sub_len] = torch.arange(sub_len, device=device)

        decision_pos = torch.tensor(
            [self._chain2d_offset(t, T) + (T - 2 - t) for t in range(T - 1)],
            dtype=torch.long, device=device,
        )
        return t_idx, j_idx, decision_pos

    @staticmethod
    def _build_chain2d_mask(n_obs: int, device) -> torch.Tensor:
        """Attention mask for the 2D chain V(t, k).

        Sequence layout:
            [x_0,...,x_{n-1} | sub(t=0) | sub(t=1) | ... | sub(t=n-2)]
        Total length: n + n*(n-1)//2

        V(t, k) at flat chain position offset(t)+j  (k = n-2-j):
          - attends to obs 0..t  (posterior includes x_t; blocks x_{t+1} onwards)
          - attends causally within sub(t): positions offset(t)..offset(t)+j

        At decision step t, the agent has observed x_0,...,x_t and must decide
        whether to stop/buy. The Bayesian posterior includes x_t, so the chain
        computing V(t,k) should see x_t. It must NOT see x_{t+1},...,x_{n-1}.

        Returns: (L, L) float mask with 0 = attend, -inf = block
        """
        n = n_obs
        L_chain = n * (n - 1) // 2
        L = n + L_chain
        mask = torch.full((L, L), float('-inf'), device=device)

        # obs-to-obs: standard causal
        mask[:n, :n] = nn.Transformer.generate_square_subsequent_mask(n, device=device)

        # chain positions: one block per sub-chain
        for t in range(n - 1):
            sub_len = n - 1 - t
            off = OnlineDecisionTransformer._chain2d_offset(t, n)
            p0, p1 = n + off, n + off + sub_len

            # obs attention: x_0,...,x_t (includes x_t, blocks x_{t+1} onwards)
            mask[p0:p1, :t + 1] = 0.0

            # within-sub-chain causal
            mask[p0:p1, p0:p1] = nn.Transformer.generate_square_subsequent_mask(
                sub_len, device=device)

        return mask

    def forward(
        self,
        x: torch.Tensor,
        chain2d_targets: torch.Tensor | None = None,
        n_horizon: int | torch.Tensor | None = None,
        B_cost: float | torch.Tensor | None = None,
        r_cost: float | torch.Tensor | None = None,
        task_id: int | torch.Tensor | None = None,
        mode: str = "teacher_forcing",
        return_attention: bool = False,
    ):
        """Forward pass with 2D chain V(t, k).

        For each decision step t (0-indexed), a sub-chain computes:
            V(t, n-2) -> V(t, n-3) -> ... -> V(t, t)
        using obs x_0,...,x_t consistently for every DP step.

        V(t) := chain_V(t, t) is the sole output used for decisions.
        Action: stopping — stop if x_t >= V(t)*M; ski — buy if V(t) >= 1.

        Three modes:
          "teacher_forcing"  — chain inputs are shifted ground-truth targets.
                               Single forward pass. Requires chain2d_targets.
          "autoregressive"   — chain inputs are the model's own predictions.
                               T-1 forward passes, gradients flow through all.
                               Use for training to reduce exposure bias.
          "inference"        — same as autoregressive but with torch.no_grad()
                               handled by caller. No functional difference.

        Args:
            x               : (B, T) int64 observations
            chain2d_targets : (B, L_chain) flat normalised targets (teacher_forcing only)
            n_horizon, B_cost, r_cost, task_id: context
            mode            : "teacher_forcing", "autoregressive", or "inference"
            return_attention: if True, also return per-layer attention weights
                              (only supported with teacher_forcing mode)

        Returns:
            chain2d_V    : (B, L_chain) chain predictions in flat order
            decision_pos : (T-1,)       flat index of V(t,t) for t=0..T-2
                           chain2d_V[:, decision_pos[t]] = V(t) for each t
            attn_weights : (only if return_attention) list of (B, n_heads, L_total, L_total)
                           per layer, where L_total = T + T*(T-1)//2
        """
        B_batch, T = x.shape
        device = x.device
        L_chain = T * (T - 1) // 2

        h_ctx = self._context_embed(n_horizon, B_batch, device, B_cost=B_cost,
                                    r_cost=r_cost, task_id=task_id)

        obs_pos = torch.arange(T, device=device).unsqueeze(0)
        h_obs = self.value_embed(x) + self.pos_embed(obs_pos)
        if h_ctx is not None:
            h_obs = h_obs + h_ctx

        t_idx, j_idx, decision_pos = self._get_chain2d_info(T, device)
        is_start = (j_idx == 0)
        causal_mask = self._build_chain2d_mask(T, device)

        def _build_h_chain(prev_values):
            """Build chain embeddings given (B, L_chain) previous values."""
            h = self.chain_value_proj(prev_values.unsqueeze(-1))
            h[:, is_start, :] = self.start_chain.view(1, 1, -1).expand(
                B_batch, int(is_start.sum()), -1)
            h = h + self.chain2d_t_embed(t_idx) + self.chain2d_j_embed(j_idx)
            if h_ctx is not None:
                h = h + h_ctx
            return h

        if mode == "teacher_forcing":
            # --- Teacher forcing: single pass, ground-truth shifted inputs ---
            assert chain2d_targets is not None, "teacher_forcing requires chain2d_targets"
            prev_targets = torch.zeros(B_batch, L_chain, device=device)
            for t in range(T - 1):
                off = self._chain2d_offset(t, T)
                sub_len = T - 1 - t
                if sub_len > 1:
                    prev_targets[:, off + 1:off + sub_len] = \
                        chain2d_targets[:, off:off + sub_len - 1]

            h_chain = _build_h_chain(prev_targets)
            h_full = self.drop(torch.cat([h_obs, h_chain], dim=1))
            h_full, attn = self._run_transformer(h_full, causal_mask,
                                                  return_attention=return_attention)

            chain2d_V = self.chain_head(h_full[:, T:]).squeeze(-1)
            if return_attention:
                return chain2d_V, decision_pos, attn
            return chain2d_V, decision_pos

        else:
            # --- Autoregressive: T-1 rounds, own predictions as inputs ---
            # Gradients flow through when mode="autoregressive" (for training).
            # For mode="inference", caller wraps in torch.no_grad().
            #
            # We avoid in-place ops so autograd works. Collect per-step
            # predictions as (index, value) pairs and scatter at the end.
            all_preds = []   # list of (active_flat_indices, preds_tensor) per step
            prev_values = torch.zeros(B_batch, L_chain, device=device)

            for step_j in range(T - 1):
                active = (j_idx == step_j)
                if not active.any():
                    break

                h_chain = _build_h_chain(prev_values)
                h_full = self.drop(torch.cat([h_obs, h_chain], dim=1))
                h_out, _ = self._run_transformer(h_full, causal_mask)

                active_flat = active.nonzero(as_tuple=True)[0]
                active_global = T + active_flat
                preds = self.chain_head(h_out[:, active_global]).squeeze(-1)  # (B, n_active)
                all_preds.append((active_flat, preds))

                # Feed predictions to next j step — build new prev_values
                # without in-place mutation
                next_active = (j_idx == step_j + 1)
                if next_active.any():
                    curr_pos = active_flat
                    next_pos = curr_pos + 1
                    in_bounds = next_pos < L_chain
                    next_pos_safe = next_pos.clamp(max=L_chain - 1)
                    valid = in_bounds & (j_idx[next_pos_safe] == step_j + 1)
                    if valid.any():
                        # Scatter preds into prev_values for next step
                        update = torch.zeros_like(prev_values)
                        # Map from active_flat positions to their prediction values
                        # preds[:, i] corresponds to active_flat[i]
                        valid_local = torch.arange(len(curr_pos), device=device)[valid]
                        update[:, next_pos[valid]] = preds[:, valid_local]
                        prev_values = prev_values + update

            # Assemble chain2d_V from collected predictions (no in-place)
            chain2d_V = torch.zeros(B_batch, L_chain, device=device)
            for active_flat, preds in all_preds:
                scatter_mask = torch.zeros(B_batch, L_chain, device=device)
                scatter_mask[:, active_flat] = 1.0
                filler = torch.zeros(B_batch, L_chain, device=device)
                filler[:, active_flat] = preds
                chain2d_V = chain2d_V * (1 - scatter_mask) + filler

            return chain2d_V, decision_pos

    def _run_transformer(self, h, causal_mask, return_attention=False):
        """Run through transformer layers with optional attention weight capture."""
        attn_weights = [] if return_attention else None

        for layer in self.layers:
            if return_attention:
                h_norm = layer.norm1(h)
                attn_out, weights = layer.self_attn(
                    h_norm, h_norm, h_norm,
                    attn_mask=causal_mask, need_weights=True,
                    average_attn_weights=False,
                )
                h = h + attn_out
                h = h + layer._ff_block(layer.norm2(h))
                attn_weights.append(weights.detach())
            else:
                h = layer(h, src_mask=causal_mask)

        h = self.final_norm(h)
        return h, attn_weights

    def _context_embed(self, n_horizon, batch_size, device, B_cost=None, r_cost=None,
                       task_id=None):
        """Compute additive context embeddings (horizon, buy cost, rent cost, task)."""
        parts = []
        if n_horizon is not None:
            if isinstance(n_horizon, (int, np.integer)):
                n_t = torch.full((batch_size, 1), int(n_horizon), dtype=torch.long, device=device)
            else:
                n_t = n_horizon.to(device).unsqueeze(1)
            parts.append(self.horizon_embed(n_t))
        if B_cost is not None:
            if isinstance(B_cost, (int, float)):
                b_t = torch.full((batch_size, 1), int(B_cost), dtype=torch.long, device=device)
            else:
                b_t = B_cost.long().to(device).unsqueeze(1)
            parts.append(self.B_embed(b_t))
        if r_cost is not None:
            if isinstance(r_cost, (int, float)):
                r_t = torch.full((batch_size, 1), int(r_cost), dtype=torch.long, device=device)
            else:
                r_t = r_cost.long().to(device).unsqueeze(1)
            parts.append(self.r_embed(r_t))
        if task_id is not None:
            if isinstance(task_id, int):
                t_t = torch.full((batch_size, 1), task_id, dtype=torch.long, device=device)
            else:
                t_t = task_id.long().to(device).unsqueeze(1)
            parts.append(self.task_embed(t_t))
        if not parts:
            return None
        return sum(parts)


if __name__ == "__main__":
    M, n, bs = 100, 8, 4
    model = OnlineDecisionTransformer(M=M, n_layers=3)
    x = torch.randint(1, M + 1, (bs, n))

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # Teacher forcing
    L = n * (n - 1) // 2
    chain2d_tgt = torch.rand(bs, L)
    chain2d_V, decision_pos = model(
        x, chain2d_targets=chain2d_tgt, n_horizon=n, task_id=0, mode="teacher_forcing")
    print(f"Teacher forcing: V={chain2d_V.shape}  V(t)={chain2d_V[:, decision_pos].shape}")

    # Autoregressive (training mode — with gradients)
    chain2d_V_ar, _ = model(x, n_horizon=n, task_id=0, mode="autoregressive")
    print(f"Autoregressive: V={chain2d_V_ar.shape}  (training, grads flow)")

    # Inference (no gradients)
    with torch.no_grad():
        chain2d_V_inf, _ = model(x, n_horizon=n, task_id=0, mode="inference")
    print(f"Inference:      V={chain2d_V_inf.shape}  (no grad)")
