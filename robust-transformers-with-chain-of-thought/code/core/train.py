"""
Training loop for both optimal stopping and ski rental.

Uses 2D chain-of-thought V(t, k) as the sole forward path.

Loss:
  L = w_value * L_value + w_chain * L_chain + w_action * L_action

  L_value  — MSE between V(t) := chain_V(t,t) and target_V(t)
  L_chain  — MSE between chain_V(t,k) and target_V(k) for all intermediate steps
  L_action — Soft CE on the action implied by V(t) vs x_t:
             stop/buy if V(t) > x_t, continue/rent otherwise
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.model import OnlineDecisionTransformer


# ═══════════════════════════════════════════════════════════════════════════
# Robust position masks — restrict training to positions the deployment
# rule actually uses. Positions outside the mask receive zero gradient.
# ═══════════════════════════════════════════════════════════════════════════

def build_stopping_robust_mask(n_horizons, beta, max_n):
    """
    Optimal stopping robust-aware training mask.

    The robust deployment rule (Algorithm 1) splits the horizon into
    three phases based on beta -> (lambda1, lambda2):

        Early (t <= n*lambda1):   always skip         -> prediction IGNORED
        Middle (n*lambda1 < t <= n*lambda2): use threshold -> prediction USED
        Late (t > n*lambda2):     accept any best      -> prediction IGNORED

    This mask is 1 only in the middle phase where the transformer's
    prediction actually influences the decision.
    """
    from deployment import find_lambdas
    lambda1, lambda2 = find_lambdas(beta)

    B = len(n_horizons)
    mask = torch.zeros(B, max_n)
    for b in range(B):
        n = int(n_horizons[b])
        for t in range(n):
            t_norm = (t + 1) / n
            if lambda1 < t_norm <= lambda2:
                mask[b, t] = 1.0
    return mask


def build_ski_robust_mask(n_horizons, lam, B_cost, r, U, max_n):
    """
    Ski rental robust-aware training mask.

    The robust deployment rule (Algorithm 2) clamps the buying day K*
    toward the breakeven point B/r. The prediction only matters when
    K* falls in [ceil(lam * B/r), ceil(B/(r*lam))].
    """
    ratio = B_cost / r
    sqrt_ratio = np.sqrt(ratio)

    lam_lower = np.ceil(lam * ratio)
    lam_upper = np.ceil(ratio / lam)

    B_batch = len(n_horizons)
    mask = torch.zeros(B_batch, max_n)
    for b in range(B_batch):
        n = int(n_horizons[b])
        for t in range(n):
            val = t + sqrt_ratio
            if lam_lower < val < lam_upper:
                mask[b, t] = 1.0
    return mask


def _build_chain2d_targets(V_target: torch.Tensor, j_idx: torch.Tensor, T: int) -> torch.Tensor:
    """Build flat 2D chain targets from the 1D DP value array.

    Flat chain position p has:
        j = j_idx[p]          — within-sub-chain step
        k = (T-2) - j         — DP step being predicted at this position

    Target at p = V_target[:, k]  (same for all decision steps t at training
    time, because the true DP values don't depend on which step t we're at).

    Args:
        V_target : (B, T) normalised DP values  (C/M for stopping, J/B for ski)
        j_idx    : (L_chain,) within-sub-chain step for each flat position
        T        : padded sequence length

    Returns:
        (B, L_chain) flat chain targets
    """
    k_idx = (T - 2) - j_idx
    k_idx = k_idx.clamp(0, T - 1)
    return V_target[:, k_idx]


def _build_chain2d_valid_mask(
    t_idx: torch.Tensor,
    j_idx: torch.Tensor,
    mask: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Per-sample validity mask for 2D chain positions.

    A flat chain position p = (t, k) is valid for sample b if:
        t < h_b   — decision step is within the actual horizon
        k < h_b   — DP step is within the actual horizon
    where k = (T-2) - j_idx[p] and h_b = mask[b].sum().
    """
    h = mask.float().sum(dim=1).long()
    k_idx = (T - 2) - j_idx

    t_valid = t_idx.unsqueeze(0) < h.unsqueeze(1)
    k_valid = k_idx.unsqueeze(0) < h.unsqueeze(1)
    return (t_valid & k_valid).float()


def compute_loss(
    chain2d_V: torch.Tensor,
    decision_pos: torch.Tensor,
    batch: dict,
    t_idx: torch.Tensor,
    j_idx: torch.Tensor,
    problem: str,
    M: float,
    B: float,
    w_value: float = 1.0,
    w_chain: float = 0.0,
    w_action: float = 0.0,
    alpha: float = 10.0,
    robust_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Loss for the 2D chain architecture.

    V(t) = chain2d_V[:, decision_pos[t]] is the sole value output.
    Action rule: stop/buy if V(t) > x_t, continue/rent otherwise.

    Three independently-weighted loss terms:

      L_value  (w_value) — MSE between V(t) and target_V(t).

      L_chain  (w_chain) — MSE between chain_V(t,k) and target_V(k) for every
                           intermediate step.

      L_action (w_action) — Soft CE on the action implied by V(t):
                           stop/buy if V(t) > x_t  (sigmoid(alpha * (V(t) - x_t)))

    If robust_mask is provided (B, T), it is intersected with the padding mask
    so that gradient only flows through positions the robust wrapper uses.
    """
    device = chain2d_V.device
    Bsz = chain2d_V.shape[0]
    T = int(t_idx.max().item()) + 2

    mask_f = batch["mask"].to(device).float()
    a_target = batch["a"].to(device)

    # Robust masking:
    # - L_value, L_action: masked to only decision steps the wrapper uses
    # - L_chain: masked to only sub-chains for useful decision steps.
    #   Within a useful sub-chain, ALL intermediate V(t,k) are supervised
    #   (even when k falls outside the wrapper's range) because the chain
    #   must be correct end-to-end for V(t,t) to be accurate.
    #   Sub-chains for non-useful decision steps get NO gradient at all.
    if robust_mask is not None:
        rmask = robust_mask.to(device)
        dec_level_mask = mask_f * rmask
        # Build chain-level mask: chain position p belongs to decision step t_idx[p].
        # Supervise it only if robust_mask[b, t_idx[p]] = 1.
        chain_robust = rmask[:, t_idx]  # (B, L_chain) — 1 if sub-chain t is useful
    else:
        dec_level_mask = mask_f
        chain_robust = None

    # ── Normalised DP targets ────────────────────────────────────────────────
    if problem == "stopping":
        V_target = batch["C"].to(device) / M
    else:
        if isinstance(B, torch.Tensor):
            B_norm = B.to(device)
            if B_norm.dim() == 1:
                B_norm = B_norm.unsqueeze(1)
        else:
            B_norm = B
        V_target = batch["J"].to(device) / B_norm

    metrics = {}
    total = torch.tensor(0.0, device=device)

    # ── Always compute all 3 losses for logging, only backprop weighted ones ──

    # L_value: MSE on V(t) vs target_V(t)
    T_dec = decision_pos.shape[0]
    V_final = chain2d_V[:, decision_pos]
    V_tgt_dec = V_target[:, :T_dec]
    dec_mask = dec_level_mask[:, :T_dec]
    mse_v = ((V_final - V_tgt_dec) ** 2 * dec_mask).sum() \
            / dec_mask.sum().clamp(min=1)
    if w_value > 0:
        total = total + w_value * mse_v
    metrics["value_loss"] = mse_v.item()

    # L_chain: MSE on chain_V(t,k) vs target_V(k)
    # Only sub-chains for useful decision steps (if robust mask provided)
    chain2d_tgt = _build_chain2d_targets(V_target, j_idx, T)
    valid = _build_chain2d_valid_mask(t_idx, j_idx, mask_f, T)
    if chain_robust is not None:
        valid = valid * chain_robust  # zero out entire sub-chains for non-useful t
    mse_c = ((chain2d_V - chain2d_tgt) ** 2 * valid).sum() \
            / valid.sum().clamp(min=1)
    if w_chain > 0:
        total = total + w_chain * mse_c
    metrics["chain_loss"] = mse_c.item()

    # L_action: soft CE on implied decision
    a_tgt = a_target[:, :T_dec]
    act_mask = dec_level_mask[:, :T_dec]
    if problem == "stopping":
        x_norm = batch["values"].to(device)[:, :T_dec].float() / M
        margin = alpha * (x_norm - V_final)
    else:
        margin = alpha * (V_final - 1.0)
    a_soft = torch.sigmoid(margin)
    eps = 1e-7
    ce = -(a_tgt * torch.log(a_soft + eps)
           + (1 - a_tgt) * torch.log(1 - a_soft + eps))
    a_loss = (ce * act_mask).sum() / act_mask.sum().clamp(min=1)
    if w_action > 0:
        total = total + w_action * a_loss
    metrics["action_loss"] = a_loss.item()

    return total, metrics


def train(
    model: OnlineDecisionTransformer,
    train_loader,
    val_loader,
    problem: str,       # "stopping" or "ski"
    n: int,
    M: int = 1000,
    B: float = 10.0,
    r: float = 1.0,
    lr: float = 1e-3,
    epochs: int = 50,
    w_value: float = 1.0,
    w_action: float = 0.5,
    w_chain: float = 0.0,
    action_alpha: float = 10.0,
    training_mode: str = "teacher_forcing",  # "teacher_forcing" or "autoregressive"
    robust_train: bool = False,
    robust_beta: float | None = None,
    robust_lambda: float | None = None,
    robust_U: int | None = None,
    device: str = "cpu",
    checkpoint_path: str = "checkpoint.pt",
    patience: int = 10,  # early stopping: stop if val loss doesn't improve for this many epochs
) -> tuple:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    epochs_without_improvement = 0
    epoch_logs = []

    def _run_batch(batch):
        """Forward + loss for one batch."""
        n_hor = batch.get("n_horizon", None)

        if problem == "stopping":
            x = batch["values"].to(device)
        else:
            x = batch["input_seq"].to(device)
        max_n = x.shape[1]

        task_id = 0 if problem == "stopping" else 1
        t_idx, j_idx, _ = model._get_chain2d_info(max_n, device)

        if problem == "stopping":
            V_target_norm = batch["C"].to(device) / M
        else:
            B_per = batch["B"].to(device).unsqueeze(1) \
                if ("B" in batch and isinstance(batch["B"], torch.Tensor)) else B
            V_target_norm = batch["J"].to(device) / B_per

        chain2d_tgt = _build_chain2d_targets(V_target_norm, j_idx, max_n)

        # Pass cost parameters for ski rental
        B_input = batch.get("B", None)
        r_input = batch.get("r", None)
        if isinstance(B_input, (int, float)):
            B_input = None  # scalar means fixed, model uses default
        if isinstance(r_input, (int, float)):
            r_input = None

        chain2d_V, decision_pos = model(
            x, chain2d_targets=chain2d_tgt if training_mode == "teacher_forcing" else None,
            n_horizon=n_hor, B_cost=B_input, r_cost=r_input,
            task_id=task_id, mode=training_mode)

        # Build robust mask if requested
        rmask = None
        if robust_train:
            n_horizons = batch.get("n_horizon", None)
            if n_horizons is None or not isinstance(n_horizons, torch.Tensor):
                n_horizons = torch.full((x.shape[0],), n, dtype=torch.long)
            if problem == "stopping" and robust_beta is not None:
                rmask = build_stopping_robust_mask(n_horizons, robust_beta, max_n)
            elif problem == "ski" and robust_lambda is not None:
                rmask = build_ski_robust_mask(
                    n_horizons, robust_lambda, B, r, robust_U, max_n)

        B_cost = (batch["B"].to(device) if ("B" in batch and
                  isinstance(batch["B"], torch.Tensor)) else B)
        loss, metrics = compute_loss(
            chain2d_V, decision_pos, batch, t_idx, j_idx,
            problem=problem, M=M, B=B_cost,
            w_value=w_value, w_chain=w_chain, w_action=w_action,
            alpha=action_alpha, robust_mask=rmask)

        return loss, metrics

    import time
    nt = len(train_loader)
    nv = len(val_loader)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr = {"total": 0.0, "value_loss": 0.0, "action_loss": 0.0, "chain_loss": 0.0}
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            loss, metrics = _run_batch(batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr["total"] += loss.item()
            tr["value_loss"] += metrics.get("value_loss", 0.0)
            tr["action_loss"] += metrics.get("action_loss", 0.0)
            tr["chain_loss"] += metrics.get("chain_loss", 0.0)

            # Progress bar (every 10 batches or last batch)
            if (bi + 1) % 10 == 0 or bi + 1 == nt:
                elapsed = time.time() - t0
                eta = elapsed / (bi + 1) * (nt - bi - 1)
                print(f"\r    batch {bi+1}/{nt}  loss={loss.item():.4f}  "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", end="", flush=True)
        print()  # newline after progress bar

        scheduler.step()

        # Validate
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = _run_batch(batch)
                val_total += loss.item()

        comp_name = "train_C" if problem == "stopping" else "train_J"
        log = {
            "epoch": epoch,
            "train_total": tr["total"] / nt,
            comp_name: tr["value_loss"] / nt,
            "train_a": tr["action_loss"] / nt,
            "train_chain": tr["chain_loss"] / nt,
            "val_total": val_total / nv,
        }
        epoch_logs.append(log)

        is_best = val_total < best_val
        chain_str = f"  chain {tr['chain_loss']/nt:.4f}" if w_chain > 0 else ""
        elapsed_epoch = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train {log['train_total']:.4f} "
            f"({comp_name.split('_')[1]} {tr['value_loss']/nt:.4f}  a {tr['action_loss']/nt:.4f}"
            f"{chain_str})  "
            f"val {log['val_total']:.4f}  "
            f"[{elapsed_epoch:.1f}s]"
            + ("  ✓ best" if is_best else f"  (no improvement {epochs_without_improvement+1}/{patience})")
        )

        if is_best:
            best_val = val_total
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    best_epoch = min(epoch_logs, key=lambda l: l["val_total"])["epoch"]
    print(f"\n  Best val loss {best_val/nv:.4f} at epoch {best_epoch}  saved → {checkpoint_path}")
    return model, epoch_logs
