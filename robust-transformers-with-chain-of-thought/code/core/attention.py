"""
Attention extraction and visualization for mechanistic interpretation.

Two-step workflow:
  1. Extract: load a model checkpoint, run on chosen instances, save attention maps
  2. Visualize: load saved attention maps, generate plots

Usage:
    # Extract attention from best value+action model on 5 diverse instances
    python attention.py extract \
        --checkpoint results/sweep/.../models/stopping_value+action.pt \
        --out attention_maps/value+action.pt

    # Extract with specific family and horizon
    python attention.py extract \
        --checkpoint model.pt \
        --family geometric --n 30 --num_instances 3

    # Visualize saved attention maps
    python attention.py plot \
        --input attention_maps/value+action.pt \
        --out_dir plots/attention_value+action/

    # Extract and plot in one go
    python attention.py both \
        --checkpoint model.pt \
        --out_dir plots/attention/
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from core.model import OnlineDecisionTransformer
from core.sampling import sample_stopping_batch, STOPPING_SAMPLERS
from core.dp import stopping_labels
from core.train import _build_chain2d_targets


# ═══════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_attention(model, instances, M, device="cpu"):
    """Run teacher-forced forward pass and extract attention weights.

    Args:
        model     : trained OnlineDecisionTransformer
        instances : list of StoppingInstance (all same horizon n)
        M         : domain size
        device    : device

    Returns:
        dict with:
            attn_weights    : list of (B, n_heads, L, L) per layer
            info            : position labels, t_idx, j_idx, decision_pos
            x               : (B, n) observation values
            V_target        : (B, n) normalized DP targets
            instance_info   : list of dicts with family, params, pmf per instance
    """
    model.eval()
    n = instances[0].n
    B = len(instances)

    x = torch.tensor(np.stack([inst.values for inst in instances]), dtype=torch.long)

    # Compute DP labels for teacher forcing
    V_targets = []
    for inst in instances:
        lbl = stopping_labels(inst.pmf, inst.values)
        V_targets.append(lbl["C"] / M)
    V_target = torch.tensor(np.stack(V_targets), dtype=torch.float32)

    t_idx, j_idx, _ = model._get_chain2d_info(n, x.device)
    chain2d_tgt = _build_chain2d_targets(V_target, j_idx, n)

    x_dev = x.to(device)
    with torch.no_grad():
        chain2d_V, decision_pos, attn = model(
            x_dev, chain2d_targets=chain2d_tgt.to(device),
            n_horizon=n, task_id=0,
            mode="teacher_forcing", return_attention=True)

    # Build position labels
    labels = [f"x_{t}" for t in range(n)]
    for p in range(len(t_idx)):
        t = int(t_idx[p])
        k = n - 2 - int(j_idx[p])
        labels.append(f"V({t},{k})")

    # Instance metadata
    instance_info = []
    for inst in instances:
        instance_info.append({
            "family": inst.dist_type,
            "params": inst.params,
            "values": inst.values.tolist(),
            "n": inst.n,
        })

    return {
        "attn_weights": [a.cpu() for a in attn],
        "info": {
            "n_obs": n,
            "t_idx": t_idx.cpu(),
            "j_idx": j_idx.cpu(),
            "decision_pos": decision_pos.cpu(),
            "position_labels": labels,
        },
        "x": x.cpu(),
        "V_target": V_target.cpu(),
        "chain2d_V": chain2d_V.cpu(),
        "instance_info": instance_info,
        "M": M,
        "n": n,
    }


def sample_diverse_instances(n, M, num_instances=5, seed=42):
    """Sample instances from different families for diverse attention patterns."""
    rng = np.random.default_rng(seed)
    families = list(STOPPING_SAMPLERS.keys())
    instances = []
    for i in range(num_instances):
        fam = families[i % len(families)]
        inst = sample_stopping_batch(1, n, M, dist_type=fam, rng=rng)[0]
        instances.append(inst)
    return instances


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_full_maps(data, out_dir, instance_idx=0):
    """Full attention heatmaps with quadrant labels."""
    attn = data["attn_weights"]
    n = data["n"]
    L = attn[0].shape[2]
    config = data.get("config", "model")
    inst_info = data["instance_info"][instance_idx]
    family = inst_info["family"]

    for layer in range(len(attn)):
        for head in range(attn[0].shape[1]):
            fig, ax = plt.subplots(figsize=(12, 10))
            A = attn[layer][instance_idx, head].numpy()
            im = ax.imshow(A, cmap="Blues", vmin=0, aspect="auto")
            ax.axhline(n - 0.5, color="red", linewidth=1.5, linestyle="--")
            ax.axvline(n - 0.5, color="red", linewidth=1.5, linestyle="--")
            mid, cmid = n // 2, n + (L - n) // 2
            ax.text(mid, mid, "obs\u2192obs", ha="center", va="center",
                    fontsize=10, color="red", alpha=0.6, fontweight="bold")
            ax.text(cmid, mid, "obs\u2192chain\n(blocked)", ha="center", va="center",
                    fontsize=9, color="gray", alpha=0.5)
            ax.text(mid, cmid, "chain\u2192obs\n(dist. inference)", ha="center", va="center",
                    fontsize=9, color="red", alpha=0.6, fontweight="bold")
            ax.text(cmid, cmid, "chain\u2192chain\n(DP recurrence)", ha="center", va="center",
                    fontsize=9, color="red", alpha=0.6, fontweight="bold")
            ax.set_xlabel("Key (attending to)")
            ax.set_ylabel("Query (attending from)")
            ax.set_title(f"Layer {layer}, Head {head} — family: {family}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / f"full_L{layer}_H{head}_inst{instance_idx}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_decision_obs_profiles(data, out_dir, instance_idx=0):
    """What observations does V(t,t) attend to at each decision step?"""
    attn = data["attn_weights"]
    n = data["n"]
    decision_pos = data["info"]["decision_pos"].tolist()
    inst_info = data["instance_info"][instance_idx]
    x_vals = data["x"][instance_idx].numpy()

    test_steps = [0, n // 4, n // 2, 3 * n // 4, min(n - 2, len(decision_pos) - 1)]

    for layer in range(len(attn)):
        n_heads = attn[0].shape[1]
        fig, axes = plt.subplots(n_heads, len(test_steps), figsize=(18, 3 * n_heads),
                                 sharey="row")
        if n_heads == 1:
            axes = axes[np.newaxis, :]

        for col, t in enumerate(test_steps):
            if t >= len(decision_pos):
                continue
            gp = n + decision_pos[t]
            for head in range(n_heads):
                ax = axes[head, col]
                weights = attn[layer][instance_idx, head, gp, :n].numpy()
                bars = ax.bar(range(n), weights, color="#90CAF9", alpha=0.7)
                # Highlight visible positions (0..t)
                for i in range(min(t + 1, n)):
                    bars[i].set_color("#1565C0")
                    bars[i].set_alpha(0.9)
                ax.set_xlim(-0.5, n - 0.5)
                if head == 0:
                    ax.set_title(f"V({t},{t})", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"Head {head}")
                if head == n_heads - 1:
                    ax.set_xlabel("Obs position")

        fig.suptitle(f"Layer {layer}: V(t,t) \u2192 observations (dark = visible x_0..x_t)\n"
                     f"Family: {inst_info['family']}, values: {x_vals[:8].tolist()}...",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"decision_obs_L{layer}_inst{instance_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_subchain_recurrence(data, out_dir, instance_idx=0):
    """Within-sub-chain attention: does V(t,k) attend to V(t,k+1)?"""
    from model import OnlineDecisionTransformer

    attn = data["attn_weights"]
    n = data["n"]
    n_heads = attn[0].shape[1]

    for t_show in [0, n // 4, n // 2]:
        sub_len = n - 1 - t_show
        if sub_len <= 1:
            continue
        off = OnlineDecisionTransformer._chain2d_offset(t_show, n)
        chain_positions = list(range(n + off, n + off + sub_len))

        for layer in range(len(attn)):
            fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5))
            if n_heads == 1:
                axes = [axes]
            for head in range(n_heads):
                ax = axes[head]
                sub_attn = np.zeros((sub_len, sub_len))
                for i in range(sub_len):
                    for j in range(sub_len):
                        sub_attn[i, j] = attn[layer][instance_idx, head,
                                                      chain_positions[i],
                                                      chain_positions[j]].item()
                im = ax.imshow(sub_attn, cmap="Blues", vmin=0)
                tick_labels = [f"V({t_show},{n - 2 - j})" for j in range(sub_len)]
                step = max(1, sub_len // 8)
                ax.set_xticks(range(0, sub_len, step))
                ax.set_xticklabels([tick_labels[i] for i in range(0, sub_len, step)],
                                   fontsize=7, rotation=45, ha="right")
                ax.set_yticks(range(0, sub_len, step))
                ax.set_yticklabels([tick_labels[i] for i in range(0, sub_len, step)], fontsize=7)
                ax.set_title(f"Head {head}")
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
            fig.suptitle(f"Layer {layer}: Sub-chain t={t_show} internal attention", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"subchain_t{t_show}_L{layer}_inst{instance_idx}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_head_specialization(data, out_dir, instance_idx=0):
    """Fraction of attention to obs vs chain at each decision step."""
    attn = data["attn_weights"]
    n = data["n"]
    decision_pos = data["info"]["decision_pos"].tolist()
    n_layers = len(attn)
    n_heads = attn[0].shape[1]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6 * n_heads, 4 * n_layers))
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes[np.newaxis, :]
    elif n_heads == 1:
        axes = axes[:, np.newaxis]

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            obs_f, chain_f = [], []
            for t in range(len(decision_pos)):
                gp = n + decision_pos[t]
                w = attn[layer][instance_idx, head, gp, :].numpy()
                obs_f.append(w[:n].sum())
                chain_f.append(w[n:].sum())
            ax.bar(range(len(decision_pos)), obs_f, label="obs", color="#42A5F5", alpha=0.8)
            ax.bar(range(len(decision_pos)), chain_f, bottom=obs_f,
                   label="chain", color="#66BB6A", alpha=0.8)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Decision step t")
            ax.set_ylabel("Attention fraction")
            ax.set_title(f"Layer {layer}, Head {head}")
            if layer == 0 and head == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Head Specialization (obs vs chain attention per decision step)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"head_specialization_inst{instance_idx}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_accuracy(data, out_dir, instance_idx=0):
    """Compare model's chain predictions V(t,t) against DP targets."""
    chain_V = data["chain2d_V"][instance_idx].numpy()
    V_target = data["V_target"][instance_idx].numpy()
    decision_pos = data["info"]["decision_pos"].tolist()
    x_vals = data["x"][instance_idx].numpy()
    M = data["M"]
    n = data["n"]
    inst_info = data["instance_info"][instance_idx]

    V_pred = chain_V[decision_pos]  # (n-1,)
    V_true = V_target[:n - 1]       # (n-1,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: V(t) predictions vs targets
    steps = range(n - 1)
    ax1.plot(steps, V_true, "o-", color="#1565C0", label="DP target (C_t/M)", markersize=4)
    ax1.plot(steps, V_pred, "s-", color="#EF5350", label="Model V(t)", markersize=4)
    ax1.fill_between(steps, V_true, V_pred, alpha=0.1, color="#EF5350")
    ax1.set_xlabel("Decision step t")
    ax1.set_ylabel("Normalized value")
    ax1.set_title("Continuation value: prediction vs target")
    ax1.legend(fontsize=8)

    # Right: observations and implied decisions
    x_norm = x_vals / M
    ax2.bar(range(n), x_norm, color="#90CAF9", alpha=0.6, label="x_t / M")
    ax2.plot(range(n - 1), V_pred, "s-", color="#EF5350", label="V(t) (threshold)", markersize=4)
    ax2.plot(range(n - 1), V_true, "o-", color="#1565C0", label="C_t / M (true threshold)",
             markersize=4, alpha=0.7)
    # Mark where x_t >= V(t)*M (model would stop)
    for t in range(n - 1):
        if x_norm[t] >= V_pred[t]:
            ax2.axvline(t, color="#66BB6A", alpha=0.3, linewidth=8)
            ax2.text(t, max(x_norm.max(), V_pred.max()) * 1.02, "stop",
                     ha="center", fontsize=7, color="#2E7D32")
            break
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Normalized value")
    ax2.set_title(f"Observations and thresholds ({inst_info['family']})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"predictions_inst{instance_idx}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(data, out_dir):
    """Generate all plots for all instances in the data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_instances = data["x"].shape[0]

    for i in range(n_instances):
        family = data["instance_info"][i]["family"]
        print(f"  Instance {i} ({family})...")
        plot_full_maps(data, out_dir, instance_idx=i)
        plot_decision_obs_profiles(data, out_dir, instance_idx=i)
        plot_subchain_recurrence(data, out_dir, instance_idx=i)
        plot_head_specialization(data, out_dir, instance_idx=i)
        plot_prediction_accuracy(data, out_dir, instance_idx=i)

    print(f"  {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Attention extraction and visualization")
    sub = parser.add_subparsers(dest="command")

    # Extract
    p_ext = sub.add_parser("extract", help="Extract attention maps from a model checkpoint")
    p_ext.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    p_ext.add_argument("--out", default="attention_maps.pt", help="Output .pt file")
    p_ext.add_argument("--M", type=int, default=1000)
    p_ext.add_argument("--n", type=int, default=20, help="Horizon for test instances")
    p_ext.add_argument("--family", type=str, default=None, help="Specific family (default: diverse sample)")
    p_ext.add_argument("--num_instances", type=int, default=5)
    p_ext.add_argument("--seed", type=int, default=42)
    p_ext.add_argument("--device", type=str, default="cpu")

    # Plot
    p_plot = sub.add_parser("plot", help="Visualize saved attention maps")
    p_plot.add_argument("--input", required=True, help="Path to saved .pt attention file")
    p_plot.add_argument("--out_dir", default="plots/attention/")
    p_plot.add_argument("--instance", type=int, default=None, help="Specific instance index (default: all)")

    # Both
    p_both = sub.add_parser("both", help="Extract and plot in one step")
    p_both.add_argument("--checkpoint", required=True)
    p_both.add_argument("--out_dir", default="plots/attention/")
    p_both.add_argument("--M", type=int, default=1000)
    p_both.add_argument("--n", type=int, default=20)
    p_both.add_argument("--family", type=str, default=None)
    p_both.add_argument("--num_instances", type=int, default=5)
    p_both.add_argument("--seed", type=int, default=42)
    p_both.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.command == "extract" or args.command == "both":
        print(f"Loading model from {args.checkpoint}...")
        model = OnlineDecisionTransformer(M=args.M)
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.to(args.device)

        print(f"Sampling {args.num_instances} instances (n={args.n}, family={args.family or 'diverse'})...")
        if args.family:
            instances = sample_stopping_batch(args.num_instances, args.n, args.M,
                                              dist_type=args.family,
                                              rng=np.random.default_rng(args.seed))
        else:
            instances = sample_diverse_instances(args.n, args.M,
                                                 num_instances=args.num_instances,
                                                 seed=args.seed)

        print("Extracting attention...")
        data = extract_attention(model, instances, args.M, device=args.device)
        data["config"] = Path(args.checkpoint).stem

        if args.command == "extract":
            torch.save(data, args.out)
            print(f"Saved to {args.out}")
        else:
            out_dir = Path(args.out_dir)
            plot_all(data, out_dir)

    elif args.command == "plot":
        print(f"Loading attention maps from {args.input}...")
        data = torch.load(args.input, weights_only=False)
        out_dir = Path(args.out_dir)
        if args.instance is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            i = args.instance
            print(f"  Instance {i} ({data['instance_info'][i]['family']})...")
            plot_full_maps(data, out_dir, instance_idx=i)
            plot_decision_obs_profiles(data, out_dir, instance_idx=i)
            plot_subchain_recurrence(data, out_dir, instance_idx=i)
            plot_head_specialization(data, out_dir, instance_idx=i)
            plot_prediction_accuracy(data, out_dir, instance_idx=i)
        else:
            plot_all(data, out_dir)


if __name__ == "__main__":
    main()
