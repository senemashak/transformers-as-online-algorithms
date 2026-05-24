"""
Plotting functions for sweep_weights results.

Reads results.json produced by sweep_weights.py and generates all plots.
Fully decoupled from training — only needs the JSON file.

Plots generated:
  1. Training curves: train/val total loss + component breakdown (per config, shaded across seeds)
  2. Weight sweep ranking: horizontal bar chart of configs ranked by CR
  3. TF vs AR comparison: grouped bars comparing training modes
  4. Per-family heatmap: CR (or loss) by config × family
  5. In-distribution bars: all configs + baselines side by side
  6. Chain supervision effect: configs with chain vs without

Usage:
    python sweep_weights_plots.py results/sweep/sweep_<timestamp>/results.json
    python sweep_weights_plots.py results.json --out_dir plots/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Style ──

# Consistent colors per config name (same config = same color across all plots)
CONFIG_PALETTE = {
    # Single-loss ablations
    "value_only":     "#42A5F5",  # blue
    "action_only":    "#EF5350",  # red
    "chain_only":     "#66BB6A",  # green
    # Pairwise
    "value+action":   "#7E57C2",  # purple
    "value+chain":    "#26A69A",  # teal
    "action+chain":   "#FFA726",  # orange
    # Balanced
    "equal_third":    "#5C6BC0",  # indigo
    "emph_value":     "#29B6F6",  # sky blue
    "emph_action":    "#EC407A",  # pink
    "emph_chain":     "#26C6DA",  # cyan
    "all_1_0.5_1":    "#8D6E63",  # brown
}

# Baselines
BASELINE_COLORS = {
    "Offline":       "#9E9E9E",
    "Bayes DP":      "#F44336",
    "Deterministic": "#FF9800",
}

# Training curve components
LOSS_COLORS = {
    "train":    "#42A5F5",  # blue
    "val":      "#AB47BC",  # purple
    "L_value":  "#42A5F5",  # blue
    "L_action": "#EF5350",  # red
    "L_chain":  "#66BB6A",  # green
}

# Fallback cycle for unknown configs
_FALLBACK = ["#42A5F5", "#66BB6A", "#AB47BC", "#26A69A", "#7E57C2",
             "#29B6F6", "#EF5350", "#FFA726", "#8D6E63", "#EC407A",
             "#5C6BC0", "#26C6DA", "#D4E157", "#78909C", "#FF7043", "#9CCC65"]


def _config_color(name, idx=0):
    """Get color for a config name, falling back to cycle."""
    return CONFIG_PALETTE.get(name, _FALLBACK[idx % len(_FALLBACK)])


def _savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Training curves (per config, shaded across seeds)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(entries, problem, out_dir):
    """One figure per config: train/val loss + component breakdown."""
    for entry in entries:
        config = entry["config"]
        all_curves = [s["training_curve"] for s in entry["seeds"]]
        n_epochs = max(len(c) for c in all_curves)
        epochs = list(range(1, n_epochs + 1))

        def _extract(key):
            # Seeds may have different epoch counts due to early stopping
            # Pad shorter seeds with their last value
            max_len = max(len(c) for c in all_curves)
            rows = []
            for curve in all_curves:
                vals = [ep.get(key, 0) for ep in curve]
                if len(vals) < max_len:
                    vals.extend([vals[-1]] * (max_len - len(vals)))
                rows.append(vals)
            arr = np.array(rows)
            return arr.mean(axis=0), arr.std(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: total train/val loss
        tr_m, tr_s = _extract("train_total")
        va_m, va_s = _extract("val_total")
        ax1.plot(epochs, tr_m, label="Train", color=LOSS_COLORS["train"])
        ax1.fill_between(epochs, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color=LOSS_COLORS["train"])
        ax1.plot(epochs, va_m, label="Val", color=LOSS_COLORS["val"])
        ax1.fill_between(epochs, va_m - va_s, va_m + va_s, alpha=0.15, color=LOSS_COLORS["val"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        n_seeds = len(all_curves)
        mode = "TF" if entry["training_mode"] == "teacher_forcing" else "AR"
        ax1.set_title(f"{config} [{mode}] ({n_seeds} seeds)")
        ax1.legend(fontsize=8)

        # Right: component breakdown
        comp_key = "train_C" if problem == "stopping" else "train_J"
        for key, label, color in [(comp_key, "L value", LOSS_COLORS["L_value"]),
                                   ("train_a", "L action", LOSS_COLORS["L_action"]),
                                   ("train_chain", "L chain", LOSS_COLORS["L_chain"])]:
            if key in all_curves[0][0]:
                m, s = _extract(key)
                # Always plot all 3 components (show as flat zero if inactive)
                ax2.plot(epochs, m, label=label, color=color)
                ax2.fill_between(epochs, m - s, m + s, alpha=0.15, color=color)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Component Loss")
        ax2.set_title("Loss Decomposition")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        safe = config.replace('/', '_').replace(' ', '_')
        _savefig(fig, out_dir / f"training_{problem}_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Weight sweep ranking (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_weight_ranking(entries, problem, out_dir):
    """Horizontal bar chart: configs ranked by best-seed CR (or loss for ski)."""
    # Collect mean CR across eval seeds for the best seed of each config
    configs = []
    for e in entries:
        best = e["seeds"][e["best_seed_idx"]]["eval"]
        cr_mean = float(np.mean(best["in_dist_cr"]))
        cr_std = float(np.std(best["in_dist_cr"]))
        mode = "TF" if e["training_mode"] == "teacher_forcing" else "AR"
        label = f"{e['config']} [{mode}]"
        configs.append((label, cr_mean, cr_std, e["training_mode"]))

    # Also collect ALL seeds' CRs for cross-seed error bars
    configs_allseed = []
    for e in entries:
        all_crs = []
        for s in e["seeds"]:
            all_crs.extend(s["eval"]["in_dist_cr"])
        mode = "TF" if e["training_mode"] == "teacher_forcing" else "AR"
        label = f"{e['config']} [{mode}]"
        configs_allseed.append((label, np.mean(all_crs), np.std(all_crs), e["training_mode"], e["config"]))

    # Sort by CR (higher = better)
    configs_allseed.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(configs_allseed) * 0.4)))
    labels = [c[0] for c in configs_allseed]
    means = [c[1] for c in configs_allseed]
    stds = [c[2] for c in configs_allseed]
    colors = [_config_color(c[4], i) for i, c in enumerate(configs_allseed)]

    ax.barh(range(len(labels)), means, xerr=stds, capsize=3,
            color=colors, edgecolor="#616161", height=0.6,
            error_kw={"linewidth": 0.8, "color": "#424242"})

    # Add baselines
    base = entries[0]["seeds"][0]["eval"].get("baselines", {})
    if "dp_cr" in base:
        dp_cr = float(np.mean(base["dp_cr"]))
        ax.axvline(dp_cr, color=BASELINE_COLORS["Bayes DP"], linestyle="--", linewidth=1.5,
                   label=f"Bayes DP ({dp_cr:.3f})")
    if "offline_cr" in base:
        off_cr = float(np.mean(base["offline_cr"]))
        ax.axvline(off_cr, color=BASELINE_COLORS["Offline"], linestyle=":", linewidth=1.5,
                   label=f"Offline ({off_cr:.3f})")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Competitive Ratio (mean ± std across all seeds)")
    title_prob = "Optimal Stopping" if problem == "stopping" else "Ski Rental"
    ax.set_title(f"Weight Sweep Ranking — {title_prob}")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_dir / f"ranking_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TF vs AR comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_tf_vs_ar(entries, problem, out_dir):
    """Grouped bar chart: TF vs AR for configs that have both."""
    # Group by weight tuple
    from collections import defaultdict
    groups = defaultdict(dict)
    for e in entries:
        key = (round(e["w_value"], 2), round(e["w_action"], 2), round(e["w_chain"], 2))
        mode = e["training_mode"]
        all_crs = []
        for s in e["seeds"]:
            all_crs.extend(s["eval"]["in_dist_cr"])
        groups[key][mode] = (np.mean(all_crs), np.std(all_crs))

    # Only keep configs that have both TF and AR
    paired = {k: v for k, v in groups.items() if "teacher_forcing" in v and "autoregressive" in v}
    if not paired:
        return

    labels = [f"({k[0]},{k[1]},{k[2]})" for k in sorted(paired.keys())]
    tf_means = [paired[k]["teacher_forcing"][0] for k in sorted(paired.keys())]
    tf_stds = [paired[k]["teacher_forcing"][1] for k in sorted(paired.keys())]
    ar_means = [paired[k]["autoregressive"][0] for k in sorted(paired.keys())]
    ar_stds = [paired[k]["autoregressive"][1] for k in sorted(paired.keys())]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - width/2, tf_means, width, yerr=tf_stds, capsize=3,
           label="Teacher Forcing", color="#42A5F5", edgecolor="#424242", linewidth=0.5,
           error_kw={"linewidth": 0.8})
    ax.bar(x + width/2, ar_means, width, yerr=ar_stds, capsize=3,
           label="Autoregressive", color="#66BB6A", edgecolor="#424242", linewidth=0.5,
           error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("(w_value, w_action, w_chain)")
    ax.set_ylabel("Competitive Ratio")
    title_prob = "Optimal Stopping" if problem == "stopping" else "Ski Rental"
    ax.set_title(f"Teacher Forcing vs Autoregressive — {title_prob}")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / f"tf_vs_ar_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-family heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_family_heatmap(entries, problem, families, out_dir):
    """Heatmap: rows=configs, columns=families.

    Generates two heatmaps:
      - CR for both problems (higher = better)
      - P(best) for stopping / additive loss for ski (secondary metric)
    """
    from matplotlib.colors import LinearSegmentedColormap

    metrics = [("cr", "per_family_cr", "Competitive Ratio", True)]
    if problem == "stopping":
        metrics.append(("prob_best", "per_family_prob_best", "P(best)", True))
    else:
        metrics.append(("additive_loss", "per_family_additive_loss", "Additive Loss", False))

    for suffix, key, metric_label, higher_better in metrics:
        configs = []
        data = []
        for e in entries:
            best = e["seeds"][e["best_seed_idx"]]["eval"]
            if key not in best:
                continue
            mode = "TF" if e["training_mode"] == "teacher_forcing" else "AR"
            configs.append(f"{e['config']} [{mode}]")
            row = [float(np.mean(best[key].get(f, [0]))) for f in families]
            data.append(row)

        if not data:
            continue

        data = np.array(data)
        fig, ax = plt.subplots(figsize=(max(10, len(families) * 0.9), max(4, len(configs) * 0.35)))

        cmap = LinearSegmentedColormap.from_list("gp", ["#E1BEE7", "#FFFFFF", "#A5D6A7"])
        if not higher_better:
            cmap = cmap.reversed()
        im = ax.imshow(data, aspect="auto", cmap=cmap)

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=7)

        for i in range(len(configs)):
            for j in range(len(families)):
                val = data[i, j]
                best_in_col = data[:, j].max() if higher_better else data[:, j].min()
                weight = "bold" if val == best_in_col else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5,
                        fontweight=weight)

        title_prob = "Optimal Stopping" if problem == "stopping" else "Ski Rental"
        ax.set_title(f"Per-Family {metric_label} — {title_prob}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        _savefig(fig, out_dir / f"per_family_{suffix}_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. In-distribution summary bars
# ═══════════════════════════════════════════════════════════════════════════

def plot_in_dist_summary(entries, problem, out_dir):
    """Bar charts: best-seed metrics for each config + baselines.

    Generates two plots:
      - CR (both problems)
      - P(best) for stopping / additive loss for ski
    """
    metrics_to_plot = [("cr", "in_dist_cr", "Competitive Ratio", True)]
    if problem == "stopping":
        metrics_to_plot.append(("prob_best", "in_dist_prob_best", "P(best)", True))
    else:
        metrics_to_plot.append(("additive_loss", "in_dist_additive_loss", "Additive Loss", False))

    baseline_map = {
        "cr": {"Offline": "offline_cr", "Bayes DP": "dp_cr", "Deterministic": "deterministic_cr"},
        "prob_best": {"Offline": "offline_prob_best", "Bayes DP": "dp_prob_best"},
        "additive_loss": {"Bayes DP": "dp_loss", "Deterministic": "deterministic_loss"},
    }

    for suffix, eval_key, ylabel, higher_better in metrics_to_plot:
        labels, means, stds = [], [], []

        # Baselines
        base = entries[0]["seeds"][0]["eval"].get("baselines", {})
        for bname, bkey in baseline_map.get(suffix, {}).items():
            if bkey in base:
                labels.append(bname)
                means.append(float(np.mean(base[bkey])))
                stds.append(float(np.std(base[bkey])))

        n_baselines = len(labels)

        # All configs sorted by metric
        sorted_entries = sorted(entries,
                                key=lambda e: np.mean(e["seeds"][e["best_seed_idx"]]["eval"].get(eval_key, [0])),
                                reverse=higher_better)
        for e in sorted_entries:
            best = e["seeds"][e["best_seed_idx"]]["eval"]
            vals = best.get(eval_key, [0])
            mode = "TF" if e["training_mode"] == "teacher_forcing" else "AR"
            labels.append(f"{e['config']} [{mode}]")
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
        bar_colors = [BASELINE_COLORS.get(labels[i], "#E0E0E0") for i in range(n_baselines)]
        bar_colors += [_config_color(e["config"], i) for i, e in enumerate(sorted_entries)]

        ax.bar(range(len(labels)), means, yerr=stds, capsize=2,
               color=bar_colors, edgecolor="#616161", linewidth=0.5,
               error_kw={"linewidth": 0.8, "color": "#424242"})
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        title_prob = "Optimal Stopping" if problem == "stopping" else "Ski Rental"
        ax.set_title(f"In-Distribution {ylabel} — {title_prob}")
        fig.tight_layout()
        _savefig(fig, out_dir / f"in_dist_{suffix}_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Chain supervision effect
# ═══════════════════════════════════════════════════════════════════════════

def plot_chain_effect(entries, problem, out_dir):
    """Compare configs with w_chain=0 vs w_chain>0 (TF only for fair comparison)."""
    tf_entries = [e for e in entries if e["training_mode"] == "teacher_forcing"]
    with_chain = [e for e in tf_entries if e["w_chain"] > 0]
    without_chain = [e for e in tf_entries if e["w_chain"] == 0]

    if not with_chain or not without_chain:
        return

    def _get_cr(e):
        all_crs = []
        for s in e["seeds"]:
            all_crs.extend(s["eval"]["in_dist_cr"])
        return np.mean(all_crs), np.std(all_crs)

    labels_w = [e["config"] for e in with_chain]
    means_w = [_get_cr(e)[0] for e in with_chain]
    stds_w = [_get_cr(e)[1] for e in with_chain]
    colors_w = [_config_color(e["config"], i) for i, e in enumerate(with_chain)]

    labels_wo = [e["config"] for e in without_chain]
    means_wo = [_get_cr(e)[0] for e in without_chain]
    stds_wo = [_get_cr(e)[1] for e in without_chain]
    colors_wo = [_config_color(e["config"], i) for i, e in enumerate(without_chain)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # With chain
    ax1.barh(range(len(labels_w)), means_w, xerr=stds_w, capsize=3,
             color=colors_w, edgecolor="#424242", height=0.6)
    ax1.set_yticks(range(len(labels_w)))
    ax1.set_yticklabels(labels_w, fontsize=8)
    ax1.set_xlabel("CR")
    ax1.set_title("w_chain > 0 (supervised chain)")
    ax1.invert_yaxis()

    # Without chain
    ax2.barh(range(len(labels_wo)), means_wo, xerr=stds_wo, capsize=3,
             color=colors_wo, edgecolor="#424242", height=0.6)
    ax2.set_yticks(range(len(labels_wo)))
    ax2.set_yticklabels(labels_wo, fontsize=8)
    ax2.set_xlabel("CR")
    ax2.set_title("w_chain = 0 (unsupervised scratchpad)")
    ax2.invert_yaxis()

    # Align x-axes
    xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)

    title_prob = "Optimal Stopping" if problem == "stopping" else "Ski Rental"
    fig.suptitle(f"Chain Supervision Effect — {title_prob}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out_dir / f"chain_effect_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main: load JSON and generate all plots
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str, help="Path to results.json from sweep_weights")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (default: plots/ next to results.json)")
    args = parser.parse_args()

    json_path = Path(args.results_json)
    with open(json_path) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else json_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = data["meta"]
    print(f"Loaded results: {len(meta['configs'])} configs, "
          f"{meta['n_train_seeds']} train seeds, {meta['n_eval_seeds']} eval seeds")
    print(f"Plots → {out_dir}\n")

    for problem in data.get("meta", {}).get("problems", ["stopping", "ski"]):
        entries = data.get(problem, [])
        if not entries:
            continue

        families = meta[f"{problem}_families_tested"]
        print(f"=== {problem.upper()} ({len(entries)} configs) ===")

        print("  Training curves...")
        plot_training_curves(entries, problem, out_dir)

        print("  Weight ranking...")
        plot_weight_ranking(entries, problem, out_dir)

        print("  Per-family heatmap...")
        plot_per_family_heatmap(entries, problem, families, out_dir)

        print("  In-distribution summary...")
        plot_in_dist_summary(entries, problem, out_dir)

        print("  Chain effect...")
        plot_chain_effect(entries, problem, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots generated.")


if __name__ == "__main__":
    main()
