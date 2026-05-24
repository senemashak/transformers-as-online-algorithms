"""
Plot horizon generalization results.

Reads results.json from eval_horizon.py and generates:
  1. CR vs horizon: DP, value+action, equal_third+wrapper, masked+wrapper
  2. Wrapper benefit: how much the wrapper helps at each horizon
  3. Standard vs masked at each horizon for a chosen beta

Usage:
    python plot_horizon.py results/eval_horizon/results.json
    python plot_horizon.py results/eval_horizon/results.json --out_dir plots/
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_json).parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = data["meta"]["horizons"]
    betas = data["meta"]["betas"]
    train_lo, train_hi = data["meta"]["train_range"]

    # =====================================================
    # 1. MAIN PLOT: CR vs horizon
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # DP oracle
    dp_means = [np.mean(data["dp"][str(n)]) for n in horizons]
    dp_stds = [np.std(data["dp"][str(n)]) for n in horizons]
    ax.plot(horizons, dp_means, "D-", color="#D32F2F", label="DP oracle (knows F)",
            linewidth=2, markersize=5)
    ax.fill_between(horizons, np.array(dp_means) - np.array(dp_stds),
                    np.array(dp_means) + np.array(dp_stds), alpha=0.1, color="#D32F2F")

    # value+action (no wrapper)
    va_means = [np.mean(data["value_action"][str(n)]["cr_learned"]) for n in horizons]
    va_stds = [np.std(data["value_action"][str(n)]["cr_learned"]) for n in horizons]
    ax.plot(horizons, va_means, "o-", color="#42A5F5", label="value+action (no wrapper)",
            linewidth=2, markersize=5)
    ax.fill_between(horizons, np.array(va_means) - np.array(va_stds),
                    np.array(va_means) + np.array(va_stds), alpha=0.1, color="#42A5F5")

    # equal_third + best wrapper
    # Find best beta at training-range horizon
    mid_h = str(min(horizons, key=lambda h: abs(h - 35)))  # middle of training range
    et_data = data["equal_third"][mid_h]
    best_beta = max(betas, key=lambda b: np.mean(et_data[f"cr_robust_{b:.2f}"]))

    et_means = [np.mean(data["equal_third"][str(n)][f"cr_robust_{best_beta:.2f}"]) for n in horizons]
    et_stds = [np.std(data["equal_third"][str(n)][f"cr_robust_{best_beta:.2f}"]) for n in horizons]
    ax.plot(horizons, et_means, "s-", color="#66BB6A",
            label=f"equal third + wrapper (beta={best_beta:.2f})",
            linewidth=2, markersize=5)
    ax.fill_between(horizons, np.array(et_means) - np.array(et_stds),
                    np.array(et_means) + np.array(et_stds), alpha=0.1, color="#66BB6A")

    # masked at best_beta + wrapper
    if f"{best_beta:.2f}" in data["masked"].get(str(horizons[0]), {}):
        mk_means = [np.mean(data["masked"][str(n)][f"{best_beta:.2f}"][f"cr_robust_{best_beta:.2f}"])
                     for n in horizons]
        mk_stds = [np.std(data["masked"][str(n)][f"{best_beta:.2f}"][f"cr_robust_{best_beta:.2f}"])
                    for n in horizons]
        ax.plot(horizons, mk_means, "^-", color="#7E57C2",
                label=f"masked (beta={best_beta:.2f}) + wrapper",
                linewidth=2, markersize=5)
        ax.fill_between(horizons, np.array(mk_means) - np.array(mk_stds),
                        np.array(mk_means) + np.array(mk_stds), alpha=0.1, color="#7E57C2")

    # Training range shading
    ax.axvspan(train_lo, train_hi, alpha=0.08, color="#FFA726",
               label=f"Training range [{train_lo}, {train_hi}]")

    ax.set_xlabel("Test horizon n", fontsize=12)
    ax.set_ylabel("Competitive Ratio", fontsize=12)
    ax.set_title("Horizon Generalization: Optimal Stopping", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "horizon_cr.png")

    # =====================================================
    # 2. WRAPPER BENEFIT: CR(with wrapper) - CR(no wrapper)
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 5))

    # equal_third: wrapper benefit at each beta
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(betas)))
    for bi, beta in enumerate(betas):
        et_no_wrap = [np.mean(data["equal_third"][str(n)]["cr_learned"]) for n in horizons]
        et_wrap = [np.mean(data["equal_third"][str(n)][f"cr_robust_{beta:.2f}"]) for n in horizons]
        benefit = [w - nw for w, nw in zip(et_wrap, et_no_wrap)]
        ax.plot(horizons, benefit, "o-", color=colors[bi], label=f"beta={beta:.2f}",
                linewidth=1.5, markersize=4, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(train_lo, train_hi, alpha=0.08, color="#FFA726")
    ax.set_xlabel("Test horizon n", fontsize=12)
    ax.set_ylabel("CR(with wrapper) - CR(no wrapper)", fontsize=12)
    ax.set_title("Wrapper Benefit vs Horizon (equal third model)", fontsize=13)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "horizon_wrapper_benefit.png")

    # =====================================================
    # 3. STANDARD vs MASKED at each horizon (chosen beta)
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 5))

    for beta in [0.15, 0.25, 0.35]:
        if f"{beta:.2f}" not in data["masked"].get(str(horizons[0]), {}):
            continue
        std_cr = [np.mean(data["equal_third"][str(n)][f"cr_robust_{beta:.2f}"]) for n in horizons]
        msk_cr = [np.mean(data["masked"][str(n)][f"{beta:.2f}"][f"cr_robust_{beta:.2f}"]) for n in horizons]
        diff = [m - s for m, s in zip(msk_cr, std_cr)]

        color = {"0.15": "#42A5F5", "0.25": "#66BB6A", "0.35": "#7E57C2"}[f"{beta:.2f}"]
        ax.plot(horizons, diff, "o-", color=color, label=f"beta={beta:.2f}",
                linewidth=2, markersize=5)

    ax.axhline(0, color="black", linewidth=0.5, label="no difference")
    ax.axvspan(train_lo, train_hi, alpha=0.08, color="#FFA726")
    ax.set_xlabel("Test horizon n", fontsize=12)
    ax.set_ylabel("CR(masked) - CR(standard)", fontsize=12)
    ax.set_title("Masked vs Standard Training: CR Difference at Each Horizon", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "horizon_masked_vs_standard.png")

    # =====================================================
    # 4. ALL MODELS SUMMARY TABLE
    # =====================================================
    print(f"\n{'Horizon':<10} {'DP':>8} {'v+a':>8} {'eq+wrap':>8} {'msk+wrap':>8}")
    print("─" * 44)
    for n in horizons:
        dp = np.mean(data["dp"][str(n)])
        va = np.mean(data["value_action"][str(n)]["cr_learned"])
        et = np.mean(data["equal_third"][str(n)][f"cr_robust_{best_beta:.2f}"])
        mk_key = f"{best_beta:.2f}"
        mk = np.mean(data["masked"][str(n)][mk_key][f"cr_robust_{best_beta:.2f}"]) if mk_key in data["masked"][str(n)] else 0
        tag = " *" if train_lo <= n <= train_hi else ""
        print(f"n={n:<6}{tag} {dp:>8.4f} {va:>8.4f} {et:>8.4f} {mk:>8.4f}")

    print(f"\n(* = within training range [{train_lo}, {train_hi}])")
    print(f"Best wrapper beta: {best_beta:.2f}")
    print(f"\nDone. {len(list(out_dir.glob('horizon_*.png')))} plots.")


if __name__ == "__main__":
    main()
