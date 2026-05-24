"""
Plot robust evaluation results: standard vs masked training at each beta.

Reads results.json files from eval_robust_p1/p2/p3 and generates:
  1. CR vs beta (standard, masked, no-wrapper, DP oracle)
  2. CR difference (masked - standard) per beta
  3. Per-family heatmap at best beta

Usage:
    python plot_robust_eval.py
    python plot_robust_eval.py --result_dirs results/eval_robust_p1 results/eval_robust_p2 results/eval_robust_p3
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def _savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+",
                        default=["results/eval_robust_p1",
                                 "results/eval_robust_p2",
                                 "results/eval_robust_p3"])
    parser.add_argument("--out_dir", default="../plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_baseline, all_masked = [], []
    for d in args.result_dirs:
        rfile = Path(d) / "results.json"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            data = json.load(f)
        all_baseline.extend(data["baseline"])
        all_masked.extend(data["masked"])

    all_baseline.sort(key=lambda e: e["beta"])
    all_masked.sort(key=lambda e: e["beta"])

    betas_std = [e["beta"] for e in all_baseline]
    cr_std = [np.mean(e["eval"]["cr_robust"]) for e in all_baseline]
    cr_std_err = [np.std(e["eval"]["cr_robust"]) for e in all_baseline]

    betas_msk = [e["beta"] for e in all_masked]
    cr_msk = [np.mean(e["eval"]["cr_robust"]) for e in all_masked]
    cr_msk_err = [np.std(e["eval"]["cr_robust"]) for e in all_masked]

    cr_learned = [np.mean(e["eval"]["cr_learned"]) for e in all_baseline]
    dp_cr = np.mean(all_baseline[0]["eval"]["baselines"]["dp_cr"])

    print(f"Loaded {len(all_baseline)} baseline evals, {len(all_masked)} masked evals")

    # 1. CR vs beta
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(betas_std, cr_std, "o-", color="#42A5F5", label="Standard training + wrapper",
            linewidth=2, markersize=6)
    ax.fill_between(betas_std, np.array(cr_std) - np.array(cr_std_err),
                    np.array(cr_std) + np.array(cr_std_err), alpha=0.15, color="#42A5F5")
    ax.plot(betas_msk, cr_msk, "s-", color="#7E57C2", label="Masked training + wrapper",
            linewidth=2, markersize=6)
    ax.fill_between(betas_msk, np.array(cr_msk) - np.array(cr_msk_err),
                    np.array(cr_msk) + np.array(cr_msk_err), alpha=0.15, color="#7E57C2")
    ax.plot(betas_std, cr_learned, "^--", color="#66BB6A", label="Standard training, no wrapper",
            linewidth=1.5, markersize=5, alpha=0.7)
    ax.axhline(dp_cr, color="#D32F2F", linestyle="--", linewidth=1.5,
               label=f"DP oracle (knows F) = {dp_cr:.3f}")
    ax.set_xlabel("Robustness parameter beta", fontsize=12)
    ax.set_ylabel("Competitive Ratio", fontsize=12)
    ax.set_title("Optimal Stopping: Robust Deployment\nStandard vs Masked Training (1/3, 1/3, 1/3 weights)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 0.39)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "robust_cr_vs_beta.png")

    # 2. Difference plot
    fig, ax = plt.subplots(figsize=(10, 4))
    diffs, diff_errs = [], []
    for beta in betas_msk:
        s = next(e for e in all_baseline if e["beta"] == beta)
        m = next(e for e in all_masked if e["beta"] == beta)
        diff = np.array(m["eval"]["cr_robust"]) - np.array(s["eval"]["cr_robust"])
        diffs.append(np.mean(diff))
        diff_errs.append(np.std(diff))
    ax.bar(range(len(betas_msk)), diffs, yerr=diff_errs, capsize=4,
           color=["#66BB6A" if d > 0 else "#EF5350" for d in diffs],
           edgecolor="#424242", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(betas_msk)))
    ax.set_xticklabels([f"{b:.2f}" for b in betas_msk])
    ax.set_xlabel("Beta")
    ax.set_ylabel("CR difference (masked - standard)")
    ax.set_title("Masked vs Standard: CR Difference per Beta\n(green = masked better, red = standard better)")
    fig.tight_layout()
    _savefig(fig, out_dir / "robust_cr_diff.png")

    # 3. Per-family heatmap at best beta (including DP oracle)
    from sampling import sample_stopping_batch, STOPPING_SAMPLERS as _SAMPLERS
    from deployment import compare_stopping

    best_idx = np.argmax(cr_std)
    best_beta = betas_std[best_idx]
    s_entry = all_baseline[best_idx]
    m_entry = next((e for e in all_masked if e["beta"] == best_beta), None)
    families = list(s_entry["eval"]["per_family_cr_robust"].keys())

    # Compute DP oracle per family (same eval seeds, no model needed)
    print("  Computing DP oracle per family...")
    eval_seeds = [50042, 50043, 50044, 50045, 50046]
    n_min, n_max, M = 20, 50, 1000
    dp_per_family = {f: [] for f in families}
    for seed in eval_seeds:
        for fam in families:
            frng = np.random.default_rng(seed + hash(fam) % 10000)
            nf = 1000 // len(families)
            fh = frng.integers(n_min, n_max + 1, size=nf)
            dr, mx = [], []
            for h in sorted(set(fh)):
                c = int((fh == h).sum())
                insts = sample_stopping_batch(c, int(h), M, dist_type=fam,
                    rng=np.random.default_rng(seed + hash(fam) % 10000 + int(h)))
                res = compare_stopping(insts, None, betas=[], r_fractions=[])
                dr.append(res["dp"]["mean_reward"] * c)
                mx.append(res["offline"]["mean_reward"] * c)
            ft = len(fh)
            fmm = sum(mx) / ft
            dp_per_family[fam].append(sum(dr) / ft / fmm if fmm > 0 else 0)

    models = ["DP oracle (knows F)", "Standard + wrapper"]
    rows = [
        [np.mean(dp_per_family[f]) for f in families],
        [np.mean(s_entry["eval"]["per_family_cr_robust"][f]) for f in families],
    ]
    if m_entry:
        models.append("Masked + wrapper")
        rows.append([np.mean(m_entry["eval"]["per_family_cr_robust"][f]) for f in families])
    data = np.array(rows)

    fig, ax = plt.subplots(figsize=(14, max(2.5, len(models) * 1.2)))
    cmap = LinearSegmentedColormap.from_list("rg", ["#EF5350", "#FFCDD2", "#FFFFFF", "#C8E6C9", "#66BB6A"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(families)):
            val = data[i, j]
            best = data[:, j].max()
            weight = "bold" if val == best else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, fontweight=weight)
    ax.set_title(f"Per-Family CR at beta={best_beta:.2f}: DP Oracle vs Standard vs Masked")
    fig.colorbar(im, ax=ax, shrink=0.8, label="CR (red=bad, green=good)")
    fig.tight_layout()
    _savefig(fig, out_dir / f"robust_per_family_beta{best_beta:.2f}.png")

    print(f"\nDone. {len(list(out_dir.glob('robust_*.png')))} plots.")


if __name__ == "__main__":
    main()
