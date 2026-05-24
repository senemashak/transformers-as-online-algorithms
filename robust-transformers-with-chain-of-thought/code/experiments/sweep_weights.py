"""
Sweep loss weight configurations for BOTH optimal stopping and ski rental.

For each (w_value, w_action, w_chain) configuration:
  - Trains one model per task × per training seed
  - Evaluates EVERY model across multiple eval seeds
  - Saves EVERY model checkpoint
  - Reports CR + P(best) for stopping, CR + additive loss for ski

Output structure (designed for decoupled plotting):
    results/sweep/sweep_<timestamp>/
        results.json                        — all metrics for all seeds (for plots)
        models/
            stopping_<config>.pt            — best model only per config
            ski_<config>.pt
        attention/
            stopping_<config>.pt            — attention maps for best model
            ski_<config>.pt

results.json structure per config per problem:
    {
        "config": "equal_third",
        "problem": "stopping",
        "w_value": 0.33, "w_action": 0.33, "w_chain": 0.33,
        "seeds": [
            {
                "train_seed": 42,
                "training_curve": [...],
                "val_loss": 0.123,
                "model_path": "models/stopping_equal_third_s0.pt",
                "eval": {
                    "in_dist_cr": [...],          # one per eval seed
                    "in_dist_prob_best": [...],
                    "per_family_cr": {"geometric": [...], ...},
                    "per_family_prob_best": {"geometric": [...], ...},
                }
            },
            ...
        ],
        "best_seed_idx": 1,   # index into seeds[] — best by CR or additive loss
    }

Usage:
    python sweep_weights.py
    python sweep_weights.py --epochs 30 --device cuda
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.sampling import (
    sample_stopping_batch, sample_ski_batch,
    STOPPING_SAMPLERS, SKI_SAMPLERS,
)
from core.dataset import StoppingDataset, SkiRentalDataset, make_dataloader
from core.model import OnlineDecisionTransformer
from core.train import train as run_train
from core.deployment import compare_stopping, compare_ski


# ═══════════════════════════════════════════════════════════════════════════
# Weight configurations to sweep
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # name,                    w_value, w_action, w_chain, training_mode
    # --- Single-loss ablations ---
    ("value_only",             1.0,     0.0,      0.0,    "teacher_forcing"),
    ("action_only",            0.0,     1.0,      0.0,    "teacher_forcing"),
    ("chain_only",             0.0,     0.0,      1.0,    "teacher_forcing"),

    # --- Pairwise ---
    ("value+action",           1.0,     0.5,      0.0,    "teacher_forcing"),
    ("value+chain",            0.5,     0.0,      0.5,    "teacher_forcing"),
    ("action+chain",           0.0,     0.5,      0.5,    "teacher_forcing"),

    # --- Equal weighting ---
    ("equal_third",            1/3,     1/3,      1/3,    "teacher_forcing"),

    # --- Variations around 1/3 ---
    ("emph_value",             0.5,     0.25,     0.25,   "teacher_forcing"),
    ("emph_action",            0.25,    0.5,      0.25,   "teacher_forcing"),
    ("emph_chain",             0.25,    0.25,     0.5,    "teacher_forcing"),

    # --- All three, heavier ---
    ("all_1_0.5_1",           1.0,     0.5,      1.0,    "teacher_forcing"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--B", type=float, default=10.0)
    p.add_argument("--r", type=float, default=1.0)
    p.add_argument("--B_min", type=int, default=10)
    p.add_argument("--B_max", type=int, default=100)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--n_max", type=int, default=50)
    p.add_argument("--d_model", type=int, default=None, help="Hidden dim (default: auto from M)")
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2, help="Transformer layers (2 = minimum for Prop 1)")
    p.add_argument("--d_ff", type=int, default=None, help="FFN width (default: auto from M, B, n)")
    p.add_argument("--max_n", type=int, default=501)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train_seeds", type=int, default=3)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--n_eval_seeds", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="results/sweep")
    p.add_argument("--only", type=str, default=None, choices=["stopping", "ski"],
                   help="Run only one problem (for parallel execution on separate GPUs)")
    return p.parse_args()


def _make_model(args):
    return OnlineDecisionTransformer(
        M=args.M, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n)


def _eval_stopping(model, args, eval_seeds):
    """Evaluate a stopping model across eval seeds.

    Instances are sampled at variable horizons n ~ U[n_min, n_max], matching
    the training distribution. Each horizon is evaluated as a separate batch
    (since compare_stopping requires fixed n per batch), then results are
    pooled across horizons.

    Returns dict with per-eval-seed arrays of CR and P(best),
    both in-dist and per-family (all 11 families).
    """
    M = args.M
    n_min, n_max = args.n_min, args.n_max
    result = {
        "in_dist_cr": [],
        "in_dist_prob_best": [],
        "per_family_cr": {fam: [] for fam in STOPPING_SAMPLERS},
        "per_family_prob_best": {fam: [] for fam in STOPPING_SAMPLERS},
        "baselines": {
            "dp_cr": [], "dp_prob_best": [],
            "offline_cr": [], "offline_prob_best": [],
        },
    }

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        # Sample horizons, then group instances by horizon
        horizons = rng.integers(n_min, n_max + 1, size=args.n_eval)
        all_learned_rewards, all_max_vals, all_is_best = [], [], []
        all_dp_rewards, all_dp_is_best = [], []

        for h in sorted(set(horizons)):
            count = int((horizons == h).sum())
            h_rng = np.random.default_rng(seed + h)
            insts = sample_stopping_batch(count, h, M, rng=h_rng)
            res = compare_stopping(insts, model, betas=[], r_fractions=[], device=args.device)
            all_learned_rewards.append(res["learned"]["mean_reward"] * count)
            all_max_vals.append(res["offline"]["mean_reward"] * count)
            all_is_best.append(res["learned"]["prob_best"] * count)
            all_dp_rewards.append(res["dp"]["mean_reward"] * count)
            all_dp_is_best.append(res["dp"]["prob_best"] * count)

        total = len(horizons)
        mean_learned = sum(all_learned_rewards) / total
        mean_max = sum(all_max_vals) / total
        result["in_dist_cr"].append(mean_learned / mean_max if mean_max > 0 else 0)
        result["in_dist_prob_best"].append(sum(all_is_best) / total)
        result["baselines"]["dp_cr"].append(sum(all_dp_rewards) / total / mean_max if mean_max > 0 else 0)
        result["baselines"]["dp_prob_best"].append(sum(all_dp_is_best) / total)
        result["baselines"]["offline_cr"].append(1.0)
        result["baselines"]["offline_prob_best"].append(1.0)

        # Per-family with variable horizons
        for fam in STOPPING_SAMPLERS:
            fam_rng = np.random.default_rng(seed + hash(fam) % 10000)
            n_fam = args.n_eval // len(STOPPING_SAMPLERS)
            fam_horizons = fam_rng.integers(n_min, n_max + 1, size=n_fam)
            fam_rewards, fam_maxes, fam_best = [], [], []
            for h in sorted(set(fam_horizons)):
                cnt = int((fam_horizons == h).sum())
                h_rng = np.random.default_rng(seed + hash(fam) % 10000 + h)
                insts = sample_stopping_batch(cnt, h, M, dist_type=fam, rng=h_rng)
                res = compare_stopping(insts, model, betas=[], r_fractions=[],
                                       device=args.device)
                fam_rewards.append(res["learned"]["mean_reward"] * cnt)
                fam_maxes.append(res["offline"]["mean_reward"] * cnt)
                fam_best.append(res["learned"]["prob_best"] * cnt)
            fam_total = len(fam_horizons)
            fam_mr = sum(fam_rewards) / fam_total
            fam_mm = sum(fam_maxes) / fam_total
            result["per_family_cr"][fam].append(fam_mr / fam_mm if fam_mm > 0 else 0)
            result["per_family_prob_best"][fam].append(sum(fam_best) / fam_total)

    return result


def _eval_ski(model, args, eval_seeds):
    """Evaluate a ski model across eval seeds.

    Instances are sampled at variable horizons n ~ U[n_min, n_max] and
    variable B ~ U{B_min, ..., B_max}, matching the training distribution.

    Returns dict with per-eval-seed arrays of CR and additive loss,
    both in-dist and per-family (all 10 families).
    """
    n_min, n_max = args.n_min, args.n_max
    B_default, r = args.B, args.r
    result = {
        "in_dist_cr": [],
        "in_dist_additive_loss": [],
        "per_family_cr": {fam: [] for fam in SKI_SAMPLERS},
        "per_family_additive_loss": {fam: [] for fam in SKI_SAMPLERS},
        "baselines": {
            "dp_cr": [], "dp_loss": [],
            "deterministic_cr": [], "deterministic_loss": [],
        },
    }

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        # Sample variable horizons, sample one B per horizon group
        # (compare_ski requires shared n and B per batch)
        horizons = rng.integers(n_min, n_max + 1, size=args.n_eval)
        all_learned_costs, all_opt_costs, all_det_costs, all_dp_costs = [], [], [], []

        for h in sorted(set(horizons)):
            count = int((horizons == h).sum())
            B_val = float(rng.integers(args.B_min, args.B_max + 1))
            h_rng = np.random.default_rng(seed + h)
            insts = sample_ski_batch(count, h, B_val, r, rng=h_rng)
            res = compare_ski(insts, model, lambdas=[], device=args.device)
            all_learned_costs.append(res["learned"]["mean_cost"] * count)
            all_opt_costs.append(res["learned"]["mean_opt_cost"] * count)
            all_dp_costs.append(res["dp"]["mean_cost"] * count)
            all_det_costs.append(res["deterministic"]["mean_cost"] * count)

        total = len(horizons)
        mean_learned = sum(all_learned_costs) / total
        mean_opt = sum(all_opt_costs) / total
        mean_dp = sum(all_dp_costs) / total
        mean_det = sum(all_det_costs) / total

        result["in_dist_cr"].append(mean_opt / mean_learned if mean_learned > 0 else 1.0)
        result["in_dist_additive_loss"].append(mean_learned - mean_opt)
        result["baselines"]["dp_cr"].append(mean_opt / mean_dp if mean_dp > 0 else 1.0)
        result["baselines"]["dp_loss"].append(mean_dp - mean_opt)
        result["baselines"]["deterministic_cr"].append(mean_opt / mean_det if mean_det > 0 else 1.0)
        result["baselines"]["deterministic_loss"].append(mean_det - mean_opt)

        # Per-family with variable horizons and B
        for fam in SKI_SAMPLERS:
            fam_rng = np.random.default_rng(seed + hash(fam) % 10000)
            n_fam = args.n_eval // len(SKI_SAMPLERS)
            fam_horizons = fam_rng.integers(n_min, n_max + 1, size=n_fam)
            fam_learned, fam_opt = [], []
            for h in sorted(set(fam_horizons)):
                cnt = int((fam_horizons == h).sum())
                B_val = float(fam_rng.integers(args.B_min, args.B_max + 1))
                h_rng = np.random.default_rng(seed + hash(fam) % 10000 + h)
                insts = sample_ski_batch(cnt, h, B_val, r, dist_type=fam, rng=h_rng)
                res = compare_ski(insts, model, lambdas=[], device=args.device)
                fam_learned.append(res["learned"]["mean_cost"] * cnt)
                fam_opt.append(res["learned"]["mean_opt_cost"] * cnt)
            fam_total = len(fam_horizons)
            fam_ml = sum(fam_learned) / fam_total
            fam_mo = sum(fam_opt) / fam_total
            result["per_family_cr"][fam].append(fam_mo / fam_ml if fam_ml > 0 else 1.0)
            result["per_family_additive_loss"][fam].append(fam_ml - fam_mo)

    return result


def _save_attention_maps(model, args, problem, out_dir, config_name):
    """Save attention maps from the best model on a representative instance.

    Saves a .pt file containing attention weights, position labels, and the
    instance data — everything needed for mechanistic interpretation later.
    """
    from deployment import get_attention_maps
    from train import _build_chain2d_targets

    model.eval()
    n = args.n  # use default horizon for interpretability
    M = args.M
    rng = np.random.default_rng(args.seed + 77777)  # fixed seed for reproducibility

    if problem == "stopping":
        insts = sample_stopping_batch(1, n, M, rng=rng)
        x = torch.tensor(np.stack([inst.values for inst in insts]), dtype=torch.long)
        # Build chain targets from DP labels
        from dp import stopping_labels
        lbl = stopping_labels(insts[0].pmf, insts[0].values)
        V_target = torch.tensor(lbl["C"], dtype=torch.float32).unsqueeze(0) / M
    else:
        insts = sample_ski_batch(1, n, args.B, args.r, rng=rng)
        x = torch.ones(1, n, dtype=torch.long)
        from dp import ski_labels
        lbl = ski_labels(insts[0].pmf_T, insts[0].n, insts[0].B, insts[0].r)
        V_target = torch.tensor(lbl["J"], dtype=torch.float32).unsqueeze(0) / insts[0].B

    t_idx, j_idx, _ = model._get_chain2d_info(n, x.device)
    chain2d_tgt = _build_chain2d_targets(V_target, j_idx, n)

    B_cost = int(insts[0].B) if problem == "ski" else None
    r_cost = int(insts[0].r) if problem == "ski" else None

    attn, info = get_attention_maps(model, x, n_horizon=n,
                                    task_id=0 if problem == "stopping" else 1,
                                    chain2d_targets=chain2d_tgt,
                                    B_cost=B_cost, r_cost=r_cost,
                                    device=args.device)

    safe = config_name.replace('/', '_').replace(' ', '_')
    attn_path = out_dir / "attention" / f"{problem}_{safe}.pt"
    attn_path.parent.mkdir(exist_ok=True)
    torch.save({
        "attn_weights": attn,         # list of (1, n_heads, L, L) per layer
        "info": info,                  # position labels, t_idx, j_idx, decision_pos
        "x": x.cpu(),                 # input observations
        "V_target": V_target.cpu(),   # ground truth normalized values
        "problem": problem,
        "config": config_name,
        "n": n,
        "M": M,
    }, attn_path)
    print(f"      attention maps → {attn_path.name}")


def _run_config(args, out_dir, model_dir, problem, name, w_v, w_a, w_c, training_mode, eval_seeds):
    """Train + eval all seeds for one config/problem.

    Saves only the best model checkpoint (not all seeds).
    Saves attention maps for the best model.
    Keeps all per-seed eval metrics in the returned entry dict.
    """
    safe_name = name.replace('/', '_').replace(' ', '_')

    entry = {
        "config": name,
        "problem": problem,
        "w_value": w_v, "w_action": w_a, "w_chain": w_c,
        "training_mode": training_mode,
        "seeds": [],
        "best_seed_idx": None,
    }

    models = []
    ckpt_paths = []

    for ts in range(args.n_train_seeds):
        train_seed = args.seed + ts * 1000
        print(f"    [{problem}] train seed {ts+1}/{args.n_train_seeds}")

        # Train
        if problem == "stopping":
            train_ds = StoppingDataset(args.train_size, args.n, args.M, seed=train_seed,
                                       n_min=args.n_min, n_max=args.n_max)
            val_ds = StoppingDataset(args.val_size, args.n, args.M, seed=train_seed + 1,
                                     n_min=args.n_min, n_max=args.n_max)
        else:
            train_ds = SkiRentalDataset(args.train_size, args.n, args.B, args.r, seed=train_seed,
                                        n_min=args.n_min, n_max=args.n_max,
                                        B_min=args.B_min, B_max=args.B_max)
            val_ds = SkiRentalDataset(args.val_size, args.n, args.B, args.r, seed=train_seed + 1,
                                      n_min=args.n_min, n_max=args.n_max,
                                      B_min=args.B_min, B_max=args.B_max)

        torch.manual_seed(train_seed)
        model = _make_model(args)

        ckpt_path = str(out_dir / f"_tmp_{problem}_s{ts}.pt")
        model, logs = run_train(
            model, make_dataloader(train_ds, args.batch_size),
            make_dataloader(val_ds, args.batch_size, shuffle=False),
            problem=problem, n=args.n, M=args.M, B=args.B,
            lr=args.lr, epochs=args.epochs,
            w_value=w_v, w_action=w_a, w_chain=w_c,
            training_mode=training_mode,
            device=args.device, checkpoint_path=ckpt_path)

        models.append(model)
        ckpt_paths.append(ckpt_path)

        # Evaluate
        print(f"      evaluating across {args.n_eval_seeds} eval seeds...")
        if problem == "stopping":
            eval_result = _eval_stopping(model, args, eval_seeds)
        else:
            eval_result = _eval_ski(model, args, eval_seeds)

        cr_mean = float(np.mean(eval_result["in_dist_cr"]))
        if problem == "stopping":
            pb_mean = float(np.mean(eval_result["in_dist_prob_best"]))
            print(f"      CR={cr_mean:.4f}  P(best)={pb_mean:.4f}")
        else:
            al_mean = float(np.mean(eval_result["in_dist_additive_loss"]))
            print(f"      CR={cr_mean:.4f}  Δloss={al_mean:.4f}")

        entry["seeds"].append({
            "train_seed": train_seed,
            "training_curve": logs,
            "val_loss": min(log["val_total"] for log in logs),
            "eval": eval_result,
        })

    # Select best seed by lowest validation loss (cheap, computed during training,
    # independent of eval horizon/procedure)
    val_losses = [s["val_loss"] for s in entry["seeds"]]
    entry["best_seed_idx"] = int(np.argmin(val_losses))
    best_idx = entry["best_seed_idx"]

    # Save only the best checkpoint
    import shutil
    best_model_name = f"{problem}_{safe_name}.pt"
    final_path = model_dir / best_model_name
    shutil.move(ckpt_paths[best_idx], final_path)
    entry["best_model_path"] = f"models/{best_model_name}"

    # Delete non-best temp checkpoints
    for i, p in enumerate(ckpt_paths):
        if i != best_idx and Path(p).exists():
            Path(p).unlink()

    # Save attention maps for best model
    _save_attention_maps(models[best_idx], args, problem, out_dir, name)

    best = entry["seeds"][best_idx]
    print(f"    [{problem}] best seed: s{best_idx} "
          f"(CR={np.mean(best['eval']['in_dist_cr']):.4f})")

    return entry


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"sweep_{stamp}"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    eval_seeds = [args.seed + 50000 + i for i in range(args.n_eval_seeds)]

    problems = [args.only] if args.only else ["stopping", "ski"]
    problems_str = " + ".join(problems)

    print(f"{'='*60}")
    print(f"  Loss Weight Sweep — {problems_str}")
    print(f"  {len(CONFIGS)} configs × {args.n_train_seeds} train seeds × {args.n_eval_seeds} eval seeds")
    print(f"  Best model + attention maps saved per config")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    output = {
        "meta": {
            "configs": [[name, w_v, w_a, w_c, tm] for name, w_v, w_a, w_c, tm in CONFIGS],
            "problems": problems,
            "n_train_seeds": args.n_train_seeds,
            "n_eval_seeds": args.n_eval_seeds,
            "eval_seeds": eval_seeds,
            "stopping_families_tested": list(STOPPING_SAMPLERS.keys()),
            "ski_families_tested": list(SKI_SAMPLERS.keys()),
            "args": vars(args),
            "timestamp": stamp,
        },
        "stopping": [],
        "ski": [],
    }

    import time
    sweep_t0 = time.time()
    total_jobs = len(CONFIGS) * len(problems)
    job = 0

    for i, (name, w_v, w_a, w_c, t_mode) in enumerate(CONFIGS):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(CONFIGS)}] {name}  (w_v={w_v:.3f}, w_a={w_a:.3f}, w_c={w_c:.3f}, {t_mode})")
        print(f"{'─'*60}")

        for problem in problems:
            job += 1
            elapsed = time.time() - sweep_t0
            if job > 1:
                eta = elapsed / (job - 1) * (total_jobs - job + 1)
                print(f"\n  >>> Job {job}/{total_jobs}  [{elapsed/3600:.1f}h elapsed, ~{eta/3600:.1f}h remaining]")
            output[problem].append(
                _run_config(args, out_dir, model_dir, problem, name, w_v, w_a, w_c, t_mode, eval_seeds))

            # Save incrementally after each config — results available for plotting immediately
            json_path = out_dir / "results.json"
            with open(json_path, "w") as f:
                json.dump(output, f, indent=2, default=float)
            print(f"      results.json updated ({len(output[problem])} {problem} configs saved)")

    # Print summary
    for problem in problems:
        entries = output[problem]
        print(f"\n{'='*72}")
        print(f"  {problem.upper()} — sorted by best-seed in-dist CR (higher = better)")
        print(f"{'='*72}")

        if problem == "stopping":
            print(f"{'Config':<24} {'mode':<4} {'w_v':>4} {'w_a':>4} {'w_c':>4} {'CR':>8} {'±':>6} {'P(best)':>8} {'±':>6}")
        else:
            print(f"{'Config':<24} {'mode':<4} {'w_v':>4} {'w_a':>4} {'w_c':>4} {'CR':>8} {'±':>6} {'Δloss':>8} {'±':>6}")
        print(f"{'─'*80}")

        ranked = sorted(entries,
                        key=lambda e: np.mean(e["seeds"][e["best_seed_idx"]]["eval"]["in_dist_cr"]),
                        reverse=True)
        for e in ranked:
            best = e["seeds"][e["best_seed_idx"]]["eval"]
            cr = best["in_dist_cr"]
            tm = "TF" if e["training_mode"] == "teacher_forcing" else "AR"
            if problem == "stopping":
                sec = best["in_dist_prob_best"]
            else:
                sec = best["in_dist_additive_loss"]
            print(f"{e['config']:<24} {tm:<4} {e['w_value']:>4.2f} {e['w_action']:>4.2f} "
                  f"{e['w_chain']:>4.2f} {np.mean(cr):>8.4f} {np.std(cr):>6.4f} "
                  f"{np.mean(sec):>8.4f} {np.std(sec):>6.4f}")

    print(f"\nResults   → {json_path}")
    print(f"Models    → {model_dir}/")
    n_models = len(list(model_dir.glob("*.pt")))
    print(f"  ({n_models} best-model checkpoints)")
    attn_dir = out_dir / "attention"
    if attn_dir.exists():
        n_attn = len(list(attn_dir.glob("*.pt")))
        print(f"Attention → {attn_dir}/")
        print(f"  ({n_attn} attention maps)")


if __name__ == "__main__":
    main()
