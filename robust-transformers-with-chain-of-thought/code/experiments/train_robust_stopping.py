"""
Train robust-masked models for optimal stopping.

For each robustness parameter beta, trains with equal loss weights (1/3, 1/3, 1/3)
but with supervision masked to only the positions the robust wrapper uses:
  - L_value: only at middle-phase decision steps
  - L_action: only at middle-phase decision steps
  - L_chain: only sub-chains for middle-phase decision steps
    (within those sub-chains, ALL intermediate V(t,k) supervised)

The existing equal_third model (trained on all positions) serves as the
standard baseline — no retraining needed for that.

Saves per beta:
  - Best model checkpoint (by val loss across 3 seeds)
  - Training curves for all 3 seeds
  - Attention maps for best model (5 diverse instances)

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_robust_stopping.py --device cuda
    CUDA_VISIBLE_DEVICES=1 python train_robust_stopping.py --device cuda --betas 0.10 0.20 0.30
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.sampling import sample_stopping_batch, STOPPING_SAMPLERS
from core.dataset import StoppingDataset, make_dataloader
from core.model import OnlineDecisionTransformer
from core.train import train as run_train
from core.deployment import find_lambdas
from core.attention import extract_attention, sample_diverse_instances


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--n_max", type=int, default=50)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--max_n", type=int, default=501)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train_seeds", type=int, default=3)
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    p.add_argument("--out_dir", type=str, default="../results/robust_stopping")
    return p.parse_args()


def _make_model(args):
    return OnlineDecisionTransformer(
        M=args.M, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n)


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{stamp}"
    model_dir = out_dir / "models"
    attn_dir = out_dir / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    attn_dir.mkdir(exist_ok=True)

    # Fixed loss weights: equal 1/3
    w_v, w_a, w_c = 1/3, 1/3, 1/3

    print(f"{'='*60}")
    print(f"  Robust-Masked Training for Optimal Stopping")
    print(f"  Loss weights: ({w_v:.2f}, {w_a:.2f}, {w_c:.2f})")
    print(f"  Betas: {args.betas}")
    print(f"  {args.n_train_seeds} seeds per beta, epochs={args.epochs}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    output = {
        "meta": {
            "betas": args.betas,
            "loss_weights": {"w_value": w_v, "w_action": w_a, "w_chain": w_c},
            "n_train_seeds": args.n_train_seeds,
            "args": vars(args),
            "timestamp": stamp,
        },
        "models": [],
    }

    sweep_t0 = time.time()

    for bi, beta in enumerate(args.betas):
        lam1, lam2 = find_lambdas(beta)
        elapsed = time.time() - sweep_t0
        remaining = elapsed / max(bi, 1) * (len(args.betas) - bi)

        print(f"\n{'='*60}")
        print(f"  [{bi+1}/{len(args.betas)}] beta={beta:.2f}  (lam1={lam1:.3f}, lam2={lam2:.3f})")
        print(f"  Middle phase: {lam1:.1%} to {lam2:.1%} of horizon")
        print(f"  [{elapsed/3600:.1f}h elapsed, ~{remaining/3600:.1f}h remaining]")
        print(f"{'='*60}")

        entry = {
            "beta": beta,
            "lambda1": lam1,
            "lambda2": lam2,
            "seeds": [],
            "best_seed_idx": None,
            "best_model_path": None,
        }

        models = []
        ckpt_paths = []

        for ts in range(args.n_train_seeds):
            train_seed = args.seed + ts * 1000
            print(f"\n  seed {ts+1}/{args.n_train_seeds}")

            train_ds = StoppingDataset(args.train_size, args.n, args.M, seed=train_seed,
                                       n_min=args.n_min, n_max=args.n_max)
            val_ds = StoppingDataset(args.val_size, args.n, args.M, seed=train_seed + 1,
                                     n_min=args.n_min, n_max=args.n_max)

            torch.manual_seed(train_seed)
            model = _make_model(args)
            ckpt = str(out_dir / f"_tmp_beta{beta:.2f}_s{ts}.pt")

            model, logs = run_train(
                model, make_dataloader(train_ds, args.batch_size),
                make_dataloader(val_ds, args.batch_size, shuffle=False),
                problem="stopping", n=args.n, M=args.M,
                lr=args.lr, epochs=args.epochs,
                w_value=w_v, w_action=w_a, w_chain=w_c,
                training_mode="teacher_forcing",
                robust_train=True, robust_beta=beta,
                device=args.device, checkpoint_path=ckpt)

            models.append(model)
            ckpt_paths.append(ckpt)

            entry["seeds"].append({
                "train_seed": train_seed,
                "training_curve": logs,
                "val_loss": min(log["val_total"] for log in logs),
            })

            # Save after every seed (crash-safe)
            # Replace or append this beta's entry
            if bi < len(output["models"]):
                output["models"][bi] = entry
            else:
                output["models"].append(entry)
            with open(out_dir / "results.json", "w") as f:
                json.dump(output, f, indent=2, default=float)
            print(f"    saved to results.json (beta={beta:.2f}, seed {ts+1}/{args.n_train_seeds})")

        # Select best by val loss
        val_losses = [s["val_loss"] for s in entry["seeds"]]
        best_idx = int(np.argmin(val_losses))
        entry["best_seed_idx"] = best_idx

        # Save best checkpoint
        safe = f"stopping_masked_beta{beta:.2f}".replace('.', 'p')
        final_path = model_dir / f"{safe}.pt"
        shutil.move(ckpt_paths[best_idx], final_path)
        entry["best_model_path"] = f"models/{safe}.pt"

        # Delete non-best
        for i, p in enumerate(ckpt_paths):
            if i != best_idx and Path(p).exists():
                Path(p).unlink()

        # Save attention maps for best model
        print(f"\n  Saving attention maps for best seed (s{best_idx})...")
        attn_instances = sample_diverse_instances(args.n, args.M, num_instances=5, seed=args.seed)
        attn_data = extract_attention(models[best_idx], attn_instances, args.M, device=args.device)
        attn_data["config"] = f"masked_beta{beta:.2f}"
        attn_data["beta"] = beta

        # Save raw attention data (plot later with attention.py)
        torch.save(attn_data, attn_dir / f"{safe}.pt")

        print(f"  best_seed=s{best_idx}  val_loss={val_losses[best_idx]:.6f}")

        # Final update for this beta (with best_seed_idx and model_path)
        if bi < len(output["models"]):
            output["models"][bi] = entry
        else:
            output["models"].append(entry)
        with open(out_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, default=float)
        print(f"  results.json updated ({bi+1}/{len(args.betas)} betas done)")

    total_time = time.time() - sweep_t0
    print(f"\n{'='*60}")
    print(f"  Done in {total_time/3600:.1f}h")
    print(f"  Results  → {out_dir / 'results.json'}")
    print(f"  Models   → {model_dir}/ ({len(list(model_dir.glob('*.pt')))} checkpoints)")
    print(f"  Attention→ {attn_dir}/ ({len(list(attn_dir.glob('*.pt')))} maps)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
