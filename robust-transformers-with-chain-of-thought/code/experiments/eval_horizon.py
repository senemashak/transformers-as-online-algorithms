"""
Horizon generalization experiment for optimal stopping.

Evaluates trained models at test horizons spanning interpolation and extrapolation.
No retraining — just inference on saved checkpoints.

Models evaluated:
  - value+action (best from sweep_weights, no wrapper)
  - equal_third (standard baseline, with wrapper at each beta)
  - 7 masked models (each with wrapper at its training beta)
  - DP oracle (upper bound)

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_horizon.py --device cuda
    CUDA_VISIBLE_DEVICES=0 python eval_horizon.py --device cuda --horizons 5 10 20 50 100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core.model import OnlineDecisionTransformer
from core.sampling import sample_stopping_batch, STOPPING_SAMPLERS
from core.deployment import compare_stopping, find_lambdas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--n_eval_seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[5, 10, 15, 20, 30, 40, 50, 75, 100])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    p.add_argument("--value_action_path", type=str,
                   default="../results/sweep/sweep_20260407_013555/models/stopping_value+action.pt")
    p.add_argument("--equal_third_path", type=str,
                   default="../results/sweep/sweep_20260407_013555/models/stopping_equal_third.pt")
    p.add_argument("--masked_dirs", type=str, nargs="+",
                   default=["../results/robust_stopping_p1",
                            "../results/robust_stopping_p2",
                            "../results/robust_stopping_p3"])
    p.add_argument("--out_dir", type=str, default="../results/eval_horizon")
    return p.parse_args()


def _load_model(path, M, device):
    model = OnlineDecisionTransformer(M=M)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _eval_at_horizon(model, n, M, n_eval, eval_seeds, device, betas=None):
    """Evaluate one model at one horizon. Returns CR for learned + each beta."""
    if betas is None:
        betas = []

    result = {"cr_learned": [], "cr_dp": []}
    for beta in betas:
        result[f"cr_robust_{beta:.2f}"] = []

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        # Reduce instances for large horizons to avoid OOM (O(n^2) chain)
        actual_n_eval = min(n_eval, max(50, 5000 // n))
        insts = sample_stopping_batch(actual_n_eval, n, M, rng=rng)
        betas_arg = [b for b in betas if b > 0]
        res = compare_stopping(insts, model, betas=betas_arg, r_fractions=[], device=device,
                               batch_size=max(1, min(512, 50000 // (n * n))))

        result["cr_learned"].append(res["learned"]["cr"])
        result["cr_dp"].append(res["dp"]["cr"])

        for beta in betas:
            key = f"robust β={beta:.2f}" if beta > 0 else "learned"
            result[f"cr_robust_{beta:.2f}"].append(res[key]["cr"])

    return result


def _eval_dp_only(n, M, n_eval, eval_seeds):
    """Evaluate DP oracle at one horizon (no model needed)."""
    crs = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        actual_n_eval = min(n_eval, max(50, 5000 // n))
        insts = sample_stopping_batch(actual_n_eval, n, M, rng=rng)
        res = compare_stopping(insts, None, betas=[], r_fractions=[])
        crs.append(res["dp"]["cr"])
    return crs


def main():
    args = parse_args()
    eval_seeds = [args.seed + 50000 + i for i in range(args.n_eval_seeds)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Horizon Generalization — Optimal Stopping")
    print(f"  Horizons: {args.horizons}")
    print(f"  Training range: [20, 50]")
    print(f"  {args.n_eval_seeds} eval seeds x {args.n_eval} instances each")
    print(f"{'='*60}\n")

    # Load models
    print("Loading models...")
    value_action = _load_model(args.value_action_path, args.M, args.device)
    print(f"  value+action: {args.value_action_path}")

    equal_third = _load_model(args.equal_third_path, args.M, args.device)
    print(f"  equal_third: {args.equal_third_path}")

    masked_models = {}
    for d in args.masked_dirs:
        for run_dir in Path(d).glob("run_*"):
            for pt in (run_dir / "models").glob("*.pt"):
                beta_str = pt.stem.split("beta")[1]
                beta = float(beta_str.replace("p", "."))
                masked_models[beta] = _load_model(str(pt), args.M, args.device)
                print(f"  masked beta={beta:.2f}: {pt}")

    output = {
        "meta": {
            "horizons": args.horizons,
            "betas": args.betas,
            "train_range": [20, 50],
            "n_eval": args.n_eval,
            "n_eval_seeds": args.n_eval_seeds,
            "args": vars(args),
        },
        "dp": {},
        "value_action": {},
        "equal_third": {},
        "masked": {},
    }

    t0 = time.time()

    for hi, n in enumerate(args.horizons):
        elapsed = time.time() - t0
        eta = elapsed / max(hi, 1) * (len(args.horizons) - hi) if hi > 0 else 0

        print(f"\n{'='*60}")
        in_range = "IN" if 20 <= n <= 50 else "OUT"
        print(f"  [{hi+1}/{len(args.horizons)}] n={n} ({in_range} training range) [{elapsed/60:.0f}m elapsed, ~{eta/60:.0f}m remaining]")
        print(f"{'='*60}")

        # DP oracle (no model)
        print(f"  DP oracle...")
        dp_crs = _eval_dp_only(n, args.M, args.n_eval, eval_seeds)
        output["dp"][str(n)] = dp_crs
        print(f"    CR={np.mean(dp_crs):.4f} ± {np.std(dp_crs):.4f}")

        # value+action (no wrapper)
        print(f"  value+action (no wrapper)...")
        va_res = _eval_at_horizon(value_action, n, args.M, args.n_eval, eval_seeds, args.device)
        output["value_action"][str(n)] = va_res
        print(f"    CR={np.mean(va_res['cr_learned']):.4f} ± {np.std(va_res['cr_learned']):.4f}")

        # equal_third with wrapper at each beta
        print(f"  equal_third (with wrappers)...")
        et_res = _eval_at_horizon(equal_third, n, args.M, args.n_eval, eval_seeds,
                                  args.device, betas=args.betas)
        output["equal_third"][str(n)] = et_res
        best_beta = max(args.betas, key=lambda b: np.mean(et_res[f"cr_robust_{b:.2f}"]))
        print(f"    no wrapper: CR={np.mean(et_res['cr_learned']):.4f}")
        print(f"    best wrapper beta={best_beta:.2f}: CR={np.mean(et_res[f'cr_robust_{best_beta:.2f}']):.4f}")

        # Masked models (each with its own beta wrapper)
        output["masked"][str(n)] = {}
        for beta in sorted(masked_models.keys()):
            print(f"  masked beta={beta:.2f} (with wrapper)...")
            m_res = _eval_at_horizon(masked_models[beta], n, args.M, args.n_eval, eval_seeds,
                                     args.device, betas=[beta])
            output["masked"][str(n)][f"{beta:.2f}"] = m_res
            print(f"    CR={np.mean(m_res[f'cr_robust_{beta:.2f}']):.4f}")

        # Incremental save
        with open(out_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, default=float)

    total = time.time() - t0
    print(f"\nDone in {total/60:.0f}m. Results → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
