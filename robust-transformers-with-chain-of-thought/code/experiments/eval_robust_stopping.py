"""
Evaluate robust-masked models vs standard baseline for optimal stopping.

Loads:
  - 7 masked models (one per beta, trained with robust mask)
  - 1 standard baseline (equal_third, trained on all positions)

Evaluates each model:
  - WITHOUT wrapper (raw learned policy): stop at first t where x_t >= V(t)*M
  - WITH wrapper at each beta (Algorithm 1): early skip, middle use V(t), late take best

Variable horizons n ~ U[n_min, n_max], all 11 families including OOD.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_robust_stopping.py --device cuda
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
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--n_max", type=int, default=50)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--n_eval_seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--baseline_path", type=str,
                   default="../results/sweep/sweep_20260407_013555/models/stopping_equal_third.pt")
    p.add_argument("--masked_dirs", type=str, nargs="+",
                   default=["../results/robust_stopping_p1",
                            "../results/robust_stopping_p2",
                            "../results/robust_stopping_p3"])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.367])
    p.add_argument("--out_dir", type=str, default="../results/eval_robust_stopping")
    return p.parse_args()


def _load_model(path, M, device):
    model = OnlineDecisionTransformer(M=M)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _eval_at_beta(model, args, beta, eval_seeds):
    """Evaluate one model at one beta with variable horizons.

    Returns dict with CR and P(best) for both learned (no wrapper) and robust (with wrapper).
    Per-family breakdown on all 11 families.
    """
    M = args.M
    n_min, n_max = args.n_min, args.n_max
    use_wrapper = beta > 0
    rob_key = f"robust β={beta:.2f}" if use_wrapper else "learned"
    betas_arg = [beta] if use_wrapper else []

    result = {
        "cr_learned": [], "cr_robust": [],
        "pbest_learned": [], "pbest_robust": [],
        "per_family_cr_learned": {f: [] for f in STOPPING_SAMPLERS},
        "per_family_cr_robust": {f: [] for f in STOPPING_SAMPLERS},
        "baselines": {"dp_cr": []},
    }

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        horizons = rng.integers(n_min, n_max + 1, size=args.n_eval)

        # In-distribution
        lr, rr, mv, ib_l, ib_r, dpr = [], [], [], [], [], []
        for h in sorted(set(horizons)):
            c = int((horizons == h).sum())
            insts = sample_stopping_batch(c, int(h), M, rng=np.random.default_rng(seed + int(h)))
            res = compare_stopping(insts, model, betas=betas_arg, r_fractions=[], device=args.device)
            lr.append(res["learned"]["mean_reward"] * c)
            rr.append(res[rob_key]["mean_reward"] * c)
            mv.append(res["offline"]["mean_reward"] * c)
            ib_l.append(res["learned"]["prob_best"] * c)
            ib_r.append(res[rob_key]["prob_best"] * c)
            dpr.append(res["dp"]["mean_reward"] * c)

        t = len(horizons)
        mm = sum(mv) / t
        result["cr_learned"].append(sum(lr) / t / mm if mm > 0 else 0)
        result["cr_robust"].append(sum(rr) / t / mm if mm > 0 else 0)
        result["pbest_learned"].append(sum(ib_l) / t)
        result["pbest_robust"].append(sum(ib_r) / t)
        result["baselines"]["dp_cr"].append(sum(dpr) / t / mm if mm > 0 else 0)

        # Per-family
        for fam in STOPPING_SAMPLERS:
            frng = np.random.default_rng(seed + hash(fam) % 10000)
            nf = args.n_eval // len(STOPPING_SAMPLERS)
            fh = frng.integers(n_min, n_max + 1, size=nf)
            flr, frr, fmv = [], [], []
            for h in sorted(set(fh)):
                c = int((fh == h).sum())
                insts = sample_stopping_batch(c, int(h), M, dist_type=fam,
                    rng=np.random.default_rng(seed + hash(fam) % 10000 + int(h)))
                res = compare_stopping(insts, model, betas=betas_arg, r_fractions=[], device=args.device)
                flr.append(res["learned"]["mean_reward"] * c)
                frr.append(res[rob_key]["mean_reward"] * c)
                fmv.append(res["offline"]["mean_reward"] * c)
            ft = len(fh)
            fmm = sum(fmv) / ft
            result["per_family_cr_learned"][fam].append(sum(flr) / ft / fmm if fmm > 0 else 0)
            result["per_family_cr_robust"][fam].append(sum(frr) / ft / fmm if fmm > 0 else 0)

    return result


def main():
    args = parse_args()
    eval_seeds = [args.seed + 50000 + i for i in range(args.n_eval_seeds)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline model
    print(f"Loading baseline: {args.baseline_path}")
    baseline_model = _load_model(args.baseline_path, args.M, args.device)

    # Load masked models
    masked_models = {}
    for d in args.masked_dirs:
        for run_dir in Path(d).glob("run_*"):
            for pt in (run_dir / "models").glob("*.pt"):
                # Extract beta from filename: stopping_masked_beta0p20.pt -> 0.20
                name = pt.stem  # stopping_masked_beta0p20
                beta_str = name.split("beta")[1]  # 0p20
                beta = float(beta_str.replace("p", "."))
                masked_models[beta] = str(pt)
                print(f"  Found masked model: beta={beta:.2f} -> {pt}")

    print(f"\nEvaluating at betas: {args.betas}")
    print(f"  {args.n_eval_seeds} eval seeds x {args.n_eval} instances each")
    print(f"  Horizons: n ~ U[{args.n_min}, {args.n_max}]")
    print(f"  Families: {len(STOPPING_SAMPLERS)} (including hard_instance)")

    output = {
        "meta": {
            "betas": args.betas,
            "n_eval_seeds": args.n_eval_seeds,
            "n_eval": args.n_eval,
            "baseline_path": args.baseline_path,
            "masked_models": masked_models,
            "args": vars(args),
        },
        "baseline": [],   # one entry per beta
        "masked": [],      # one entry per beta (only for betas with masked models)
    }

    t0 = time.time()

    for bi, beta in enumerate(args.betas):
        elapsed = time.time() - t0
        if bi > 0:
            eta = elapsed / bi * (len(args.betas) - bi)
        else:
            eta = 0

        if beta > 0 and beta < 0.367:
            lam1, lam2 = find_lambdas(beta)
            phase = f"lam1={lam1:.3f}, lam2={lam2:.3f}"
        elif beta == 0:
            phase = "no wrapper"
        else:
            phase = "near 1/e"

        print(f"\n{'='*60}")
        print(f"  [{bi+1}/{len(args.betas)}] beta={beta:.3f} ({phase}) [{elapsed/60:.0f}m elapsed, ~{eta/60:.0f}m remaining]")
        print(f"{'='*60}")

        # Evaluate baseline (standard model) at this beta
        print(f"  Baseline (equal_third)...")
        base_result = _eval_at_beta(baseline_model, args, beta, eval_seeds)
        base_entry = {
            "beta": beta,
            "mode": "standard",
            "eval": base_result,
        }
        cr = float(np.mean(base_result["cr_robust"]))
        print(f"    CR(robust)={cr:.4f}")
        output["baseline"].append(base_entry)

        # Evaluate masked model at this beta (if we have one)
        if beta in masked_models:
            print(f"  Masked (beta={beta:.2f})...")
            masked_model = _load_model(masked_models[beta], args.M, args.device)
            mask_result = _eval_at_beta(masked_model, args, beta, eval_seeds)
            mask_entry = {
                "beta": beta,
                "mode": "robust_masked",
                "eval": mask_result,
            }
            cr_m = float(np.mean(mask_result["cr_robust"]))
            print(f"    CR(robust)={cr_m:.4f}")
            winner = "masked" if cr_m > cr else "standard"
            print(f"    Winner: {winner} ({cr_m:.4f} vs {cr:.4f})")
            output["masked"].append(mask_entry)

        # Incremental save
        with open(out_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, default=float)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Standard vs Masked (robust CR)")
    print(f"{'='*60}")
    print(f"{'beta':<8} {'Standard':>12} {'Masked':>12} {'Winner':>10}")
    print(f"{'─'*44}")
    for beta in args.betas:
        base = next((e for e in output["baseline"] if e["beta"] == beta), None)
        mask = next((e for e in output["masked"] if e["beta"] == beta), None)
        s_cr = float(np.mean(base["eval"]["cr_robust"])) if base else 0
        if mask:
            m_cr = float(np.mean(mask["eval"]["cr_robust"]))
            winner = "masked" if m_cr > s_cr else "standard"
            print(f"{beta:<8.3f} {s_cr:>12.4f} {m_cr:>12.4f} {winner:>10}")
        else:
            print(f"{beta:<8.3f} {s_cr:>12.4f} {'—':>12} {'—':>10}")

    total = time.time() - t0
    print(f"\nDone in {total/60:.0f}m. Results → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
