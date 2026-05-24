"""
Evaluation: policies, metrics, and comparison for both problems.

Optimal stopping policies: offline, dp, learned, robust (Alg 1), dynkin
Ski rental policies: offline_opt, dp, learned, robust (Alg 2 of Cui et al.), deterministic

Decision rule: stop/buy if V(t) > x_t, continue/rent otherwise.
"""

import numpy as np
import torch

from core.sampling import StoppingInstance, SkiInstance
from core.dp import (
    stopping_continuation_values, stopping_expected_max,
    ski_value_to_go, ski_labels, ski_optimal_cost, ski_policy_cost,
)
from core.model import OnlineDecisionTransformer


# ═══════════════════════════════════════════════════════════════════════════
# Lambda computation (optimal stopping robust rule)
# ═══════════════════════════════════════════════════════════════════════════

def find_lambdas(beta: float) -> tuple[float, float]:
    """
    Two roots lambda1 < 1/e < lambda2 of  -lambda * ln(lambda) = beta.
    """
    from scipy.optimize import brentq

    inv_e = 1.0 / np.e
    if not (0 < beta < inv_e):
        raise ValueError(f"beta must be in (0, 1/e ≈ {inv_e:.4f}), got {beta}")

    f = lambda lam: -lam * np.log(lam) - beta
    lambda1 = brentq(f, 1e-12, inv_e)
    lambda2 = brentq(f, inv_e, 1 - 1e-12)
    return lambda1, lambda2


# ═══════════════════════════════════════════════════════════════════════════
# Optimal stopping policies
# ═══════════════════════════════════════════════════════════════════════════

def stop_policy_offline(values):
    idx = int(np.argmax(values))
    return idx, float(values[idx])


def stop_policy_dp(values, C):
    n = len(values)
    for t in range(n - 1):
        if values[t] >= C[t]:
            return t, float(values[t])
    return n - 1, float(values[n - 1])


def stop_policy_learned(values, V_hat, M):
    """Stop at first t where x_t >= V(t)*M (i.e., value exceeds continuation threshold)."""
    n = len(values)
    threshold = V_hat * M
    for t in range(n - 1):
        if values[t] >= threshold[t]:
            return t, float(values[t])
    return n - 1, float(values[n - 1])


def stop_policy_robust(values, V_hat, beta, M):
    """Robust wrapper: x_t >= threshold in the middle phase only."""
    n = len(values)
    lambda1, lambda2 = find_lambdas(beta)
    threshold = V_hat * M
    m = -np.inf
    for t in range(n):
        x = float(values[t])
        is_best = x > m
        m = max(m, x)
        t_norm = (t + 1) / n
        if t_norm <= lambda1:
            continue
        elif t_norm <= lambda2:
            if is_best and x >= threshold[t]:
                return t, x
        else:
            if is_best:
                return t, x
    return n - 1, float(values[n - 1])


def stop_policy_dynkin(values, r):
    """Classical secretary / Dynkin's algorithm.

    Reject the first r items (observation phase), then accept the first item
    that is the best so far (better than ALL previous items, not just the first r).
    At r = floor(n/e), this gives the classical 1/e competitive ratio.
    """
    n = len(values)
    if r >= n:
        return n - 1, float(values[n - 1])
    best_seen = float(values[:r].max()) if r > 0 else -np.inf
    for t in range(r, n - 1):
        if values[t] > best_seen:
            return t, float(values[t])
        best_seen = max(best_seen, float(values[t]))
    return n - 1, float(values[n - 1])


# ═══════════════════════════════════════════════════════════════════════════
# Ski rental policies
# ═══════════════════════════════════════════════════════════════════════════

def ski_policy_dp(inst: SkiInstance) -> int:
    """Optimal buying day from exact DP. Returns K (0-indexed: rent K days, buy on K+1)."""
    lbl = ski_labels(inst.pmf_T, inst.n, inst.B, inst.r)
    return lbl["K_star"]


def ski_policy_learned(V_hat: np.ndarray) -> int:
    """Buy at first t where V(t) >= 1 (i.e., J_hat >= B). Returns buying day K (0-indexed)."""
    for t in range(len(V_hat)):
        if V_hat[t] >= 1.0:
            return t
    return len(V_hat)  # never buy


def ski_policy_deterministic(B: float, r: float) -> int:
    """Classic 2-competitive: buy at day floor(B/r). Returns buying day K."""
    return int(np.floor(B / r))


def compute_training_pmf(n, grid_size=200):
    """Compute the marginal PMF of T under the uniform mixture of 10 ski
    training families by numerical integration over each family's
    hyperparameter prior. Fully deterministic (no sampling)."""
    from sampling import _make_ski_pmf

    t = np.arange(1, n + 1, dtype=float)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    G = grid_size
    family_pmfs = []

    # 1. Geometric
    acc = np.zeros(n)
    for p in np.linspace(0.05, 0.5, G):
        acc += _make_ski_pmf((t - 1) * np.log1p(-p) + np.log(p))
    family_pmfs.append(acc / G)

    # 2. Poisson
    acc = np.zeros(n)
    for lam in np.linspace(1.0, n / 2, G):
        acc += _make_ski_pmf(t * np.log(lam) - lam - log_fact[t.astype(int)])
    family_pmfs.append(acc / G)

    # 3. Binomial
    acc = np.zeros(n)
    s = np.arange(n)
    for p in np.linspace(0.1, 0.9, G):
        log_w = (log_fact[n-1] - log_fact[s] - log_fact[n-1-s]
                 + s * np.log(p + 1e-300) + (n-1-s) * np.log(1 - p + 1e-300))
        acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / G)

    # 4. Uniform
    acc = np.zeros(n)
    for K in range(2, n + 1):
        pmf = np.zeros(n); pmf[:K] = 1.0 / K; acc += pmf
    family_pmfs.append(acc / (n - 1))

    # 5. Zipf
    acc = np.zeros(n)
    for alpha in np.linspace(1.0, 3.0, G):
        acc += _make_ski_pmf(-alpha * np.log(t))
    family_pmfs.append(acc / G)

    # 6. Log-normal
    G2 = max(G // 5, 40)
    acc = np.zeros(n)
    for mu in np.linspace(0, np.log(n), G2):
        for sigma in np.linspace(0.3, 1.5, G2):
            acc += _make_ski_pmf(-(np.log(t) - mu)**2 / (2*sigma**2) - np.log(t))
    family_pmfs.append(acc / (G2 * G2))

    # 7. Weibull
    acc = np.zeros(n)
    for beta in np.linspace(0.3, 2.0, G2):
        for lam in np.linspace(1.0, n / 2, G2):
            acc += _make_ski_pmf(-(t / lam)**beta)
    family_pmfs.append(acc / (G2 * G2))

    # 8. Bimodal
    G3 = max(G // 10, 20)
    acc = np.zeros(n)
    for lam1 in np.linspace(1.0, n/4, G3):
        for lam2 in np.linspace(n/2, float(n), G3):
            for p_high in np.linspace(0.1, 0.4, G3):
                pmf1 = _make_ski_pmf(t*np.log(lam1) - lam1 - log_fact[np.minimum(t.astype(int), n)])
                pmf2 = _make_ski_pmf(t*np.log(lam2) - lam2 - log_fact[np.minimum(t.astype(int), n)])
                pmf = (1 - p_high) * pmf1 + p_high * pmf2; pmf /= pmf.sum(); acc += pmf
    family_pmfs.append(acc / (G3**3))

    # 9. Spike
    acc = np.zeros(n); count = 0
    for k in range(1, n + 1):
        for eps in np.linspace(0.01, 0.2, G2):
            pmf = np.full(n, eps/n); pmf[k-1] += (1 - eps); pmf /= pmf.sum(); acc += pmf; count += 1
    family_pmfs.append(acc / count)

    # 10. Two-point
    acc = np.zeros(n); count = 0
    for a in range(1, max(1, n//2) + 1):
        for b in range(max(2, n//2), n + 1):
            for p_b in np.linspace(0.05, 0.5, G3):
                pmf = np.zeros(n); pmf[a-1] = 1 - p_b; pmf[b-1] += p_b; pmf /= pmf.sum(); acc += pmf; count += 1
    family_pmfs.append(acc / count)

    mixture_pmf = np.mean(family_pmfs, axis=0)
    mixture_pmf /= mixture_pmf.sum()
    return mixture_pmf


def compute_U(pmf_train: np.ndarray, B: float, r: float) -> int:
    """
    Compute the tail threshold U from the training distribution P.

    U is the smallest 0-indexed day such that P(T > U+1) <= 1/sqrt(B/r),
    i.e., the tail beyond day U+1 is negligible.

    Returns 0-indexed to match K_hat convention (K=0 means buy on day 1).
    """
    threshold = 1.0 / np.sqrt(B / r)
    n = len(pmf_train)
    cdf = np.cumsum(pmf_train)  # cdf[t] = P(T <= t+1), 0-indexed
    for t in range(n):
        tail = 1.0 - cdf[t]    # P(T > t+1)
        if tail <= threshold:
            return t            # 0-indexed
    return n - 1


def ski_policy_robust(inst: SkiInstance, K_hat: int, lam: float,
                      U: int | None = None, pmf_train: np.ndarray | None = None) -> int:
    """Algorithm 2 of Cui et al. — lambda-robust wrapper."""
    B, r = inst.B, inst.r
    ratio = B / r
    sqrt_ratio = np.sqrt(ratio)

    if lam >= 1:
        return ski_policy_deterministic(B, r)

    if U is None:
        if pmf_train is not None:
            U = compute_U(pmf_train, B, r)
        else:
            U = compute_U(inst.pmf_T, B, r)

    K_star = min(K_hat + sqrt_ratio, U + sqrt_ratio)

    if lam <= 0:
        # Full trust: d = K* + 1 (Algorithm 2, line 5)
        d = int(K_star) + 1
    elif K_star <= ratio:
        # K* before breakeven: clamp up (Algorithm 2, line 7)
        d = max(int(K_star) + 1, int(np.ceil(lam * ratio)))
    else:
        # K* after breakeven: clamp down (Algorithm 2, line 8)
        d = min(int(K_star) + 1, int(np.ceil(ratio / lam)))

    return max(0, d - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Model inference
# ═══════════════════════════════════════════════════════════════════════════

def get_predictions(model, instances, task_id: int,
                    device="cpu", batch_size=512):
    """Run the 2D chain model on a batch of instances.

    Returns a list of V_hat arrays (normalised, shape (n,) each), where
    V_hat[t] = chain_V(t, t) = V(t).

    For stopping: V(t) approx C[t]/M  -> stop  if x_t/M >= V(t)
    For ski     : V(t) approx J[t]/B  -> buy   if V(t) >= 1.0

    The last position (t = n-1) is a forced accept; its value is set to 0.
    """
    n = instances[0].n
    model.eval()
    all_V = []

    for start in range(0, len(instances), batch_size):
        chunk = instances[start:start + batch_size]
        if task_id == 0:
            x = torch.tensor(
                np.stack([inst.values for inst in chunk]), dtype=torch.long
            ).to(device)
            B_cost, r_cost = None, None
        else:
            x = torch.ones(len(chunk), n, dtype=torch.long).to(device)
            B_cost = torch.tensor([inst.B for inst in chunk], dtype=torch.long).to(device)
            r_cost = torch.tensor([inst.r for inst in chunk], dtype=torch.long).to(device)

        with torch.no_grad():
            chain2d_V, decision_pos = model(x, n_horizon=n, task_id=task_id,
                                            B_cost=B_cost, r_cost=r_cost,
                                            mode="inference")

        V_decision = chain2d_V[:, decision_pos].cpu().numpy()
        V_padded = np.concatenate(
            [V_decision, np.zeros((V_decision.shape[0], 1))], axis=1)
        all_V.extend(V_padded)

    return all_V


def get_attention_maps(model, x, n_horizon, task_id, chain2d_targets,
                       device="cpu", B_cost=None, r_cost=None):
    """Run a teacher-forced forward pass and return per-layer attention weights.

    Args:
        model           : trained OnlineDecisionTransformer
        x               : (B, T) int64 observation tokens
        n_horizon       : int or (B,) tensor
        task_id         : 0 (stopping) or 1 (ski)
        chain2d_targets : (B, L_chain) normalised targets for teacher forcing
        device          : device
        B_cost          : float or (B,) tensor — buy cost (ski rental)
        r_cost          : float or (B,) tensor — rent cost (ski rental)

    Returns:
        attn_weights : list of (B, n_heads, L_total, L_total) tensors, one per layer
                       L_total = T + T*(T-1)//2
        info         : dict with index mappings for interpretation:
            "n_obs"        : T (number of observation positions)
            "t_idx"        : (L_chain,) decision step t per chain position
            "j_idx"        : (L_chain,) within-sub-chain step j per chain position
            "decision_pos" : (T-1,) flat index of V(t,t) in chain
            "position_labels" : list of str labels for all L_total positions
                                e.g. ["x_0", "x_1", ..., "V(0,7)", "V(0,6)", ..., "V(6,6)"]
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        chain2d_V, decision_pos, attn = model(
            x, chain2d_targets=chain2d_targets.to(device),
            n_horizon=n_horizon, task_id=task_id,
            B_cost=B_cost, r_cost=r_cost,
            mode="teacher_forcing", return_attention=True)

    T = x.shape[1]
    t_idx, j_idx, _ = model._get_chain2d_info(T, x.device)

    # Build human-readable position labels
    labels = [f"x_{t}" for t in range(T)]
    for p in range(len(t_idx)):
        t = int(t_idx[p])
        k = T - 2 - int(j_idx[p])
        labels.append(f"V({t},{k})")

    info = {
        "n_obs": T,
        "t_idx": t_idx.cpu(),
        "j_idx": j_idx.cpu(),
        "decision_pos": decision_pos.cpu(),
        "position_labels": labels,
    }

    return [a.cpu() for a in attn], info


def get_stopping_predictions(model, instances, device="cpu", batch_size=512,
                             use_chain=True):
    """Run model on stopping instances. Returns list of V_hat arrays (normalised)."""
    return get_predictions(model, instances, task_id=0,
                           device=device, batch_size=batch_size)


def get_ski_predictions(model, instances, device="cpu", batch_size=512,
                        use_chain=True):
    """Run model on ski instances. Returns list of V_hat arrays (normalised)."""
    return get_predictions(model, instances, task_id=1,
                           device=device, batch_size=batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# Full comparison: Optimal stopping
# ═══════════════════════════════════════════════════════════════════════════

def compare_stopping(instances, model=None, betas=None, r_fractions=None,
                     device="cpu", use_chain=True, batch_size=512):
    if betas is None:
        betas = [0.1, 0.15, 0.2, 0.25]
    if r_fractions is None:
        r_fractions = [0.1, 0.2, 0.368, 0.5]

    n = instances[0].n
    M = len(instances[0].pmf)
    results = {}

    prophets = np.array([stopping_expected_max(inst.pmf, n) for inst in instances])
    realized_maxes = np.array([float(inst.values.max()) for inst in instances])
    C_dp = [stopping_continuation_values(inst.pmf, n) for inst in instances]

    def _run(policy_fn):
        rewards, is_best = [], []
        for i, inst in enumerate(instances):
            _, reward = policy_fn(i, inst)
            rewards.append(reward)
            is_best.append(float(reward == inst.values.max()))
        mean_r = float(np.mean(rewards))
        mean_max = float(np.mean(realized_maxes))
        return {
            "mean_reward": mean_r,
            "mean_prophet": float(np.mean(prophets)),
            "cr": mean_r / mean_max if mean_max > 0 else 0.0,
            "prob_best": float(np.mean(is_best)),
        }

    results["offline"] = _run(lambda i, inst: stop_policy_offline(inst.values))
    results["dp"] = _run(lambda i, inst: stop_policy_dp(inst.values, C_dp[i]))

    if model is not None:
        V_hat_all = get_stopping_predictions(model, instances, device=device,
                                                batch_size=batch_size)
        results["learned"] = _run(
            lambda i, inst: stop_policy_learned(inst.values, V_hat_all[i], M))
        for beta in betas:
            lam1, lam2 = find_lambdas(beta)
            results[f"robust β={beta:.2f}"] = {
                **_run(lambda i, inst, b=beta: stop_policy_robust(inst.values, V_hat_all[i], b, M)),
                "lambda1": round(lam1, 4), "lambda2": round(lam2, 4),
            }

    for frac in r_fractions:
        r_val = max(0, int(frac * n))
        actual_frac = r_val / n if n > 0 else 0
        beta_dyn = -actual_frac * np.log(actual_frac) if actual_frac > 0 else 0.0
        results[f"dynkin β={beta_dyn:.2f}"] = _run(
            lambda i, inst, _r=r_val: stop_policy_dynkin(inst.values, _r))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Full comparison: Ski rental
# ═══════════════════════════════════════════════════════════════════════════

def compare_ski(instances, model=None, lambdas=None, device="cpu", use_chain=True,
                pmf_train=None):
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.5, 0.8, 1.0]

    results = {}
    n = instances[0].n
    B = instances[0].B
    r = instances[0].r

    U = compute_U(pmf_train, B, r) if pmf_train is not None else None

    opt_costs = []
    dp_buy_days = []
    for inst in instances:
        opt_costs.append(ski_optimal_cost(inst.pmf_T, inst.n, inst.B, inst.r))
        dp_buy_days.append(ski_policy_dp(inst))
    opt_costs = np.array(opt_costs)

    def _eval_policy(buying_days):
        costs = []
        for i, inst in enumerate(instances):
            costs.append(ski_policy_cost(buying_days[i], inst.pmf_T, inst.n, inst.B, inst.r))
        costs = np.array(costs)
        losses = costs - opt_costs
        # CR = mean(OPT) / mean(policy cost) — ratio of means
        mean_cost = float(np.mean(costs))
        mean_opt = float(np.mean(opt_costs))
        return {
            "mean_cost": mean_cost,
            "mean_opt_cost": mean_opt,
            "mean_additive_loss": float(np.mean(losses)),
            "median_additive_loss": float(np.median(losses)),
            "cr": mean_opt / mean_cost if mean_cost > 0 else 1.0,
        }

    results["offline_opt"] = _eval_policy(
        [int(np.argmin([ski_policy_cost(K, inst.pmf_T, inst.n, inst.B, inst.r)
                        for K in range(inst.n + 1)])) for inst in instances])

    results["dp"] = _eval_policy(dp_buy_days)

    det_K = ski_policy_deterministic(B, r)
    results["deterministic"] = _eval_policy([det_K] * len(instances))

    if model is not None:
        V_hat_all = get_ski_predictions(model, instances, device=device)
        learned_days = [ski_policy_learned(V_hat_all[i]) for i in range(len(instances))]
        results["learned"] = _eval_policy(learned_days)

        for lam in lambdas:
            robust_days = []
            for i, inst in enumerate(instances):
                K_hat = learned_days[i]
                robust_days.append(ski_policy_robust(inst, K_hat, lam, U=U))
            results[f"robust λ={lam:.1f}"] = _eval_policy(robust_days)

    return results


def print_stopping_results(results):
    header = f"{'Policy':<28} {'E[X_τ]':>9} {'E[max]':>9} {'CR':>7} {'P(best)':>9}"
    print(header)
    print("─" * len(header))
    for name, m in results.items():
        print(f"{name:<28} {m['mean_reward']:>9.3f} {m['mean_prophet']:>9.3f} "
              f"{m['cr']:>7.4f} {m['prob_best']:>9.4f}")


def print_ski_results(results):
    header = f"{'Policy':<28} {'Cost':>10} {'OPT':>10} {'Δ(loss)':>10}"
    print(header)
    print("─" * len(header))
    for name, m in results.items():
        print(f"{name:<28} {m['mean_cost']:>10.3f} {m['mean_opt_cost']:>10.3f} "
              f"{m['mean_additive_loss']:>10.3f}")
