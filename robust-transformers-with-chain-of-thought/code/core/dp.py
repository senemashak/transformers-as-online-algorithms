"""
Exact dynamic programming for both optimal stopping and ski rental.

Optimal stopping:
    V_t(x) = max(x, C_t)     C_t = E[V_{t+1}(X)]     V_n = x_n

Ski rental:
    J_t = min{B, r + q_t * J_{t+1}}    J_{n+1} = 0
    q_t = Pr(T > t | T >= t)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Optimal stopping DP
# ═══════════════════════════════════════════════════════════════════════════

def stopping_continuation_values(pmf: np.ndarray, n: int) -> np.ndarray:
    """
    Backward induction for IID prophet on {1,...,M}.

    DP recurrence (i.i.d. case, C_t depends only on t, not on history):
        V_{t+1}(x) = max(x, C_{t+1})
        C_t = E[V_{t+1}(X)] = sum_k pmf(k) * max(k, C_{t+1})

    Returns C of shape (n,):
      C[t] = continuation value at step t+1 (0-indexed)
      C[n-1] = 0.0 sentinel (must accept at last step)
    """
    if n == 1:
        return np.zeros(1)
    M = len(pmf)
    k = np.arange(1, M + 1, dtype=float)  # all possible values X can take
    C = np.zeros(n)
    # At step n-1 (second-to-last), continuing gives one more draw → C = E[X]
    C[n - 2] = float(np.dot(k, pmf))
    # Earlier steps: C_t = E[max(X, C_{t+1})]
    #   np.maximum(k, C[t+1]) = V_{t+1}(k) for each realization k
    #   np.dot(pmf, ...) takes the expectation over X ~ pmf
    for t in range(n - 3, -1, -1):
        C[t] = float(np.dot(pmf, np.maximum(k, C[t + 1])))
    return C


def stopping_labels(pmf: np.ndarray, values: np.ndarray) -> dict:
    """
    Compute DP labels for one realized stopping sequence.

    Labels per time step t:
      C[t] — continuation value at step t
      a[t] — binary optimal decision: 1 if x_t >= C_t (stop), 0 otherwise
    """
    n = len(values)
    x = values.astype(float)
    C = stopping_continuation_values(pmf, n)
    a = (x >= C).astype(float)
    a[n - 1] = 1.0

    return {"C": C, "a": a}


def stopping_expected_max(pmf: np.ndarray, n: int) -> float:
    """E[max{X_1,...,X_n}] for n IID draws from pmf."""
    M = len(pmf)
    k = np.arange(1, M + 1, dtype=float)
    F = np.cumsum(pmf)
    F_prev = np.concatenate([[0.0], F[:-1]])
    return float(np.dot(k, F ** n - F_prev ** n))


# ═══════════════════════════════════════════════════════════════════════════
# Ski rental DP
# ═══════════════════════════════════════════════════════════════════════════

def ski_value_to_go(pmf_T: np.ndarray, n: int, B: float, r: float) -> np.ndarray:
    """
    Backward induction for Bayesian ski rental.

    Args:
        pmf_T : (n,) PMF of T ∈ {1,...,n}
        n     : horizon
        B     : buying cost
        r     : rental cost per day

    Returns:
        J : (n+1,) where J[t] = value-to-go at day t+1 (0-indexed), J[n]=0
    """
    # Compute survival function S(t) = Pr(T >= t)
    S = np.zeros(n + 2)
    S[1] = 1.0
    for t in range(2, n + 2):
        S[t] = S[t - 1] - pmf_T[t - 2]   # pmf_T[k-1] = Pr(T=k)
    S = np.maximum(S, 0.0)

    # Conditional survival: q_t = Pr(T > t | T >= t) = S(t+1) / S(t)
    q = np.zeros(n + 1)
    for t in range(1, n + 1):
        q[t] = S[t + 1] / S[t] if S[t] > 1e-15 else 0.0

    # Backward induction: J[n] = 0, J[t] = min{B, r + q[t] * J[t+1]}
    J = np.zeros(n + 1)  # J[0] unused, J[1]..J[n]
    for t in range(n, 0, -1):
        J[t] = min(B, r + q[t] * J[t + 1]) if t < n else min(B, r)
    # J[0] unused
    return J[1:]  # shape (n,): J[0]=J_1, J[1]=J_2, ..., J[n-1]=J_n


def ski_labels(pmf_T: np.ndarray, n: int, B: float, r: float) -> dict:
    """
    Compute DP labels for one ski rental instance.

    Labels per time step t:
      J[t]   — cost-to-go at step t
      a[t]   — binary optimal decision: 1 if B <= J[t] (buy), 0 otherwise
      K_star — optimal buying day (0-indexed), or n if never buy
    """
    J = ski_value_to_go(pmf_T, n, B, r)
    a = (B <= J).astype(float)
    buy_times = np.where(B <= J)[0]
    K_star = int(buy_times[0]) if len(buy_times) > 0 else n

    return {"J": J, "a": a, "K_star": K_star}


def ski_optimal_cost(pmf_T: np.ndarray, n: int, B: float, r: float) -> float:
    """
    Offline optimal cost: min over all buying days K of c(A_K, P).
    K=0 means buy immediately, K=n means never buy.
    """
    best = float("inf")
    cumP = np.cumsum(pmf_T)  # cumP[k-1] = Pr(T <= k)
    for K in range(n + 1):
        if K == 0:
            cost = B  # buy immediately
        else:
            # cost = sum_{t=1}^{K} t*r*P(T=t) + sum_{t=K+1}^{n} (K*r+B)*P(T=t)
            cost = 0.0
            for t in range(1, K + 1):
                cost += t * r * pmf_T[t - 1]
            prob_after_K = 1.0 - cumP[K - 1] if K <= n else 0.0
            cost += (K * r + B) * max(0, prob_after_K)
        best = min(best, cost)
    return best


def ski_policy_cost(K: int, pmf_T: np.ndarray, n: int, B: float, r: float) -> float:
    """Expected cost of policy that rents for K days then buys on day K+1."""
    if K >= n:
        # Never buy: pay r per day
        return sum((t + 1) * r * pmf_T[t] for t in range(n))
    cost = 0.0
    for t in range(K):
        cost += (t + 1) * r * pmf_T[t]
    cumP = np.cumsum(pmf_T)
    prob_after_K = 1.0 - cumP[K - 1] if K > 0 else 1.0
    cost += (K * r + B) * max(0, prob_after_K)
    return cost


# ═══════════════════════════════════════════════════════════════════════════
# Quick demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sampling import sample_stopping_instance, sample_ski_instance

    rng = np.random.default_rng(42)

    # Stopping
    inst = sample_stopping_instance(6, 20, "geometric", rng)
    lbl = stopping_labels(inst.pmf, inst.values)
    print("=== Stopping ===")
    print(f"  values: {inst.values}")
    print(f"  C:      {np.round(lbl['C'], 2)}")
    print(f"  E[max]: {stopping_expected_max(inst.pmf, inst.n):.2f}")

    # Ski rental
    sinst = sample_ski_instance(20, 10, 1, "ski_geometric", rng)
    slbl = ski_labels(sinst.pmf_T, sinst.n, sinst.B, sinst.r)
    print("\n=== Ski rental ===")
    print(f"  T_realized: {sinst.T_realized}")
    print(f"  J:          {np.round(slbl['J'][:5], 2)}...")
    print(f"  K_star:     {slbl['K_star']}")
    print(f"  OPT cost:   {ski_optimal_cost(sinst.pmf_T, sinst.n, sinst.B, sinst.r):.2f}")
