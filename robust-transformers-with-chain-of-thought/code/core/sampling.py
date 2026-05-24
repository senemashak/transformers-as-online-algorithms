"""
Instance samplers for both optimal stopping and ski rental.

Optimal stopping: distributions over values X ∈ {1,...,M}.
Ski rental: distributions over lifetime T ∈ {1,...,n}.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

# ═══════════════════════════════════════════════════════════════════════════
# Optimal stopping
# ═══════════════════════════════════════════════════════════════════════════

StoppingDistType = Literal[
    "geometric", "zipf", "binomial", "hard_instance",
    "lognormal", "bimodal", "weibull",
    "uniform_small", "poisson_small",
    "geometric_steep", "sparse_low",
]


@dataclass
class StoppingInstance:
    dist_type: StoppingDistType
    params: dict
    n: int
    values: np.ndarray          # (n,) integers in {1,...,M}
    pmf: np.ndarray             # (M,) PMF over {1,...,M}


def _make_pmf(log_weights: np.ndarray) -> np.ndarray:
    w = np.exp(log_weights - log_weights.max())
    return w / w.sum()


def _draw(rng: np.random.Generator, pmf: np.ndarray, M: int, n: int) -> np.ndarray:
    return rng.choice(np.arange(1, M + 1), size=n, p=pmf)


def _subsample_support(pmf: np.ndarray, M: int, n: int, rng: np.random.Generator) -> np.ndarray:
    lo = min(max(n, 10), M - 1)
    hi = max(lo + 1, M // 3)
    K = int(rng.integers(lo, hi))
    support_idx = rng.choice(M, size=K, replace=False)
    masked = np.zeros(M)
    masked[support_idx] = pmf[support_idx]
    total = masked.sum()
    if total == 0:
        masked[support_idx] = 1.0 / K
    else:
        masked /= total
    return masked


def _log_factorial(M: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, M + 1)))])


# --- Stopping families ---

def _stop_geometric(n, M, rng):
    p = rng.uniform(0.01, 0.5)
    k = np.arange(1, M + 1)
    log_w = (k - 1) * np.log1p(-p) + np.log(p)
    pmf = _subsample_support(_make_pmf(log_w), M, n, rng)
    return StoppingInstance("geometric", {"M": M, "p": float(p)}, n, _draw(rng, pmf, M, n), pmf)


def _stop_zipf(n, M, rng):
    alpha = rng.uniform(1.0, 3.0)
    k = np.arange(1, M + 1)
    log_w = -alpha * np.log(k)
    pmf = _subsample_support(_make_pmf(log_w), M, n, rng)
    return StoppingInstance("zipf", {"M": M, "alpha": float(alpha)}, n, _draw(rng, pmf, M, n), pmf)


def _stop_binomial(n, M, rng):
    p = rng.uniform(0.1, 0.9)
    log_fact = _log_factorial(M)
    k = np.arange(M)
    log_w = (log_fact[M - 1] - log_fact[k] - log_fact[M - 1 - k]
             + k * np.log(p + 1e-300) + (M - 1 - k) * np.log(1 - p + 1e-300))
    pmf = _subsample_support(_make_pmf(log_w), M, n, rng)
    return StoppingInstance("binomial", {"M": M, "p": float(p)}, n, _draw(rng, pmf, M, n), pmf)


def _stop_hard_instance(n, M, rng):
    n3 = n ** 3
    M_small = max(1, M // n3)
    n3_actual = min(n3, M_small)
    v = rng.choice(np.arange(1, M_small + 1), size=n3_actual, replace=False)
    u = M
    p_small = (1.0 / n3_actual) * (1.0 - 1.0 / n ** 2)
    p_u = 1.0 / n ** 2
    pmf = np.zeros(M)
    for vi in v:
        pmf[vi - 1] += p_small
    pmf[u - 1] += p_u
    pmf /= pmf.sum()
    outcomes = np.append(v, u)
    probs = np.full(n3_actual + 1, p_small)
    probs[-1] = p_u
    probs /= probs.sum()
    indices = rng.choice(len(outcomes), size=n, p=probs)
    values = outcomes[indices].astype(int)
    return StoppingInstance("hard_instance",
                            {"M": M, "M_small": int(M_small), "u": int(u), "n3": n3_actual},
                            n, values, pmf)


def _stop_lognormal(n, M, rng):
    log_M = np.log(M)
    mu = rng.uniform(log_M / 4, 3 * log_M / 4)
    sigma = rng.uniform(0.5, 1.5)
    k = np.arange(1, M + 1, dtype=float)
    log_w = -(np.log(k) - mu) ** 2 / (2 * sigma ** 2) - np.log(k)
    pmf = _subsample_support(_make_pmf(log_w), M, n, rng)
    return StoppingInstance("lognormal", {"M": M, "mu": float(mu), "sigma": float(sigma)},
                            n, _draw(rng, pmf, M, n), pmf)


def _stop_bimodal(n, M, rng):
    p_high = rng.uniform(0.05, 0.35)
    mid = M // 2
    lo_max = mid  # pool size for low values
    hi_max = M - mid  # pool size for high values
    K_lo = int(rng.integers(min(max(n // 2, 5), lo_max), min(max(n // 2 + 1, mid // 3), lo_max) + 1))
    K_hi = int(rng.integers(min(max(n // 2, 5), hi_max), min(max(n // 2 + 1, (M - mid) // 3), hi_max) + 1))
    lo_vals = rng.choice(np.arange(1, mid + 1), size=K_lo, replace=False)
    hi_vals = rng.choice(np.arange(mid + 1, M + 1), size=K_hi, replace=False)
    pmf = np.zeros(M)
    pmf[lo_vals - 1] = (1 - p_high) / K_lo
    pmf[hi_vals - 1] = p_high / K_hi
    pmf /= pmf.sum()
    return StoppingInstance("bimodal", {"M": M, "p_high": float(p_high)},
                            n, _draw(rng, pmf, M, n), pmf)


def _stop_weibull(n, M, rng):
    beta = rng.uniform(0.3, 0.8)
    lam = rng.uniform(M / 10, M / 2)
    k = np.arange(1, M + 1, dtype=float)
    log_w = -(k / lam) ** beta
    pmf = _subsample_support(_make_pmf(log_w), M, n, rng)
    return StoppingInstance("weibull", {"M": M, "beta": float(beta), "lam": float(lam)},
                            n, _draw(rng, pmf, M, n), pmf)


def _stop_uniform_small(n, M, rng):
    lo = min(n, M)
    K = int(rng.integers(lo, min(5 * n, M) + 1)) if lo < min(5 * n, M) + 1 else M
    pmf = np.zeros(M)
    pmf[:K] = 1.0 / K
    values = rng.integers(1, K + 1, size=n)
    return StoppingInstance("uniform_small", {"M": M, "K": K}, n, values, pmf)


def _stop_poisson_small(n, M, rng):
    lam = rng.uniform(1.0, float(n))
    k = np.arange(1, M + 1)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, M + 1)))])
    log_w = k * np.log(lam) - lam - log_fact[k]
    pmf = _make_pmf(log_w)
    values = _draw(rng, pmf, M, n)
    return StoppingInstance("poisson_small", {"M": M, "lambda": float(lam)}, n, values, pmf)


def _stop_geometric_steep(n, M, rng):
    p = rng.uniform(0.5, 0.95)
    k = np.arange(1, M + 1)
    log_w = (k - 1) * np.log1p(-p) + np.log(p)
    pmf = _make_pmf(log_w)
    return StoppingInstance("geometric_steep", {"M": M, "p": float(p)},
                            n, _draw(rng, pmf, M, n), pmf)


def _stop_sparse_low(n, M, rng):
    L = int(rng.integers(3, min(M, max(4, 2 * n + 1))))
    K = int(rng.integers(2, L + 1))
    support = rng.choice(np.arange(1, L + 1), size=K, replace=False)
    pmf = np.zeros(M)
    for v in support:
        pmf[v - 1] = 1.0 / K
    values = rng.choice(support, size=n)
    return StoppingInstance("sparse_low", {"M": M, "L": L, "K": K}, n, values, pmf)


STOPPING_SAMPLERS = {
    "geometric": _stop_geometric,
    "zipf": _stop_zipf,
    "binomial": _stop_binomial,
    "lognormal": _stop_lognormal,
    "bimodal": _stop_bimodal,
    "weibull": _stop_weibull,
    "hard_instance": _stop_hard_instance,
    "uniform_small": _stop_uniform_small,
    "poisson_small": _stop_poisson_small,
    "geometric_steep": _stop_geometric_steep,
    "sparse_low": _stop_sparse_low,
}

STOPPING_TRAIN_FAMILIES = [k for k in STOPPING_SAMPLERS if k != "hard_instance"]


def sample_stopping_instance(n, M, dist_type, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return STOPPING_SAMPLERS[dist_type](n, M, rng)


def sample_stopping_batch(num, n, M, dist_type=None, rng=None, families=None):
    if rng is None:
        rng = np.random.default_rng()
    pool = families if families is not None else list(STOPPING_SAMPLERS.keys())
    return [STOPPING_SAMPLERS[dist_type if dist_type else rng.choice(pool)](n, M, rng)
            for _ in range(num)]


# ═══════════════════════════════════════════════════════════════════════════
# Ski rental
# ═══════════════════════════════════════════════════════════════════════════

SkiDistType = Literal[
    "ski_geometric", "ski_poisson", "ski_binomial", "ski_uniform",
    "ski_zipf", "ski_lognormal", "ski_weibull",
    "ski_bimodal", "ski_spike", "ski_twopoint",
]


@dataclass
class SkiInstance:
    dist_type: SkiDistType
    params: dict
    n: int
    B: float
    r: float
    T_realized: int             # realized lifetime
    pmf_T: np.ndarray           # (n,) PMF over T ∈ {1,...,n}


def _make_ski_pmf(log_weights: np.ndarray) -> np.ndarray:
    w = np.exp(log_weights - log_weights.max())
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total == 0:
        return np.ones(len(w)) / len(w)
    return w / total


def _draw_T(rng, pmf_T, n):
    return int(rng.choice(np.arange(1, n + 1), p=pmf_T))


# --- Ski rental families ---

def _ski_geometric(n, B, r, rng):
    p = rng.uniform(0.05, 0.5)
    t = np.arange(1, n + 1)
    log_w = (t - 1) * np.log1p(-p) + np.log(p)
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_geometric", {"p": float(p)}, n, B, r,
                       _draw_T(rng, pmf, n), pmf)


def _ski_poisson(n, B, r, rng):
    lam = rng.uniform(1.0, n / 2)
    t = np.arange(1, n + 1)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    log_w = t * np.log(lam) - lam - log_fact[t]
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_poisson", {"lambda": float(lam)}, n, B, r,
                       _draw_T(rng, pmf, n), pmf)


def _ski_binomial(n, B, r, rng):
    p = rng.uniform(0.1, 0.9)
    t = np.arange(n)  # t = 0,...,n-1 representing T-1
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    log_w = (log_fact[n - 1] - log_fact[t] - log_fact[n - 1 - t]
             + t * np.log(p + 1e-300) + (n - 1 - t) * np.log(1 - p + 1e-300))
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_binomial", {"p": float(p)}, n, B, r,
                       _draw_T(rng, pmf, n), pmf)


def _ski_uniform(n, B, r, rng):
    K = int(rng.integers(2, n + 1))
    pmf = np.zeros(n)
    pmf[:K] = 1.0 / K
    return SkiInstance("ski_uniform", {"K": K}, n, B, r,
                       _draw_T(rng, pmf, n), pmf)


def _ski_zipf(n, B, r, rng):
    alpha = rng.uniform(1.0, 3.0)
    t = np.arange(1, n + 1, dtype=float)
    log_w = -alpha * np.log(t)
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_zipf", {"alpha": float(alpha)}, n, B, r,
                       _draw_T(rng, pmf, n), pmf)


def _ski_lognormal(n, B, r, rng):
    mu = rng.uniform(0, np.log(n))
    sigma = rng.uniform(0.3, 1.5)
    t = np.arange(1, n + 1, dtype=float)
    log_w = -(np.log(t) - mu) ** 2 / (2 * sigma ** 2) - np.log(t)
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_lognormal", {"mu": float(mu), "sigma": float(sigma)},
                       n, B, r, _draw_T(rng, pmf, n), pmf)


def _ski_weibull(n, B, r, rng):
    beta = rng.uniform(0.3, 2.0)
    lam = rng.uniform(1.0, n / 2)
    t = np.arange(1, n + 1, dtype=float)
    log_w = -(t / lam) ** beta
    pmf = _make_ski_pmf(log_w)
    return SkiInstance("ski_weibull", {"beta": float(beta), "lam": float(lam)},
                       n, B, r, _draw_T(rng, pmf, n), pmf)


def _ski_bimodal(n, B, r, rng):
    lam1 = rng.uniform(1.0, n / 4)
    lam2 = rng.uniform(n / 2, float(n))
    p_high = rng.uniform(0.1, 0.4)
    t = np.arange(1, n + 1, dtype=float)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    # Two Poisson components
    log_w1 = t * np.log(lam1) - lam1 - log_fact[np.minimum(t.astype(int), n)]
    log_w2 = t * np.log(lam2) - lam2 - log_fact[np.minimum(t.astype(int), n)]
    pmf1 = _make_ski_pmf(log_w1)
    pmf2 = _make_ski_pmf(log_w2)
    pmf = (1 - p_high) * pmf1 + p_high * pmf2
    pmf /= pmf.sum()
    return SkiInstance("ski_bimodal", {"lam1": float(lam1), "lam2": float(lam2),
                                       "p_high": float(p_high)},
                       n, B, r, _draw_T(rng, pmf, n), pmf)


def _ski_spike(n, B, r, rng):
    k = int(rng.integers(1, n + 1))
    eps = rng.uniform(0.01, 0.2)
    pmf = np.full(n, eps / n)
    pmf[k - 1] += (1 - eps)
    pmf /= pmf.sum()
    return SkiInstance("ski_spike", {"k": k, "eps": float(eps)},
                       n, B, r, _draw_T(rng, pmf, n), pmf)


def _ski_twopoint(n, B, r, rng):
    a = int(rng.integers(1, max(2, n // 2 + 1)))
    b = int(rng.integers(max(2, n // 2), n + 1))
    p_b = rng.uniform(0.05, 0.5)
    pmf = np.zeros(n)
    pmf[a - 1] = 1 - p_b
    pmf[b - 1] += p_b
    pmf /= pmf.sum()
    return SkiInstance("ski_twopoint", {"a": a, "b": b, "p_b": float(p_b)},
                       n, B, r, _draw_T(rng, pmf, n), pmf)


SKI_SAMPLERS = {
    "ski_geometric": _ski_geometric,
    "ski_poisson": _ski_poisson,
    "ski_binomial": _ski_binomial,
    "ski_uniform": _ski_uniform,
    "ski_zipf": _ski_zipf,
    "ski_lognormal": _ski_lognormal,
    "ski_weibull": _ski_weibull,
    "ski_bimodal": _ski_bimodal,
    "ski_spike": _ski_spike,
    "ski_twopoint": _ski_twopoint,
}

SKI_TRAIN_FAMILIES = list(SKI_SAMPLERS.keys())

SKI_HEAVY_TAIL_FAMILIES = ["ski_zipf", "ski_lognormal", "ski_weibull"]


def sample_ski_instance(n, B, r, dist_type, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return SKI_SAMPLERS[dist_type](n, B, r, rng)


def sample_ski_batch(num, n, B, r, dist_type=None, rng=None, families=None):
    if rng is None:
        rng = np.random.default_rng()
    pool = families if families is not None else SKI_TRAIN_FAMILIES
    return [SKI_SAMPLERS[dist_type if dist_type else rng.choice(pool)](n, B, r, rng)
            for _ in range(num)]


# ═══════════════════════════════════════════════════════════════════════════
# Quick demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Stopping families ===")
    for family in STOPPING_SAMPLERS:
        inst = sample_stopping_instance(8, 1000, family, rng)
        print(f"  [{family}]  values={inst.values[:5]}...  E[X]={np.dot(np.arange(1,1001), inst.pmf):.1f}")

    print("\n=== Ski rental families ===")
    for family in SKI_SAMPLERS:
        inst = sample_ski_instance(20, 10, 1, family, rng)
        print(f"  [{family}]  T={inst.T_realized}  E[T]={np.dot(np.arange(1,21), inst.pmf_T):.1f}")
