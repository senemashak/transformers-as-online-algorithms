"""Experiment B: per-sequence regime-shift evaluation.

B.1 (`--pairs b1`, default): pairs from training-prior σ ∈ {1, 10, 100}.
B.2 (`--pairs b2`):           pairs from OOD σ ∈ {3, 30, 300}.

For each pair (σ_a, σ_b) we generate N = 10^4 sequences with
X_t ~ N(μ_i, σ_a²) for t ∈ [1, 128] and X_t ~ N(μ_i, σ_b²) for
t ∈ [129, 256], same μ_i ~ N(0, τ_0²) throughout.
Random-variance cv models {D_disc_cv, D_logu_cv} are evaluated on each
cache. Per-sequence static-σ reference oracles are computed at σ_a and σ_b
(Algorithm 1 tables already cached: `D{1,2,3}_static_*` for B.1 σ values,
`D_ood_{3,30,300}_static_*` for B.2 σ values).

For each (model, pair) cell we compute:
  1. Pre-shift tracking error: mean |Ĉ_t - C*_{σ_a, t}| over t ∈ [50, 128].
  2. Per-sequence t_{1/2}: smallest Δt ≥ 1 such that ρ_{seq, 128+Δt} ≥ 0.5,
     where ρ_t := (log |Ĉ_t| - log <|C*_σ_a, t|>) /
                  (log <|C*_σ_b, t|> - log <|C*_σ_a, t|>)
     and <·> denotes the iid-σ mean reference. ρ_t is NaN for t at which
     |log <|C*_σ_b, t|> - log <|C*_σ_a, t|>| < 0.05 (numerical-stability
     clip from the B.2 spec).
  3. Population mean t_{1/2}: same calculation on the across-sequences mean
     trajectory.
  4. Post-shift tracking error: mean |Ĉ_t - C*_{σ_b, t}| over t ∈ [200, 255].
  5. Settled ρ at t=200 (population + per-seq quantiles).
  6. Final ρ at t=255 (population + per-seq quantiles); reported as NaN if
     the t=255 reference denominator falls inside the 0.05 clip.

Outputs (suffix `_b2` when --pairs b2):
  results/regime-shift/caches/D_shift_<a>_<b>_test.npz
  results/regime-shift/raw{,_b2}.json                 — metrics per (model, pair)
  results/regime-shift/trajectories{,_b2}.npz         — focal mean/std + per-seq
                                                        thresholds & references
  results/regime-shift/t_half_per_seq{,_b2}.npz       — per-seq t_{1/2} arrays
  results/regime-shift/iid_refs/iid_ref_sigma_<s>.npz — cached iid-σ mean
                                                        references for OOD σ

The σ²-ablation checkpoints are used for D_disc_cv / D_logu_cv (matches §6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from data.distributions import MU_0, TAU0_2, TAU_0
from eval.policies import (
    map_sigma_plugin,
    mle_sigma_plugin,
    model_action,
    random_oracle,
    secretary,
    static_oracle,
)
from oracle.conjugate import compute_eta
from oracle.random_adp import load_table as load_random_table
from oracle.random_adp_torch import make_sigma_grid as make_logu_grid
from oracle.static_adp import solve_adp
from train.io import load_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAIRS_B1: List[Tuple[float, float]] = [
    (1.0, 100.0),                                                # sharp upward
    (100.0, 1.0),                                                # sharp downward
    (10.0, 100.0),                                               # moderate upward
    (100.0, 10.0),                                               # moderate downward
]
PAIRS_B2: List[Tuple[float, float]] = [
    (3.0, 30.0), (30.0, 3.0),                                    # 10× shifts, OOD endpoints
    (3.0, 300.0), (300.0, 3.0),                                  # 100× shifts, OOD endpoints
    (30.0, 300.0), (300.0, 30.0),                                # 10× shifts, one endpoint OOD
]
PAIRS = PAIRS_B1                                                 # backward-compat alias
SHIFT_T = 128                                                    # 1-indexed; t ≤ 128 is σ_a
N_TEST = 10_000
N = 256                                                          # sequence length
SEED_BASE_B1 = 6000                                              # disjoint from existing 42xx-54xx
SEED_BASE_B2 = 7000                                              # disjoint from B.1 caches
SEED_BASE = SEED_BASE_B1                                         # backward-compat alias

OOD_SIGMAS = {3.0, 30.0, 300.0}
TRAIN_SIGMAS = {1.0, 10.0, 100.0}

# Reference-difference clip for ρ_t (spec correction): set ρ_t to NaN where
# |log <C*_σ_b,t> - log <C*_σ_a,t>| < RHO_DENOM_CLIP. Was 1e-12 in B.1.
RHO_DENOM_CLIP = 0.05

MODELS = ['D_disc_cv', 'D_logu_cv']                              # act lacks a threshold readout
SIG2_SWAP = {'D_disc_cv': 'D_disc_cv_sig2', 'D_logu_cv': 'D_logu_cv_sig2'}

K_PROD, J_PROD = 2048, 128

RS_ROOT = V3_ROOT / 'results' / 'regime-shift'
CACHE_DIR = RS_ROOT / 'caches'
ORACLE_TABLES_DIR = CODE_ROOT / 'oracle' / 'tables'
PHASE5 = V3_ROOT / 'results' / 'phase5'

# Pre-shift / post-shift error windows.
PRE_RANGE = (50, 128)                                            # 1-indexed inclusive
POST_RANGE = (200, 255)                                          # 1-indexed inclusive


def _sigma_tag(sigma: float) -> str:
    """For cache naming: integer sigmas only here ({1,10,100,3,30,300})."""
    s = int(sigma)
    if s != sigma:
        raise ValueError(f'unexpected non-integer sigma {sigma}')
    return str(s)


def _suffix(pair_set: str) -> str:
    """Filename suffix per pair-set (`b1` -> '', `b2` -> '_b2')."""
    if pair_set == 'b1':
        return ''
    if pair_set == 'b2':
        return '_b2'
    raise ValueError(f'unknown pair_set {pair_set!r}')


def _pair_tag(sigma_a: float, sigma_b: float) -> str:
    return f'{_sigma_tag(sigma_a)}_{_sigma_tag(sigma_b)}'


# ---------------------------------------------------------------------------
# Step 1: caches
# ---------------------------------------------------------------------------

def _load_or_build_cache(sigma_a: float, sigma_b: float, idx: int,
                         seed_base: int = SEED_BASE_B1) -> np.ndarray:
    tag = _pair_tag(sigma_a, sigma_b)
    path = CACHE_DIR / f'D_shift_{tag}_test.npz'
    if path.exists():
        z = np.load(path, allow_pickle=False)
        print(f'  loaded {path.name} (seed={int(z["seed"][0])})')
        return z['X']
    seed = seed_base + idx
    rng = np.random.default_rng(seed=seed)
    mu_i = rng.normal(MU_0, TAU_0, size=N_TEST)
    pre = rng.standard_normal(size=(N_TEST, SHIFT_T)) * sigma_a + mu_i[:, None]
    post = rng.standard_normal(size=(N_TEST, N - SHIFT_T)) * sigma_b + mu_i[:, None]
    X = np.concatenate([pre, post], axis=1).astype(np.float64)
    sigma_per_t = np.concatenate([np.full(SHIFT_T, sigma_a), np.full(N - SHIFT_T, sigma_b)])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        sigma_a=np.float64(sigma_a),
        sigma_b=np.float64(sigma_b),
        sigma_per_t=sigma_per_t.astype(np.float64),
        mu_i=mu_i.astype(np.float64),
        seed=np.array([seed], dtype=np.int64),
        shift_t=np.array([SHIFT_T], dtype=np.int64),
    )
    print(f'  wrote {path.name} (seed={seed}, σ_a={sigma_a}, σ_b={sigma_b})')
    return X


def step1_caches(pairs: Iterable[Tuple[float, float]] = PAIRS_B1,
                 seed_base: int = SEED_BASE_B1) -> Dict[Tuple[float, float], np.ndarray]:
    print(f'[regime-shift] step 1: caches  (seed_base={seed_base})')
    caches: Dict[Tuple[float, float], np.ndarray] = {}
    for i, (sa, sb) in enumerate(pairs, start=1):
        caches[(sa, sb)] = _load_or_build_cache(sa, sb, i, seed_base=seed_base)
    return caches


# ---------------------------------------------------------------------------
# Step 2: per-sequence reference oracles + iid mean references
# ---------------------------------------------------------------------------

def _load_static_table(sigma: float) -> dict:
    """Load the existing static ADP table for σ ∈ {1, 10, 100} ∪ {3, 30, 300}."""
    if sigma in TRAIN_SIGMAS:
        sigma_to_idx = {1.0: 1, 10.0: 2, 100.0: 3}
        path = ORACLE_TABLES_DIR / f'D{sigma_to_idx[sigma]}_static_K{K_PROD}_J{J_PROD}.npz'
    elif sigma in OOD_SIGMAS:
        path = ORACLE_TABLES_DIR / f'D_ood_{int(sigma)}_static_K{K_PROD}_J{J_PROD}.npz'
    else:
        raise ValueError(f'no oracle table available for sigma={sigma}')
    z = np.load(path, allow_pickle=False)
    return {'C_hat': z['C_hat'], 'grids': z['grids']}


# Cached iid-σ mean reference trajectories live alongside regime-shift
# artifacts so the OOD builds run once and persist across re-runs.
IID_REF_DIR = RS_ROOT / 'iid_refs'
IID_REF_SEED_BASE = 8000


def _build_iid_mean_ref_ood(sigma: float, table: dict) -> np.ndarray:
    """Build the iid-σ mean threshold trajectory for an OOD σ ∈ {3, 30, 300}.

    Generates one fresh iid-σ cache (N_TEST sequences, n=256, μ_i ~ N(0, τ_0²)),
    runs the static σ-oracle, and returns the across-sequence mean (256,).
    """
    seed = IID_REF_SEED_BASE + int(sigma)
    rng = np.random.default_rng(seed=seed)
    mu = rng.normal(MU_0, TAU_0, size=N_TEST)
    X = rng.standard_normal(size=(N_TEST, N)) * sigma + mu[:, None]
    _, thr = static_oracle(X, sigma, table)
    mean_thr = thr.mean(axis=0)
    IID_REF_DIR.mkdir(parents=True, exist_ok=True)
    cache = IID_REF_DIR / f'iid_ref_sigma_{int(sigma)}.npz'
    np.savez_compressed(
        cache, mean=mean_thr,
        sigma=np.float64(sigma), seed=np.array([seed], dtype=np.int64),
        n_test=np.array([N_TEST], dtype=np.int64),
    )
    print(f'  built iid-σ mean ref for σ={sigma:g} '
          f'(seed={seed}, N={N_TEST}, cached at {cache.name})')
    return mean_thr


def _load_iid_mean_reference(sigmas: Iterable[float]) -> Dict[float, np.ndarray]:
    """Mean threshold trajectory of static-σ oracle on iid-σ test data.
    σ ∈ {1, 10, 100} loaded from phase5/trajectories.npz; σ ∈ {3, 30, 300}
    built (or loaded from cache) via a fresh iid-σ N=10^4 sample.
    Each entry is shape (256,) with thresholds at t = 1..255 and 0 at t = 256.
    """
    train_keys = {
        1.0:   'baseline__D_1__oracle_static__mean',
        10.0:  'baseline__D_2__oracle_static__mean',
        100.0: 'baseline__D_3__oracle_static__mean',
    }
    out: Dict[float, np.ndarray] = {}
    z_phase5 = None
    for sg in sigmas:
        if sg in TRAIN_SIGMAS:
            if z_phase5 is None:
                z_phase5 = np.load(PHASE5 / 'trajectories.npz', allow_pickle=False)
            out[sg] = z_phase5[train_keys[sg]].copy()
        elif sg in OOD_SIGMAS:
            cache = IID_REF_DIR / f'iid_ref_sigma_{int(sg)}.npz'
            if cache.exists():
                out[sg] = np.load(cache, allow_pickle=False)['mean'].copy()
                print(f'  loaded iid-σ mean ref for σ={sg:g} ({cache.name})')
            else:
                out[sg] = _build_iid_mean_ref_ood(sg, _load_static_table(sg))
        else:
            raise ValueError(f'no iid-σ reference available for sigma={sg}')
    return out


# ---------------------------------------------------------------------------
# Step 3+4: model evaluation, metrics
# ---------------------------------------------------------------------------

def _model_threshold(model, X: np.ndarray, head: str, device) -> np.ndarray:
    """Returns (N, 256) threshold trajectory for one model on one cache."""
    _, thr = model_action(model, X, head, device)
    return thr


def _safe_log(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """log of |x| with a small floor (handles negative thresholds per spec)."""
    return np.log(np.maximum(np.abs(x), floor))


def _rho_per_seq(thr_seq: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray,
                 denom_clip: float = RHO_DENOM_CLIP) -> np.ndarray:
    """Per-sequence ρ_t.

    ρ_t = (log |Ĉ_t| - log <|C*_σ_a,t|>) /
          (log <|C*_σ_b,t|> - log <|C*_σ_a,t|>).

    thr_seq: (N, n) — model's per-sequence threshold trajectory.
    ref_a: (n,) — iid-σ_a oracle's mean threshold trajectory.
    ref_b: (n,) — iid-σ_b oracle's mean threshold trajectory.

    Returns (N, n). NaN at any t where |log <|C*_σ_b,t|> − log <|C*_σ_a,t|>|
    < `denom_clip` (numerical-stability spec: 0.05).
    """
    log_a = _safe_log(ref_a)
    log_b = _safe_log(ref_b)
    log_t = _safe_log(thr_seq)
    den = log_b - log_a                                          # (n,)
    rho = np.full(thr_seq.shape, np.nan, dtype=np.float64)
    nz = np.abs(den) >= denom_clip
    rho[:, nz] = (log_t[:, nz] - log_a[None, nz]) / den[None, nz]
    return rho


def _t_half_per_seq(rho: np.ndarray, shift_t: int = SHIFT_T,
                    threshold: float = 0.5) -> np.ndarray:
    """Per-sequence t_{1/2}: smallest Δt ∈ [1, n - shift_t] such that
    ρ_{shift_t + Δt - 1} ≥ threshold (0-indexed t = shift_t + Δt - 1).
    Returns Δt ∈ [1, n - shift_t]; n - shift_t when never crossed.
    """
    N, n = rho.shape
    n_post = n - shift_t                                         # = 128
    post = rho[:, shift_t:]                                      # (N, n_post): t = 129..256, 0-indexed
    crossed = post >= threshold
    any_cross = crossed.any(axis=1)
    first_idx = crossed.argmax(axis=1) + 1                       # Δt is 1-indexed
    return np.where(any_cross, first_idx, n_post).astype(np.int64)


def _t_half_population(thr_mean: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray,
                       shift_t: int = SHIFT_T, threshold: float = 0.5) -> int:
    """t_{1/2} computed from the population-mean threshold trajectory."""
    rho = _rho_per_seq(thr_mean[None, :], ref_a, ref_b)[0]       # (n,)
    n = rho.shape[0]
    n_post = n - shift_t
    for delta in range(1, n_post + 1):
        idx = shift_t + delta - 1
        if not np.isnan(rho[idx]) and rho[idx] >= threshold:
            return delta
    return n_post


def _tracking_error(thr_seq: np.ndarray, ref_per_seq: np.ndarray,
                    t_lo: int, t_hi: int) -> float:
    """Mean |Ĉ_t - C*_{σ, t}| over t ∈ [t_lo, t_hi] (1-indexed, inclusive),
    averaged over sequences."""
    lo = t_lo - 1                                                # 0-indexed
    hi = t_hi - 1
    return float(np.mean(np.abs(thr_seq[:, lo:hi + 1] - ref_per_seq[:, lo:hi + 1])))


# ---------------------------------------------------------------------------
# Shift-cache baselines (added per "include all baselines" request)
# ---------------------------------------------------------------------------

def _build_random_oracle_table(distribution: str) -> dict:
    """Mirror of run_ood._build_random_oracle_table."""
    name = (
        'D_disc_K256_J64.npz' if distribution == 'D_disc'
        else 'D_logu_K256_J64_Js64.npz'
    )
    return load_random_table(ORACLE_TABLES_DIR / name)


def _baseline_mean_std(threshold: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std over sequences, ignoring NaN/inf entries (mirrors run_ood).
    threshold: (N, n) — returns (mean(n,), std(n,))."""
    finite = np.where(np.isfinite(threshold), threshold, np.nan)
    return np.nanmean(finite, axis=0), np.nanstd(finite, axis=0)


def _compute_shift_baselines(caches: Dict[Tuple[float, float], np.ndarray],
                             static_tables: Dict[float, dict],
                             random_tables: Dict[str, dict],
                             eta: np.ndarray,
                             sigma_grid_disc: np.ndarray,
                             log_omega_disc: np.ndarray,
                             sigma_grid_logu: np.ndarray,
                             log_omega_logu: np.ndarray) -> Dict[str, np.ndarray]:
    """For each shift cache, compute the eight baselines that work natively
    on a switching-σ cache (skips plug-in / prior-only / myopic — those
    require a single known σ).

    Returns a dict of mean/std arrays keyed:
      baseline__shift_<tag>__<bname>__mean / __std
    where bname ∈ {oracle_static_a, oracle_static_b, random_oracle_disc,
                   random_oracle_logu, MAP_sigma_disc, MAP_sigma_logu,
                   MLE_sigma, secretary}.
    """
    out: Dict[str, np.ndarray] = {}
    for (sa, sb), X in caches.items():
        tag = _pair_tag(sa, sb)
        print(f'  baselines for shift_{tag} '
              f'(σ_a={sa:g}, σ_b={sb:g})')

        # Two known-σ static oracles (anchor lines).
        _, thr = static_oracle(X, sa, static_tables[sa])
        out[f'baseline__shift_{tag}__oracle_static_a__mean'], \
            out[f'baseline__shift_{tag}__oracle_static_a__std'] = \
                _baseline_mean_std(thr)
        _, thr = static_oracle(X, sb, static_tables[sb])
        out[f'baseline__shift_{tag}__oracle_static_b__mean'], \
            out[f'baseline__shift_{tag}__oracle_static_b__std'] = \
                _baseline_mean_std(thr)

        # Random-ADP oracles under the two training priors.
        _, thr = random_oracle(X, random_tables['D_disc'])
        out[f'baseline__shift_{tag}__random_oracle_disc__mean'], \
            out[f'baseline__shift_{tag}__random_oracle_disc__std'] = \
                _baseline_mean_std(thr)
        _, thr = random_oracle(X, random_tables['D_logu'])
        out[f'baseline__shift_{tag}__random_oracle_logu__mean'], \
            out[f'baseline__shift_{tag}__random_oracle_logu__std'] = \
                _baseline_mean_std(thr)

        # MAP-σ plug-in baselines (D_disc and D_logu priors).
        _, thr = map_sigma_plugin(X, sigma_grid_disc, log_omega_disc, eta)
        out[f'baseline__shift_{tag}__MAP_sigma_disc__mean'], \
            out[f'baseline__shift_{tag}__MAP_sigma_disc__std'] = \
                _baseline_mean_std(thr)
        _, thr = map_sigma_plugin(X, sigma_grid_logu, log_omega_logu, eta)
        out[f'baseline__shift_{tag}__MAP_sigma_logu__mean'], \
            out[f'baseline__shift_{tag}__MAP_sigma_logu__std'] = \
                _baseline_mean_std(thr)

        # MLE-σ plug-in (no prior).
        _, thr = mle_sigma_plugin(X, eta)
        out[f'baseline__shift_{tag}__MLE_sigma__mean'], \
            out[f'baseline__shift_{tag}__MLE_sigma__std'] = \
                _baseline_mean_std(thr)

        # Secretary (scale-free).
        _, thr = secretary(X)
        out[f'baseline__shift_{tag}__secretary__mean'], \
            out[f'baseline__shift_{tag}__secretary__std'] = \
                _baseline_mean_std(thr)

    return out


def step3_eval(caches: Dict[Tuple[float, float], np.ndarray],
               pair_set: str = 'b1') -> dict:
    """Evaluate each (model, pair). Saves trajectories{,_b2}.npz,
    t_half_per_seq{,_b2}.npz, raw{,_b2}.json. Returns the raw dict."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    suffix = _suffix(pair_set)
    print(f'[regime-shift] step 2-4: eval (device={device}, pair_set={pair_set})')

    sigmas_used = sorted({s for pair in caches.keys() for s in pair})
    static_tables = {s: _load_static_table(s) for s in sigmas_used}
    iid_ref = _load_iid_mean_reference(sigmas_used)              # σ -> (256,)

    # Baseline machinery (random-ADP oracles + MAP-σ plug-ins + ...).
    print('[regime-shift] loading random-ADP oracle tables (disc, logu)')
    random_tables = {
        'D_disc': _build_random_oracle_table('D_disc'),
        'D_logu': _build_random_oracle_table('D_logu'),
    }
    eta = compute_eta(N)
    sigma_grid_disc = np.array([1.0, 10.0, 100.0])
    log_omega_disc = np.zeros(3)
    sigma_grid_logu, log_omega_logu = make_logu_grid('logu', J_sigma=64)

    # Per-sequence reference threshold per pair (using regime-shift cache).
    print('  computing per-sequence reference oracles')
    refs_per_seq: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
    for (sa, sb), X in caches.items():
        _, thr_a = static_oracle(X, sa, static_tables[sa])
        _, thr_b = static_oracle(X, sb, static_tables[sb])
        refs_per_seq[(sa, sb)] = {'ref_a': thr_a, 'ref_b': thr_b}

    # Per-cell containers
    raw: Dict[str, Dict[str, dict]] = {}
    traj_payload: Dict[str, np.ndarray] = {}                     # for trajectories.npz
    t_half_payload: Dict[str, np.ndarray] = {}                   # for t_half_per_seq.npz

    for run_name in MODELS:
        print(f'\n[regime-shift] === {run_name} ===')
        actual = SIG2_SWAP.get(run_name, run_name)
        if actual != run_name:
            print(f'  (loading {actual} for {run_name})')
        model, head, _ = load_checkpoint(actual, which='best')
        model = model.to(device).eval()

        raw[run_name] = {}
        for (sa, sb), X in caches.items():
            tag = _pair_tag(sa, sb)
            t0 = time.perf_counter()
            thr_seq = _model_threshold(model, X, head, device)   # (N, 256)
            dt = time.perf_counter() - t0
            print(f'  pair=({sa:>5g},{sb:<5g})  forward {dt:.1f}s', end='', flush=True)

            # Per-sequence references for tracking errors.
            ref_a_seq = refs_per_seq[(sa, sb)]['ref_a']           # (N, 256), σ_a oracle on cache
            ref_b_seq = refs_per_seq[(sa, sb)]['ref_b']           # (N, 256), σ_b oracle on cache

            # Population mean references (for ρ_t).
            ref_a_pop = iid_ref[sa]                               # (256,)
            ref_b_pop = iid_ref[sb]                               # (256,)

            # Trajectories.
            thr_mean = thr_seq.mean(axis=0)                       # (256,)
            thr_std = thr_seq.std(axis=0)                         # (256,)

            # ρ_t per sequence (uses iid-σ mean references for the denominator).
            rho_seq = _rho_per_seq(thr_seq, ref_a_pop, ref_b_pop)  # (N, 256)
            rho_mean = np.nanmean(rho_seq, axis=0)                # (256,)
            rho_std = np.nanstd(rho_seq, axis=0)                  # (256,)

            # t_{1/2} per sequence.
            t_half_seq = _t_half_per_seq(rho_seq, SHIFT_T, threshold=0.5)
            t_half_pop = _t_half_population(thr_mean, ref_a_pop, ref_b_pop,
                                             SHIFT_T, threshold=0.5)
            never_crossed = int((t_half_seq == (N - SHIFT_T)).sum())

            # Tracking errors.
            pre_err = _tracking_error(thr_seq, ref_a_seq, *PRE_RANGE)
            post_err = _tracking_error(thr_seq, ref_b_seq, *POST_RANGE)

            # ρ at t=200 ("settled" post-shift; iid references still have full
            # σ-magnitude — they collapse to ~0 in the final ~10 steps as the
            # horizon shrinks, which would make log ρ degenerate at t=255).
            RHO_PROBE_T = 200                                     # 1-indexed
            rho_settled_seq = rho_seq[:, RHO_PROBE_T - 1]
            rho_pop_traj = _rho_per_seq(thr_mean[None, :],
                                         ref_a_pop, ref_b_pop)[0]
            rho_settled_population = float(rho_pop_traj[RHO_PROBE_T - 1])
            rho_settled_median = float(np.nanmedian(rho_settled_seq))
            rho_settled_p10 = float(np.nanpercentile(rho_settled_seq, 10))
            rho_settled_p90 = float(np.nanpercentile(rho_settled_seq, 90))

            # ρ_255 (final adaptation, per B.2 spec). ρ at the last decision
            # step. NaN when the t=255 reference denominator is below the
            # 0.05 clip — propagated from _rho_per_seq.
            RHO_FINAL_T = 255                                     # 1-indexed
            rho_final_seq = rho_seq[:, RHO_FINAL_T - 1]
            rho_final_clipped = bool(np.all(np.isnan(rho_final_seq)))
            if rho_final_clipped:
                rho_final_population = float('nan')
                rho_final_median = float('nan')
                rho_final_p10 = float('nan')
                rho_final_p90 = float('nan')
            else:
                rho_final_population = float(rho_pop_traj[RHO_FINAL_T - 1])
                rho_final_median = float(np.nanmedian(rho_final_seq))
                rho_final_p10 = float(np.nanpercentile(rho_final_seq, 10))
                rho_final_p90 = float(np.nanpercentile(rho_final_seq, 90))

            raw[run_name][tag] = {
                'sigma_a': sa,
                'sigma_b': sb,
                'pre_shift_tracking_error':  pre_err,
                'post_shift_tracking_error': post_err,
                't_half_population_mean_traj': int(t_half_pop),
                't_half_per_seq_mean':       float(t_half_seq.mean()),
                't_half_per_seq_median':     float(np.median(t_half_seq)),
                't_half_per_seq_p10':        float(np.percentile(t_half_seq, 10)),
                't_half_per_seq_p90':        float(np.percentile(t_half_seq, 90)),
                't_half_never_crossed':       never_crossed,
                'rho_settled_t':              RHO_PROBE_T,
                'rho_settled_population':     rho_settled_population,
                'rho_settled_per_seq_median': rho_settled_median,
                'rho_settled_per_seq_p10':    rho_settled_p10,
                'rho_settled_per_seq_p90':    rho_settled_p90,
                'rho_final_t':                RHO_FINAL_T,
                'rho_final_clipped':          rho_final_clipped,
                'rho_final_population':       rho_final_population,
                'rho_final_per_seq_median':   rho_final_median,
                'rho_final_per_seq_p10':      rho_final_p10,
                'rho_final_per_seq_p90':      rho_final_p90,
            }
            rho255_str = (f'{rho_final_population:.3f}'
                          if not rho_final_clipped else 'NaN(clipped)')
            print(f'  pre_err={pre_err:.3f}  post_err={post_err:.3f}  '
                  f't1/2={t_half_seq.mean():.1f}  ρ_200={rho_settled_population:.3f}  '
                  f'ρ_255={rho255_str}')

            # Save into npz payloads.
            traj_payload[f'{run_name}__shift_{tag}__thr_mean'] = thr_mean
            traj_payload[f'{run_name}__shift_{tag}__thr_std'] = thr_std
            traj_payload[f'{run_name}__shift_{tag}__rho_mean'] = rho_mean
            traj_payload[f'{run_name}__shift_{tag}__rho_std'] = rho_std
            t_half_payload[f'{run_name}__shift_{tag}__t_half'] = t_half_seq

        del model
        torch.cuda.empty_cache()

    # Save references too — needed for plotting reference lines.
    for (sa, sb), X in caches.items():
        tag = _pair_tag(sa, sb)
        traj_payload[f'shift_{tag}__ref_a_iid'] = iid_ref[sa]
        traj_payload[f'shift_{tag}__ref_b_iid'] = iid_ref[sb]

    # Baseline trajectories (per cache, model-independent).
    print('[regime-shift] computing baseline trajectories per cache')
    t0 = time.perf_counter()
    baseline_traj = _compute_shift_baselines(
        caches, static_tables, random_tables, eta,
        sigma_grid_disc, log_omega_disc,
        sigma_grid_logu, log_omega_logu,
    )
    traj_payload.update(baseline_traj)
    print(f'  baselines done in {time.perf_counter() - t0:.1f}s')

    RS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RS_ROOT / f'trajectories{suffix}.npz', **traj_payload)
    np.savez_compressed(RS_ROOT / f't_half_per_seq{suffix}.npz', **t_half_payload)

    pairs_list = list(caches.keys())
    seed_base = SEED_BASE_B2 if pair_set == 'b2' else SEED_BASE_B1
    raw_meta = {
        'pair_set': pair_set,
        'pairs': [{'sigma_a': a, 'sigma_b': b} for a, b in pairs_list],
        'shift_t': SHIFT_T,
        'pre_range': PRE_RANGE,
        'post_range': POST_RANGE,
        'rho_denom_clip': RHO_DENOM_CLIP,
        'n_test': N_TEST,
        'seeds': {_pair_tag(a, b): seed_base + i
                  for i, (a, b) in enumerate(pairs_list, start=1)},
        'sig2_swap': SIG2_SWAP,
    }
    out_payload = {'meta': raw_meta, 'metrics': raw}
    (RS_ROOT / f'raw{suffix}.json').write_text(json.dumps(out_payload, indent=2))

    return out_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summary(label: str, raw: dict, pairs: List[Tuple[float, float]]) -> None:
    print('\n' + '=' * 60)
    print(f'Summary ({label}): mean t_{{1/2}}, post-shift tracking error, ρ_255')
    print('=' * 60)
    for run in MODELS:
        for (sa, sb) in pairs:
            tag = _pair_tag(sa, sb)
            r = raw['metrics'][run][tag]
            rho255 = ('NaN(clip)' if r['rho_final_clipped']
                      else f'{r["rho_final_population"]:+.3f}')
            print(f'  {run:14s}  ({sa:>4g}->{sb:<4g})  '
                  f't1/2={r["t_half_per_seq_mean"]:>5.1f}  '
                  f'post={r["post_shift_tracking_error"]:>6.2f}  '
                  f'ρ_200={r["rho_settled_population"]:+.3f}  '
                  f'ρ_255={rho255}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('--pairs', choices=['b1', 'b2', 'both'], default='b1',
                        help='Pair set: b1 = training-prior σ ∈ {1,10,100}, '
                             'b2 = OOD σ ∈ {3,30,300}, both = run both back-to-back.')
    args = parser.parse_args()

    print('=' * 60)
    print(f'Experiment B: per-sequence regime shift  (pairs={args.pairs})')
    print('=' * 60)

    if args.pairs in ('b1', 'both'):
        caches_b1 = step1_caches(PAIRS_B1, SEED_BASE_B1)
        raw_b1 = step3_eval(caches_b1, pair_set='b1')
        _print_summary('B.1', raw_b1, PAIRS_B1)

    if args.pairs in ('b2', 'both'):
        caches_b2 = step1_caches(PAIRS_B2, SEED_BASE_B2)
        raw_b2 = step3_eval(caches_b2, pair_set='b2')
        _print_summary('B.2', raw_b2, PAIRS_B2)

    return 0


if __name__ == '__main__':
    sys.exit(main())
