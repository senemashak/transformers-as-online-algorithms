"""
V3 batch streamer for training.

stream_batches(distribution, batch_size, table, rng) yields infinite
batches of (X, sigma_i, y_cv, y_act, cv_mask, act_mask).

Mask shapes are (n,):
    cv_mask:  False at t=n always; for D_disc/D_logu also False at t=1.
              True elsewhere.
    act_mask: False at t=n only.

Cache helpers `build_cache` and `load_cache` populate / read the
pre-generated val and test sets in `v3/data/cache/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from data.distributions import (
    ALL_DISTRIBUTIONS,
    N,
    RANDOM_DISTRIBUTIONS,
    STATIC_DISTRIBUTIONS,
    sample,
    static_sigma,
)
from data.labeling import label_random, label_static


def make_cv_mask(distribution: str) -> np.ndarray:
    """(n,) bool mask. False at t=n; for random distributions also False at t=1."""
    if distribution not in ALL_DISTRIBUTIONS:
        raise ValueError(f"unknown distribution: {distribution!r}")
    mask = np.ones(N, dtype=bool)
    mask[N - 1] = False
    if distribution in RANDOM_DISTRIBUTIONS:
        mask[0] = False
    return mask


def make_act_mask() -> np.ndarray:
    """(n,) bool mask. False at t=n only."""
    mask = np.ones(N, dtype=bool)
    mask[N - 1] = False
    return mask


def _make_labeler(distribution: str, table: dict):
    if distribution in STATIC_DISTRIBUTIONS:
        sigma = static_sigma(distribution)
        return lambda X: label_static(X, sigma, table)
    return lambda X: label_random(X, table)


def stream_batches(
    distribution: str,
    batch_size: int,
    table: dict,
    rng: np.random.Generator,
    compute_labels: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Infinite generator of (X, sigma_i, y_cv, y_act, cv_mask, act_mask).

    When `compute_labels=False`, y_cv and y_act are returned as `None` and the
    CPU labeling step is skipped entirely. The trainer uses this for D_disc
    and D_logu (random distributions), where it labels on GPU instead — the
    CPU bilinear lookup over n stages is the per-step bottleneck on those
    distributions, and skipping it shaves ~100 ms per batch.
    """
    cv_mask = make_cv_mask(distribution)
    act_mask = make_act_mask()
    if compute_labels:
        label_fn = _make_labeler(distribution, table)
    else:
        label_fn = None
    while True:
        X, sigma_i, _ = sample(distribution, batch_size, rng)
        if label_fn is not None:
            y_cv, y_act = label_fn(X)
        else:
            y_cv, y_act = None, None
        yield X, sigma_i, y_cv, y_act, cv_mask, act_mask


# ---------------------------------------------------------------------------
# Pre-generated val / test caches
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT = Path(__file__).resolve().parent / 'cache'

VAL_SEED = 42
TEST_SEED = 43
N_VAL = 10_000
N_TEST = 10_000


def cache_path(
    distribution: str, kind: str, cache_dir: Path = CACHE_DIR_DEFAULT,
) -> Path:
    """`{distribution}_{kind}.npz`. kind is 'val' or 'test'."""
    if kind not in ('val', 'test'):
        raise ValueError(f"kind must be 'val' or 'test', got {kind!r}")
    return cache_dir / f'{distribution}_{kind}.npz'


def build_cache_one(
    distribution: str, kind: str, N_seq: int, seed: int,
    cache_dir: Path = CACHE_DIR_DEFAULT,
) -> Path:
    """Sample (X, sigma_i, mu_i), save to cache, return path."""
    rng = np.random.default_rng(seed)
    X, sigma_i, mu_i = sample(distribution, N_seq, rng)
    path = cache_path(distribution, kind, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float64),
        sigma_i=sigma_i.astype(np.float64),
        mu_i=mu_i.astype(np.float64),
    )
    return path


def load_cache(
    distribution: str, kind: str, cache_dir: Path = CACHE_DIR_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, sigma_i, mu_i) from cache."""
    z = np.load(cache_path(distribution, kind, cache_dir), allow_pickle=False)
    return z['X'], z['sigma_i'], z['mu_i']


def build_all_caches(cache_dir: Path = CACHE_DIR_DEFAULT) -> None:
    """Build 5 val sets (one per training distribution) + 3 test sets
    (per regime). Each set gets its own deterministic seed derived from
    the global VAL_SEED / TEST_SEED plus a fixed per-distribution offset.
    """
    seed_offsets = {
        'D_1': 1, 'D_2': 2, 'D_3': 3, 'D_disc': 4, 'D_logu': 5,
    }
    for dist in ALL_DISTRIBUTIONS:
        path = build_cache_one(
            dist, 'val', N_VAL, seed=VAL_SEED * 100 + seed_offsets[dist],
            cache_dir=cache_dir,
        )
        print(f'  wrote {path.name}')
    for dist in STATIC_DISTRIBUTIONS:
        path = build_cache_one(
            dist, 'test', N_TEST, seed=TEST_SEED * 100 + seed_offsets[dist],
            cache_dir=cache_dir,
        )
        print(f'  wrote {path.name}')


if __name__ == '__main__':
    build_all_caches()
