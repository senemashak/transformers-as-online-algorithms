"""
Pre-Step-5 data separation checks.

Three explicit checks, hash-based:

  Check A: val_set ∩ training_stream = ∅, per training distribution.
  Check B: test_set ∩ val_set = ∅ and test_set ∩ training_stream = ∅.
  Check C: per-cell test-cache routing for the eval driver — verify the
           test cache files are present and document which goes where.

Hashing: SHA-256 of each sequence's float64 byte representation. With
continuous-distribution iid samples and IEEE754 float64, expected
collisions are essentially zero across distinct seeds; finding any in a
finite sample is informative.

For the training-stream side we sample 100k sequences per (distribution,
supervision) — the user-specified probabilistic cap. The training streams
use seed `1000 + run_idx`; val uses `42*100 + offset`; test uses
`43*100 + offset` (distinct seed scheme, see train/configs.py and
data/streaming.py).

Writes v3/results/phase4/data_separation_check.md.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent
sys.path.insert(0, str(V3_ROOT))

from data.distributions import (
    ALL_DISTRIBUTIONS,
    STATIC_DISTRIBUTIONS,
    sample,
)
from data.streaming import CACHE_DIR_DEFAULT, cache_path, load_cache
from train.configs import RESULTS_PHASE4, make_run_configs


N_TRAIN_SAMPLE = 100_000   # sequences per (distribution, supervision)
TRAIN_BATCH = 1024
RESULTS_DIR = RESULTS_PHASE4


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def hash_row(x: np.ndarray) -> str:
    """SHA-256 of a single float64 sequence's byte representation."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def hash_array(X: np.ndarray) -> set[str]:
    return {hash_row(row) for row in X}


# ---------------------------------------------------------------------------
# Cache hash sets (val + test)
# ---------------------------------------------------------------------------

def build_val_hashes() -> dict[str, set[str]]:
    out = {}
    for dist in ALL_DISTRIBUTIONS:
        X, _, _ = load_cache(dist, 'val')
        out[dist] = hash_array(X)
    return out


def build_test_hashes() -> dict[str, set[str]]:
    out = {}
    for regime in STATIC_DISTRIBUTIONS:
        X, _, _ = load_cache(regime, 'test')
        out[regime] = hash_array(X)
    return out


# ---------------------------------------------------------------------------
# Training-stream sample
# ---------------------------------------------------------------------------

def stream_seed_for_run(run_name: str) -> int:
    """Recover the training seed for a run from its config."""
    for stage in ('full',):
        for cfg in make_run_configs(stage):
            if cfg.run_name == run_name:
                return cfg.seed
    raise KeyError(run_name)


def sample_training_stream(distribution: str, seed: int, n_seq: int) -> set[str]:
    """Hash the first `n_seq` sequences from the training stream of
    distribution+seed (matches what the trainer saw at run start)."""
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    while len(seen) < n_seq:
        take = min(TRAIN_BATCH, n_seq - len(seen))
        X, _, _ = sample(distribution, take, rng)
        seen.update(hash_row(row) for row in X)
    return seen


# ---------------------------------------------------------------------------
# Check A — val ∩ training_stream
# ---------------------------------------------------------------------------

def check_a(val_hashes: dict[str, set[str]]) -> list[dict]:
    rows = []
    for dist in ALL_DISTRIBUTIONS:
        for sup in ('cv', 'act'):
            run_name = f'{dist}_{sup}'
            seed = stream_seed_for_run(run_name)
            t0 = time.perf_counter()
            train_hashes = sample_training_stream(dist, seed, N_TRAIN_SAMPLE)
            wall = time.perf_counter() - t0
            overlap = val_hashes[dist] & train_hashes
            rows.append({
                'distribution': dist,
                'run_name': run_name,
                'seed': seed,
                'val_set_size': len(val_hashes[dist]),
                'training_sample_size': len(train_hashes),
                'overlap': len(overlap),
                'wall_s': wall,
            })
    return rows


# ---------------------------------------------------------------------------
# Check B — test ∩ val and test ∩ training_stream
# ---------------------------------------------------------------------------

def check_b_test_vs_val(test_hashes, val_hashes):
    rows = []
    for regime in STATIC_DISTRIBUTIONS:
        for dist in ALL_DISTRIBUTIONS:
            overlap = test_hashes[regime] & val_hashes[dist]
            rows.append({
                'test_regime': regime,
                'val_distribution': dist,
                'test_set_size': len(test_hashes[regime]),
                'val_set_size': len(val_hashes[dist]),
                'overlap': len(overlap),
            })
    return rows


def check_b_test_vs_training(test_hashes):
    rows = []
    for regime in STATIC_DISTRIBUTIONS:
        for dist in ALL_DISTRIBUTIONS:
            for sup in ('cv', 'act'):
                run_name = f'{dist}_{sup}'
                seed = stream_seed_for_run(run_name)
                t0 = time.perf_counter()
                train_hashes = sample_training_stream(dist, seed, N_TRAIN_SAMPLE)
                wall = time.perf_counter() - t0
                overlap = test_hashes[regime] & train_hashes
                rows.append({
                    'test_regime': regime,
                    'train_run': run_name,
                    'seed': seed,
                    'test_set_size': len(test_hashes[regime]),
                    'training_sample_size': len(train_hashes),
                    'overlap': len(overlap),
                    'wall_s': wall,
                })
    return rows


# ---------------------------------------------------------------------------
# Check C — eval routing
# ---------------------------------------------------------------------------

def check_c() -> dict:
    """Catalog the cache file paths the eval driver should use."""
    out = {
        'payoff_matrix_test_caches': {},
        'per_sigma_payoff_caches': {},
        'val_caches': {},
    }
    for regime in STATIC_DISTRIBUTIONS:
        p = cache_path(regime, 'test', CACHE_DIR_DEFAULT)
        out['payoff_matrix_test_caches'][regime] = {
            'path': str(p), 'exists': p.exists(),
            'size_bytes': p.stat().st_size if p.exists() else 0,
        }
    for dist in ALL_DISTRIBUTIONS:
        # val caches (used during training; pre-existing from Step 3)
        pv = cache_path(dist, 'val', CACHE_DIR_DEFAULT)
        out['val_caches'][dist] = {
            'path': str(pv), 'exists': pv.exists(),
            'size_bytes': pv.stat().st_size if pv.exists() else 0,
        }
        # For random-variance distributions the per-σ payoff figure (Step 5
        # figure 4) needs a test set drawn from the *training* distribution.
        # We use the val cache for that (it's drawn from the training
        # distribution, seed 42*100 + offset; the existing cache satisfies
        # the spec — no separate "training-distribution test set" file).
        if dist in ('D_disc', 'D_logu'):
            out['per_sigma_payoff_caches'][dist] = {
                'note': 'Use the val cache as the per-σ test set '
                        '(drawn from training distribution).',
                'path': str(pv), 'exists': pv.exists(),
            }
    return out


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_md(
    out: Path, *, a_rows, b_val_rows, b_train_rows, c_routing,
    val_sizes, test_sizes,
) -> None:
    lines: list[str] = []
    lines.append('# Pre-Step-5 data-separation check\n\n')
    lines.append(f'Generated: {time.strftime("%Y-%m-%dT%H:%M:%S")}\n\n')
    lines.append(
        f'Hashing: SHA-256 of each float64 sequence\'s byte representation.\n'
        f'Per Step-5 prelude, training streams are sampled probabilistically '
        f'at {N_TRAIN_SAMPLE:,} sequences per (distribution, supervision) — '
        'the same RNG state the trainer started from, so this is the start '
        'of each stream rather than uniformly across; with continuous iid '
        'sampling the collision probability is uniform across the run, and '
        'finding any here would already be diagnostic.\n\n'
    )

    # --- Check A ---
    lines.append('## Check A — val ∩ training stream\n\n')
    lines.append(
        '| training distribution | run | seed | val set size | training sample size | overlap | wall (s) | result |\n'
        '|---|---|---:|---:|---:|---:|---:|---|\n'
    )
    a_pass = True
    for r in a_rows:
        passed = (r['overlap'] == 0)
        a_pass &= passed
        lines.append(
            f'| {r["distribution"]} | {r["run_name"]} | {r["seed"]} | '
            f'{r["val_set_size"]:,} | {r["training_sample_size"]:,} | '
            f'{r["overlap"]} | {r["wall_s"]:.2f} | '
            f'{"PASS" if passed else "FAIL"} |\n'
        )
    lines.append(f'\nCheck A overall: **{"PASS" if a_pass else "FAIL"}** '
                 '(expected: zero overlap on every row).\n\n')

    # --- Check B (test vs val) ---
    lines.append('## Check B(i) — test ∩ val\n\n')
    lines.append(
        '| test regime | val distribution | test size | val size | overlap | result |\n'
        '|---|---|---:|---:|---:|---|\n'
    )
    b_val_pass = True
    for r in b_val_rows:
        passed = (r['overlap'] == 0)
        b_val_pass &= passed
        lines.append(
            f'| {r["test_regime"]} | {r["val_distribution"]} | '
            f'{r["test_set_size"]:,} | {r["val_set_size"]:,} | '
            f'{r["overlap"]} | {"PASS" if passed else "FAIL"} |\n'
        )
    lines.append(f'\nCheck B(i) overall: **{"PASS" if b_val_pass else "FAIL"}**.\n\n')

    # --- Check B (test vs training) ---
    lines.append('## Check B(ii) — test ∩ training stream\n\n')
    lines.append(
        '| test regime | training run | seed | test size | training sample size | overlap | wall (s) | result |\n'
        '|---|---|---:|---:|---:|---:|---:|---|\n'
    )
    b_train_pass = True
    for r in b_train_rows:
        passed = (r['overlap'] == 0)
        b_train_pass &= passed
        lines.append(
            f'| {r["test_regime"]} | {r["train_run"]} | {r["seed"]} | '
            f'{r["test_set_size"]:,} | {r["training_sample_size"]:,} | '
            f'{r["overlap"]} | {r["wall_s"]:.2f} | '
            f'{"PASS" if passed else "FAIL"} |\n'
        )
    lines.append(f'\nCheck B(ii) overall: **{"PASS" if b_train_pass else "FAIL"}**.\n\n')

    # --- Check C ---
    lines.append('## Check C — eval cache routing\n\n')
    lines.append(
        'The Step-5 payoff matrix evaluates every model on the three test '
        'regimes D_1, D_2, D_3 (each: 10⁴ sequences from σ ∈ {1, 10, 100}, '
        'seed 43-derived). The per-σ payoff breakdown for random-variance '
        'models uses the corresponding training-distribution val cache '
        '(seed 42-derived) as the σ-binned test set — those caches already '
        'exist from Step 3.\n\n'
    )
    lines.append('### Payoff-matrix test caches (used for the 5 × 3 cell grid)\n\n')
    lines.append('| test regime | path | exists | size (bytes) |\n|---|---|---|---:|\n')
    for regime, info in c_routing['payoff_matrix_test_caches'].items():
        lines.append(
            f'| {regime} | `{info["path"]}` | '
            f'{"yes" if info["exists"] else "**NO**"} | '
            f'{info["size_bytes"]:,} |\n'
        )
    lines.append('\n### Per-σ payoff caches (Step-5 figure 4, random-variance only)\n\n')
    lines.append('| training distribution | path (val cache) | exists |\n|---|---|---|\n')
    for dist, info in c_routing['per_sigma_payoff_caches'].items():
        lines.append(
            f'| {dist} | `{info["path"]}` | '
            f'{"yes" if info["exists"] else "**NO**"} |\n'
        )
    lines.append('\n### Val caches (background, used during training)\n\n')
    lines.append('| training distribution | path | exists | size (bytes) |\n|---|---|---|---:|\n')
    for dist, info in c_routing['val_caches'].items():
        lines.append(
            f'| {dist} | `{info["path"]}` | '
            f'{"yes" if info["exists"] else "**NO**"} | '
            f'{info["size_bytes"]:,} |\n'
        )

    all_pass = a_pass and b_val_pass and b_train_pass
    lines.append(f'\n## Overall verdict: **{"PASS" if all_pass else "FAIL"}**\n')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(lines))


def main() -> int:
    print('Building val + test hash sets...')
    val_hashes = build_val_hashes()
    test_hashes = build_test_hashes()
    val_sizes = {k: len(v) for k, v in val_hashes.items()}
    test_sizes = {k: len(v) for k, v in test_hashes.items()}
    print(f'  val sizes: {val_sizes}')
    print(f'  test sizes: {test_sizes}')

    print('\nCheck A: val ∩ training stream...')
    a_rows = check_a(val_hashes)
    for r in a_rows:
        print(f'  {r["run_name"]:14s}  overlap={r["overlap"]}  '
              f'(checked {r["training_sample_size"]:,} train, {r["wall_s"]:.1f}s)')

    print('\nCheck B(i): test ∩ val...')
    b_val_rows = check_b_test_vs_val(test_hashes, val_hashes)
    for r in b_val_rows:
        print(f'  {r["test_regime"]:5s} ∩ val[{r["val_distribution"]:>6s}]  overlap={r["overlap"]}')

    print('\nCheck B(ii): test ∩ training stream...')
    b_train_rows = check_b_test_vs_training(test_hashes)
    for r in b_train_rows:
        print(f'  {r["test_regime"]:5s} ∩ train[{r["train_run"]:14s}]  overlap={r["overlap"]}')

    print('\nCheck C: routing...')
    c_routing = check_c()

    out = RESULTS_DIR / 'data_separation_check.md'
    write_md(
        out, a_rows=a_rows, b_val_rows=b_val_rows, b_train_rows=b_train_rows,
        c_routing=c_routing, val_sizes=val_sizes, test_sizes=test_sizes,
    )
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
