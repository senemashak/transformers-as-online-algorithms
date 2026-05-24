"""
Variant-aware paths for the probing pipeline.

The probe scripts (`probe_data`, `probe_cache`, `probe_train`,
`probe_run`, `probe_render`) all read paths from this module, so adding
a new training distribution / model to probe is a single-line addition
to the `VARIANTS` table below plus an environment variable at launch:

    PROBE_VARIANT=disc-act python -m interp.probe_run --run D_disc_act ...

Backwards compatibility: the default is `logu-act`, which matches the
existing on-disk layout of `results/probing/logu-act/`.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
V3_ROOT = CODE_ROOT.parent


# Each variant pins:
#   distribution: the sampling distribution name (used by data.distributions.sample)
#   sigma_grid_kind: argument to oracle.random_adp_torch.make_sigma_grid
#   random_adp_table: filename inside oracle/tables/ for the random-ADP table
#   probe_seeds: train / val / test seeds (disjoint across variants and other caches)
VARIANTS = {
    'logu-act': {
        'distribution': 'D_logu',
        'sigma_grid_kind': 'logu',
        'sigma_grid_J': 64,
        'random_adp_table': 'D_logu_K256_J64_Js64.npz',
        'probe_seeds': {'train': 7001, 'val': 7002, 'test': 7003},
    },
    'disc-act': {
        'distribution': 'D_disc',
        'sigma_grid_kind': 'disc',
        'sigma_grid_J': 3,                       # discrete prior has 3 atoms {1,10,100}
        'random_adp_table': 'D_disc_K256_J64.npz',
        'probe_seeds': {'train': 8001, 'val': 8002, 'test': 8003},
    },
}

PROBE_VARIANT = os.environ.get('PROBE_VARIANT', 'logu-act')
if PROBE_VARIANT not in VARIANTS:
    raise ValueError(
        f'PROBE_VARIANT={PROBE_VARIANT!r} not recognized; '
        f'known: {sorted(VARIANTS)}'
    )

_v = VARIANTS[PROBE_VARIANT]
DISTRIBUTION = _v['distribution']
SIGMA_GRID_KIND = _v['sigma_grid_kind']
SIGMA_GRID_J = _v['sigma_grid_J']
RANDOM_ADP_TABLE = _v['random_adp_table']
PROBE_SEEDS = dict(_v['probe_seeds'])


# Directory roots. All probe outputs for a given variant live under one tree.
RESULTS_ROOT = V3_ROOT / 'results' / 'probing' / PROBE_VARIANT
DATA_ROOT = RESULTS_ROOT / 'data'
CACHE_ROOT = RESULTS_ROOT / 'cache'
RUNS_ROOT = RESULTS_ROOT / 'runs'

SPLITS = {
    'train': (50_000, PROBE_SEEDS['train']),
    'val':   (5_000,  PROBE_SEEDS['val']),
    'test':  (10_000, PROBE_SEEDS['test']),
}
