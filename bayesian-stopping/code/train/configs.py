"""
Step 4 sweep configs.

One RunConfig per (distribution, supervision). Stage A pilot uses 5000
steps + 500-step val cadence; Stage B is the spec-pinned full sweep.

Run names:
    Stage A: D_<dist>_<sup>_pilot   (e.g. D_disc_cv_pilot)
    Stage B: D_<dist>_<sup>         (e.g. D_disc_cv)

Wave layout (4 GPUs available — 0, 1, 2, 4; GPU 3 reserved for another job):

    Wave 1: D_1_cv     D_2_cv     D_3_cv     D_disc_cv
    Wave 2: D_logu_cv  D_1_act    D_2_act    D_3_act
    Wave 3: D_disc_act D_logu_act
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from data.distributions import (
    ALL_DISTRIBUTIONS,
    RANDOM_DISTRIBUTIONS,
    STATIC_DISTRIBUTIONS,
)


# Path roots. Source code lives under v3/code/ (CODE_ROOT); checkpoints
# and results live one level up at v3/ (V3_ROOT).
CODE_ROOT = Path(__file__).resolve().parent.parent       # v3/code/
V3_ROOT = CODE_ROOT.parent                                # v3/
CHECKPOINT_ROOT = V3_ROOT / 'checkpoints'
RESULTS_PHASE4 = V3_ROOT / 'results' / 'phase4'
RESULTS_PHASE4_PILOT = V3_ROOT / 'results' / 'phase4_pilot'
DATA_CACHE = CODE_ROOT / 'data' / 'cache'
ORACLE_TABLES = CODE_ROOT / 'oracle' / 'tables'


# ---------------------------------------------------------------------------
# Spec-pinned hyperparameters (Section 5.2 of v3 spec, plus the pilot deltas)
# ---------------------------------------------------------------------------

LR = 1e-4
BATCH_SIZE = 64
WARMUP_FRAC = 0.20
TRAIN_LOG_EVERY = 100

PILOT_STEPS = 5_000
PILOT_VAL_EVERY = 500
PILOT_PERIODIC_EVERY = None         # No periodic checkpoints in pilot

FULL_VAL_EVERY_STATIC = 2_000
FULL_VAL_EVERY_RANDOM = 3_000
FULL_PERIODIC_EVERY = 50_000

# Step counts for Stage B.
_FULL_STEPS = {
    'cv':  {'D_1': 200_000, 'D_2': 200_000, 'D_3': 200_000,
            'D_disc': 300_000, 'D_logu': 300_000},
    'act': {'D_1': 100_000, 'D_2': 100_000, 'D_3': 100_000,
            'D_disc': 150_000, 'D_logu': 150_000},
}


# Wave layout: each tuple lists run names, length must match GPU_IDS for full
# parallelism (Wave 3 has only 2 runs, uses first 2 GPUs).
WAVES = [
    ('D_1_cv',     'D_2_cv',     'D_3_cv',     'D_disc_cv'),    # Wave 1
    ('D_logu_cv',  'D_1_act',    'D_2_act',    'D_3_act'),       # Wave 2
    ('D_disc_act', 'D_logu_act'),                                # Wave 3
]
GPU_IDS = (0, 1, 2, 4)              # NEVER 3 — that GPU is in use


# ---------------------------------------------------------------------------
# RunConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    distribution: str
    supervision: str                 # 'cv' or 'act'
    stage: str                       # 'pilot' or 'full'
    step_count: int
    val_every: int
    periodic_every: int | None       # None = no periodic checkpoints
    seed: int
    train_log_every: int = TRAIN_LOG_EVERY
    lr: float = LR
    batch_size: int = BATCH_SIZE
    warmup_frac: float = WARMUP_FRAC
    n: int = 256
    d_emb: int = 128
    n_layers: int = 8
    n_heads: int = 4
    sigma_power: int = 1             # cv-loss denominator exponent: σ_i ** sigma_power
    # Label source for both cv and act heads. 'oracle' = ADP-table labels
    # (original Section 3.4 convention). 'offline' = hindsight prophet
    # labels computed from the suffix of each sampled sequence.
    label_source: str = 'oracle'

    @property
    def base_name(self) -> str:
        if self.label_source == 'offline':
            # Offline cv on random distributions implicitly uses σ² (the
            # OOD-reporting convention); the suffix is just '_offline'.
            return f'{self.distribution}_{self.supervision}_offline'
        if self.label_source == 'control_zero':
            return f'{self.distribution}_{self.supervision}_control'
        suffix = '_sig2' if self.sigma_power == 2 else ''
        return f'{self.distribution}_{self.supervision}{suffix}'

    @property
    def run_name(self) -> str:
        return f'{self.base_name}_pilot' if self.stage == 'pilot' else self.base_name

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_ROOT / self.run_name

    @property
    def log_dir(self) -> Path:
        root = RESULTS_PHASE4_PILOT if self.stage == 'pilot' else RESULTS_PHASE4
        return root / self.run_name

    @property
    def is_random(self) -> bool:
        return self.distribution in RANDOM_DISTRIBUTIONS

    def model_kwargs(self) -> dict:
        return {
            'n': self.n, 'd_emb': self.d_emb,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d['run_name'] = self.run_name
        d['base_name'] = self.base_name
        d['checkpoint_dir'] = str(self.checkpoint_dir)
        d['log_dir'] = str(self.log_dir)
        return d


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_run_configs(stage: str) -> List[RunConfig]:
    """All 10 RunConfig objects for `stage` ('pilot' or 'full')."""
    if stage not in ('pilot', 'full'):
        raise ValueError(f'stage must be pilot or full, got {stage!r}')
    cfgs: List[RunConfig] = []
    run_idx = 0
    for dist in ALL_DISTRIBUTIONS:
        for sup in ('cv', 'act'):
            if stage == 'pilot':
                cfg = RunConfig(
                    distribution=dist, supervision=sup, stage='pilot',
                    step_count=PILOT_STEPS, val_every=PILOT_VAL_EVERY,
                    periodic_every=PILOT_PERIODIC_EVERY, seed=1000 + run_idx,
                )
            else:
                val_every = (FULL_VAL_EVERY_RANDOM if dist in RANDOM_DISTRIBUTIONS
                             else FULL_VAL_EVERY_STATIC)
                cfg = RunConfig(
                    distribution=dist, supervision=sup, stage='full',
                    step_count=_FULL_STEPS[sup][dist],
                    val_every=val_every,
                    periodic_every=FULL_PERIODIC_EVERY, seed=1000 + run_idx,
                )
            cfgs.append(cfg)
            run_idx += 1
    return cfgs


def get_run_config(base_name: str, stage: str) -> RunConfig:
    """Look up a config by base name (e.g. 'D_disc_cv', no stage suffix).

    Cv-loss σ² ablation: base names ending in '_sig2' (e.g. 'D_disc_cv_sig2',
    'D_logu_cv_sig2') reuse the matching σ¹ run's hyperparameters and seed,
    flipping only `sigma_power` from 1 to 2. Cv supervision only — there is
    no σ² ablation for act supervision (act loss has no σ normalizer).

    Offline-supervision variants: base names ending in '_offline' (e.g.
    'D_disc_cv_offline', 'D_disc_act_offline') reuse the matching oracle
    run's hyperparameters but flip `label_source` from 'oracle' to
    'offline'. For cv on random distributions the offline variants also
    inherit the σ² (`sigma_power=2`) convention used in OOD reporting —
    see results/offline-supervision/README.md. Seeds shift by +1000 to
    stay disjoint from the existing runs.
    """
    base = base_name.removesuffix('_pilot')
    if base.endswith('_control'):
        underlying = base[: -len('_control')]                            # e.g. 'D_logu_act'
        for cfg in make_run_configs(stage):
            if cfg.base_name != underlying:
                continue
            from dataclasses import replace as _replace
            return _replace(
                cfg,
                label_source='control_zero',
                seed=cfg.seed + 2000,           # disjoint from oracle (1xxx) and offline (2xxx)
            )
        raise KeyError(f'No oracle run named {underlying!r} to derive {base!r} from')
    if base.endswith('_offline'):
        underlying = base[: -len('_offline')]                       # e.g. 'D_disc_cv'
        # Offline cv on random distributions matches the σ² oracle baseline;
        # offline act has no normalizer choice.
        promote_sig2 = underlying in ('D_disc_cv', 'D_logu_cv')
        for cfg in make_run_configs(stage):
            if cfg.base_name != underlying:
                continue
            from dataclasses import replace as _replace
            return _replace(
                cfg,
                label_source='offline',
                sigma_power=2 if promote_sig2 else cfg.sigma_power,
                seed=cfg.seed + 1000,
            )
        raise KeyError(f'No oracle run named {underlying!r} to derive {base!r} from')
    if base.endswith('_sig2'):
        underlying = base[: -len('_sig2')]
        for cfg in make_run_configs(stage):
            if cfg.base_name == underlying:
                if cfg.supervision != 'cv':
                    raise ValueError(
                        f'σ² ablation is cv-only; got supervision={cfg.supervision!r} '
                        f'for {base_name!r}'
                    )
                from dataclasses import replace as _replace
                return _replace(cfg, sigma_power=2)
        raise KeyError(f'No σ¹ run named {underlying!r} to derive {base!r} from')
    for cfg in make_run_configs(stage):
        if cfg.base_name == base:
            return cfg
    raise KeyError(f'No run for base_name={base!r}, stage={stage!r}')


def parse_run_name(name: str) -> Tuple[str, str]:
    """'D_disc_cv' -> ('D_disc', 'cv'); 'D_logu_act_pilot' -> ('D_logu', 'act');
    'D_disc_cv_sig2' -> ('D_disc', 'cv') (σ² ablation suffix is stripped);
    'D_disc_cv_offline' -> ('D_disc', 'cv') (offline suffix is stripped)."""
    base = (
        name.removesuffix('_pilot')
            .removesuffix('_offline')
            .removesuffix('_sig2')
    )
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f'Bad run name: {name!r}')
    sup = parts[-1]
    dist = '_'.join(parts[:-1])
    return dist, sup
