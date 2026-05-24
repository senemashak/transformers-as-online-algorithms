"""
Probe sweep orchestrator.

Sweeps `train_probe` across (target, layer, variant) for a single trained
model. Designed so a per-layer activation memmap is loaded once and shared
across the 14 probes (7 targets × 2 variants) trained on that layer.

CLI:
    python -m interp.probe_run --run D_logu_act --targets C_star --layers all
    python -m interp.probe_run --run D_logu_act --targets all   --layers all

Outputs (per `--run`) go under
    results/probing/logu-act/runs/<run>/probe_results.json
and a flat CSV under the same dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from interp.probe_data import PROBE_DATA_ROOT, SPLITS
from interp.probe_train import (
    CACHE_ROOT, RESULTS_ROOT, _load_activations, _load_target, train_probe,
)


ALL_TARGETS = (
    'S_t', 'Q_t',
    'X_bar', 'sigma_hat2', 'sigma_MAP', 'sigma_MLE',
    'C_star', 'C_plugin_MAP', 'C_plugin_MLE',
    'log_lik_sigma_1', 'log_lik_sigma_10', 'log_lik_sigma_100',
    'log_lik_true_sigma',
)
N_LAYERS = 9                     # ℓ ∈ {0, ..., 8}
VARIANTS = ('linear', 'mlp')


def _parse_list(value: str, allowed: tuple) -> List[str]:
    if value == 'all':
        return list(allowed)
    items = [v.strip() for v in value.split(',') if v.strip()]
    for it in items:
        if it not in allowed:
            raise ValueError(f'unknown {it!r}; allowed: {allowed}')
    return items


def _parse_layer_spec(spec: str) -> List[int]:
    if spec == 'all':
        return list(range(N_LAYERS))
    return [int(x) for x in spec.split(',')]


def run_sweep(
    run_name: str,
    targets: List[str],
    layers: List[int],
    device: torch.device,
    out_root: Path,
    batch_size: int = 256,
    probe_kind: str = 'attn',
    selectivity: bool = False,
) -> None:
    """Run probes for the given (targets × layers × {linear, mlp}) grid.

    Activation cache is loaded once per layer (memmap); target arrays are
    loaded once per target. Results are appended to
    `out_root / 'probe_results{,_noattn}.json'` after each (target, layer, variant)
    cell completes; the suffix depends on `probe_kind`.
    """
    cache_dir = CACHE_ROOT / run_name
    if not cache_dir.exists():
        raise FileNotFoundError(f'activation cache missing: {cache_dir}')

    out_root.mkdir(parents=True, exist_ok=True)
    suffix = '' if probe_kind == 'attn' else f'_{probe_kind}'
    if selectivity:
        suffix += '_sel'
    results_path = out_root / f'probe_results{suffix}.json'
    if results_path.exists():
        results: Dict = json.loads(results_path.read_text())
    else:
        results = {'run': run_name, 'cells': []}

    seen = {(c['target'], c['layer'], c['variant']) for c in results['cells']}

    target_cache: Dict[str, dict] = {t: _load_target(t, shuffled=selectivity) for t in targets}

    t_global = time.perf_counter()
    for layer in layers:
        print(f'\n[probe-run] === layer {layer} (cache: {cache_dir.name}) ===')
        t_layer = time.perf_counter()
        # Load each split's activations once and put on GPU as fp16 (saves
        # ~4× GPU memory vs fp32; the probe upcasts on read). This
        # eliminates per-batch host→device copy, which is the dominant
        # bottleneck on a CPU-contended box.
        H_arrs_mm = _load_activations(cache_dir, layer)
        t_io = time.perf_counter()
        H_arrs = {
            k: torch.as_tensor(np.asarray(v), dtype=torch.float16, device=device)
            for k, v in H_arrs_mm.items()
        }
        gpu_mem_gb = sum(t.numel() * t.element_size() for t in H_arrs.values()) / 1e9
        print(f'  loaded layer {layer} into GPU ({gpu_mem_gb:.2f} GB) '
              f'in {time.perf_counter() - t_io:.1f}s')
        for target in targets:
            for variant in VARIANTS:
                if (target, layer, variant) in seen:
                    print(f'  [{target:>13s} L{layer} {variant:>6s}]  skip (already done)')
                    continue
                res = train_probe(
                    layer=layer, target_name=target, variant=variant,
                    cache_dir=cache_dir, device=device,
                    target_arrs=target_cache[target],
                    H_arrs=H_arrs,
                    batch_size=batch_size,
                    probe_kind=probe_kind,
                )
                results['cells'].append(res.as_dict())
                # Persist after each cell so progress survives interruptions.
                results_path.write_text(json.dumps(results, indent=2))
                print(
                    f'  [{target:>13s} L{layer} {variant:>6s}]  '
                    f'test_norm_mse={res.test_norm_mse:.4f}  '
                    f'val_norm_mse={res.val_norm_mse:.4f}  '
                    f'wall={res.wall_s:.1f}s  best_ep={res.best_epoch}'
                )
        print(f'[probe-run] layer {layer} wall={time.perf_counter() - t_layer:.1f}s')

    # Dump a flat CSV for quick perusal.
    csv_path = out_root / f'probe_results{suffix}.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        fields = (
            'run', 'target', 'layer', 'variant', 'n_epochs',
            'train_mse', 'val_mse', 'test_mse',
            'train_norm_mse', 'val_norm_mse', 'test_norm_mse',
            'target_var', 'best_epoch', 'attn_arg_mean', 'wall_s',
        )
        w.writerow(fields)
        for c in results['cells']:
            w.writerow([
                run_name, c['target'], c['layer'], c['variant'], c['n_epochs'],
                f'{c["train_mse"]:.6g}', f'{c["val_mse"]:.6g}', f'{c["test_mse"]:.6g}',
                f'{c["train_norm_mse"]:.6g}', f'{c["val_norm_mse"]:.6g}', f'{c["test_norm_mse"]:.6g}',
                f'{c["target_var"]:.6g}', c['best_epoch'],
                f'{c["attention_argmax_per_t_mean"]:.4f}', f'{c["wall_s"]:.1f}',
            ])
    print(f'\n[probe-run] wrote {results_path}')
    print(f'[probe-run] wrote {csv_path}')
    print(f'[probe-run] total wall {time.perf_counter() - t_global:.1f}s')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='D_logu_act',
                   help='checkpoint run name; activation cache is read from '
                        '`results/probing/logu-act/cache/<run>/`')
    p.add_argument('--targets', default='C_star',
                   help='comma-separated target names or "all"')
    p.add_argument('--layers', default='all',
                   help='comma-separated layer indices or "all" (0..8)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out-subdir', default=None,
                   help='subdir under results/probing/logu-act/runs/; '
                        'defaults to the --run name')
    p.add_argument('--probe-kind', choices=['attn', 'noattn'], default='attn',
                   help='attn = attention-pooled probe (default, existing); '
                        'noattn = per-timestep probe (FF(H_t) only)')
    p.add_argument('--selectivity', action='store_true',
                   help='Train probes on shuffled-target labels (Hewitt-Liang '
                        'control task). Output filenames get a `_sel` suffix.')
    args = p.parse_args()

    out_subdir = args.out_subdir or args.run
    out_root = RESULTS_ROOT / 'runs' / out_subdir
    targets = _parse_list(args.targets, ALL_TARGETS)
    layers = _parse_layer_spec(args.layers)
    device = torch.device(args.device)

    print(f'[probe-run] run={args.run}, targets={targets}, layers={layers}, '
          f'probe_kind={args.probe_kind}, selectivity={args.selectivity}')
    print(f'[probe-run] out_root={out_root}')
    run_sweep(args.run, targets, layers, device, out_root,
              batch_size=args.batch_size, probe_kind=args.probe_kind,
              selectivity=args.selectivity)
    return 0


if __name__ == '__main__':
    sys.exit(main())
