#!/usr/bin/env bash
# Create the sigma_hat output run dir + manifest. No symlinks (the driver
# reads hidden_cache/probe_data directly from the source run dir).
#
# Usage:
#   bash setup_sigma_hat.sh <dst_run_dir> <src_run_dir>

set -e

DST="$(readlink -f "${1:?usage: $0 <dst_run_dir> <src_run_dir>}")"
SRC="$(readlink -f "${2:?usage: $0 <dst_run_dir> <src_run_dir>}")"

mkdir -p "${DST}"/{members,logs,stats,stats_attn,comparison,figures}

python3 - <<EOF
import json, pathlib, subprocess, datetime
src_manifest_path = pathlib.Path("${SRC}/manifest.json")
src_manifest = json.loads(src_manifest_path.read_text()) if src_manifest_path.exists() else {}
commit = subprocess.check_output([
    'git','-C','/home/senemi/transformers-as-online-algorithms/bayesian-stopping',
    'rev-parse','HEAD',
]).decode().strip()
m = {
    'experiment': 'sigma_hat (online unbiased prefix std) probes with exp-output, '
                  'relative sigma-space MSE loss',
    'target': 'sigma_hat_exp_relmse',
    'target_definition':
        'sigma_hat_t = sqrt(max(eps, (Q_t - t*mean_t^2) / (t-1))), valid t >= 2; '
        'distinct from existing sigma_mle_sqrt which divides by t.',
    'probe_parameterization':
        'a_hat = probe(hidden_state); sigma_pred = exp(a_hat).',
    'loss':
        'rel_sigma_mse = mean(((sigma_pred - sigma_hat_t) / sigma_hat_t)^2)',
    'val_selection':
        'best validation rel_sigma_mse, every step_count/8 steps, 1 epoch cap',
    'diagnostic_test_metrics':
        ['rel_sigma_mse', 'raw_rmse', 'raw_mae', 'log_mse', 'log_mae'],
    'probe_families': ['timestep_shared_simple_linear_sigma_hat_exp_relmse',
                       'timestep_shared_attention_pooled_sigma_hat_exp_relmse'],
    'source_run': "${SRC}",
    'source_run_manifest_commit': src_manifest.get('git_commit'),
    'reuse_policy': 'reads hidden_cache and probe_data from source run dir '
                    'in-place (no symlinks); only writes new probe outputs '
                    'under this dst dir. Source run is read-only.',
    'ensemble_size_total_possible': 100,
    'M_train_budgets': [64, 128, 256, 512, 1024],
    'M_val': 512, 'M_test': 10000,
    'reps': ['true', 'perm_glob'],
    'epsilon_var_clamp': 1e-12,
    'git_commit': commit,
    'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
}
pathlib.Path("${DST}/manifest.json").write_text(json.dumps(m, indent=2))
print(f'wrote manifest: ${DST}/manifest.json')
EOF
echo "[setup_sigma_hat] dst=${DST}"
echo "[setup_sigma_hat] src=${SRC}"
