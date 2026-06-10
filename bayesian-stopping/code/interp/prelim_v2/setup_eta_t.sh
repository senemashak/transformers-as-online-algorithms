#!/usr/bin/env bash
# Create the eta_t output run dir + manifest.
#
# Usage:
#   bash setup_eta_t.sh <dst_run_dir> <src_run_dir>

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
    'experiment': 'eta_t (plug-in threshold from known-mu Gaussian stopping recursion) '
                  'probing with raw MSE',
    'target': 'eta_t_raw_mse',
    'target_definition':
        'eta_{n-1}=0; eta_t=psi(eta_{t+1}) for t=n-2..1; '
        'psi(z) = z * Phi(z) + phi(z). '
        'Target depends only on timestep, not on X_{1:t} or sigma_i.',
    'probe_parameterization':
        'eta_pred = probe(hidden_state) (no exp, no normalization).',
    'loss':
        'raw_mse = mean((eta_pred - eta_t)^2)',
    'val_selection':
        'best validation eta_mse, every step_count/8 steps, 1 epoch cap',
    'diagnostic_test_metrics':
        ['eta_mse', 'eta_rmse', 'eta_mae'],
    'probe_families': ['timestep_shared_simple_linear_eta_t_raw_mse',
                       'timestep_shared_attention_pooled_eta_t_raw_mse'],
    'source_run': "${SRC}",
    'source_run_manifest_commit': src_manifest.get('git_commit'),
    'reuse_policy': 'reads hidden_cache directly from source run dir in-place; '
                    'only writes new probe outputs under this dst dir. '
                    'Source run is read-only.',
    'note_about_controls':
        'eta_t depends only on timestep. Both true and perm_glob have the '
        'same positional embeddings (paired ensemble shares init), so '
        'control performance is expected to be strong, especially at L0 '
        'where absolute position embeddings dominate. eta_t does not test '
        'in-context inference by itself.',
    'M_train_budgets': [64, 128, 256, 512, 1024],
    'M_val': 512, 'M_test': 10000,
    'reps': ['true', 'perm_glob'],
    'valid_prefixes_t': 't in 1..n-1 (col 0..n-2); col n-1 = NaN (no eta needed)',
    'git_commit': commit,
    'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
}
pathlib.Path("${DST}/manifest.json").write_text(json.dumps(m, indent=2))
print(f'wrote manifest: ${DST}/manifest.json')
EOF
echo "[setup_eta_t] dst=${DST}"
echo "[setup_eta_t] src=${SRC}"
