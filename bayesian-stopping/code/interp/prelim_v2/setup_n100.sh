#!/usr/bin/env bash
# Create the unified N=100 run dir, symlinking existing N=20 artifacts and
# copying their probe outputs.
#
# Usage:
#   bash setup_n100.sh <n100_run_dir> <n20_simple_run_dir> <n20_attn_run_dir>

set -e

N100_DIR="$(readlink -f "${1:?usage: $0 <n100_run_dir> <n20_simple_run_dir> <n20_attn_run_dir>}")"
N20_SIMPLE="$(readlink -f "${2:?usage: $0 <n100_run_dir> <n20_simple_run_dir> <n20_attn_run_dir>}")"
N20_ATTN="$(readlink -f "${3:?usage: $0 <n100_run_dir> <n20_simple_run_dir> <n20_attn_run_dir>}")"

echo "[setup_n100] n100_dir         = ${N100_DIR}"
echo "[setup_n100] n20_simple_dir   = ${N20_SIMPLE}"
echo "[setup_n100] n20_attn_dir     = ${N20_ATTN}"

mkdir -p "${N100_DIR}"/{members,logs,stats,figures,comparison}

# Manifest.
python3 - <<EOF
import json, pathlib, subprocess, datetime
src = json.loads(pathlib.Path("${N20_SIMPLE}/manifest.json").read_text())
commit = subprocess.check_output(['git','-C',
    '/home/senemi/transformers-as-online-algorithms/bayesian-stopping',
    'rev-parse','HEAD']).decode().strip()
m = {
    'experiment': 'preliminary_mechanistic_interpretability_v2 (UNIFIED N=100)',
    'predecessor_simple_n20': "${N20_SIMPLE}",
    'predecessor_attn_n20':   "${N20_ATTN}",
    'predecessor_git_commit': src.get('git_commit'),
    'reuse_policy': 'members 0..19 symlinked from predecessor simple_n20 dirs; '
                    'probe_results.json (simple and attn) copied into '
                    'probe_runs/ and probe_runs_attn/ respectively. '
                    'Members 20..99 trained fresh.',
    'probe_families': ['timestep_shared_simple_linear',
                       'timestep_shared_attention_pooled'],
    'output_subdirs': {'simple': 'probe_runs', 'attn': 'probe_runs_attn'},
    'ensemble_size': 100,
    'base_seed': '2_000_000 + i for i in 0..99',
    'targets': src['scope']['targets'],
    'M_train_budgets': src['scope']['M_train_budgets'],
    'M_val': src['scope']['M_val'], 'M_test': src['scope']['M_test'],
    'reps': ['true', 'perm_glob'],
    'parallelism': '5-wave bash driver on GPUs 0..4',
    'git_commit': commit,
    'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
}
pathlib.Path("${N100_DIR}/manifest.json").write_text(json.dumps(m, indent=2))
print(f'wrote manifest: ${N100_DIR}/manifest.json')
EOF

# Symlink + copy for members 0..19.
for i in $(seq 0 19); do
  ii=$(printf "%03d" $i)
  src_member="${N20_SIMPLE}/members/${ii}"
  attn_member="${N20_ATTN}/members/${ii}"
  dst_member="${N100_DIR}/members/${ii}"
  mkdir -p "${dst_member}"

  # Symlink heavy artifacts.
  for sub in true perm_glob hidden_cache probe_data; do
    if [[ -e "${dst_member}/${sub}" ]]; then
      echo "[setup_n100] ${ii}: ${sub} already exists, skipping"
    else
      ln -s "${src_member}/${sub}" "${dst_member}/${sub}"
    fi
  done

  # Symlink the simple member_result.json too (for reference).
  if [[ ! -e "${dst_member}/member_result_n20_simple.json" ]]; then
    ln -s "${src_member}/member_result.json" "${dst_member}/member_result_n20_simple.json"
  fi

  # Copy simple probe results into probe_runs/.
  for rep in true perm_glob; do
    mkdir -p "${dst_member}/probe_runs/${rep}"
    if [[ ! -e "${dst_member}/probe_runs/${rep}/probe_results.json" ]]; then
      cp "${src_member}/probe_runs/${rep}/probe_results.json" \
         "${dst_member}/probe_runs/${rep}/probe_results.json"
    fi
  done

  # Copy attn probe results into probe_runs_attn/ (from the separate N=20 attn run).
  for rep in true perm_glob; do
    mkdir -p "${dst_member}/probe_runs_attn/${rep}"
    if [[ ! -e "${dst_member}/probe_runs_attn/${rep}/probe_results.json" ]]; then
      cp "${attn_member}/probe_runs/${rep}/probe_results.json" \
         "${dst_member}/probe_runs_attn/${rep}/probe_results.json"
    fi
  done
done

echo "[setup_n100] done. Members 0..19 populated; members 20..99 left for sweep."
ls "${N100_DIR}/members" | head -25
