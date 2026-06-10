#!/usr/bin/env bash
# Watch for member completion events and re-probe under the new sigma-MSE loss.
#
# Use case: wave 2 of the main sweep (members 4..7) imported the OLD probe.py
# at startup (before the sigma-normalized MSE loss was added on disk). Their
# in-memory probe code is stale, so their probe_results.json will be in the
# old units when wave 2 finishes. This watcher polls for each member's
# member_result.json to appear, then queues a reprobe on a dedicated GPU.
#
# Usage:
#   bash reprobe_watcher.sh <run_dir> <gpu> <member_id...>
# Example:
#   bash reprobe_watcher.sh /path/to/run 3 4 5 6 7
#
# Behaviour:
#   - Polls every 60s for member_result.json to appear for each id.
#   - Once it appears, deletes both probe_runs/<rep>/probe_results.json
#     for that member and re-invokes run_member --skip train probe_data hidden_cache.
#   - Re-probes are serialized (one at a time on the given GPU).
#   - Each member's reprobe overwrites its member_result.json with new probe stats.
#   - Logs to <run_dir>/logs/reprobe_watcher.log; per-member logs in the same dir.

set -u

RUN_DIR="${1:?usage: $0 <run_dir> <gpu> <member_id...>}"
GPU="${2:?usage: $0 <run_dir> <gpu> <member_id...>}"
shift 2
MEMBERS=("$@")

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${RUN_DIR}/logs"
WATCHER_LOG="${LOG_DIR}/reprobe_watcher.log"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -u +%FT%T)Z] $*" | tee -a "${WATCHER_LOG}"; }

log "watcher start: GPU=${GPU} members=${MEMBERS[*]}"
log "code_dir=${CODE_DIR}  run_dir=${RUN_DIR}"

cd "${CODE_DIR}"

for member in "${MEMBERS[@]}"; do
  member_idx=$(printf "%03d" "${member}")
  member_dir="${RUN_DIR}/members/${member_idx}"
  result_file="${member_dir}/member_result.json"

  log "waiting for member ${member_idx} to complete (${result_file})..."
  while [ ! -f "${result_file}" ]; do
    sleep 60
  done
  log "member ${member_idx} completed; preparing reprobe"

  # Delete stale probe_results.json so run_member's skip logic re-runs probes.
  for rep in true perm_glob; do
    p="${member_dir}/probe_runs/${rep}/probe_results.json"
    if [ -f "${p}" ]; then
      rm "${p}"
      log "  removed stale ${p}"
    fi
  done

  reprobe_log="${LOG_DIR}/reprobe_member${member_idx}.log"
  log "  launching reprobe on GPU ${GPU} -> ${reprobe_log}"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m interp.prelim_v2.run_member \
    --run-dir "${RUN_DIR}" --ensemble-idx "${member}" --gpu 0 \
    --skip train probe_data hidden_cache \
    > "${reprobe_log}" 2>&1
  rc=$?
  if [ ${rc} -eq 0 ]; then
    log "  member ${member_idx} reprobe DONE"
  else
    log "  member ${member_idx} reprobe FAILED rc=${rc}"
  fi
done

log "watcher exit: all targeted members reprobed"
