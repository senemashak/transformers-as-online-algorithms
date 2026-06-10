#!/usr/bin/env bash
# Sequential sweep over members for the two new targets (sigma_hat_exp_relmse
# and eta_t_raw_mse), one GPU at a time so the running N=100 base sweep
# only loses ~15-20% throughput on the shared GPU.
#
# For each member in [start, end):
#   1. sigma_hat: simple + attn probe families
#   2. eta_t:     simple + attn probe families
#
# At the end, aggregate paired stats + figures for each target.
#
# Usage:
#   bash run_sweep_new_targets.sh <src_run_dir> <sigma_dst_dir> <eta_dst_dir> <start> <end> <gpu>

set -e

SRC="$(readlink -f "${1:?usage: $0 <src> <sigma_dst> <eta_dst> <start> <end> <gpu>}")"
SIGMA_DST="$(readlink -f "${2:?usage}")"
ETA_DST="$(readlink -f "${3:?usage}")"
START="${4:?usage}"
END="${5:?usage}"
GPU="${6:-0}"

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR_SIG="${SIGMA_DST}/logs"
LOG_DIR_ETA="${ETA_DST}/logs"
mkdir -p "${LOG_DIR_SIG}" "${LOG_DIR_ETA}"

echo "[new-targets sweep] src=${SRC}"
echo "[new-targets sweep] sigma_dst=${SIGMA_DST}"
echo "[new-targets sweep] eta_dst=${ETA_DST}"
echo "[new-targets sweep] members [${START}, ${END})  gpu=${GPU}"

cd "${CODE_DIR}"

for ((i=${START}; i<${END}; i++)); do
  echo "[new-targets sweep] member ${i}: sigma_hat"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m interp.prelim_v2.run_member_sigma_hat \
    --src-run-dir "${SRC}" --dst-run-dir "${SIGMA_DST}" \
    --ensemble-idx "${i}" --gpu 0 \
    > "${LOG_DIR_SIG}/member_${i}.out" 2> "${LOG_DIR_SIG}/member_${i}.err" \
    || { echo "[sigma_hat] member ${i} FAILED"; cat "${LOG_DIR_SIG}/member_${i}.err"; }

  echo "[new-targets sweep] member ${i}: eta_t"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m interp.prelim_v2.run_member_eta_t \
    --src-run-dir "${SRC}" --dst-run-dir "${ETA_DST}" \
    --ensemble-idx "${i}" --gpu 0 \
    > "${LOG_DIR_ETA}/member_${i}.out" 2> "${LOG_DIR_ETA}/member_${i}.err" \
    || { echo "[eta_t] member ${i} FAILED"; cat "${LOG_DIR_ETA}/member_${i}.err"; }
done

echo "[new-targets sweep] all members in [${START}, ${END}) done; aggregating..."

# Aggregate simple-probe (probe_runs/) stats + figures
python3 -m interp.prelim_v2.stats   --run-dir "${SIGMA_DST}"
python3 -m interp.prelim_v2.figures --run-dir "${SIGMA_DST}"
python3 -m interp.prelim_v2.stats   --run-dir "${ETA_DST}"
python3 -m interp.prelim_v2.figures --run-dir "${ETA_DST}"

# Also aggregate attn-probe (probe_runs_attn/) — using the unified attn aggregators
python3 -m interp.prelim_v2.stats_attn   --run-dir "${SIGMA_DST}"
python3 -m interp.prelim_v2.figures_attn --run-dir "${SIGMA_DST}"
python3 -m interp.prelim_v2.stats_attn   --run-dir "${ETA_DST}"
python3 -m interp.prelim_v2.figures_attn --run-dir "${ETA_DST}"

echo "[new-targets sweep] complete:"
echo "  sigma_hat: ${SIGMA_DST}/stats/summary.md, ${SIGMA_DST}/stats_attn/summary.md"
echo "  eta_t:     ${ETA_DST}/stats/summary.md,   ${ETA_DST}/stats_attn/summary.md"
