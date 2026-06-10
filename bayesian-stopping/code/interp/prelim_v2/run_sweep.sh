#!/usr/bin/env bash
# 4-wave parallel sweep over ensemble members for the preliminary
# mechanistic-interpretability run.
#
# Usage:
#   bash run_sweep.sh <run_dir> <N> [smoke]
#
# Launches members 0..N-1, 4 at a time across GPUs 0, 1, 2, 4 (skipping 3),
# matching the original train/run_sweep.sh convention. Each member runs the
# full end-to-end (train true + perm_glob, cache hiddens, train probes).
#
# Logs go to <run_dir>/logs/member_<i>_gpu<g>.{out,err}.

set -e

RUN_DIR="${1:?usage: $0 <run_dir> <N> [smoke]}"
N="${2:?usage: $0 <run_dir> <N> [smoke]}"
MODE="${3:-full}"

if [[ "${MODE}" == "smoke" ]]; then
  EXTRA_ARGS="--smoke"
else
  EXTRA_ARGS=""
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# GPUs to use (skip 3 by convention).
GPUS=(0 1 2 4)
WAVE_SIZE=${#GPUS[@]}

echo "[sweep] code_dir=${CODE_DIR}  run_dir=${RUN_DIR}  N=${N}  mode=${MODE}"
echo "[sweep] using GPUs: ${GPUS[*]}  wave_size=${WAVE_SIZE}"

cd "${CODE_DIR}"

i=0
while (( i < N )); do
  pids=()
  for slot in "${!GPUS[@]}"; do
    member=$((i + slot))
    if (( member >= N )); then
      break
    fi
    gpu="${GPUS[$slot]}"
    out="${LOG_DIR}/member_${member}_gpu${gpu}.out"
    err="${LOG_DIR}/member_${member}_gpu${gpu}.err"
    echo "[sweep] wave start: member=${member} gpu=${gpu} -> ${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -m interp.prelim_v2.run_member \
      --run-dir "${RUN_DIR}" --ensemble-idx "${member}" --gpu 0 ${EXTRA_ARGS} \
      > "${out}" 2> "${err}" &
    pids+=($!)
  done
  echo "[sweep] waiting for wave (members $((i))..$((i + WAVE_SIZE - 1))), pids=${pids[*]}"
  for pid in "${pids[@]}"; do
    wait "${pid}" || echo "[sweep] WARNING: pid ${pid} exited non-zero"
  done
  i=$((i + WAVE_SIZE))
done

echo "[sweep] all members done"
echo "[sweep] computing paired statistics..."
python3 -m interp.prelim_v2.stats --run-dir "${RUN_DIR}"
echo "[sweep] rendering figures..."
python3 -m interp.prelim_v2.figures --run-dir "${RUN_DIR}"
echo "[sweep] complete: see ${RUN_DIR}/stats/summary.md"
