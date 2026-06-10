#!/usr/bin/env bash
# 4-wave parallel sweep over ensemble members for the attention-pooled
# preliminary probing run. Trains the timestep-shared attention-pooled
# probe on top of an existing simple-probe run (which is read-only).
#
# Usage:
#   bash run_sweep_attn.sh <src_run_dir> <attn_run_dir> <N> [smoke]

set -e

SRC_RUN_DIR="$(readlink -f "${1:?usage: $0 <src_run_dir> <attn_run_dir> <N> [smoke]}")"
ATTN_RUN_DIR="$(readlink -f "${2:?usage: $0 <src_run_dir> <attn_run_dir> <N> [smoke]}")"
N="${3:?usage: $0 <src_run_dir> <attn_run_dir> <N> [smoke]}"
MODE="${4:-full}"

if [[ "${MODE}" == "smoke" ]]; then
  EXTRA_ARGS="--smoke"
else
  EXTRA_ARGS=""
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ATTN_RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

GPUS=(0 1 2 4)
WAVE_SIZE=${#GPUS[@]}

echo "[attn-sweep] code_dir=${CODE_DIR}"
echo "[attn-sweep] src_run_dir=${SRC_RUN_DIR}"
echo "[attn-sweep] attn_run_dir=${ATTN_RUN_DIR}"
echo "[attn-sweep] N=${N}  mode=${MODE}  GPUs=${GPUS[*]}  wave_size=${WAVE_SIZE}"

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
    echo "[attn-sweep] wave start: member=${member} gpu=${gpu} -> ${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -m interp.prelim_v2.run_member_attn \
      --src-run-dir "${SRC_RUN_DIR}" --attn-run-dir "${ATTN_RUN_DIR}" \
      --ensemble-idx "${member}" --gpu 0 ${EXTRA_ARGS} \
      > "${out}" 2> "${err}" &
    pids+=($!)
  done
  echo "[attn-sweep] waiting for wave (members $((i))..$((i + WAVE_SIZE - 1))), pids=${pids[*]}"
  for pid in "${pids[@]}"; do
    wait "${pid}" || echo "[attn-sweep] WARNING: pid ${pid} exited non-zero"
  done
  i=$((i + WAVE_SIZE))
done

echo "[attn-sweep] all members done"
echo "[attn-sweep] computing paired statistics (attn-alone + simple-vs-attn)..."
python3 -m interp.prelim_v2.stats_attn --src-run-dir "${SRC_RUN_DIR}" --attn-run-dir "${ATTN_RUN_DIR}"
echo "[attn-sweep] rendering figures..."
python3 -m interp.prelim_v2.figures_attn --src-run-dir "${SRC_RUN_DIR}" --attn-run-dir "${ATTN_RUN_DIR}"
echo "[attn-sweep] complete: see ${ATTN_RUN_DIR}/stats/"
