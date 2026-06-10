#!/usr/bin/env bash
# Parallel sweep over a range of ensemble members for the unified N=100 run.
#
# Usage:
#   bash run_sweep_n100.sh <run_dir> <start> <end> [smoke]
#
# Launches members [start, end), 5 at a time across GPUs 0,1,2,3,4 (the N=100
# config uses all 5 unlike the N=20 sweep which skipped GPU 3). Each member
# runs the full end-to-end (train true + perm_glob, cache hiddens, simple
# probes, attn probes).
#
# Logs go to <run_dir>/logs/member_<i>_gpu<g>.{out,err}.

set -e

RUN_DIR="$(readlink -f "${1:?usage: $0 <run_dir> <start> <end> [smoke]}")"
START="${2:?usage: $0 <run_dir> <start> <end> [smoke]}"
END="${3:?usage: $0 <run_dir> <start> <end> [smoke]}"
MODE="${4:-full}"

if [[ "${MODE}" == "smoke" ]]; then
  EXTRA_ARGS="--smoke"
else
  EXTRA_ARGS=""
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

GPUS=(0 1 2 3 4)
WAVE_SIZE=${#GPUS[@]}

echo "[sweep-n100] code_dir=${CODE_DIR}  run_dir=${RUN_DIR}"
echo "[sweep-n100] members [${START}, ${END})  mode=${MODE}  GPUs=${GPUS[*]}"

cd "${CODE_DIR}"

i="${START}"
while (( i < END )); do
  pids=()
  for slot in "${!GPUS[@]}"; do
    member=$((i + slot))
    if (( member >= END )); then
      break
    fi
    gpu="${GPUS[$slot]}"
    out="${LOG_DIR}/member_${member}_gpu${gpu}.out"
    err="${LOG_DIR}/member_${member}_gpu${gpu}.err"
    echo "[sweep-n100] wave start: member=${member} gpu=${gpu} -> ${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -m interp.prelim_v2.run_member \
      --run-dir "${RUN_DIR}" --ensemble-idx "${member}" --gpu 0 ${EXTRA_ARGS} \
      > "${out}" 2> "${err}" &
    pids+=($!)
  done
  echo "[sweep-n100] waiting for wave (members $((i))..$((i + WAVE_SIZE - 1))), pids=${pids[*]}"
  for pid in "${pids[@]}"; do
    wait "${pid}" || echo "[sweep-n100] WARNING: pid ${pid} exited non-zero"
  done
  i=$((i + WAVE_SIZE))
done

echo "[sweep-n100] all members in [${START}, ${END}) done"
echo "[sweep-n100] computing N=100 paired statistics + figures..."
python3 -m interp.prelim_v2.stats     --run-dir "${RUN_DIR}"
python3 -m interp.prelim_v2.figures   --run-dir "${RUN_DIR}"
python3 -m interp.prelim_v2.stats_attn    --run-dir "${RUN_DIR}"
python3 -m interp.prelim_v2.figures_attn  --run-dir "${RUN_DIR}"
echo "[sweep-n100] complete: see ${RUN_DIR}/stats/summary.md and ${RUN_DIR}/comparison/summary.md"
