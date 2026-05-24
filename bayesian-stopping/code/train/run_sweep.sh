#!/bin/bash
# Step 4 sweep launcher.
#
# Usage:
#   bash run_sweep.sh pilot    # Stage A — 5000 steps per run, ~30 min
#   bash run_sweep.sh full     # Stage B — full step counts, ~7 h
#
# Three sequential waves, each running multiple runs in parallel on
# physical GPUs 0, 1, 2, 4 (NEVER 3 — that's reserved). Each subprocess
# runs `python -m train.sweep --run <name> --gpu <id> --stage <stage>`
# with CUDA_VISIBLE_DEVICES set to the physical id.

set -u
set -o pipefail

STAGE="${1:?usage: $0 pilot|full}"
if [[ "${STAGE}" != "pilot" && "${STAGE}" != "full" ]]; then
    echo "stage must be 'pilot' or 'full', got: ${STAGE}" >&2
    exit 2
fi

cd "$(dirname "$0")/.."

# GPU pool — order matters; matches WAVES tuples in configs.py.
GPUS=(0 1 2 4)

# Wave layout. Indices align with GPUS.
WAVE1=("D_1_cv"     "D_2_cv"     "D_3_cv"     "D_disc_cv")
WAVE2=("D_logu_cv"  "D_1_act"    "D_2_act"    "D_3_act")
WAVE3=("D_disc_act" "D_logu_act")

LOG_ROOT="results/sweep_logs/${STAGE}"
mkdir -p "${LOG_ROOT}"

run_wave() {
    local wave_name="$1"; shift
    local -a runs=("$@")
    echo "[run_sweep] ===== ${wave_name} (${#runs[@]} runs) =====" | tee -a "${LOG_ROOT}/sweep.log"
    local pids=()
    local i
    for i in "${!runs[@]}"; do
        local run="${runs[$i]}"
        local gpu="${GPUS[$i]}"
        local out="${LOG_ROOT}/${run}.out"
        local err="${LOG_ROOT}/${run}.err"
        echo "[run_sweep]   launching ${run} on GPU ${gpu} (logs: ${out})" \
             | tee -a "${LOG_ROOT}/sweep.log"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            python3 -u -m train.sweep --run "${run}" --gpu "${gpu}" --stage "${STAGE}" \
            > "${out}" 2> "${err}" &
        pids+=($!)
    done
    # Wait for all of this wave to finish.
    local pid
    local rc=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            rc=$?
            echo "[run_sweep]   ERROR: pid ${pid} exited with code ${rc}" \
                 | tee -a "${LOG_ROOT}/sweep.log"
        fi
    done
    if [[ $rc -ne 0 ]]; then
        echo "[run_sweep]   ${wave_name} had failures; aborting sweep" \
             | tee -a "${LOG_ROOT}/sweep.log"
        return $rc
    fi
    echo "[run_sweep]   ${wave_name} done" | tee -a "${LOG_ROOT}/sweep.log"
}

start=$(date +%s)
run_wave "Wave 1" "${WAVE1[@]}" || exit $?
run_wave "Wave 2" "${WAVE2[@]}" || exit $?
run_wave "Wave 3" "${WAVE3[@]}" || exit $?
elapsed=$(( $(date +%s) - start ))
echo "[run_sweep] sweep finished — ${STAGE} stage, total wall ${elapsed}s" \
     | tee -a "${LOG_ROOT}/sweep.log"
