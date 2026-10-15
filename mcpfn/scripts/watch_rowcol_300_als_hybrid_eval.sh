#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"

WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-alshybrid768-rank10-p02-target300-cdf05-huber30-sigma08-lr5e5-cos50k-from-target300cdf25k}"
FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
LOG_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${LOG_DIR}"

wait_for_checkpoint() {
  local checkpoint="$1"
  while [[ ! -f "${checkpoint}" ]]; do
    date -u +"%Y-%m-%dT%H:%M:%SZ waiting for ${checkpoint}"
    sleep "${POLL_SECONDS:-60}"
  done
}

run_eval() {
  local tag="$1"
  shift
  python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
    --sizes 300 \
    --seeds 0 1 2 3 4 \
    --rank 10 \
    --missing-rate 0.2 \
    --generator gaussian_factors \
    --scale rank \
    --skip-softimpute \
    --direct-names \
    --direct-prediction-stat median \
    "$@" \
    --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-alshybrid-${tag}_eval"
}

wait_for_checkpoint "${WORK_DIR}/checkpoint_phase1_5000.pth"
run_eval "phase005k" \
  --extra-direct "RowCol ALS hybrid target300 CDF05 Huber30 sigma08 phase005k|${WORK_DIR}/checkpoint_phase1_5000.pth|300|rowcol_als_hybrid" \
  2>&1 | tee "${LOG_DIR}/rowcol_300_als_hybrid_phase005k_eval.log"

wait_for_checkpoint "${WORK_DIR}/checkpoint_phase2_10000.pth"
run_eval "phase010k" \
  --extra-direct "RowCol ALS hybrid target300 CDF05 Huber30 sigma08 phase010k|${WORK_DIR}/checkpoint_phase2_10000.pth|300|rowcol_als_hybrid" \
  2>&1 | tee "${LOG_DIR}/rowcol_300_als_hybrid_phase010k_eval.log"

wait_for_checkpoint "${WORK_DIR}/checkpoint_phase3_25000.pth"
run_eval "phase025k" \
  --extra-direct "RowCol ALS hybrid target300 CDF05 Huber30 sigma08 phase025k|${WORK_DIR}/checkpoint_phase3_25000.pth|300|rowcol_als_hybrid" \
  2>&1 | tee "${LOG_DIR}/rowcol_300_als_hybrid_phase025k_eval.log"
