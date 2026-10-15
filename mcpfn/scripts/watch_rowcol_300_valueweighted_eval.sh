#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"

WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-valueweighted-cdf2-huber50-lr3e5-cos200k-from-target300cdf25k}"
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
  local output_prefix="${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-valueweighted-${tag}_eval"
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
    --output-prefix "${output_prefix}"
}

wait_for_checkpoint "${WORK_DIR}/checkpoint_phase4_150000.pth"
run_eval "phase150k" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase150k|${WORK_DIR}/checkpoint_phase4_150000.pth|300|rowcol_latent_attention" \
  2>&1 | tee "${LOG_DIR}/rowcol_300_valueweighted_phase150k_eval.log"

wait_for_checkpoint "${WORK_DIR}/checkpoint_phase5_200000.pth"
run_eval "phase200k" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase200k|${WORK_DIR}/checkpoint_phase5_200000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 trainloss best|${WORK_DIR}/best_checkpoint.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 final|${WORK_DIR}/checkpoint_200000.pth|300|rowcol_latent_attention" \
  2>&1 | tee "${LOG_DIR}/rowcol_300_valueweighted_phase200k_eval.log"
