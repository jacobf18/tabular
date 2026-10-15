#!/usr/bin/env bash
set -uo pipefail

cd /root/tabular/benchmark-selection
export PYTHONPATH=/root/tabular/mcpfn/src
export BENCHSELECT_TABIMPUTE_DEVICE="${BENCHSELECT_TABIMPUTE_DEVICE:-cuda}"
export BENCHSELECT_TABIMPUTE_MODEL=compressed

export BENCHSELECT_METHODS=mi
export BENCHSELECT_K_MAX=5
export BENCHSELECT_MAX_ROWS=1000
export BENCHSELECT_TEST_BATCH=500

unset BENCHSELECT_ROW_LIMIT || true
unset BENCHSELECT_N_FOLDS || true
unset BENCHSELECT_COMPRESSED_CHUNK_ROWS || true
unset BENCHSELECT_COMPRESSED_CHUNK_COLS || true

log_step() {
  printf '\n[%s] %s\n' "$(date -u --iso-8601=seconds)" "$*"
}

run_model() {
  local label="$1"
  local checkpoint="$2"
  export BENCHSELECT_TABIMPUTE_LABEL="${label}"
  export BENCHSELECT_COMPRESSED_CHECKPOINT="${checkpoint}"
  log_step "Starting compressor quick MI@5 model ${label}"
  if python -u code/eval_tabimpute.py; then
    log_step "Finished compressor quick MI@5 model ${label}"
  else
    local status=$?
    log_step "FAILED compressor quick MI@5 model ${label} with exit status ${status}; continuing"
  fi
}

run_model \
  "quick_mi5_comp_recsys_v2_broad_best" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-comp-rank10-recsys-v2-broadrect100to700x400to1700-100chunks-random-k20-100k-from-v2best-rmse/best_checkpoint.pth"

run_model \
  "quick_mi5_comp_recsys150to300_clean_best" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-comp-rank10-recsys150to300-100x100chunks-random-k20-100k-from-phase1best-lr2e5-min5e6-clean/best_checkpoint.pth"

run_model \
  "quick_mi5_comp_pure_rabel64_curriculum_best" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-compressed-pure-rabel64-rank10-k20-curriculum-160k/best_checkpoint.pth"

log_step "Compressor quick MI@5 queue complete"
