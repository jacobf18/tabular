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
  log_step "Starting low-rank compressor quick MI@5 model ${label}"
  if python -u code/eval_tabimpute.py; then
    log_step "Finished low-rank compressor quick MI@5 model ${label}"
  else
    local status=$?
    log_step "FAILED low-rank compressor quick MI@5 model ${label} with exit status ${status}; continuing"
  fi
}

run_model \
  "quick_mi5_comp_lowrank_phase2_random_k20_100k" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-comp-rank10-100x100-50x50chunks-phase2-random-k20-100k-warmstable-from45k/checkpoint_100000.pth"

run_model \
  "quick_mi5_comp_lowrank_random50to300_randomfrac_k20_100k" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-comp-rank10-random50to300-randomfrac10to50-min10-k20-from-200comp-v4/checkpoint_100000.pth"

run_model \
  "quick_mi5_comp_lowrank_200x200_random_k20_100k" \
  "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-comp-rank10-200x200-100x100chunks-random-k20-100k-from-100comp/checkpoint_100000.pth"

log_step "Low-rank compressor quick MI@5 queue complete"
