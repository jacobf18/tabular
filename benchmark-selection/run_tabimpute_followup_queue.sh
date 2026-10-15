#!/usr/bin/env bash
set -uo pipefail

WAIT_PID="${1:-387390}"

cd /root/tabular/benchmark-selection
export PYTHONPATH=/root/tabular/mcpfn/src
export BENCHSELECT_TABIMPUTE_DEVICE="${BENCHSELECT_TABIMPUTE_DEVICE:-cuda}"

log_step() {
  printf '\n[%s] %s\n' "$(date -u --iso-8601=seconds)" "$*"
}

run_stage() {
  local label="$1"
  shift
  log_step "Starting ${label}"
  if "$@"; then
    log_step "Finished ${label}"
  else
    local status=$?
    log_step "FAILED ${label} with exit status ${status}; continuing queue"
  fi
}

if [[ -n "${WAIT_PID}" ]]; then
  log_step "Waiting for existing TabImpute evaluation PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  log_step "Existing TabImpute evaluation PID ${WAIT_PID} has exited"
fi

unset BENCHSELECT_ROW_LIMIT || true
unset BENCHSELECT_METHODS || true
unset BENCHSELECT_K_MAX || true
unset BENCHSELECT_N_FOLDS || true
unset BENCHSELECT_MAX_ROWS || true
unset BENCHSELECT_TEST_BATCH || true

unset BENCHSELECT_TABIMPUTE_MANIFEST_NAME || true
unset BENCHSELECT_TABIMPUTE_MANIFEST || true
unset BENCHSELECT_IMPUTER_MAX_COLS || true

export BENCHSELECT_TABIMPUTE_MODEL=current
export BENCHSELECT_TABIMPUTE_LABEL=current_public_chunked
export BENCHSELECT_IMPUTER_MAX_ROWS="${BENCHSELECT_CURRENT_MAX_ROWS:-500}"
run_stage "public current TabImpute checkpoint with row chunking" \
  python -u code/eval_tabimpute.py

# Direct matrix-to-matrix checkpoints were trained/evaluated with chunked
# windows; keep those limits explicit for the benchmark-selection matrices.
export BENCHSELECT_TABIMPUTE_MODEL=manifest
export BENCHSELECT_IMPUTER_MAX_ROWS="${BENCHSELECT_IMPUTER_MAX_ROWS:-50}"
export BENCHSELECT_IMPUTER_MAX_COLS="${BENCHSELECT_IMPUTER_MAX_COLS:-50}"
export BENCHSELECT_TABIMPUTE_MANIFEST="${BENCHSELECT_TABIMPUTE_MANIFEST:-/root/tabular/mcpfn/benchmark/movielens/configs/tabimpute_models.json}"

for model_name in dwip-v2-1 dwip-v2-2 dwip-v1; do
  export BENCHSELECT_TABIMPUTE_MANIFEST_NAME="${model_name}"
  export BENCHSELECT_TABIMPUTE_LABEL="${model_name}"
  run_stage "TabImpute manifest model ${model_name}" \
    python -u code/eval_tabimpute.py
done

unset BENCHSELECT_TABIMPUTE_MODEL
unset BENCHSELECT_TABIMPUTE_LABEL
unset BENCHSELECT_TABIMPUTE_MANIFEST_NAME
unset BENCHSELECT_TABIMPUTE_MANIFEST
unset BENCHSELECT_IMPUTER_MAX_ROWS
unset BENCHSELECT_IMPUTER_MAX_COLS

run_stage "benchmark-selection Gaussian entropy-vs-MI baseline" \
  python -u code/eval_entropy_vs_mi.py

run_stage "benchmark-selection greedy CV baseline" \
  python -u code/eval_greedy_all.py

log_step "Follow-up queue complete"
