#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

BASE_RUN_NAME="${BASE_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-large-80k-from100k}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/${BASE_RUN_NAME}/checkpoint_80000.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-sharpen-60k-fromlarge80k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

DEFAULT_PHASES_JSON='[{"name":"50_to_125","end_step":15000,"rows_low":50,"rows_high":125,"cols_low":50,"cols_high":125,"batch_size":8},{"name":"125_to_300","end_step":40000,"rows_low":125,"rows_high":300,"cols_low":125,"cols_high":300,"batch_size":3},{"name":"300_to_500","end_step":60000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}]'
PHASES_JSON="${PHASES_JSON:-${DEFAULT_PHASES_JSON}}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
else
  EXTRA_ARGS+=(--init-model-checkpoint "${INIT_MODEL_CHECKPOINT:-${BASE_CHECKPOINT}}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-47}" \
  --prior-kind latent_factor \
  --missingness-type "${MISSINGNESS_TYPE:-mcar}" \
  --p-missing "${P_MISSING:-0.2}" \
  --latent-rank-low "${LATENT_RANK_LOW:-10}" \
  --latent-rank-high "${LATENT_RANK_HIGH:-10}" \
  --lr "${LR:-2e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.02}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --stable-steps "${STABLE_STEPS:-44000}" \
  --decay-steps "${DECAY_STEPS:-15000}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.25}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --point-loss-weight "${POINT_LOSS_WEIGHT:-50.0}" \
  --point-loss-kind "${POINT_LOSS_KIND:-mse}" \
  --point-loss-prediction "${POINT_LOSS_PREDICTION:-mean}" \
  --phases-json "${PHASES_JSON}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
