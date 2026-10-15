#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-latent512-randomrank-lowrank-100k-scratch}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

DEFAULT_PHASES_JSON='[{"name":"50_square","end_step":25000,"rows_low":50,"rows_high":50,"cols_low":50,"cols_high":50,"batch_size":32},{"name":"50_to_100","end_step":50000,"rows_low":50,"rows_high":100,"cols_low":50,"cols_high":100,"batch_size":16},{"name":"100_to_200","end_step":75000,"rows_low":100,"rows_high":200,"cols_low":100,"cols_high":200,"batch_size":8},{"name":"200_to_300","end_step":100000,"rows_low":200,"rows_high":300,"cols_low":200,"cols_high":300,"batch_size":4}]'
PHASES_JSON="${PHASES_JSON:-${DEFAULT_PHASES_JSON}}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${INIT_MODEL_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--init-model-checkpoint "${INIT_MODEL_CHECKPOINT}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "${REPO_DIR}/scripts/train_direct_rowcol_latent_random_rank_curriculum.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-43}" \
  --prior-kind latent_factor \
  --missingness-type "${MISSINGNESS_TYPE:-mcar}" \
  --p-missing "${P_MISSING:-0.4}" \
  --latent-rank-low "${LATENT_RANK_LOW:-1}" \
  --latent-rank-high "${LATENT_RANK_HIGH:-40}" \
  --lr "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.03}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
  --warmup-steps "${WARMUP_STEPS:-5000}" \
  --stable-steps "${STABLE_STEPS:-75000}" \
  --decay-steps "${DECAY_STEPS:-20000}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --phases-json "${PHASES_JSON}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
