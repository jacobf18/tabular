#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-pure-rabel64-scaledattn-tallskinny-50k-from-tallbest}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
else
  EXTRA_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-tallskinny-predictor-only-50k-from-oosbest/best_checkpoint.pth}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "${REPO_DIR}/scripts/train_direct_pure_rabel_scaled_attention_joint_matrix_causal.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --device "${DEVICE:-cuda}" \
  --steps "${STEPS:-50000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --rows-low "${ROWS_LOW:-70}" \
  --rows-high "${ROWS_HIGH:-2200}" \
  --cols-low "${COLS_LOW:-3}" \
  --cols-high "${COLS_HIGH:-4}" \
  --causal-prob "${CAUSAL_PROB:-0.5}" \
  --cate-covariates-low "${CATE_COVARIATES_LOW:-1}" \
  --cate-covariates-high "${CATE_COVARIATES_HIGH:-2}" \
  --skinny-block-prob "${SKINNY_BLOCK_PROB:-0.5}" \
  --skinny-cols-low "${SKINNY_COLS_LOW:-3}" \
  --skinny-cols-high "${SKINNY_COLS_HIGH:-4}" \
  --skinny-rank-low "${SKINNY_RANK_LOW:-2}" \
  --skinny-rank-high "${SKINNY_RANK_HIGH:-3}" \
  --skinny-extra-mcar-prob "${SKINNY_EXTRA_MCAR_PROB:-0.0}" \
  --cate-tau-scale "${CATE_TAU_SCALE:-1.5}" \
  --cate-query-noise-scale "${CATE_QUERY_NOISE_SCALE:-0.2}" \
  --query-treatment-mode "${QUERY_TREATMENT_MODE:-paired}" \
  --lr "${LR:-5e-6}" \
  --min-lr "${MIN_LR:-2e-6}" \
  --weight-decay "${WEIGHT_DECAY:-0.03}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --stable-steps "${STABLE_STEPS:-39000}" \
  --decay-steps "${DECAY_STEPS:-10000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --seed "${SEED:-41}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
