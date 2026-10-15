#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-pure-rabel64-joint-spectrum-cateblock-60k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
else
  EXTRA_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT:-/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-lr2e5/best_checkpoint.pth}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "${REPO_DIR}/scripts/train_direct_pure_rabel_joint_matrix_causal.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --device "${DEVICE:-cuda}" \
  --steps "${STEPS:-60000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --rows-low "${ROWS_LOW:-150}" \
  --rows-high "${ROWS_HIGH:-300}" \
  --cols-low "${COLS_LOW:-50}" \
  --cols-high "${COLS_HIGH:-300}" \
  --causal-prob "${CAUSAL_PROB:-0.35}" \
  --cate-covariates-low "${CATE_COVARIATES_LOW:-4}" \
  --cate-covariates-high "${CATE_COVARIATES_HIGH:-14}" \
  --skinny-block-prob "${SKINNY_BLOCK_PROB:-0.0}" \
  --skinny-cols-low "${SKINNY_COLS_LOW:-6}" \
  --skinny-cols-high "${SKINNY_COLS_HIGH:-20}" \
  --skinny-rank-low "${SKINNY_RANK_LOW:-2}" \
  --skinny-rank-high "${SKINNY_RANK_HIGH:-8}" \
  --skinny-extra-mcar-prob "${SKINNY_EXTRA_MCAR_PROB:-0.0}" \
  --cate-tau-scale "${CATE_TAU_SCALE:-1.0}" \
  --cate-query-noise-scale "${CATE_QUERY_NOISE_SCALE:-1.0}" \
  --query-treatment-mode "${QUERY_TREATMENT_MODE:-factual}" \
  --lr "${LR:-2e-5}" \
  --min-lr "${MIN_LR:-5e-6}" \
  --weight-decay "${WEIGHT_DECAY:-0.03}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --warmup-steps "${WARMUP_STEPS:-2000}" \
  --stable-steps "${STABLE_STEPS:-45000}" \
  --decay-steps "${DECAY_STEPS:-13000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --seed "${SEED:-41}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
