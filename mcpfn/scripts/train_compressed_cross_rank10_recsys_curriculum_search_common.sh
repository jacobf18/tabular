#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-recsys-exposed-fitted_bias-broadrect100to700x400to1700-100chunks-random-k20-50k-from-broadbest/best_checkpoint.pth}"

: "${RUN_NAME:?Set RUN_NAME for the search configuration.}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
STABLE_STEPS="${STABLE_STEPS:-4250}"
DECAY_STEPS="${DECAY_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"

K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-1e-5}"
COMPRESSOR_LR="${COMPRESSOR_LR:-1e-5}"
ENCODER_LR="${ENCODER_LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.03}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-2.0}"
RANDOM_TARGETS_PER_STEP="${RANDOM_TARGETS_PER_STEP:-1}"
MATRIX_ROWS_MIN="${MATRIX_ROWS_MIN:-150}"
MATRIX_ROWS_MAX="${MATRIX_ROWS_MAX:-1000}"
MATRIX_COLS_MIN="${MATRIX_COLS_MIN:-300}"
MATRIX_COLS_MAX="${MATRIX_COLS_MAX:-1800}"
RECSYS_CURRICULUM="${RECSYS_CURRICULUM:-movielens_proxy_20k}"
EXPOSURE_DENSITY_LOW="${EXPOSURE_DENSITY_LOW:-0.04}"
EXPOSURE_DENSITY_HIGH="${EXPOSURE_DENSITY_HIGH:-0.06}"
SKIP_FINAL_CHECKPOINT="${SKIP_FINAL_CHECKPOINT:-0}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
RABEL_NUM_KERNELS="${RABEL_NUM_KERNELS:-64}"
RABEL_SIGMA="${RABEL_SIGMA:-1.0}"
RABEL_EXPONENT_DIGITS="${RABEL_EXPONENT_DIGITS:-1}"
RABEL_EXPONENT_EMBEDDING_SIZE="${RABEL_EXPONENT_EMBEDDING_SIZE:-32}"
RABEL_INITIAL_SCALE="${RABEL_INITIAL_SCALE:-0.05}"

WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_MODEL_ARGS=()
if [[ -n "${ENCODER_LR}" ]]; then
  EXTRA_MODEL_ARGS+=(--encoder-lr "${ENCODER_LR}")
fi
if [[ "${MODEL_FAMILY}" == "rabel" ]]; then
  EXTRA_MODEL_ARGS+=(
    --rabel-num-kernels "${RABEL_NUM_KERNELS}"
    --rabel-sigma "${RABEL_SIGMA}"
    --rabel-exponent-digits "${RABEL_EXPONENT_DIGITS}"
    --rabel-exponent-embedding-size "${RABEL_EXPONENT_EMBEDDING_SIZE}"
    --rabel-initial-scale "${RABEL_INITIAL_SCALE}"
  )
fi

python "${REPO_DIR}/scripts/train_compressed_cross_rank10.py" \
  --predictor-checkpoint "${CONFIG_CHECKPOINT}" \
  --config-checkpoint "${CONFIG_CHECKPOINT}" \
  --compressed-checkpoint "${COMPRESSED_CHECKPOINT}" \
  --allow-compressor-query-resize \
  --train-from-scratch \
  --work-dir "${WORK_DIR}" \
  --run-name "${RUN_NAME}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --randomize-matrix-size \
  --matrix-rows-min "${MATRIX_ROWS_MIN}" \
  --matrix-rows-max "${MATRIX_ROWS_MAX}" \
  --matrix-cols-min "${MATRIX_COLS_MIN}" \
  --matrix-cols-max "${MATRIX_COLS_MAX}" \
  --chunk-rows 100 \
  --chunk-cols 100 \
  --target-chunk-sampling random \
  --random-targets-per-step "${RANDOM_TARGETS_PER_STEP}" \
  --prior-kind rec_sys_bias_residual_exposed_fitted_bias \
  --best-metric exposed_holdout_rmse \
  --recsys-curriculum "${RECSYS_CURRICULUM}" \
  --recsys-holdout-target-fraction-low 0.10 \
  --recsys-holdout-target-fraction-high 0.30 \
  --recsys-holdout-bias-ridge-reg 2.0 \
  --recsys-holdout-bias-ridge-iters 30 \
  --recsys-bias-global-mean-low 3.3 \
  --recsys-bias-global-mean-high 3.7 \
  --recsys-bias-user-std-low 0.25 \
  --recsys-bias-user-std-high 0.60 \
  --recsys-bias-item-std-low 0.35 \
  --recsys-bias-item-std-high 0.80 \
  --recsys-bias-user-distribution normal \
  --recsys-bias-item-distribution lognormal \
  --recsys-bias-residual-rank-low 5 \
  --recsys-bias-residual-rank-high 30 \
  --recsys-bias-residual-scale-low 0.15 \
  --recsys-bias-residual-scale-high 0.60 \
  --recsys-bias-noise-std-low 0.05 \
  --recsys-bias-noise-std-high 0.25 \
  --recsys-bias-rating-quantization-probability 0.8 \
  --recsys-exposure-density-low "${EXPOSURE_DENSITY_LOW}" \
  --recsys-exposure-density-high "${EXPOSURE_DENSITY_HIGH}" \
  --recsys-exposure-preference-weight-low 0.20 \
  --recsys-exposure-preference-weight-high 0.75 \
  --recsys-exposure-cluster-bonus-std 0.5 \
  --k-row "${K_ROW}" \
  --k-col "${K_COL}" \
  --compressor-layers "${COMPRESSOR_LAYERS}" \
  --predictor-lr "${PREDICTOR_LR}" \
  --compressor-lr "${COMPRESSOR_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --scheduler-name warmup_stable_cosine \
  --warmup-steps "${WARMUP_STEPS}" \
  --stable-steps "${STABLE_STEPS}" \
  --decay-steps "${DECAY_STEPS}" \
  --min-lr-ratio "${MIN_LR_RATIO}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --checkpoint-every 0 \
  --lightweight-checkpoints \
  --disable-best-checkpoint \
  --device cuda \
  --model-family "${MODEL_FAMILY}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --disable-cudnn-sdpa \
  $(if [[ "${SKIP_FINAL_CHECKPOINT}" == "1" ]]; then printf '%s' "--skip-final-checkpoint"; fi)
