#!/usr/bin/env bash
set -euo pipefail

# 20k exposed-holdout curriculum:
# - starts from the current exposed fitted-bias checkpoint
# - increases rows/cols over four phases
# - decreases exposure density toward MovieLens-like sparsity

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-recsys-exposed-fitted_bias-broadrect100to700x400to1700-100chunks-random-k20-50k-from-broadbest/best_checkpoint.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-recsys-exposed-fitted-curriculum20k-100chunks-random-k20-from-exposedbest}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
STABLE_STEPS="${STABLE_STEPS:-14000}"
DECAY_STEPS="${DECAY_STEPS:-5000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"

K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-5e-6}"
COMPRESSOR_LR="${COMPRESSOR_LR:-1e-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-2.0}"

RECSYS_HOLDOUT_TARGET_FRACTION_LOW="${RECSYS_HOLDOUT_TARGET_FRACTION_LOW:-0.10}"
RECSYS_HOLDOUT_TARGET_FRACTION_HIGH="${RECSYS_HOLDOUT_TARGET_FRACTION_HIGH:-0.30}"
RECSYS_HOLDOUT_BIAS_RIDGE_REG="${RECSYS_HOLDOUT_BIAS_RIDGE_REG:-2.0}"
RECSYS_HOLDOUT_BIAS_RIDGE_ITERS="${RECSYS_HOLDOUT_BIAS_RIDGE_ITERS:-30}"

RECSYS_BIAS_USER_STD_LOW="${RECSYS_BIAS_USER_STD_LOW:-0.25}"
RECSYS_BIAS_USER_STD_HIGH="${RECSYS_BIAS_USER_STD_HIGH:-0.60}"
RECSYS_BIAS_ITEM_STD_LOW="${RECSYS_BIAS_ITEM_STD_LOW:-0.35}"
RECSYS_BIAS_ITEM_STD_HIGH="${RECSYS_BIAS_ITEM_STD_HIGH:-0.80}"
RECSYS_BIAS_RESIDUAL_RANK_LOW="${RECSYS_BIAS_RESIDUAL_RANK_LOW:-5}"
RECSYS_BIAS_RESIDUAL_RANK_HIGH="${RECSYS_BIAS_RESIDUAL_RANK_HIGH:-30}"
RECSYS_BIAS_RESIDUAL_SCALE_LOW="${RECSYS_BIAS_RESIDUAL_SCALE_LOW:-0.15}"
RECSYS_BIAS_RESIDUAL_SCALE_HIGH="${RECSYS_BIAS_RESIDUAL_SCALE_HIGH:-0.60}"
RECSYS_BIAS_NOISE_STD_LOW="${RECSYS_BIAS_NOISE_STD_LOW:-0.05}"
RECSYS_BIAS_NOISE_STD_HIGH="${RECSYS_BIAS_NOISE_STD_HIGH:-0.25}"
RECSYS_BIAS_QUANTIZATION_PROB="${RECSYS_BIAS_QUANTIZATION_PROB:-0.8}"
RECSYS_EXPOSURE_PREF_WEIGHT_LOW="${RECSYS_EXPOSURE_PREF_WEIGHT_LOW:-0.20}"
RECSYS_EXPOSURE_PREF_WEIGHT_HIGH="${RECSYS_EXPOSURE_PREF_WEIGHT_HIGH:-0.75}"
RECSYS_EXPOSURE_CLUSTER_BONUS_STD="${RECSYS_EXPOSURE_CLUSTER_BONUS_STD:-0.5}"

DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
DISABLE_CUDNN_SDPA="${DISABLE_CUDNN_SDPA:-1}"

python "${REPO_DIR}/scripts/train_compressed_cross_rank10.py" \
  --predictor-checkpoint "${CONFIG_CHECKPOINT}" \
  --config-checkpoint "${CONFIG_CHECKPOINT}" \
  --compressed-checkpoint "${COMPRESSED_CHECKPOINT}" \
  --train-from-scratch \
  --work-dir "${WORK_DIR}" \
  --run-name "${RUN_NAME}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --randomize-matrix-size \
  --matrix-rows-min 150 \
  --matrix-rows-max 1000 \
  --matrix-cols-min 300 \
  --matrix-cols-max 1800 \
  --chunk-rows "${CHUNK_ROWS}" \
  --chunk-cols "${CHUNK_COLS}" \
  --target-chunk-sampling random \
  --prior-kind rec_sys_bias_residual_exposed_fitted_bias \
  --best-metric exposed_holdout_rmse \
  --recsys-curriculum movielens_proxy_20k \
  --recsys-holdout-target-fraction-low "${RECSYS_HOLDOUT_TARGET_FRACTION_LOW}" \
  --recsys-holdout-target-fraction-high "${RECSYS_HOLDOUT_TARGET_FRACTION_HIGH}" \
  --recsys-holdout-bias-ridge-reg "${RECSYS_HOLDOUT_BIAS_RIDGE_REG}" \
  --recsys-holdout-bias-ridge-iters "${RECSYS_HOLDOUT_BIAS_RIDGE_ITERS}" \
  --recsys-bias-global-mean-low 3.3 \
  --recsys-bias-global-mean-high 3.7 \
  --recsys-bias-user-std-low "${RECSYS_BIAS_USER_STD_LOW}" \
  --recsys-bias-user-std-high "${RECSYS_BIAS_USER_STD_HIGH}" \
  --recsys-bias-item-std-low "${RECSYS_BIAS_ITEM_STD_LOW}" \
  --recsys-bias-item-std-high "${RECSYS_BIAS_ITEM_STD_HIGH}" \
  --recsys-bias-user-distribution normal \
  --recsys-bias-item-distribution lognormal \
  --recsys-bias-residual-rank-low "${RECSYS_BIAS_RESIDUAL_RANK_LOW}" \
  --recsys-bias-residual-rank-high "${RECSYS_BIAS_RESIDUAL_RANK_HIGH}" \
  --recsys-bias-residual-scale-low "${RECSYS_BIAS_RESIDUAL_SCALE_LOW}" \
  --recsys-bias-residual-scale-high "${RECSYS_BIAS_RESIDUAL_SCALE_HIGH}" \
  --recsys-bias-noise-std-low "${RECSYS_BIAS_NOISE_STD_LOW}" \
  --recsys-bias-noise-std-high "${RECSYS_BIAS_NOISE_STD_HIGH}" \
  --recsys-bias-rating-quantization-probability "${RECSYS_BIAS_QUANTIZATION_PROB}" \
  --recsys-exposure-preference-weight-low "${RECSYS_EXPOSURE_PREF_WEIGHT_LOW}" \
  --recsys-exposure-preference-weight-high "${RECSYS_EXPOSURE_PREF_WEIGHT_HIGH}" \
  --recsys-exposure-cluster-bonus-std "${RECSYS_EXPOSURE_CLUSTER_BONUS_STD}" \
  --k-row "${K_ROW}" \
  --k-col "${K_COL}" \
  --compressor-layers "${COMPRESSOR_LAYERS}" \
  --predictor-lr "${PREDICTOR_LR}" \
  --compressor-lr "${COMPRESSOR_LR}" \
  --scheduler-name warmup_stable_cosine \
  --warmup-steps "${WARMUP_STEPS}" \
  --stable-steps "${STABLE_STEPS}" \
  --decay-steps "${DECAY_STEPS}" \
  --min-lr-ratio "${MIN_LR_RATIO}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --device "${DEVICE}" \
  --model-family "${MODEL_FAMILY}" \
  --wandb-project "${WANDB_PROJECT}" \
  $(if [[ "${DISABLE_CUDNN_SDPA}" == "1" ]]; then printf '%s' "--disable-cudnn-sdpa"; fi)
