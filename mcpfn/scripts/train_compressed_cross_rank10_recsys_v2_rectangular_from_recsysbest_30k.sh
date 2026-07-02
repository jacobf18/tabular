#!/usr/bin/env bash
set -euo pipefail

# Rec-sys exposure-prior v2 continuation.
# Changes vs v1:
# - rectangular user x item matrices
# - marginal row/column degree calibration
# - exposure coupled to full preference utility, not residual only
# - heavy-tailed user activity and item popularity

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-recsys150to300-100x100chunks-random-k20-100k-from-phase1best-lr2e5-min5e6-clean/best_checkpoint.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-recsys-v2-rect150to300x600to1200-100chunks-random-k20-30k-from-recsysbest}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-30000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
STABLE_STEPS="${STABLE_STEPS:-20000}"
DECAY_STEPS="${DECAY_STEPS:-8000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"

BATCH_SIZE="${BATCH_SIZE:-2}"
MATRIX_ROWS_MIN="${MATRIX_ROWS_MIN:-150}"
MATRIX_ROWS_MAX="${MATRIX_ROWS_MAX:-300}"
MATRIX_COLS_MIN="${MATRIX_COLS_MIN:-600}"
MATRIX_COLS_MAX="${MATRIX_COLS_MAX:-1200}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"

K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-2e-5}"
COMPRESSOR_LR="${COMPRESSOR_LR:-2e-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.25}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-2.0}"

RECSYS_DENSITY_LOW="${RECSYS_DENSITY_LOW:-0.04}"
RECSYS_DENSITY_HIGH="${RECSYS_DENSITY_HIGH:-0.06}"
RECSYS_POP_QUALITY_CORR_LOW="${RECSYS_POP_QUALITY_CORR_LOW:-0.1}"
RECSYS_POP_QUALITY_CORR_HIGH="${RECSYS_POP_QUALITY_CORR_HIGH:-0.5}"
RECSYS_MARGINAL_CALIBRATION_STEPS="${RECSYS_MARGINAL_CALIBRATION_STEPS:-6}"
RECSYS_MARGINAL_BINARY_STEPS="${RECSYS_MARGINAL_BINARY_STEPS:-24}"

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
  --matrix-rows-min "${MATRIX_ROWS_MIN}" \
  --matrix-rows-max "${MATRIX_ROWS_MAX}" \
  --matrix-cols-min "${MATRIX_COLS_MIN}" \
  --matrix-cols-max "${MATRIX_COLS_MAX}" \
  --chunk-rows "${CHUNK_ROWS}" \
  --chunk-cols "${CHUNK_COLS}" \
  --target-chunk-sampling random \
  --prior-kind rec_sys_spectrum_exposure \
  --recsys-density-low "${RECSYS_DENSITY_LOW}" \
  --recsys-density-high "${RECSYS_DENSITY_HIGH}" \
  --recsys-exposure-calibration marginal \
  --recsys-exposure-uses-full-preference \
  --recsys-user-activity-distribution lognormal \
  --recsys-item-popularity-distribution lognormal \
  --recsys-popularity-quality-corr-low "${RECSYS_POP_QUALITY_CORR_LOW}" \
  --recsys-popularity-quality-corr-high "${RECSYS_POP_QUALITY_CORR_HIGH}" \
  --recsys-exposure-item-quality-weight 0.5 \
  --recsys-exposure-user-bias-weight 0.1 \
  --recsys-preference-temperature 1.0 \
  --recsys-marginal-calibration-steps "${RECSYS_MARGINAL_CALIBRATION_STEPS}" \
  --recsys-marginal-binary-steps "${RECSYS_MARGINAL_BINARY_STEPS}" \
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
