#!/usr/bin/env bash
set -euo pipefail

# Long RMSE-focused v2 continuation.
# Memory safety: batch size defaults to 1 while rows/cols are randomized up to
# 700 x 1700. This keeps the H200 peak comfortably below the 140GB cap in the
# max-shape smoke test while still exposing the compressor to full-context scale.

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-recsys-v2-rect150to300x600to1200-100chunks-random-k20-30k-from-recsysbest/best_checkpoint.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-recsys-v2-broadrect100to700x400to1700-100chunks-random-k20-100k-from-v2best-rmse}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
STABLE_STEPS="${STABLE_STEPS:-70000}"
DECAY_STEPS="${DECAY_STEPS:-25000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MATRIX_ROWS_MIN="${MATRIX_ROWS_MIN:-100}"
MATRIX_ROWS_MAX="${MATRIX_ROWS_MAX:-700}"
MATRIX_COLS_MIN="${MATRIX_COLS_MIN:-400}"
MATRIX_COLS_MAX="${MATRIX_COLS_MAX:-1700}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"

K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-1e-5}"
COMPRESSOR_LR="${COMPRESSOR_LR:-1e-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-2.0}"

RECSYS_DENSITY_LOW="${RECSYS_DENSITY_LOW:-0.04}"
RECSYS_DENSITY_HIGH="${RECSYS_DENSITY_HIGH:-0.06}"
RECSYS_POP_QUALITY_CORR_LOW="${RECSYS_POP_QUALITY_CORR_LOW:-0.1}"
RECSYS_POP_QUALITY_CORR_HIGH="${RECSYS_POP_QUALITY_CORR_HIGH:-0.5}"
RECSYS_MARGINAL_CALIBRATION_STEPS="${RECSYS_MARGINAL_CALIBRATION_STEPS:-5}"
RECSYS_MARGINAL_BINARY_STEPS="${RECSYS_MARGINAL_BINARY_STEPS:-22}"

# RMSE-focused rating distribution. All batches are clipped to the MovieLens
# range; 30% are half-star quantized and 70% remain continuous clipped targets.
RECSYS_GLOBAL_MEAN="${RECSYS_GLOBAL_MEAN:-3.5}"
RECSYS_RATING_CLIP_PROBABILITY="${RECSYS_RATING_CLIP_PROBABILITY:-1.0}"
RECSYS_RATING_QUANTIZATION_PROBABILITY="${RECSYS_RATING_QUANTIZATION_PROBABILITY:-0.3}"
RECSYS_RATING_MIN="${RECSYS_RATING_MIN:-1.0}"
RECSYS_RATING_MAX="${RECSYS_RATING_MAX:-5.0}"

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
  --recsys-global-mean "${RECSYS_GLOBAL_MEAN}" \
  --recsys-rating-min "${RECSYS_RATING_MIN}" \
  --recsys-rating-max "${RECSYS_RATING_MAX}" \
  --recsys-rating-clip-probability "${RECSYS_RATING_CLIP_PROBABILITY}" \
  --recsys-rating-quantization-probability "${RECSYS_RATING_QUANTIZATION_PROBABILITY}" \
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
