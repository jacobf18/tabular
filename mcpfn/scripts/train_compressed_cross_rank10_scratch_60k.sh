#!/usr/bin/env bash
set -euo pipefail

# Train the compressed cross-chunk TabImpute architecture fully from scratch.
#
# The config checkpoint is used only as an architecture/optimizer/prior template.
# No predictor weights are loaded, and compressor weights are not copied from the
# predictor. All three models train jointly for 60k steps.
#
# Useful overrides:
#   K_ROW=20 K_COL=20 bash scripts/train_compressed_cross_rank10_scratch_60k.sh
#   BATCH_SIZE=2 bash scripts/train_compressed_cross_rank10_scratch_60k.sh
#   WANDB_MODE=offline bash scripts/train_compressed_cross_rank10_scratch_60k.sh

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-200x200-100x100chunks-scratch-k20-60k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MATRIX_ROWS="${MATRIX_ROWS:-200}"
MATRIX_COLS="${MATRIX_COLS:-200}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"
RANK="${RANK:-10}"
K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR_MULTIPLIER="${PREDICTOR_LR_MULTIPLIER:-1.0}"
PREDICTOR_LR="${PREDICTOR_LR:-}"
COMPRESSOR_LR="${COMPRESSOR_LR:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=(--train-from-scratch)
if [[ -n "${PREDICTOR_LR}" ]]; then
  EXTRA_ARGS+=(--predictor-lr "${PREDICTOR_LR}")
fi
if [[ -n "${COMPRESSOR_LR}" ]]; then
  EXTRA_ARGS+=(--compressor-lr "${COMPRESSOR_LR}")
fi

python "${REPO_DIR}/scripts/train_compressed_cross_rank10.py" \
  --predictor-checkpoint "${CONFIG_CHECKPOINT}" \
  --config-checkpoint "${CONFIG_CHECKPOINT}" \
  --work-dir "${WORK_DIR}" \
  --run-name "${RUN_NAME}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --matrix-rows "${MATRIX_ROWS}" \
  --matrix-cols "${MATRIX_COLS}" \
  --chunk-rows "${CHUNK_ROWS}" \
  --chunk-cols "${CHUNK_COLS}" \
  --rank "${RANK}" \
  --k-row "${K_ROW}" \
  --k-col "${K_COL}" \
  --compressor-layers "${COMPRESSOR_LAYERS}" \
  --predictor-lr-multiplier "${PREDICTOR_LR_MULTIPLIER}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --device "${DEVICE}" \
  --model-family "${MODEL_FAMILY}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${EXTRA_ARGS[@]}"
