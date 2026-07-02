#!/usr/bin/env bash
set -euo pipefail

# Continue the 100x100 / 50x50-chunk scratch compressed run from a full
# compressed checkpoint with a flat LR. This is meant to test whether the first
# 15k run plateaued because the cosine LR had decayed too aggressively.

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-100x100-50x50chunks-scratch-k20-15k/checkpoint_15000.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-100x100-50x50chunks-scratch-k20-flatlr-from15k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MATRIX_ROWS="${MATRIX_ROWS:-100}"
MATRIX_COLS="${MATRIX_COLS:-100}"
CHUNK_ROWS="${CHUNK_ROWS:-50}"
CHUNK_COLS="${CHUNK_COLS:-50}"
RANK="${RANK:-10}"
K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-1e-4}"
COMPRESSOR_LR="${COMPRESSOR_LR:-1e-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

python "${REPO_DIR}/scripts/train_compressed_cross_rank10.py" \
  --predictor-checkpoint "${CONFIG_CHECKPOINT}" \
  --config-checkpoint "${CONFIG_CHECKPOINT}" \
  --compressed-checkpoint "${COMPRESSED_CHECKPOINT}" \
  --train-from-scratch \
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
  --predictor-lr "${PREDICTOR_LR}" \
  --compressor-lr "${COMPRESSOR_LR}" \
  --scheduler-name constant \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --device "${DEVICE}" \
  --model-family "${MODEL_FAMILY}" \
  --wandb-project "${WANDB_PROJECT}"
