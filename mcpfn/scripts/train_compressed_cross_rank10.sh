#!/usr/bin/env bash
set -euo pipefail

# Train the compressed cross-chunk TabImpute architecture.
#
# This warm-starts the main predictor from the 100x100 rank-10 curriculum
# checkpoint, initializes row/column compressors randomly, and trains on
# fixed-rank 100x100 matrices partitioned into 50x50 chunks.
#
# Useful overrides:
#   STEPS=30000 BATCH_SIZE=2 bash scripts/train_compressed_cross_rank10.sh
#   K_ROW=8 K_COL=8 bash scripts/train_compressed_cross_rank10.sh
#   COMPRESSOR_LAYERS=4 bash scripts/train_compressed_cross_rank10.sh
#   PREDICTOR_LR_MULTIPLIER=0.05 bash scripts/train_compressed_cross_rank10.sh
#   FREEZE_PREDICTOR=1 bash scripts/train_compressed_cross_rank10.sh
#   INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR=0 bash scripts/train_compressed_cross_rank10.sh
#   INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR=1 bash scripts/train_compressed_cross_rank10.sh

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-bigger-rank10-200x200-100x100chunks-cls-frozen}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MATRIX_ROWS="${MATRIX_ROWS:-200}"
MATRIX_COLS="${MATRIX_COLS:-200}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"
RANK="${RANK:-10}"
K_ROW="${K_ROW:-4}"
K_COL="${K_COL:-4}"
# Match the loaded 100x100 predictor's attention depth by default.
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR_MULTIPLIER="${PREDICTOR_LR_MULTIPLIER:-0.1}"
PREDICTOR_LR="${PREDICTOR_LR:-}"
COMPRESSOR_LR="${COMPRESSOR_LR:-2e-4}"
FREEZE_PREDICTOR="${FREEZE_PREDICTOR:-1}"
INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR="${INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR:-1}"
INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR="${INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=()
if [[ -n "${PREDICTOR_LR}" ]]; then
  EXTRA_ARGS+=(--predictor-lr "${PREDICTOR_LR}")
fi
if [[ -n "${COMPRESSOR_LR}" ]]; then
  EXTRA_ARGS+=(--compressor-lr "${COMPRESSOR_LR}")
fi
if [[ "${FREEZE_PREDICTOR}" == "1" || "${FREEZE_PREDICTOR}" == "true" || "${FREEZE_PREDICTOR}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--freeze-predictor)
fi
if [[ "${INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR}" == "0" || "${INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR}" == "false" || "${INIT_COMPRESSOR_EMBEDDERS_FROM_PREDICTOR}" == "FALSE" ]]; then
  EXTRA_ARGS+=(--no-init-compressor-embedders-from-predictor)
fi
if [[ "${INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR}" == "1" || "${INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR}" == "true" || "${INIT_COMPRESSOR_TRANSFORMERS_FROM_PREDICTOR}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--init-compressor-transformers-from-predictor)
fi

python "${REPO_DIR}/scripts/train_compressed_cross_rank10.py" \
  --predictor-checkpoint "${PREDICTOR_CHECKPOINT}" \
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
