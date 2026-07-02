#!/usr/bin/env bash
set -euo pipefail

# Short experiment: train compressed cross-chunk TabImpute with compressor
# scalar encoders, CLS tokens, and axial transformer blocks initialized from
# the pretrained 100x100 predictor checkpoint.

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-200x200-100x100chunks-predictor-init-4k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"

STEPS="${STEPS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MATRIX_ROWS="${MATRIX_ROWS:-200}"
MATRIX_COLS="${MATRIX_COLS:-200}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"
RANK="${RANK:-10}"
K_ROW="${K_ROW:-4}"
K_COL="${K_COL:-4}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
COMPRESSOR_LR="${COMPRESSOR_LR:-2e-4}"
PREDICTOR_LR_MULTIPLIER="${PREDICTOR_LR_MULTIPLIER:-0.1}"
FREEZE_PREDICTOR="${FREEZE_PREDICTOR:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-stable}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=(--init-compressor-transformers-from-predictor)
if [[ -n "${COMPRESSOR_LR}" ]]; then
  EXTRA_ARGS+=(--compressor-lr "${COMPRESSOR_LR}")
fi
if [[ "${FREEZE_PREDICTOR}" == "1" || "${FREEZE_PREDICTOR}" == "true" || "${FREEZE_PREDICTOR}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--freeze-predictor)
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
