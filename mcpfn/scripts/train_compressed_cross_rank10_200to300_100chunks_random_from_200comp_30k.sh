#!/usr/bin/env bash
set -euo pipefail

# Continue from the strong compressed 200x200 model, but train on random
# rectangular/square-ish matrix sizes with fixed requested 100x100 chunks.
#
# Per batch:
#   rows ~ UniformInteger[200, 300]
#   cols ~ UniformInteger[200, 300]
#   chunk target = 100 x 100
#
# The compressed model code balances non-divisible chunk spans, so a 250-wide
# axis becomes similarly sized chunks instead of ending with a tiny remainder.

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_45000.pth}"
COMPRESSED_CHECKPOINT="${COMPRESSED_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-comp-rank10-200x200-100x100chunks-random-k20-100k-from-100comp/checkpoint_100000.pth}"

RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-200to300-100x100chunks-random-k20-30k-from-200comp}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

STEPS="${STEPS:-30000}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
STABLE_STEPS="${STABLE_STEPS:-0}"
DECAY_STEPS="${DECAY_STEPS:-27000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MATRIX_ROWS_MIN="${MATRIX_ROWS_MIN:-200}"
MATRIX_ROWS_MAX="${MATRIX_ROWS_MAX:-300}"
MATRIX_COLS_MIN="${MATRIX_COLS_MIN:-200}"
MATRIX_COLS_MAX="${MATRIX_COLS_MAX:-300}"
CHUNK_ROWS="${CHUNK_ROWS:-100}"
CHUNK_COLS="${CHUNK_COLS:-100}"
RANK="${RANK:-10}"
K_ROW="${K_ROW:-20}"
K_COL="${K_COL:-20}"
COMPRESSOR_LAYERS="${COMPRESSOR_LAYERS:-8}"
PREDICTOR_LR="${PREDICTOR_LR:-1e-4}"
COMPRESSOR_LR="${COMPRESSOR_LR:-1e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-2.0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
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
  --rank "${RANK}" \
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
