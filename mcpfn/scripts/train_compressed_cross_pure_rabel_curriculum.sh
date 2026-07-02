#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-tabimpute-compressed-pure-rabel64-rank10-k20-curriculum-160k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-rank10-curriculum-160k/checkpoint_160000.pth}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "${REPO_DIR}/scripts/train_compressed_cross_pure_rabel_curriculum.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --predictor-checkpoint "${PREDICTOR_CHECKPOINT}" \
  --k-row "${K_ROW:-20}" \
  --k-col "${K_COL:-20}" \
  --predictor-lr "${PREDICTOR_LR:-2e-5}" \
  --compressor-lr "${COMPRESSOR_LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.03}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
  --warmup-steps "${WARMUP_STEPS:-5000}" \
  --stable-steps "${STABLE_STEPS:-125000}" \
  --decay-steps "${DECAY_STEPS:-30000}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.1}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --device "${DEVICE:-cuda}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
