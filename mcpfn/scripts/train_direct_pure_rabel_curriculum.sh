#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-pure-rabel64-rank10-curriculum-160k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

python "${REPO_DIR}/scripts/train_direct_pure_rabel_curriculum.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --lr "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.03}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-0.5}" \
  --warmup-steps "${WARMUP_STEPS:-5000}" \
  --stable-steps "${STABLE_STEPS:-125000}" \
  --decay-steps "${DECAY_STEPS:-30000}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.1}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
  --device "${DEVICE:-cuda}" \
  --wandb-project "${WANDB_PROJECT:-tabimpute}" \
  "${EXTRA_ARGS[@]}"
