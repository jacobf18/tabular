#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-lr2e5}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi
if [[ -n "${POST_WARMUP_LR:-}" ]]; then
  EXTRA_ARGS+=(--post-warmup-lr "${POST_WARMUP_LR}")
fi

python "${REPO_DIR}/scripts/train_direct_pure_rabel_curriculum.py" \
  --run-name "${RUN_NAME}" \
  --work-dir "${WORK_DIR}" \
  --prior-kind spectrum_decay \
  --p-missing "${P_MISSING:-0.4}" \
  --spectrum-rank-low "${SPECTRUM_RANK_LOW:-80}" \
  --spectrum-rank-high "${SPECTRUM_RANK_HIGH:-128}" \
  --spectrum-target-rank "${SPECTRUM_TARGET_RANK:-40}" \
  --spectrum-alpha-low "${SPECTRUM_ALPHA_LOW:-0.30}" \
  --spectrum-alpha-high "${SPECTRUM_ALPHA_HIGH:-0.50}" \
  --spectrum-noise-std-low "${SPECTRUM_NOISE_STD_LOW:-0.001}" \
  --spectrum-noise-std-high "${SPECTRUM_NOISE_STD_HIGH:-0.005}" \
  --lr "${LR:-2e-5}" \
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
