#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-lr1e4-drop5e5}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
export RUN_NAME WORK_DIR WANDB_MODE

LR="${LR:-1e-4}" \
POST_WARMUP_LR="${POST_WARMUP_LR:-5e-5}" \
WARMUP_STEPS="${WARMUP_STEPS:-5000}" \
STABLE_STEPS="${STABLE_STEPS:-125000}" \
DECAY_STEPS="${DECAY_STEPS:-30000}" \
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" \
bash "${REPO_DIR}/scripts/train_direct_pure_rabel_spectrum_curriculum.sh"
