#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5}"
INIT_CKPT="${INIT_CKPT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-svthybrid768-spectral-rank80_steps0_50k_lr2e5/best_checkpoint.pth}"

PHASES_JSON='[
  {
    "name": "mixed_300_500_050k",
    "end_step": 50000,
    "rows_low": 300,
    "rows_high": 500,
    "cols_low": 300,
    "cols_high": 500,
    "batch_size": 1
  }
]'

MODEL_OVERRIDES_JSON='{
  "svt_steps": 0,
  "svt_init_rank": 80,
  "svt_max_rank": 80,
  "svt_init_scale": 0.02,
  "svt_threshold_fraction_init": 0.006,
  "svt_threshold_fraction_min": 0.004,
  "svt_threshold_fraction_max": 0.02,
  "svt_threshold_mode": "inverse_size",
  "svt_threshold_size_scale_init": 1.2
}'

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
  --run-name "${RUN_NAME}" \
  --prior-kind mixed_matrix \
  --phases-json "${PHASES_JSON}" \
  --init-model-checkpoint "${INIT_CKPT}" \
  --seed 61 \
  --lr 2e-5 \
  --weight-decay 0.02 \
  --grad-clip-norm 1.0 \
  --warmup-steps 500 \
  --stable-steps 0 \
  --decay-steps 49500 \
  --min-lr-ratio 0.1 \
  --checkpoint-every 5000 \
  --nll-loss-weight 0.02 \
  --point-loss-weight 1000.0 \
  --point-loss-kind huber \
  --point-loss-prediction mean \
  --huber-delta 0.02 \
  --cdf-loss-weight 0.2 \
  --distribution-loss-chunk-size 2048 \
  --model-overrides-json "${MODEL_OVERRIDES_JSON}"
