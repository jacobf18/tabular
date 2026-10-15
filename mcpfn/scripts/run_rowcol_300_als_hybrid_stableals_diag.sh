#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-cdf2-huber50-lr2e5-cos200k-from-nlldecay200k/checkpoint_phase1_25000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-alshybrid768-rank10-p02-target300-stableals-scaledridge-clip2-sigma08-lr5e5-cos25k-from-target300cdf25k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
mkdir -p "${WORK_DIR}"

PHASES_300_25K='[
  {"name":"target300_005k","end_step":5000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_010k","end_step":10000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_025k","end_step":25000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1}
]'

MODEL_OVERRIDES='{"als_rank":10,"als_steps":4,"als_ridge":0.001,"als_update_damping":0.85,"als_ridge_scale_by_obs":true,"als_solve_method":"cholesky","als_jitter":0.00001,"als_factor_norm_clip":2.0,"als_init_scale":0.035,"logit_sigma_init":0.08,"logit_sigma_min":0.008,"use_neural_residual":false}'

if [[ ! -f "${WORK_DIR}/checkpoint_25000.pth" ]]; then
  python "${REPO_DIR}/scripts/train_direct_rowcol_als_hybrid_random_rank_curriculum.py" \
    --run-name "${RUN_NAME}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-53}" \
    --prior-kind latent_factor \
    --missingness-type mcar \
    --p-missing 0.2 \
    --latent-rank-low 10 \
    --latent-rank-high 10 \
    --weight-decay "${WEIGHT_DECAY:-0.02}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
    --warmup-steps "${WARMUP_STEPS:-500}" \
    --stable-steps "${STABLE_STEPS:-0}" \
    --decay-steps "${DECAY_STEPS:-24500}" \
    --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
    --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-2048}" \
    --wandb-project "${WANDB_PROJECT:-tabimpute}" \
    --lr "${LR:-5e-5}" \
    --phases-json "${PHASES_300_25K}" \
    --init-model-checkpoint "${INIT_CHECKPOINT}" \
    --allow-partial-init \
    --model-overrides-json "${MODEL_OVERRIDES}" \
    --cdf-loss-weight 0.5 \
    --point-loss-weight 30.0 \
    --point-loss-kind huber \
    --point-loss-prediction median \
    --huber-delta 0.02
fi

python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
  --sizes 300 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --generator gaussian_factors \
  --scale rank \
  --skip-softimpute \
  --direct-names \
  --direct-prediction-stat median \
  --extra-direct "RowCol ALS hybrid target300 stableALS phase005k|${WORK_DIR}/checkpoint_phase1_5000.pth|300|rowcol_als_hybrid" \
  --extra-direct "RowCol ALS hybrid target300 stableALS phase010k|${WORK_DIR}/checkpoint_phase2_10000.pth|300|rowcol_als_hybrid" \
  --extra-direct "RowCol ALS hybrid target300 stableALS phase025k|${WORK_DIR}/checkpoint_phase3_25000.pth|300|rowcol_als_hybrid" \
  --extra-direct "RowCol ALS hybrid target300 stableALS trainloss best|${WORK_DIR}/best_checkpoint.pth|300|rowcol_als_hybrid" \
  --extra-direct "RowCol ALS hybrid target300 stableALS final|${WORK_DIR}/checkpoint_25000.pth|300|rowcol_als_hybrid" \
  --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-alshybrid-stableals-scaledridge-clip2-lr5e5-cos25k_diag_eval"
