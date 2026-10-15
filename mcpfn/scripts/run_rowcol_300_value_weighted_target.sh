#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-cdf2-huber50-lr2e5-cos200k-from-nlldecay200k/checkpoint_phase1_25000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-valueweighted-cdf2-huber50-lr3e5-cos200k-from-target300cdf25k}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
mkdir -p "${WORK_DIR}"

PHASES_300_200K='[
  {"name":"target300_025k","end_step":25000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_050k","end_step":50000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_100k","end_step":100000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_150k","end_step":150000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_200k","end_step":200000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1}
]'

MODEL_OVERRIDES='{"use_value_weighted_messages":true,"value_weighted_message_scale_init":0.01}'

if [[ ! -f "${WORK_DIR}/checkpoint_200000.pth" ]]; then
  python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
    --run-name "${RUN_NAME}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-52}" \
    --prior-kind latent_factor \
    --missingness-type mcar \
    --p-missing 0.2 \
    --latent-rank-low 10 \
    --latent-rank-high 10 \
    --weight-decay "${WEIGHT_DECAY:-0.02}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
    --warmup-steps "${WARMUP_STEPS:-1000}" \
    --stable-steps "${STABLE_STEPS:-0}" \
    --decay-steps "${DECAY_STEPS:-199000}" \
    --min-lr-ratio "${MIN_LR_RATIO:-0.05}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-25000}" \
    --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-4096}" \
    --wandb-project "${WANDB_PROJECT:-tabimpute}" \
    --lr "${LR:-3e-5}" \
    --phases-json "${PHASES_300_200K}" \
    --init-model-checkpoint "${INIT_CHECKPOINT}" \
    --allow-partial-init \
    --model-overrides-json "${MODEL_OVERRIDES}" \
    --cdf-loss-weight 2.0 \
    --point-loss-weight 50.0 \
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
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase025k|${WORK_DIR}/checkpoint_phase1_25000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase050k|${WORK_DIR}/checkpoint_phase2_50000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase100k|${WORK_DIR}/checkpoint_phase3_100000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase150k|${WORK_DIR}/checkpoint_phase4_150000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 phase200k|${WORK_DIR}/checkpoint_phase5_200000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 trainloss best|${WORK_DIR}/best_checkpoint.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 value-weighted CDF2 Huber50 lr3e-5 final|${WORK_DIR}/checkpoint_200000.pth|300|rowcol_latent_attention" \
  --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-valueweighted-cdf2-huber50-lr3e5-cos200k_target300_eval"
