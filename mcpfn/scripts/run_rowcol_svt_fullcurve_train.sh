#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-svthybrid768-rank10-p02-target300-evaloverride-steps32-tau006/checkpoint_10000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-rank10-p02-fullcurve-steps32-tau006}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
SEED="${SEED:-52}"

PHASES_FULLCURVE='[
  {"name":"warm_100_250_002k","end_step":2000,"rows_low":100,"rows_high":250,"cols_low":100,"cols_high":250,"batch_size":1},
  {"name":"mid_200_400_005k","end_step":5000,"rows_low":200,"rows_high":400,"cols_low":200,"cols_high":400,"batch_size":1},
  {"name":"full_300_500_010k","end_step":10000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}
]'

MODEL_OVERRIDES='{
  "svt_init_rank": 10,
  "svt_steps": 32,
  "svt_max_rank": 20,
  "svt_threshold_fraction_init": 0.006,
  "svt_threshold_fraction_min": 0.0001,
  "svt_update_damping": 1.0,
  "svt_init_scale": 0.035,
  "logit_sigma_init": 0.08,
  "logit_sigma_min": 0.008
}'

mkdir -p "${WORK_DIR}"

if [[ ! -f "${WORK_DIR}/checkpoint_10000.pth" ]]; then
  python "${REPO_DIR}/scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py" \
    --run-name "${RUN_NAME}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED}" \
    --prior-kind latent_factor \
    --missingness-type mcar \
    --p-missing 0.2 \
    --latent-rank-low 10 \
    --latent-rank-high 10 \
    --weight-decay "${WEIGHT_DECAY:-0.02}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
    --warmup-steps "${WARMUP_STEPS:-500}" \
    --stable-steps "${STABLE_STEPS:-0}" \
    --decay-steps "${DECAY_STEPS:-9500}" \
    --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-2000}" \
    --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-2048}" \
    --wandb-project "${WANDB_PROJECT:-tabimpute}" \
    --lr "${LR:-5e-6}" \
    --phases-json "${PHASES_FULLCURVE}" \
    --init-model-checkpoint "${INIT_CHECKPOINT}" \
    --allow-partial-init \
    --model-overrides-json "${MODEL_OVERRIDES}" \
    --nll-loss-weight "${NLL_LOSS_WEIGHT:-0.05}" \
    --cdf-loss-weight "${CDF_LOSS_WEIGHT:-0.5}" \
    --point-loss-weight "${POINT_LOSS_WEIGHT:-300.0}" \
    --point-loss-kind huber \
    --point-loss-prediction mean \
    --huber-delta "${HUBER_DELTA:-0.02}" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_fullcurve_steps32_tau006_train.log"
fi

python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
  --sizes 50 100 150 200 300 400 500 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --generator gaussian_factors \
  --scale rank \
  --direct-names \
  --direct-prediction-stat median \
  --extra-direct "RowCol SVT fullcurve steps32 tau0.006 phase002k|${WORK_DIR}/checkpoint_phase1_2000.pth|500|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT fullcurve steps32 tau0.006 phase005k|${WORK_DIR}/checkpoint_phase2_5000.pth|500|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT fullcurve steps32 tau0.006 phase010k|${WORK_DIR}/checkpoint_phase3_10000.pth|500|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT fullcurve steps32 tau0.006 trainloss best|${WORK_DIR}/best_checkpoint.pth|500|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT fullcurve steps32 tau0.006 final|${WORK_DIR}/checkpoint_10000.pth|500|rowcol_svt_hybrid" \
  --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_svt_fullcurve_steps32_tau006_eval" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_fullcurve_steps32_tau006_eval.log"
