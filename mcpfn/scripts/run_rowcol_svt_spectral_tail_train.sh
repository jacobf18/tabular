#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-svthybrid768-rank10-p02-target300-evaloverride-steps32-adaptive-tau/checkpoint_10000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-spectral-hardtail-300to500-trainsteps8-adaptive}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
SEED="${SEED:-52}"

PHASES_SPECTRAL='[
  {"name":"spectral_300_500_002k","end_step":2000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"spectral_300_500_006k","end_step":6000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"spectral_300_500_010k","end_step":10000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}
]'

MODEL_OVERRIDES='{
  "svt_init_rank": 10,
  "svt_steps": 8,
  "svt_max_rank": 80,
  "svt_threshold_fraction_init": 0.006,
  "svt_threshold_fraction_min": 0.004,
  "svt_threshold_fraction_max": 0.02,
  "svt_threshold_mode": "inverse_size",
  "svt_threshold_size_scale_init": 1.2,
  "svt_update_damping": 1.0,
  "svt_init_scale": 0.035,
  "logit_sigma_init": 0.08,
  "logit_sigma_min": 0.008
}'

PRIOR_OVERRIDES='{
  "spectrum_rank_low": 80,
  "spectrum_rank_high": 160,
  "spectrum_alpha_low": 0.25,
  "spectrum_alpha_high": 0.55,
  "spectrum_head_weight_low": 2.0,
  "spectrum_head_weight_high": 6.0,
  "spectrum_head_exp_scale_low": 2.5,
  "spectrum_head_exp_scale_high": 7.0,
  "spectrum_head_exp_power_low": 1.6,
  "spectrum_head_exp_power_high": 3.0,
  "spectrum_head_handoff_scale_low": 1.0,
  "spectrum_head_handoff_scale_high": 8.0,
  "spectrum_tail_amplitude_low": 0.25,
  "spectrum_tail_amplitude_high": 0.60,
  "spectrum_taper_strength_low": 0.6,
  "spectrum_taper_strength_high": 1.8,
  "spectrum_taper_power_low": 2.5,
  "spectrum_taper_power_high": 4.5,
  "spectrum_scale_low": 1.0,
  "spectrum_scale_high": 1.0,
  "spectrum_noise_std_low": 0.001,
  "spectrum_noise_std_high": 0.01,
  "spectrum_target_rank": 60
}'

mkdir -p "${WORK_DIR}"

if [[ ! -f "${WORK_DIR}/checkpoint_10000.pth" ]]; then
  python "${REPO_DIR}/scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py" \
    --run-name "${RUN_NAME}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED}" \
    --prior-kind spectrum_decay \
    --missingness-type mcar \
    --p-missing 0.2 \
    --weight-decay "${WEIGHT_DECAY:-0.02}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
    --warmup-steps "${WARMUP_STEPS:-500}" \
    --stable-steps "${STABLE_STEPS:-0}" \
    --decay-steps "${DECAY_STEPS:-9500}" \
    --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-2000}" \
    --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-2048}" \
    --wandb-project "${WANDB_PROJECT:-tabimpute}" \
    --lr "${LR:-3e-6}" \
    --phases-json "${PHASES_SPECTRAL}" \
    --init-model-checkpoint "${INIT_CHECKPOINT}" \
    --allow-partial-init \
    --model-overrides-json "${MODEL_OVERRIDES}" \
    --prior-overrides-json "${PRIOR_OVERRIDES}" \
    --nll-loss-weight "${NLL_LOSS_WEIGHT:-0.05}" \
    --cdf-loss-weight "${CDF_LOSS_WEIGHT:-0.5}" \
    --point-loss-weight "${POINT_LOSS_WEIGHT:-300.0}" \
    --point-loss-kind huber \
    --point-loss-prediction mean \
    --huber-delta "${HUBER_DELTA:-0.02}" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectral_tail_train.log"
fi

python "${REPO_DIR}/scripts/eval_rowcol_svt_spectral_tails.py" \
  --preset hard_tail \
  --sizes 300 500 750 1000 \
  --seeds 0 1 2 3 4 \
  --missing-rate 0.2 \
  --softimpute-max-iters 200 \
  --svt-steps 32 \
  --svt-max-rank 80 \
  --svt-tau-fractions 0.004 0.006 0.008 0.01 \
  --adaptive-svt \
  --extra-direct "RowCol SVT spectral hardtail phase002k|${WORK_DIR}/checkpoint_phase1_2000.pth|1000|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT spectral hardtail phase006k|${WORK_DIR}/checkpoint_phase2_6000.pth|1000|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT spectral hardtail phase010k|${WORK_DIR}/checkpoint_phase3_10000.pth|1000|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT spectral hardtail trainloss best|${WORK_DIR}/best_checkpoint.pth|1000|rowcol_svt_hybrid" \
  --extra-direct "RowCol SVT spectral hardtail final|${WORK_DIR}/checkpoint_10000.pth|1000|rowcol_svt_hybrid" \
  --output-prefix "${FIG_DIR}/hardtail_curve_to1000_rowcol_svt_spectral_train" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectral_tail_eval.log"
