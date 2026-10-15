#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5/best_checkpoint.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-10k-frommixedbest}"
WORK_DIR="${WORK_ROOT}/${RUN_NAME}"

PHASES_JSON='[
  {
    "name": "mixed_300_010k",
    "end_step": 10000,
    "rows_low": 300,
    "rows_high": 300,
    "cols_low": 300,
    "cols_high": 300,
    "batch_size": 1
  }
]'

MODEL_OVERRIDES_JSON='{
  "svt_steps": 0,
  "svt_init_rank": 80,
  "svt_max_rank": 100,
  "svt_init_scale": 0.02,
  "svt_threshold_fraction_init": 0.006,
  "svt_threshold_fraction_min": 0.0001,
  "svt_threshold_fraction_max": 0.02,
  "svt_threshold_mode": "inverse_size",
  "svt_threshold_size_scale_init": 1.2,
  "svt_mixture_steps": [0, 16, 32, 64],
  "svt_mixture_hidden_size": 512,
  "svt_mixture_temperature": 1.0,
  "svt_mixture_initial_logit": 2.0,
  "svt_mixture_other_logit": 0.0,
  "svt_mixture_detach_refinement": true
}'

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}" "${CURVE_DIR}" "${SPECTRAL_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${WORK_DIR}/best_checkpoint.pth" ]]; then
  python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
    --run-name "${RUN_NAME}" \
    --prior-kind mixed_matrix \
    --phases-json "${PHASES_JSON}" \
    --init-model-checkpoint "${INIT_CKPT}" \
    --allow-partial-init \
    --seed 93 \
    --lr 5e-5 \
    --post-warmup-lr 2e-5 \
    --weight-decay 0.02 \
    --grad-clip-norm 1.0 \
    --warmup-steps 500 \
    --stable-steps 8500 \
    --decay-steps 1000 \
    --min-lr-ratio 0.5 \
    --checkpoint-every 5000 \
    --nll-loss-weight 0.02 \
    --point-loss-weight 1000.0 \
    --point-loss-kind huber \
    --point-loss-prediction mean \
    --huber-delta 0.02 \
    --cdf-loss-weight 0.2 \
    --distribution-loss-chunk-size 2048 \
    --per-prior-loss-ema-normalize \
    --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_probe_train.log"
else
  echo "Skipping train; found ${WORK_DIR}/best_checkpoint.pth"
fi

CKPT="${WORK_DIR}/best_checkpoint.pth"

python scripts/eval_train_level_chunks_interface.py \
  --sizes 300 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --scale rank \
  --generator gaussian_factors \
  --normalize-softimpute \
  --direct-names \
  --no-old-preprocessors \
  --direct-prediction-stat mean \
  --extra-direct "Adaptive SVT mixer rank10|${CKPT}|1000|rowcol_svt_hybrid" \
  --output-prefix "${CURVE_DIR}/rowcol_svt_adaptive_mixer_probe_rank10_size300" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_probe_rank10_eval.log"

python scripts/eval_train_level_chunks_interface.py \
  --sizes 300 \
  --seeds 0 1 2 3 4 \
  --rank 80 \
  --missing-rate 0.2 \
  --scale rank \
  --generator gaussian_factors \
  --normalize-softimpute \
  --direct-names \
  --no-old-preprocessors \
  --direct-prediction-stat mean \
  --extra-direct "Adaptive SVT mixer rank80|${CKPT}|1000|rowcol_svt_hybrid" \
  --output-prefix "${CURVE_DIR}/rowcol_svt_adaptive_mixer_probe_rank80_size300" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_probe_rank80_eval.log"

python scripts/eval_rowcol_svt_spectral_tails.py \
  --preset hard_tail \
  --sizes 300 500 \
  --seeds 0 1 2 3 4 \
  --skip-softimpute \
  --skip-svt \
  --direct-prediction-stat mean \
  --extra-direct "Adaptive SVT mixer spectral|${CKPT}|1000|rowcol_svt_hybrid" \
  --output-prefix "${SPECTRAL_DIR}/spectral_adaptive_svt_mixer_probe_default_300_500" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_probe_spectral_eval.log"
