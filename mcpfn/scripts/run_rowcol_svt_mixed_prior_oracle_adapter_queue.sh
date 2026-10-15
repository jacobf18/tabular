#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5/best_checkpoint.pth}"

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
  "svt_threshold_size_scale_init": 1.2,
  "num_prior_adapters": 4,
  "prior_adapter_scale_init": 0.05,
  "prior_adapter_calibration_scale_init": 0.02
}'

HARDER_SPECTRAL_OVERRIDES='{
  "p_missing": 0.2,
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
  "spectrum_noise_std_low": 0.001,
  "spectrum_noise_std_high": 0.01,
  "spectrum_target_rank": 60
}'

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}" "${CURVE_DIR}" "${SPECTRAL_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

train_one() {
  local tag="$1"
  local seed="$2"
  shift 2
  local run_name="tabimpute-direct-rowcol-svthybrid768-mixed-prior-oracle-${tag}-50k-frommixedbest"
  local work_dir="${WORK_ROOT}/${run_name}"
  local train_log="${QUEUE_DIR}/rowcol_svt_mixed_prior_oracle_${tag}_train.log"
  local ckpt="${work_dir}/best_checkpoint.pth"

  if [[ -f "${ckpt}" ]]; then
    echo "=== Skipping training for ${run_name}; found ${ckpt} ==="
  else
    echo "=== Training ${run_name} ==="
    python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
      --run-name "${run_name}" \
      --prior-kind mixed_matrix \
      --phases-json "${PHASES_JSON}" \
      --init-model-checkpoint "${INIT_CKPT}" \
      --allow-partial-init \
      --seed "${seed}" \
      --lr 2e-5 \
      --weight-decay 0.02 \
      --grad-clip-norm 1.0 \
      --warmup-steps 500 \
      --stable-steps 0 \
      --decay-steps 49500 \
      --min-lr-ratio 0.1 \
      --checkpoint-every 10000 \
      --nll-loss-weight 0.02 \
      --point-loss-weight 1000.0 \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.02 \
      --cdf-loss-weight 0.2 \
      --distribution-loss-chunk-size 2048 \
      --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
      "$@" \
      2>&1 | tee "${train_log}"
  fi

  echo "=== Latent rank10 eval for ${run_name} ==="
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
    --extra-direct "RowCol oracle ${tag} rank10 adapter|${ckpt}|1000|rowcol_svt_hybrid|0" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_oracle_${tag}_fixed_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_mixed_prior_oracle_${tag}_latent_rank10_eval.log"

  echo "=== Latent rank80 eval for ${run_name} ==="
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
    --extra-direct "RowCol oracle ${tag} rank80 adapter|${ckpt}|1000|rowcol_svt_hybrid|1" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_oracle_${tag}_fixed_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_mixed_prior_oracle_${tag}_latent_rank80_eval.log"

  echo "=== Default spectral-tail eval for ${run_name} ==="
  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 750 1000 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "RowCol oracle ${tag} default spectral adapter|${ckpt}|1000|rowcol_svt_hybrid|2" \
    --output-prefix "${SPECTRAL_DIR}/spectral_oracle_${tag}_default" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_mixed_prior_oracle_${tag}_spectral_default_eval.log"

  echo "=== Harder spectral-tail eval for ${run_name} ==="
  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --sizes 300 500 750 1000 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "RowCol oracle ${tag} harder spectral adapter|${ckpt}|1000|rowcol_svt_hybrid|3" \
    --output-prefix "${SPECTRAL_DIR}/spectral_oracle_${tag}_harder" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_mixed_prior_oracle_${tag}_spectral_harder_eval.log"
}

train_one "adapters" 73
train_one "adapters-balanced" 74 --per-prior-loss-ema-normalize
