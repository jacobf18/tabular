#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5/best_checkpoint.pth}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_oracle_adapter_queue}"

PHASES_JSON='[
  {
    "name": "mixed_300_500_020k",
    "end_step": 20000,
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

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}" "${CURVE_DIR}" "${SPECTRAL_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
  echo "Waiting for tmux session ${WAIT_FOR_TMUX} to finish before scheduler sweep."
  while tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; do
    sleep 300
  done
fi

train_and_eval() {
  local tag="$1"
  local seed="$2"
  local lr="$3"
  local warmup="$4"
  local stable="$5"
  local decay="$6"
  local min_lr_ratio="$7"
  local post_warmup_lr="$8"
  local run_name="tabimpute-direct-rowcol-svthybrid768-mixed-prior-oracle-sched-${tag}-20k"
  local work_dir="${WORK_ROOT}/${run_name}"
  local ckpt="${work_dir}/best_checkpoint.pth"
  local post_args=()
  if [[ "${post_warmup_lr}" != "none" ]]; then
    post_args+=(--post-warmup-lr "${post_warmup_lr}")
  fi

  if [[ -f "${ckpt}" ]]; then
    echo "=== Skipping scheduler probe ${tag}; found ${ckpt} ==="
  else
    echo "=== Scheduler probe ${tag}: lr=${lr}, warmup=${warmup}, stable=${stable}, decay=${decay}, min_ratio=${min_lr_ratio}, post=${post_warmup_lr} ==="
    python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
      --run-name "${run_name}" \
      --prior-kind mixed_matrix \
      --phases-json "${PHASES_JSON}" \
      --init-model-checkpoint "${INIT_CKPT}" \
      --allow-partial-init \
      --seed "${seed}" \
      --lr "${lr}" \
      "${post_args[@]}" \
      --weight-decay 0.02 \
      --grad-clip-norm 1.0 \
      --warmup-steps "${warmup}" \
      --stable-steps "${stable}" \
      --decay-steps "${decay}" \
      --min-lr-ratio "${min_lr_ratio}" \
      --checkpoint-every 5000 \
      --nll-loss-weight 0.02 \
      --point-loss-weight 1000.0 \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.02 \
      --cdf-loss-weight 0.2 \
      --distribution-loss-chunk-size 2048 \
      --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_oracle_scheduler_${tag}_train.log"
  fi

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
    --extra-direct "RowCol oracle sched ${tag} rank10|${ckpt}|1000|rowcol_svt_hybrid|0" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_oracle_scheduler_${tag}_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_oracle_scheduler_${tag}_rank10_eval.log"

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
    --extra-direct "RowCol oracle sched ${tag} rank80|${ckpt}|1000|rowcol_svt_hybrid|1" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_oracle_scheduler_${tag}_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_oracle_scheduler_${tag}_rank80_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "RowCol oracle sched ${tag} spectral|${ckpt}|1000|rowcol_svt_hybrid|2" \
    --output-prefix "${SPECTRAL_DIR}/spectral_oracle_scheduler_${tag}_default_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_oracle_scheduler_${tag}_spectral_eval.log"
}

train_and_eval "cosine" 83 2e-5 500 0 19500 0.1 none
train_and_eval "flat2e5" 84 2e-5 500 18000 1500 0.5 none
train_and_eval "warmdrop5e5to2e5" 85 5e-5 500 18000 1500 0.5 2e-5
