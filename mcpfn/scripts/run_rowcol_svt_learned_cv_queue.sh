#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
ROUND4_CKPT="${ROUND4_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-round4_full_uniform_warmdrop-10k/best_checkpoint.pth}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_adaptive_threshold_queue}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-}"
PRIOR_SELECTION_MODE="${PRIOR_SELECTION_MODE:-random}"
PRIOR_OVERRIDES_JSON="$(printf '{"selection_mode":"%s"}' "${PRIOR_SELECTION_MODE}")"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
METRICS_JSONL_EVERY="${METRICS_JSONL_EVERY:-50}"

PHASES_JSON='[
  {
    "name": "mixed_mcar_mnar_300_010k",
    "end_step": 10000,
    "rows_low": 300,
    "rows_high": 300,
    "cols_low": 300,
    "cols_high": 300,
    "batch_size": 1
  }
]'

MIXED_PRIORS_JSON='[
  {
    "name": "latent_rank10_mcar20",
    "kind": "latent_factor",
    "weight": 1.0,
    "missingness_type": "mcar",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 10,
      "latent_rank_high": 10,
      "latent_spike_p": 0.3,
      "latent_slab_sigma": 2.0
    }
  },
  {
    "name": "latent_rank80_mcar20",
    "kind": "latent_factor",
    "weight": 1.0,
    "missingness_type": "mcar",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 80,
      "latent_rank_high": 80,
      "latent_spike_p": 0.0,
      "latent_slab_sigma": 1.0
    }
  },
  {
    "name": "latent_rank10_mnar_latent",
    "kind": "latent_factor",
    "weight": 0.75,
    "missingness_type": "mnar_latent_factor",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 10,
      "latent_rank_high": 10,
      "latent_spike_p": 0.3,
      "latent_slab_sigma": 2.0
    }
  },
  {
    "name": "latent_rank80_mnar_spatial",
    "kind": "latent_factor",
    "weight": 0.75,
    "missingness_type": "mnar_spatial_block",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 80,
      "latent_rank_high": 80,
      "latent_spike_p": 0.0,
      "latent_slab_sigma": 1.0,
      "spatial_block_n_blocks": 16,
      "spatial_block_p_geom": 0.035
    }
  },
  {
    "name": "latent_rank10_mnar_censor",
    "kind": "latent_factor",
    "weight": 0.50,
    "missingness_type": "mnar_censoring",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 10,
      "latent_rank_high": 10,
      "latent_spike_p": 0.3,
      "latent_slab_sigma": 2.0,
      "censor_quantile": 0.2
    }
  },
  {
    "name": "latent_rank80_mnar_cluster",
    "kind": "latent_factor",
    "weight": 0.50,
    "missingness_type": "mnar_cluster_level",
    "config": {
      "p_missing": 0.2,
      "latent_rank_low": 80,
      "latent_rank_high": 80,
      "latent_spike_p": 0.0,
      "latent_slab_sigma": 1.0,
      "cluster_level_n_row_clusters": 8,
      "cluster_level_n_col_clusters": 8,
      "cluster_level_tau_r_std": 1.5,
      "cluster_level_tau_c_std": 1.5,
      "cluster_level_epsilon_std": 0.5
    }
  },
  {
    "name": "spectrum_default_mcar20",
    "kind": "spectrum_decay",
    "weight": 1.0,
    "missingness_type": "mcar",
    "config": {
      "p_missing": 0.2,
      "spectrum_rank_low": 80,
      "spectrum_rank_high": 128,
      "spectrum_alpha_low": 0.30,
      "spectrum_alpha_high": 0.50,
      "spectrum_head_weight_low": 3.0,
      "spectrum_head_weight_high": 6.0,
      "spectrum_head_exp_scale_low": 2.5,
      "spectrum_head_exp_scale_high": 4.5,
      "spectrum_head_exp_power_low": 2.2,
      "spectrum_head_exp_power_high": 3.0,
      "spectrum_head_handoff_scale_low": 1.0,
      "spectrum_head_handoff_scale_high": 4.0,
      "spectrum_tail_amplitude_low": 0.25,
      "spectrum_tail_amplitude_high": 0.45,
      "spectrum_taper_strength_low": 1.2,
      "spectrum_taper_strength_high": 1.8,
      "spectrum_taper_power_low": 3.0,
      "spectrum_taper_power_high": 4.5,
      "spectrum_noise_std_low": 0.001,
      "spectrum_noise_std_high": 0.005,
      "spectrum_target_rank": 40
    }
  },
  {
    "name": "spectrum_harder_mcar20",
    "kind": "spectrum_decay",
    "weight": 1.0,
    "missingness_type": "mcar",
    "config": {
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
    }
  }
]'

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
  "spectrum_taper_power_low": 2.5,
  "spectrum_taper_power_high": 4.5,
  "spectrum_noise_std_low": 0.001,
  "spectrum_noise_std_high": 0.01,
  "spectrum_target_rank": 60
}'

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}" "${CURVE_DIR}" "${SPECTRAL_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

if tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
  echo "Waiting for active session ${WAIT_FOR_TMUX} to finish."
  while tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; do
    sleep 300
  done
fi

diagnose_policy() {
  local tag="$1"
  local checkpoint="$2"
  python scripts/diagnose_svt_mixture_policy.py \
    --checkpoint "${checkpoint}" \
    --model-family rowcol_svt_hybrid \
    --chunk-size 1000 \
    --seeds 0 1 2 3 4 \
    --harder-spectral-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --oracle-threshold-scales 0.25 0.5 1.0 2.0 4.0 \
    --output-csv "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_policy.csv" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_policy.log"
}

eval_round() {
  local tag="$1"
  local label="$2"
  local checkpoint="$3"

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
    --extra-direct "${label} rank10|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${tag}_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_rank10_eval.log"

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
    --extra-direct "${label} rank80|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${tag}_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_rank80_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "${label} default spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_learned_cv_${tag}_default_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_spectral_default_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --extra-direct "${label} harder spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_learned_cv_${tag}_harder_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_spectral_harder_eval.log"

  diagnose_policy "${tag}" "${checkpoint}"

  python scripts/eval_rowcol_svt_mnar_missingness.py \
    --sizes 300 \
    --ranks 10 80 \
    --seeds 0 1 2 3 4 \
    --missingness-types mnar_latent_factor mnar_spatial_block mnar_censoring mnar_cluster_level \
    --direct-prediction-stat mean \
    --extra-direct "${label} MNAR|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${tag}_mnar_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${tag}_mnar_eval.log"
}

train_round() {
  local tag="$1"
  local label="$2"
  local init_checkpoint="$3"
  local seed="$4"
  local policy="$5"
  local lr="$6"
  local post_warmup_lr="$7"
  local warmup="$8"
  local stable="$9"
  local decay="${10}"
  local min_ratio="${11}"
  local distill_weight="${12}"
  local distill_temperature="${13}"
  local cv_regret_weight="${14}"
  local nll_weight="${15}"
  local point_weight="${16}"
  local cdf_weight="${17}"
  local trainable_substrings="${18}"
  local effective_tag="${RUN_TAG_PREFIX}${tag}"
  local run_name="tabimpute-direct-rowcol-svthybrid768-learnedcv-svtmixer-${effective_tag}-10k"
  local work_dir="${WORK_ROOT}/${run_name}"
  local checkpoint="${work_dir}/best_checkpoint.pth"
  local model_overrides
  model_overrides="$(printf '{"svt_steps":0,"svt_init_rank":80,"svt_max_rank":100,"svt_init_scale":0.02,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.0001,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_mixture_steps":[0,16,32,64],"svt_mixture_hidden_size":768,"svt_mixture_temperature":1.0,"svt_mixture_initial_logit":0.0,"svt_mixture_other_logit":0.0,"svt_mixture_detach_refinement":true,"svt_mixture_spectral_feature_count":24,"svt_mixture_missingness_feature_count":28,"svt_mixture_policy":"%s","svt_mixture_cv_fraction":0.10,"svt_mixture_cv_temperature":0.01,"svt_mixture_cv_min_count":32,"svt_mixture_include_cv_features":true}' "${policy}")"
  local post_args=()
  local freeze_args=()
  if [[ "${post_warmup_lr}" != "none" ]]; then
    post_args+=(--post-warmup-lr "${post_warmup_lr}")
  fi
  if [[ "${trainable_substrings}" != "none" ]]; then
    freeze_args+=(--trainable-param-substrings "${trainable_substrings}")
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "=== Train/eval learned-CV round ${tag}: ${label} ==="
    python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
      --run-name "${run_name}" \
      --prior-kind mixed_matrix \
      --mixed-priors-json "${MIXED_PRIORS_JSON}" \
      --prior-overrides-json "${PRIOR_OVERRIDES_JSON}" \
      --phases-json "${PHASES_JSON}" \
      --init-model-checkpoint "${init_checkpoint}" \
      --allow-partial-init \
      --seed "${seed}" \
      --lr "${lr}" \
      "${post_args[@]}" \
      --weight-decay 0.02 \
      --grad-clip-norm 1.0 \
      --warmup-steps "${warmup}" \
      --stable-steps "${stable}" \
      --decay-steps "${decay}" \
      --min-lr-ratio "${min_ratio}" \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --metrics-jsonl-every "${METRICS_JSONL_EVERY}" \
      --nll-loss-weight "${nll_weight}" \
      --point-loss-weight "${point_weight}" \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.02 \
      --cdf-loss-weight "${cdf_weight}" \
      --distribution-loss-chunk-size 2048 \
      --per-prior-loss-ema-normalize \
      --svt-mixture-depth-distill-weight "${distill_weight}" \
      --svt-mixture-depth-distill-temperature "${distill_temperature}" \
      --svt-mixture-cv-regret-weight "${cv_regret_weight}" \
      "${freeze_args[@]}" \
      --model-overrides-json "${model_overrides}" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${effective_tag}_train.log"
  else
    echo "Skipping train/eval learned-CV round ${tag}; found ${checkpoint}"
  fi
  eval_round "${effective_tag}" "${label}" "${checkpoint}"
}

train_round \
  "round17c_cvfeatures_mnar_ctrl_cvreg_t10_lr1e4" \
  "Learned CV features round17c CV-regret controller" \
  "${ROUND4_CKPT}" \
  601 \
  controller \
  1e-4 \
  none \
  200 \
  8800 \
  1000 \
  0.2 \
  0.25 \
  0.10 \
  1.0 \
  0.0 \
  0.0 \
  0.0 \
  svt_mixture_controller

ROUND17_TAG="round17c_cvfeatures_mnar_ctrl_cvreg_t10_lr1e4"
ROUND17_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-learnedcv-svtmixer-${RUN_TAG_PREFIX}${ROUND17_TAG}-10k/best_checkpoint.pth"

train_round \
  "round18_cvfeatures_mnar_ctrl_pred_t01_lr5e5" \
  "Learned CV features round18 prediction" \
  "${ROUND17_CKPT}" \
  602 \
  controller \
  5e-5 \
  none \
  500 \
  8500 \
  1000 \
  0.3 \
  0.5 \
  0.05 \
  1.0 \
  0.02 \
  1000.0 \
  0.2 \
  svt_mixture_controller

train_round \
  "round19_cvfeatures_mnar_ctrlpluscv_pred_t01_lr5e5" \
  "Learned CV features round19 controller plus CV" \
  "${ROUND17_CKPT}" \
  603 \
  controller_plus_cv \
  5e-5 \
  none \
  500 \
  8500 \
  1000 \
  0.3 \
  0.25 \
  0.05 \
  0.75 \
  0.02 \
  1000.0 \
  0.2 \
  svt_mixture_controller

python scripts/summarize_rowcol_svt_adaptive_runs.py \
  --runs-root runs \
  --output-tsv "${QUEUE_DIR}/rowcol_svt_adaptive_all_summary.tsv" \
  | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${RUN_TAG_PREFIX}summary.tsv"
