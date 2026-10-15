#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"

TAG="${TAG:-round17_cvfeatures_mnar_ctrl_distill_t005_lr1e4_step5000}"
LABEL="${LABEL:-Learned CV round17 step5000}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-learnedcv-svtmixer-round17_cvfeatures_mnar_ctrl_distill_t005_lr1e4-10k}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoint_5000.pth}"
METRICS_JSONL="${METRICS_JSONL:-${RUN_DIR}/metrics_stream.jsonl}"
WAIT_FOR_TMUX_BEFORE_EVAL="${WAIT_FOR_TMUX_BEFORE_EVAL:-rowcol_svt_learned_cv_queue}"
ALLOW_CONCURRENT_EVAL="${ALLOW_CONCURRENT_EVAL:-false}"

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

echo "Waiting for ${CHECKPOINT}"
until [[ -f "${CHECKPOINT}" ]]; do
  sleep 300
done
echo "Found ${CHECKPOINT}; running mid-evals for ${TAG}"

if [[ "${ALLOW_CONCURRENT_EVAL}" != "true" && -n "${WAIT_FOR_TMUX_BEFORE_EVAL}" ]]; then
  if tmux has-session -t "${WAIT_FOR_TMUX_BEFORE_EVAL}" 2>/dev/null; then
    echo "Waiting for ${WAIT_FOR_TMUX_BEFORE_EVAL} before launching GPU evals."
    while tmux has-session -t "${WAIT_FOR_TMUX_BEFORE_EVAL}" 2>/dev/null; do
      sleep 300
    done
  fi
fi

if [[ -f "${METRICS_JSONL}" ]]; then
  python scripts/summarize_svt_learned_cv_metrics.py \
    "${METRICS_JSONL}" \
    --last-n 1000 \
    --output-csv "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_train_metrics_summary.csv" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_train_metrics_summary.log"
fi

if [[ ! -f "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_rank10_size300_summary.csv" ]]; then
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
    --extra-direct "${LABEL} rank10|${CHECKPOINT}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_rank10_eval.log"
fi

if [[ ! -f "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_rank80_size300_summary.csv" ]]; then
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
    --extra-direct "${LABEL} rank80|${CHECKPOINT}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_rank80_eval.log"
fi

if [[ ! -f "${SPECTRAL_DIR}/spectral_svt_learned_cv_${TAG}_default_300_500.summary.csv" ]]; then
  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "${LABEL} default spectral|${CHECKPOINT}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_learned_cv_${TAG}_default_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_spectral_default_eval.log"
fi

if [[ ! -f "${SPECTRAL_DIR}/spectral_svt_learned_cv_${TAG}_harder_300_500.summary.csv" ]]; then
  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --extra-direct "${LABEL} harder spectral|${CHECKPOINT}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_learned_cv_${TAG}_harder_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_spectral_harder_eval.log"
fi

if [[ ! -f "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_mnar_size300.summary.csv" ]]; then
  python scripts/eval_rowcol_svt_mnar_missingness.py \
    --sizes 300 \
    --ranks 10 80 \
    --seeds 0 1 2 3 4 \
    --missingness-types mnar_latent_factor mnar_spatial_block mnar_censoring mnar_cluster_level \
    --direct-prediction-stat mean \
    --extra-direct "${LABEL} MNAR|${CHECKPOINT}|1000|rowcol_svt_hybrid" \
    --output-prefix "${CURVE_DIR}/rowcol_svt_learned_cv_${TAG}_mnar_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_mnar_eval.log"
fi

python scripts/diagnose_svt_mixture_policy.py \
  --checkpoint "${CHECKPOINT}" \
  --model-family rowcol_svt_hybrid \
  --chunk-size 1000 \
  --seeds 0 1 2 3 4 \
  --harder-spectral-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
  --oracle-threshold-scales 0.25 0.5 1.0 2.0 4.0 \
  --output-csv "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_policy.csv" \
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_policy.log"

python scripts/summarize_rowcol_svt_adaptive_runs.py \
  --runs-root runs \
  --output-tsv "${QUEUE_DIR}/rowcol_svt_adaptive_all_summary.tsv" \
  | tee "${QUEUE_DIR}/rowcol_svt_learned_cv_${TAG}_summary.tsv"
