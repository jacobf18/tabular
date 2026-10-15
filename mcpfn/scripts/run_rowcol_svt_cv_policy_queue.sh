#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_learned_cv_queue}"

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

materialize_policy_checkpoint() {
  local tag="$1"
  local source_checkpoint="$2"
  local policy="$3"
  local cv_fraction="$4"
  local cv_temperature="$5"
  local source_dir
  local dest_dir
  source_dir="$(dirname "${source_checkpoint}")"
  dest_dir="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-cv-svtmixer-${tag}"
  if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Skipping ${tag}; missing checkpoint ${source_checkpoint}"
    return 1
  fi
  if [[ ! -f "${source_dir}/config.json" ]]; then
    echo "Skipping ${tag}; missing config ${source_dir}/config.json"
    return 1
  fi
  if [[ -f "${dest_dir}/best_checkpoint.pth" && -f "${dest_dir}/config.json" ]]; then
    echo "Using existing CV-policy checkpoint ${dest_dir}"
    return 0
  fi
  TAG="${tag}" \
  SOURCE_DIR="${source_dir}" \
  DEST_DIR="${dest_dir}" \
  POLICY="${policy}" \
  CV_FRACTION="${cv_fraction}" \
  CV_TEMPERATURE="${cv_temperature}" \
  python - <<'PY'
import json
import os
import shutil
from pathlib import Path

tag = os.environ["TAG"]
source_dir = Path(os.environ["SOURCE_DIR"])
dest_dir = Path(os.environ["DEST_DIR"])
policy = os.environ["POLICY"]
cv_fraction = float(os.environ["CV_FRACTION"])
cv_temperature = float(os.environ["CV_TEMPERATURE"])

if dest_dir.exists():
    shutil.rmtree(dest_dir)
dest_dir.mkdir(parents=True)
shutil.copy2(source_dir / "best_checkpoint.pth", dest_dir / "best_checkpoint.pth")
config = json.loads((source_dir / "config.json").read_text())
config["run_name"] = f"tabimpute-direct-rowcol-svthybrid768-cv-svtmixer-{tag}"
model = config.setdefault("model", {})
model["svt_mixture_policy"] = policy
model["svt_mixture_cv_fraction"] = cv_fraction
model["svt_mixture_cv_temperature"] = cv_temperature
model["svt_mixture_cv_seed"] = 17
model["svt_mixture_cv_min_count"] = 32
config.setdefault("training", {})["cv_policy_materialized_from"] = str(source_dir)
(dest_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
print(dest_dir)
PY
}

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
    --output-csv "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_policy.csv" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_policy.log"
}

eval_round() {
  local tag="$1"
  local label="$2"
  local checkpoint="$3"

  if [[ ! -f "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_rank10_size300_summary.csv" ]]; then
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
      --output-prefix "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_rank10_size300" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_rank10_eval.log"
  fi

  if [[ ! -f "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_rank80_size300_summary.csv" ]]; then
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
      --output-prefix "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_rank80_size300" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_rank80_eval.log"
  fi

  if [[ ! -f "${SPECTRAL_DIR}/spectral_svt_cv_mixer_${tag}_default_300_500.summary.csv" ]]; then
    python scripts/eval_rowcol_svt_spectral_tails.py \
      --preset hard_tail \
      --sizes 300 500 \
      --seeds 0 1 2 3 4 \
      --skip-softimpute \
      --skip-svt \
      --direct-prediction-stat mean \
      --extra-direct "${label} default spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
      --output-prefix "${SPECTRAL_DIR}/spectral_svt_cv_mixer_${tag}_default_300_500" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_spectral_default_eval.log"
  fi

  if [[ ! -f "${SPECTRAL_DIR}/spectral_svt_cv_mixer_${tag}_harder_300_500.summary.csv" ]]; then
    python scripts/eval_rowcol_svt_spectral_tails.py \
      --preset hard_tail \
      --sizes 300 500 \
      --seeds 0 1 2 3 4 \
      --skip-softimpute \
      --skip-svt \
      --direct-prediction-stat mean \
      --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
      --extra-direct "${label} harder spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
      --output-prefix "${SPECTRAL_DIR}/spectral_svt_cv_mixer_${tag}_harder_300_500" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_spectral_harder_eval.log"
  fi

  diagnose_policy "${tag}" "${checkpoint}"

  if [[ ! -f "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_mnar_size300.summary.csv" ]]; then
    python scripts/eval_rowcol_svt_mnar_missingness.py \
      --sizes 300 \
      --ranks 10 80 \
      --seeds 0 1 2 3 4 \
      --missingness-types mnar_latent_factor mnar_spatial_block mnar_censoring mnar_cluster_level \
      --direct-prediction-stat mean \
      --extra-direct "${label} MNAR|${checkpoint}|1000|rowcol_svt_hybrid" \
      --output-prefix "${CURVE_DIR}/rowcol_svt_cv_mixer_${tag}_mnar_size300" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_cv_mixer_${tag}_mnar_eval.log"
  fi
}

run_policy_eval() {
  local tag="$1"
  local label="$2"
  local source_checkpoint="$3"
  local policy="$4"
  local cv_fraction="$5"
  local cv_temperature="$6"
  local checkpoint="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-cv-svtmixer-${tag}/best_checkpoint.pth"
  if materialize_policy_checkpoint \
    "${tag}" \
    "${source_checkpoint}" \
    "${policy}" \
    "${cv_fraction}" \
    "${cv_temperature}"; then
    eval_round "${tag}" "${label}" "${checkpoint}"
  fi
}

ROUND4_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-round4_full_uniform_warmdrop-10k/best_checkpoint.pth"
ROUND11_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectrumenc-svtmixer-round11_spec64_enc128_ctrl_distill_t005_lr1e4-10k/best_checkpoint.pth"
ROUND14_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-adaptivethresh-svtmixer-round14_spec64_enc128_tauctrl_steps32_lr2e5-10k/best_checkpoint.pth"

run_policy_eval \
  round4_cvonly_f10_t01_fromround4 \
  "CV-only round4 f10 t01" \
  "${ROUND4_CKPT}" \
  cv \
  0.10 \
  0.01

run_policy_eval \
  round4_controllerpluscv_f10_t01_fromround4 \
  "Controller plus CV round4 f10 t01" \
  "${ROUND4_CKPT}" \
  controller_plus_cv \
  0.10 \
  0.01

run_policy_eval \
  round11_cvonly_f10_t01_fromspectrumenc \
  "CV-only round11 spectrum encoder f10 t01" \
  "${ROUND11_CKPT}" \
  cv \
  0.10 \
  0.01

run_policy_eval \
  round14_controllerpluscv_f10_t01_fromadaptivethresh \
  "Controller plus CV round14 adaptive threshold f10 t01" \
  "${ROUND14_CKPT}" \
  controller_plus_cv \
  0.10 \
  0.01

python scripts/summarize_rowcol_svt_adaptive_runs.py \
  --runs-root runs \
  --output-tsv "${QUEUE_DIR}/rowcol_svt_adaptive_all_summary.tsv" \
  | tee "${QUEUE_DIR}/rowcol_svt_cv_policy_summary.tsv"
