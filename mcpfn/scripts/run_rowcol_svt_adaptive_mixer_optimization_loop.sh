#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5/best_checkpoint.pth}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_adaptive_mixer_probe}"

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
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

if tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
  echo "Waiting for active adaptive mixer session ${WAIT_FOR_TMUX} to finish."
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
    --output-csv "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_policy.csv" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_policy.log"
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
    --output-prefix "${CURVE_DIR}/rowcol_svt_adaptive_mixer_${tag}_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_rank10_eval.log"

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
    --output-prefix "${CURVE_DIR}/rowcol_svt_adaptive_mixer_${tag}_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_rank80_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "${label} default spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_adaptive_mixer_${tag}_default_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_spectral_default_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "${label} harder spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_adaptive_mixer_${tag}_harder_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_spectral_harder_eval.log"

  diagnose_policy "${tag}" "${checkpoint}"
}

train_round() {
  local tag="$1"
  local label="$2"
  local seed="$3"
  local lr="$4"
  local post_warmup_lr="$5"
  local warmup="$6"
  local stable="$7"
  local decay="$8"
  local min_ratio="$9"
  local init_logit="${10}"
  local other_logit="${11}"
  local temperature="${12}"
  local trainable_substrings="${13}"
  local extra_train_args="${14}"
  local run_name="tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-${tag}-10k"
  local work_dir="${WORK_ROOT}/${run_name}"
  local checkpoint="${work_dir}/best_checkpoint.pth"
  local model_overrides
  model_overrides="$(printf '{"svt_steps":0,"svt_init_rank":80,"svt_max_rank":100,"svt_init_scale":0.02,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.0001,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_mixture_steps":[0,16,32,64],"svt_mixture_hidden_size":512,"svt_mixture_temperature":%s,"svt_mixture_initial_logit":%s,"svt_mixture_other_logit":%s,"svt_mixture_detach_refinement":true}' "${temperature}" "${init_logit}" "${other_logit}")"
  local post_args=()
  local freeze_args=()
  if [[ "${post_warmup_lr}" != "none" ]]; then
    post_args+=(--post-warmup-lr "${post_warmup_lr}")
  fi
  if [[ "${trainable_substrings}" != "none" ]]; then
    freeze_args+=(--trainable-param-substrings "${trainable_substrings}")
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "=== Train/eval round ${tag}: ${label} ==="
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
      --min-lr-ratio "${min_ratio}" \
      --checkpoint-every 5000 \
      --nll-loss-weight 0.02 \
      --point-loss-weight 1000.0 \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.02 \
      --cdf-loss-weight 0.2 \
      --distribution-loss-chunk-size 2048 \
      --per-prior-loss-ema-normalize \
      "${freeze_args[@]}" \
      ${extra_train_args} \
      --model-overrides-json "${model_overrides}" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_adaptive_mixer_${tag}_train.log"
  else
    echo "Skipping train/eval round ${tag}; found ${checkpoint}"
  fi
  eval_round "${tag}" "${label}" "${checkpoint}"
}

if [[ -f "${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-10k-frommixedbest/best_checkpoint.pth" ]]; then
  eval_round \
    "round1_full_biased" \
    "Adaptive mixer round1 full biased" \
    "${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-adaptive-svtmixer-10k-frommixedbest/best_checkpoint.pth"
fi

train_round \
  "round2_controller_uniform_lr5e4" \
  "Adaptive mixer round2 controller uniform lr5e-4" \
  101 \
  5e-4 \
  none \
  200 \
  8800 \
  1000 \
  0.2 \
  0.0 \
  0.0 \
  1.0 \
  svt_mixture_controller \
  ""

train_round \
  "round3_controller_sharp_lr2e4" \
  "Adaptive mixer round3 controller sharp lr2e-4" \
  102 \
  2e-4 \
  none \
  200 \
  8800 \
  1000 \
  0.2 \
  0.0 \
  0.0 \
  0.5 \
  svt_mixture_controller \
  ""

train_round \
  "round4_full_uniform_warmdrop" \
  "Adaptive mixer round4 full uniform warmdrop" \
  103 \
  5e-5 \
  2e-5 \
  500 \
  8500 \
  1000 \
  0.5 \
  0.0 \
  0.0 \
  1.0 \
  none \
  ""

python - <<'PY' | tee runs/objective_queues/rowcol_svt_adaptive_mixer_optimization_summary.tsv
from pathlib import Path
import csv

root = Path("runs/curriculum_eval")
sroot = Path("runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts")
qroot = Path("runs/objective_queues")
tags = [
    "round1_full_biased",
    "round2_controller_uniform_lr5e4",
    "round3_controller_sharp_lr2e4",
    "round4_full_uniform_warmdrop",
]

def model_rmse(path):
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    model = [row for row in rows if row["method"] != "SoftImpute"]
    return f"{float(model[0]['rmse']):.6g}" if model else ""

def spectral(path, size):
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(row["size"]) == size:
            return f"{float(row['rmse']):.6g}"
    return ""

def policy(path):
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    pieces = []
    for case in ["gaussian_rank10", "gaussian_rank80", "spectral_default", "spectral_harder"]:
        vals = [float(row["expected_steps"]) for row in rows if row["case"] == case]
        if vals:
            pieces.append(f"{case}:{sum(vals)/len(vals):.1f}")
    return "; ".join(pieces)

print("tag\trank10_rmse\trank80_rmse\tspectral300\tspectral500\tharder300\tharder500\texpected_steps")
for tag in tags:
    print(
        tag,
        model_rmse(root / f"rowcol_svt_adaptive_mixer_{tag}_rank10_size300_summary.csv"),
        model_rmse(root / f"rowcol_svt_adaptive_mixer_{tag}_rank80_size300_summary.csv"),
        spectral(sroot / f"spectral_adaptive_mixer_{tag}_default_300_500.summary.csv", 300),
        spectral(sroot / f"spectral_adaptive_mixer_{tag}_default_300_500.summary.csv", 500),
        spectral(sroot / f"spectral_adaptive_mixer_{tag}_harder_300_500.summary.csv", 300),
        spectral(sroot / f"spectral_adaptive_mixer_{tag}_harder_300_500.summary.csv", 500),
        policy(qroot / f"rowcol_svt_adaptive_mixer_{tag}_policy.csv"),
        sep="\t",
    )
PY
