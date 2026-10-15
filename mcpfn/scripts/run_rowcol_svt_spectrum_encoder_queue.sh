#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
CURVE_DIR="${CURVE_DIR:-${REPO_DIR}/runs/curriculum_eval}"
SPECTRAL_DIR="${SPECTRAL_DIR:-${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts}"
BASE_CKPT="${BASE_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-mixed-prior-rank80_steps0_50k_fromhard50k_lr2e5/best_checkpoint.pth}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_depth_distill_queue}"

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
    --output-csv "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_policy.csv" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_policy.log"
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
    --output-prefix "${CURVE_DIR}/rowcol_svt_spectrum_encoder_${tag}_rank10_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_rank10_eval.log"

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
    --output-prefix "${CURVE_DIR}/rowcol_svt_spectrum_encoder_${tag}_rank80_size300" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_rank80_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --extra-direct "${label} default spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_spectrum_encoder_${tag}_default_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_spectral_default_eval.log"

  python scripts/eval_rowcol_svt_spectral_tails.py \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 3 4 \
    --skip-softimpute \
    --skip-svt \
    --direct-prediction-stat mean \
    --preset-overrides-json "${HARDER_SPECTRAL_OVERRIDES}" \
    --extra-direct "${label} harder spectral|${checkpoint}|1000|rowcol_svt_hybrid" \
    --output-prefix "${SPECTRAL_DIR}/spectral_svt_spectrum_encoder_${tag}_harder_300_500" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_spectral_harder_eval.log"

  diagnose_policy "${tag}" "${checkpoint}"
}

train_round() {
  local tag="$1"
  local label="$2"
  local init_checkpoint="$3"
  local seed="$4"
  local lr="$5"
  local post_warmup_lr="$6"
  local warmup="$7"
  local stable="$8"
  local decay="$9"
  local min_ratio="${10}"
  local distill_weight="${11}"
  local distill_temperature="${12}"
  local nll_weight="${13}"
  local point_weight="${14}"
  local cdf_weight="${15}"
  local trainable_substrings="${16}"
  local feature_count="${17}"
  local encoder_size="${18}"
  local encoder_layers="${19}"
  local encoder_heads="${20}"
  local run_name="tabimpute-direct-rowcol-svthybrid768-spectrumenc-svtmixer-${tag}-10k"
  local work_dir="${WORK_ROOT}/${run_name}"
  local checkpoint="${work_dir}/best_checkpoint.pth"
  local model_overrides
  model_overrides="$(printf '{"svt_steps":0,"svt_init_rank":80,"svt_max_rank":100,"svt_init_scale":0.02,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.0001,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_mixture_steps":[0,16,32,64],"svt_mixture_hidden_size":768,"svt_mixture_temperature":1.0,"svt_mixture_initial_logit":0.0,"svt_mixture_other_logit":0.0,"svt_mixture_detach_refinement":true,"svt_mixture_spectral_feature_count":%s,"svt_mixture_spectrum_encoder_size":%s,"svt_mixture_spectrum_encoder_layers":%s,"svt_mixture_spectrum_encoder_heads":%s,"svt_mixture_spectrum_encoder_dropout":0.0}' "${feature_count}" "${encoder_size}" "${encoder_layers}" "${encoder_heads}")"
  local post_args=()
  local freeze_args=()
  if [[ "${post_warmup_lr}" != "none" ]]; then
    post_args+=(--post-warmup-lr "${post_warmup_lr}")
  fi
  if [[ "${trainable_substrings}" != "none" ]]; then
    freeze_args+=(--trainable-param-substrings "${trainable_substrings}")
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "=== Train/eval spectrum-encoder round ${tag}: ${label} ==="
    python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
      --run-name "${run_name}" \
      --prior-kind mixed_matrix \
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
      --checkpoint-every 5000 \
      --metrics-jsonl-every 50 \
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
      "${freeze_args[@]}" \
      --model-overrides-json "${model_overrides}" \
      2>&1 | tee "${QUEUE_DIR}/rowcol_svt_spectrum_encoder_${tag}_train.log"
  else
    echo "Skipping train/eval spectrum-encoder round ${tag}; found ${checkpoint}"
  fi
  eval_round "${tag}" "${label}" "${checkpoint}"
}

train_round \
  "round11_spec64_enc128_ctrl_distill_t005_lr1e4" \
  "Spectrum encoder round11 selector" \
  "${BASE_CKPT}" \
  401 \
  1e-4 \
  none \
  200 \
  8800 \
  1000 \
  0.2 \
  5.0 \
  0.005 \
  0.0 \
  0.0 \
  0.0 \
  svt_mixture_controller,svt_mixture_spectrum \
  64 \
  128 \
  2 \
  4

train_round \
  "round12_spec64_enc128_ctrl_pred_distill_t01_lr5e5" \
  "Spectrum encoder round12 prediction distill" \
  "${BASE_CKPT}" \
  402 \
  5e-5 \
  none \
  500 \
  8500 \
  1000 \
  0.2 \
  5.0 \
  0.01 \
  0.02 \
  1000.0 \
  0.2 \
  svt_mixture_controller,svt_mixture_spectrum \
  64 \
  128 \
  2 \
  4

ROUND11_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectrumenc-svtmixer-round11_spec64_enc128_ctrl_distill_t005_lr1e4-10k/best_checkpoint.pth"

train_round \
  "round13_spec64_enc128_full_fromround11_distill_t005_lr5e5" \
  "Spectrum encoder round13 full" \
  "${ROUND11_CKPT}" \
  403 \
  5e-5 \
  2e-5 \
  500 \
  8500 \
  1000 \
  0.5 \
  2.0 \
  0.005 \
  0.02 \
  1000.0 \
  0.2 \
  none \
  64 \
  128 \
  2 \
  4

python - <<'PY' | tee runs/objective_queues/rowcol_svt_spectrum_encoder_summary.tsv
from pathlib import Path
import csv

root = Path("runs/curriculum_eval")
sroot = Path("runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts")
qroot = Path("runs/objective_queues")
tags = [
    "round11_spec64_enc128_ctrl_distill_t005_lr1e4",
    "round12_spec64_enc128_ctrl_pred_distill_t01_lr5e5",
    "round13_spec64_enc128_full_fromround11_distill_t005_lr5e5",
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
        model_rmse(root / f"rowcol_svt_spectrum_encoder_{tag}_rank10_size300_summary.csv"),
        model_rmse(root / f"rowcol_svt_spectrum_encoder_{tag}_rank80_size300_summary.csv"),
        spectral(sroot / f"spectral_svt_spectrum_encoder_{tag}_default_300_500.summary.csv", 300),
        spectral(sroot / f"spectral_svt_spectrum_encoder_{tag}_default_300_500.summary.csv", 500),
        spectral(sroot / f"spectral_svt_spectrum_encoder_{tag}_harder_300_500.summary.csv", 300),
        spectral(sroot / f"spectral_svt_spectrum_encoder_{tag}_harder_300_500.summary.csv", 500),
        policy(qroot / f"rowcol_svt_spectrum_encoder_{tag}_policy.csv"),
        sep="\t",
    )
PY
