#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
FIG_DIR="${REPO_DIR}/runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts"
WORK_ROOT="${REPO_DIR}/src/tabimpute/workdir"
mkdir -p "${QUEUE_DIR}" "${FIG_DIR}"

BASE_LATENT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-rank10-p02-target300-evaloverride-steps32-adaptive-tau/checkpoint_10000.pth"
SPECTRAL_CKPT="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectral-hardtail-300to500-trainsteps8-adaptive/checkpoint_10000.pth"
SPECTRAL_DIR="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectral-hardtail-300to500-trainsteps8-adaptive"

PHASES_2K='[
  {"name":"spectral_300_500_002k","end_step":2000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}
]'

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

eval_hardtail() {
  local tag="$1"
  shift
  python "${REPO_DIR}/scripts/eval_rowcol_svt_spectral_tails.py" \
    --preset hard_tail \
    --sizes 300 500 \
    --seeds 0 1 2 \
    --missing-rate 0.2 \
    --softimpute-max-iters 100 \
    --svt-steps 32 \
    --svt-max-rank 80 \
    --svt-tau-fractions 0.004 0.008 0.01 0.015 0.02 0.03 \
    --adaptive-svt \
    "$@" \
    --output-prefix "${FIG_DIR}/${tag}" \
    2>&1 | tee "${QUEUE_DIR}/${tag}.log"
}

make_override_checkpoint() {
  local source_dir="$1"
  local out_dir="$2"
  local steps="$3"
  local max_rank="$4"
  python - "$source_dir" "$out_dir" "$steps" "$max_rank" <<'PY'
from pathlib import Path
import json
import sys
import torch

source_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
steps = int(sys.argv[3])
max_rank = int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
for name in ["checkpoint_10000.pth", "best_checkpoint.pth"]:
    src = source_dir / name
    if src.exists():
        ck = torch.load(src, map_location="cpu")
        torch.save(ck, out_dir / name)
cfg = json.loads((source_dir / "config.json").read_text())
cfg["model"]["svt_steps"] = steps
cfg["model"]["svt_max_rank"] = max_rank
cfg["eval_override_note"] = f"Same weights with svt_steps={steps}, svt_max_rank={max_rank}."
(out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
PY
}

train_variant() {
  local tag="$1"
  local init_ckpt="$2"
  local model_overrides="$3"
  local lr="$4"
  local point_weight="$5"
  local work_dir="${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectral-attempt-${tag}"
  local run_name="tabimpute-direct-rowcol-svthybrid768-spectral-attempt-${tag}"
  mkdir -p "${work_dir}"
  if [[ ! -f "${work_dir}/checkpoint_2000.pth" ]]; then
    python "${REPO_DIR}/scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py" \
      --run-name "${run_name}" \
      --work-dir "${work_dir}" \
      --device "${DEVICE:-cuda}" \
      --seed "${SEED:-52}" \
      --prior-kind spectrum_decay \
      --missingness-type mcar \
      --p-missing 0.2 \
      --weight-decay "${WEIGHT_DECAY:-0.02}" \
      --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
      --warmup-steps 200 \
      --stable-steps 0 \
      --decay-steps 1800 \
      --min-lr-ratio 0.10 \
      --checkpoint-every 1000 \
      --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-2048}" \
      --wandb-project "${WANDB_PROJECT:-tabimpute}" \
      --lr "${lr}" \
      --phases-json "${PHASES_2K}" \
      --init-model-checkpoint "${init_ckpt}" \
      --allow-partial-init \
      --model-overrides-json "${model_overrides}" \
      --prior-overrides-json "${PRIOR_OVERRIDES}" \
      --nll-loss-weight 0.02 \
      --cdf-loss-weight 0.2 \
      --point-loss-weight "${point_weight}" \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.02 \
      2>&1 | tee "${QUEUE_DIR}/spectral_attempt_${tag}_train.log"
  fi

  eval_hardtail "spectral_attempt_${tag}_eval" \
    --skip-softimpute \
    --skip-svt \
    --extra-direct "attempt ${tag} final|${work_dir}/checkpoint_2000.pth|1000|rowcol_svt_hybrid" \
    --extra-direct "attempt ${tag} best|${work_dir}/best_checkpoint.pth|1000|rowcol_svt_hybrid"
}

echo "Attempt 1: trained spectral checkpoint with deeper eval-time unroll."
make_override_checkpoint "${SPECTRAL_DIR}" "${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectral-hardtail-evaloverride-steps32-r80" 32 80
eval_hardtail "spectral_attempt_1_deeper_eval" \
  --extra-direct "spectral trained evaloverride s32 r80|${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-spectral-hardtail-evaloverride-steps32-r80/checkpoint_10000.pth|1000|rowcol_svt_hybrid"

echo "Attempt 2: broader algorithmic SVT sweep only."
eval_hardtail "spectral_attempt_2_svt_sweep" --skip-softimpute

echo "Attempt 3: high-rank initializer, short unroll, higher LR."
train_variant "rank80_steps4_lr3e5" \
  "${BASE_LATENT}" \
  '{"svt_init_rank":80,"svt_steps":4,"svt_max_rank":80,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.004,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_update_damping":1.0,"svt_init_scale":0.02,"logit_sigma_init":0.08,"logit_sigma_min":0.008}' \
  "3e-5" \
  "1000.0"

echo "Attempt 4: no-SVT high-rank initializer probe."
train_variant "rank80_steps0_lr3e5" \
  "${BASE_LATENT}" \
  '{"svt_init_rank":80,"svt_steps":0,"svt_max_rank":80,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.004,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_update_damping":1.0,"svt_init_scale":0.02,"logit_sigma_init":0.08,"logit_sigma_min":0.008}' \
  "3e-5" \
  "1000.0"

echo "Attempt 5: post-SVT neural residual correction."
train_variant "residual40_steps8_lr3e5" \
  "${SPECTRAL_CKPT}" \
  '{"svt_init_rank":10,"svt_steps":8,"svt_max_rank":80,"svt_threshold_fraction_init":0.006,"svt_threshold_fraction_min":0.004,"svt_threshold_fraction_max":0.02,"svt_threshold_mode":"inverse_size","svt_threshold_size_scale_init":1.2,"svt_update_damping":1.0,"svt_init_scale":0.02,"logit_sigma_init":0.08,"logit_sigma_min":0.008,"use_neural_residual":true,"neural_residual_rank":40,"neural_residual_scale_init":0.001}' \
  "3e-5" \
  "1000.0"

python - <<'PY'
from pathlib import Path
import csv

root = Path("runs/figures_by_setup/05_spectral_tails_rowcol_svt/attempts")
rows = []
for path in sorted(root.glob("spectral_attempt_*_eval.summary.csv")) + sorted(root.glob("spectral_attempt_*_*.summary.csv")):
    if not path.exists():
        continue
    for row in csv.DictReader(path.open()):
        rows.append({"source": path.name, **row})
if rows:
    print("source\tmethod\tsize\trmse\tsem")
    for row in sorted(rows, key=lambda r: (r["source"], r["method"], int(r["size"]))):
        print(f"{row['source']}\t{row['method']}\t{row['size']}\t{float(row['rmse']):.6g}\t{float(row['rmse_sem']):.3g}")
PY
