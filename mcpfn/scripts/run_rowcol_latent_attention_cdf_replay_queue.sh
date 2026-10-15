#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

LARGE_RUN_NAME="${LARGE_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-large-80k-from100k}"
LARGE_CHECKPOINT="${LARGE_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/${LARGE_RUN_NAME}/checkpoint_80000.pth}"
CDF2_RUN_NAME="${CDF2_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-cdf2-60k-fromlarge80k}"
CDF2_CHECKPOINT="${CDF2_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/${CDF2_RUN_NAME}/checkpoint_60000.pth}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
DIAG_DIR="${REPO_DIR}/runs/diagnostics"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${DIAG_DIR}" "${QUEUE_DIR}"

COMMON_TRAIN_ARGS=(
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-52}"
  --prior-kind latent_factor
  --missingness-type "${MISSINGNESS_TYPE:-mcar}"
  --p-missing "${P_MISSING:-0.2}"
  --latent-rank-low "${LATENT_RANK_LOW:-10}"
  --latent-rank-high "${LATENT_RANK_HIGH:-10}"
  --lr "${LR:-1e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.02}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}"
  --warmup-steps "${WARMUP_STEPS:-1000}"
  --min-lr-ratio "${MIN_LR_RATIO:-0.25}"
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}"
  --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-4096}"
  --wandb-project "${WANDB_PROJECT:-tabimpute}"
)

eval_variant() {
  local slug="$1"
  local label="$2"
  local final_step="$3"
  local output_prefix="${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_${slug}_eval"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-${slug}"
  local final_ckpt="${work_dir}/checkpoint_${final_step}.pth"
  local best_ckpt="${work_dir}/best_checkpoint.pth"

  python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
    --sizes 20 50 75 100 150 200 300 400 500 \
    --seeds 0 1 2 3 4 \
    --rank 10 \
    --missing-rate 0.2 \
    --generator gaussian_factors \
    --scale rank \
    --skip-softimpute \
    --direct-names \
    --extra-direct "${label} final|${final_ckpt}|500|rowcol_latent_attention" \
    --extra-direct "${label} best|${best_ckpt}|500|rowcol_latent_attention" \
    --output-prefix "${output_prefix}"

  python "${REPO_DIR}/scripts/diagnose_rowcol_information_flow.py" \
    --spec "${label} final|${final_ckpt}|rowcol_latent_attention" \
    --sizes 50 200 500 \
    --seeds 0 1 2 \
    --flow-seeds 0 \
    --rank 10 \
    --missing-rate 0.2 \
    --scale rank \
    --output-prefix "${DIAG_DIR}/rowcol_information_flow_${slug}_p02_rank10"
}

run_variant() {
  local slug="$1"
  local label="$2"
  local init_checkpoint="$3"
  local phases_json="$4"
  local final_step="$5"
  local stable_steps="$6"
  local decay_steps="$7"
  shift 7

  local run_name="tabimpute-direct-rowcol-latentattn768-rank10-p02-${slug}"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"
  local final_ckpt="${work_dir}/checkpoint_${final_step}.pth"

  mkdir -p "${work_dir}"
  if [[ ! -f "${final_ckpt}" ]]; then
    python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
      --run-name "${run_name}" \
      --work-dir "${work_dir}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --stable-steps "${stable_steps}" \
      --decay-steps "${decay_steps}" \
      --phases-json "${phases_json}" \
      --init-model-checkpoint "${init_checkpoint}" \
      "$@"
  fi

  eval_variant "${slug}" "${label}" "${final_step}"
}

PHASES_MIXED20='[{"name":"mixed_50_to_500","end_step":20000,"rows_low":50,"rows_high":500,"cols_low":50,"cols_high":500,"batch_size":1}]'
PHASES_MID_THEN_MIXED40='[{"name":"repair_50_to_300","end_step":20000,"rows_low":50,"rows_high":300,"cols_low":50,"cols_high":300,"batch_size":3},{"name":"mixed_50_to_500","end_step":40000,"rows_low":50,"rows_high":500,"cols_low":50,"cols_high":500,"batch_size":1}]'
PHASES_BALANCED80='[{"name":"50_to_125","end_step":15000,"rows_low":50,"rows_high":125,"cols_low":50,"cols_high":125,"batch_size":8},{"name":"125_to_300","end_step":35000,"rows_low":125,"rows_high":300,"cols_low":125,"cols_high":300,"batch_size":3},{"name":"300_to_500","end_step":60000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},{"name":"mixed_50_to_500","end_step":80000,"rows_low":50,"rows_high":500,"cols_low":50,"cols_high":500,"batch_size":1}]'

run_variant \
  "cdf2-mixedreplay20k-fromcdf2" \
  "RowCol CDF w2 mixed replay 20k" \
  "${CDF2_CHECKPOINT}" \
  "${PHASES_MIXED20}" \
  20000 \
  14000 \
  5000 \
  --cdf-loss-weight 2.0

run_variant \
  "cdf2-midthenmixed40k-fromcdf2" \
  "RowCol CDF w2 mid+mixed replay 40k" \
  "${CDF2_CHECKPOINT}" \
  "${PHASES_MID_THEN_MIXED40}" \
  40000 \
  30000 \
  9000 \
  --cdf-loss-weight 2.0

run_variant \
  "cdf1-mixedreplay20k-fromcdf2" \
  "RowCol CDF w1 mixed replay 20k" \
  "${CDF2_CHECKPOINT}" \
  "${PHASES_MIXED20}" \
  20000 \
  14000 \
  5000 \
  --cdf-loss-weight 1.0

run_variant \
  "cdf2-balanced80k-fromlarge80k" \
  "RowCol CDF w2 balanced 80k" \
  "${LARGE_CHECKPOINT}" \
  "${PHASES_BALANCED80}" \
  80000 \
  60000 \
  19000 \
  --cdf-loss-weight 2.0

python - <<'PY'
import csv
from pathlib import Path

fig_dir = Path("/root/tabular/mcpfn/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds")
out = Path("/root/tabular/mcpfn/runs/objective_queues/rowcol_cdf_replay_queue_summary.csv")
patterns = [
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2-mixedreplay20k-fromcdf2_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2-midthenmixed40k-fromcdf2_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf1-mixedreplay20k-fromcdf2_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2-balanced80k-fromlarge80k_eval_summary.csv",
]
rows = []
for name in patterns:
    path = fig_dir / name
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
else:
    print("No replay queue summaries found yet.")
PY
