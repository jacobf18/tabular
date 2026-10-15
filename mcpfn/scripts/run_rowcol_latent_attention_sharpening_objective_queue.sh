#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

BASE_RUN_NAME="${BASE_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-large-80k-from100k}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/${BASE_RUN_NAME}/checkpoint_80000.pth}"
FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
DIAG_DIR="${REPO_DIR}/runs/diagnostics"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${DIAG_DIR}" "${QUEUE_DIR}"

PHASES_JSON='[{"name":"50_to_125","end_step":15000,"rows_low":50,"rows_high":125,"cols_low":50,"cols_high":125,"batch_size":8},{"name":"125_to_300","end_step":40000,"rows_low":125,"rows_high":300,"cols_low":125,"cols_high":300,"batch_size":3},{"name":"300_to_500","end_step":60000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}]'

COMMON_TRAIN_ARGS=(
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-52}"
  --prior-kind latent_factor
  --missingness-type "${MISSINGNESS_TYPE:-mcar}"
  --p-missing "${P_MISSING:-0.2}"
  --latent-rank-low "${LATENT_RANK_LOW:-10}"
  --latent-rank-high "${LATENT_RANK_HIGH:-10}"
  --lr "${LR:-2e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.02}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}"
  --warmup-steps "${WARMUP_STEPS:-1000}"
  --stable-steps "${STABLE_STEPS:-44000}"
  --decay-steps "${DECAY_STEPS:-15000}"
  --min-lr-ratio "${MIN_LR_RATIO:-0.25}"
  --checkpoint-every "${CHECKPOINT_EVERY:-5000}"
  --phases-json "${PHASES_JSON}"
  --init-model-checkpoint "${INIT_MODEL_CHECKPOINT:-${BASE_CHECKPOINT}}"
  --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-4096}"
  --wandb-project "${WANDB_PROJECT:-tabimpute}"
)

run_variant() {
  local slug="$1"
  local label="$2"
  shift 2
  local run_name="tabimpute-direct-rowcol-latentattn768-rank10-p02-${slug}-60k-fromlarge80k"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"
  local final_ckpt="${work_dir}/checkpoint_60000.pth"
  local best_ckpt="${work_dir}/best_checkpoint.pth"
  local output_prefix="${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_${slug}_eval"

  mkdir -p "${work_dir}"
  if [[ ! -f "${final_ckpt}" ]]; then
    python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
      --run-name "${run_name}" \
      --work-dir "${work_dir}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      "$@"
  fi

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
    --sizes 50 300 500 \
    --seeds 0 1 2 \
    --flow-seeds 0 \
    --rank 10 \
    --missing-rate 0.2 \
    --scale rank \
    --output-prefix "${DIAG_DIR}/rowcol_information_flow_${slug}_p02_rank10"
}

run_variant \
  "cdf2" \
  "RowCol CDF w2 60k" \
  --cdf-loss-weight 2.0

run_variant \
  "softce010_sigma002" \
  "RowCol softCE w0.10 sigma0.02 60k" \
  --soft-target-loss-weight 0.10 \
  --soft-target-sigma 0.02

run_variant \
  "cdf1_entropy005" \
  "RowCol CDF w1 entropy w0.05 60k" \
  --cdf-loss-weight 1.0 \
  --entropy-loss-weight 0.05

run_variant \
  "huber25_median" \
  "RowCol median Huber w25 60k" \
  --point-loss-weight 25.0 \
  --point-loss-kind huber \
  --point-loss-prediction median \
  --huber-delta 0.05

python - <<'PY'
import csv
from pathlib import Path

fig_dir = Path("/root/tabular/mcpfn/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds")
out = Path("/root/tabular/mcpfn/runs/objective_queues/rowcol_sharpening_objective_queue_summary.csv")
patterns = [
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_softce010_sigma002_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf1_entropy005_eval_summary.csv",
    "interface_train_level_chunks_p02_rankscale_5seeds_rowcol_huber25_median_eval_summary.csv",
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
    print("No objective queue summaries found yet.")
PY
