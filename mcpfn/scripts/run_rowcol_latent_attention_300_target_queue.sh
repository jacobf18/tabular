#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

NLL_DECAY_RUN_NAME="${NLL_DECAY_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-nll-tail300to500-1m-lr5e5-cosdecay-fromcdf2}"
NLL_DECAY_WORK_DIR="${REPO_DIR}/src/tabimpute/workdir/${NLL_DECAY_RUN_NAME}"
NLL_DECAY_200K="${NLL_DECAY_200K:-${NLL_DECAY_WORK_DIR}/checkpoint_phase6_200000.pth}"

SOFTIMPUTE_SUMMARY="${SOFTIMPUTE_SUMMARY:-${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_with_rowcol_summary.csv}"
SOFTIMPUTE_300_RMSE="${SOFTIMPUTE_300_RMSE:-0.008498957101356263}"

COMMON_TRAIN_ARGS=(
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-52}"
  --prior-kind latent_factor
  --missingness-type "${MISSINGNESS_TYPE:-mcar}"
  --p-missing "${P_MISSING:-0.2}"
  --latent-rank-low "${LATENT_RANK_LOW:-10}"
  --latent-rank-high "${LATENT_RANK_HIGH:-10}"
  --weight-decay "${WEIGHT_DECAY:-0.02}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}"
  --warmup-steps "${WARMUP_STEPS:-1000}"
  --stable-steps "${STABLE_STEPS:-0}"
  --decay-steps "${DECAY_STEPS:-199000}"
  --min-lr-ratio "${MIN_LR_RATIO:-0.05}"
  --checkpoint-every "${CHECKPOINT_EVERY:-25000}"
  --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-4096}"
  --wandb-project "${WANDB_PROJECT:-tabimpute}"
)

PHASES_300_200K='[
  {"name":"target300_025k","end_step":25000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_050k","end_step":50000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_100k","end_step":100000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_150k","end_step":150000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_200k","end_step":200000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1}
]'

should_run_variant() {
  local slug="$1"
  local selected="${SELECTED_VARIANTS:-all}"
  if [[ "${selected}" == "all" ]]; then
    return 0
  fi
  [[ " ${selected} " == *" ${slug} "* ]]
}

train_variant() {
  local slug="$1"
  local label="$2"
  local init_checkpoint="$3"
  local lr="$4"
  local model_overrides_json="$5"
  shift 5

  if ! should_run_variant "${slug}"; then
    echo "Skipping ${slug}; SELECTED_VARIANTS=${SELECTED_VARIANTS:-all}"
    return 0
  fi

  local run_name="tabimpute-direct-rowcol-latentattn768-rank10-p02-${slug}"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"
  local final_ckpt="${work_dir}/checkpoint_200000.pth"
  mkdir -p "${work_dir}"

  if [[ ! -f "${final_ckpt}" ]]; then
    local cmd=(
      python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py"
      --run-name "${run_name}"
      --work-dir "${work_dir}"
      "${COMMON_TRAIN_ARGS[@]}"
      --lr "${lr}"
      --phases-json "${PHASES_300_200K}"
    )
    if [[ "${init_checkpoint}" != "none" ]]; then
      cmd+=(--init-model-checkpoint "${init_checkpoint}")
    fi
    if [[ -n "${model_overrides_json}" ]]; then
      cmd+=(--model-overrides-json "${model_overrides_json}")
    fi
    cmd+=("$@")
    echo "Training ${label}"
    "${cmd[@]}"
  else
    echo "Found ${final_ckpt}; skipping training for ${label}"
  fi

  eval_variant "${slug}" "${label}" "${work_dir}"
}

eval_variant() {
  local slug="$1"
  local label="$2"
  local work_dir="$3"
  local output_prefix="${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_${slug}_target300_eval"

  python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
    --sizes 300 \
    --seeds 0 1 2 3 4 \
    --rank 10 \
    --missing-rate 0.2 \
    --generator gaussian_factors \
    --scale rank \
    --skip-softimpute \
    --direct-names \
    --extra-direct "${label} phase025k|${work_dir}/checkpoint_phase1_25000.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} phase050k|${work_dir}/checkpoint_phase2_50000.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} phase100k|${work_dir}/checkpoint_phase3_100000.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} phase150k|${work_dir}/checkpoint_phase4_150000.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} phase200k|${work_dir}/checkpoint_phase5_200000.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} trainloss best|${work_dir}/best_checkpoint.pth|300|rowcol_latent_attention" \
    --extra-direct "${label} final|${work_dir}/checkpoint_200000.pth|300|rowcol_latent_attention" \
    --output-prefix "${output_prefix}"
}

if [[ ! -f "${NLL_DECAY_200K}" ]]; then
  echo "Missing warm-start checkpoint: ${NLL_DECAY_200K}" >&2
  exit 1
fi

# Optimization-only target specialization from the best current 300x300 checkpoint.
train_variant \
  "target300-nll-lr5e5-cos200k-from-nlldecay200k" \
  "RowCol target300 NLL lr5e-5" \
  "${NLL_DECAY_200K}" \
  "5e-5" \
  ""

# Directly train the point statistic used by eval: median prediction.
train_variant \
  "target300-huber50-lr2e5-cos200k-from-nlldecay200k" \
  "RowCol target300 Huber50 lr2e-5" \
  "${NLL_DECAY_200K}" \
  "2e-5" \
  "" \
  --point-loss-weight 50.0 \
  --point-loss-kind huber \
  --point-loss-prediction median \
  --huber-delta 0.02

# Combine ordinal distribution alignment with the point-estimate pressure.
train_variant \
  "target300-cdf2-huber50-lr2e5-cos200k-from-nlldecay200k" \
  "RowCol target300 CDF2 Huber50 lr2e-5" \
  "${NLL_DECAY_200K}" \
  "2e-5" \
  "" \
  --cdf-loss-weight 2.0 \
  --point-loss-weight 50.0 \
  --point-loss-kind huber \
  --point-loss-prediction median \
  --huber-delta 0.02

# Capacity / information-flow test. This cannot warm-start because the shape changes.
# The row/column latents may be the main bottleneck, so make them much wider
# while keeping the network shallower to avoid paying for many message-passing
# rounds before we know the wider latent state helps.
SHALLOW_WIDE_MODEL_OVERRIDES='{"embedding_size":1536,"num_attention_heads":24,"mlp_hidden_size":4096,"num_layers":6,"rabel_num_kernels":160,"rabel_exponent_embedding_size":96,"count_hidden_size":384,"drop_path_rate":0.025,"residual_scale_init":0.10,"decoder_interaction_scale_init":0.30}'
train_variant \
  "target300-shallow1536x6-cdf2-huber50-lr5e5-cos200k-scratch" \
  "RowCol target300 shallow1536x6 CDF2 Huber50 scratch" \
  "none" \
  "5e-5" \
  "${SHALLOW_WIDE_MODEL_OVERRIDES}" \
  --cdf-loss-weight 2.0 \
  --point-loss-weight 50.0 \
  --point-loss-kind huber \
  --point-loss-prediction median \
  --huber-delta 0.02

python - <<'PY'
import csv
from pathlib import Path

fig_dir = Path("/root/tabular/mcpfn/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds")
queue_dir = Path("/root/tabular/mcpfn/runs/objective_queues")
summary_out = queue_dir / "rowcol_300_target_queue_summary.csv"
best_out = queue_dir / "rowcol_300_target_queue_best.csv"
softimpute_rmse = float("0.008498957101356263")

patterns = sorted(fig_dir.glob("interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-*_target300_eval_summary.csv"))
rows = []
for path in patterns:
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

if not rows:
    print("No target300 summaries found yet.")
    raise SystemExit(0)

with summary_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best = min(rows, key=lambda row: float(row["rmse"]))
with best_out.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "method",
        "size",
        "rmse",
        "rmse_sem",
        "softimpute_rmse",
        "rowcol_minus_softimpute",
        "rowcol_over_softimpute",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    rmse = float(best["rmse"])
    writer.writerow(
        {
            "method": best["method"],
            "size": best["size"],
            "rmse": best["rmse"],
            "rmse_sem": best["rmse_sem"],
            "softimpute_rmse": f"{softimpute_rmse:.9f}",
            "rowcol_minus_softimpute": f"{rmse - softimpute_rmse:.9f}",
            "rowcol_over_softimpute": f"{rmse / softimpute_rmse:.3f}",
        }
    )
print(f"Wrote {summary_out}")
print(f"Wrote {best_out}")
print(
    f"Best target300: {best['method']} rmse={float(best['rmse']):.6f}; "
    f"SoftImpute={softimpute_rmse:.6f}; ratio={float(best['rmse']) / softimpute_rmse:.3f}"
)
PY
