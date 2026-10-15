#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ "${WAIT_FOR_PHASE_EVAL:-1}" == "1" ]]; then
  while tmux has-session -t rowcol_cdf_lr_phase_eval 2>/dev/null; do
    sleep 60
  done
fi

CDF2_RUN_NAME="${CDF2_RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-cdf2-60k-fromlarge80k}"
CDF2_CHECKPOINT="${CDF2_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/${CDF2_RUN_NAME}/checkpoint_60000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-latentattn768-rank10-p02-cdf2-tail300to500-1m-lr5e6-fromcdf2}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
DIAG_DIR="${REPO_DIR}/runs/diagnostics"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${WORK_DIR}" "${FIG_DIR}" "${DIAG_DIR}" "${QUEUE_DIR}"

PHASES_JSON='[
  {"name":"tail_300_to_500_010k","end_step":10000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_025k","end_step":25000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_050k","end_step":50000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_100k","end_step":100000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_150k","end_step":150000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_200k","end_step":200000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_300k","end_step":300000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_400k","end_step":400000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_500k","end_step":500000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_650k","end_step":650000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_800k","end_step":800000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1},
  {"name":"tail_300_to_500_1000k","end_step":1000000,"rows_low":300,"rows_high":500,"cols_low":300,"cols_high":500,"batch_size":1}
]'

FINAL_CKPT="${WORK_DIR}/checkpoint_1000000.pth"
if [[ ! -f "${FINAL_CKPT}" ]]; then
  python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
    --run-name "${RUN_NAME}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-52}" \
    --prior-kind latent_factor \
    --missingness-type "${MISSINGNESS_TYPE:-mcar}" \
    --p-missing "${P_MISSING:-0.2}" \
    --latent-rank-low "${LATENT_RANK_LOW:-10}" \
    --latent-rank-high "${LATENT_RANK_HIGH:-10}" \
    --lr "${LR:-5e-6}" \
    --weight-decay "${WEIGHT_DECAY:-0.02}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
    --warmup-steps "${WARMUP_STEPS:-1000}" \
    --stable-steps "${STABLE_STEPS:-800000}" \
    --decay-steps "${DECAY_STEPS:-199000}" \
    --min-lr-ratio "${MIN_LR_RATIO:-0.25}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-25000}" \
    --phases-json "${PHASES_JSON}" \
    --init-model-checkpoint "${CDF2_CHECKPOINT}" \
    --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-4096}" \
    --cdf-loss-weight "${CDF_LOSS_WEIGHT:-2.0}" \
    --wandb-project "${WANDB_PROJECT:-tabimpute}"
fi

OUT_PREFIX="${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2_tail1m_phase_checkpoints"

python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
  --sizes 20 50 75 100 150 200 300 400 500 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --generator gaussian_factors \
  --scale rank \
  --skip-softimpute \
  --direct-names \
  --extra-direct "RowCol CDF w2 tail1m phase010k|${WORK_DIR}/checkpoint_phase1_10000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase025k|${WORK_DIR}/checkpoint_phase2_25000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase050k|${WORK_DIR}/checkpoint_phase3_50000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase100k|${WORK_DIR}/checkpoint_phase4_100000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase150k|${WORK_DIR}/checkpoint_phase5_150000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase200k|${WORK_DIR}/checkpoint_phase6_200000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase300k|${WORK_DIR}/checkpoint_phase7_300000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase400k|${WORK_DIR}/checkpoint_phase8_400000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase500k|${WORK_DIR}/checkpoint_phase9_500000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase650k|${WORK_DIR}/checkpoint_phase10_650000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase800k|${WORK_DIR}/checkpoint_phase11_800000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m phase1000k|${WORK_DIR}/checkpoint_phase12_1000000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m trainloss best|${WORK_DIR}/best_checkpoint.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 tail1m final|${FINAL_CKPT}|500|rowcol_latent_attention" \
  --output-prefix "${OUT_PREFIX}"

python "${REPO_DIR}/scripts/diagnose_rowcol_information_flow.py" \
  --spec "RowCol CDF w2 tail1m final|${FINAL_CKPT}|rowcol_latent_attention" \
  --sizes 300 400 500 \
  --seeds 0 1 2 \
  --flow-seeds 0 \
  --rank 10 \
  --missing-rate 0.2 \
  --scale rank \
  --output-prefix "${DIAG_DIR}/rowcol_information_flow_cdf2_tail1m_p02_rank10"

python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

path = Path("/root/tabular/mcpfn/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2_tail1m_phase_checkpoints_summary.csv")
rows = list(csv.DictReader(path.open()))
by_size = defaultdict(list)
for row in rows:
    by_size[int(row["size"])].append(row)

best_out = path.with_name(path.stem + "_best_by_size.csv")
with best_out.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["size", "method", "rmse", "rmse_sem", "mae", "nrmse"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for size in sorted(by_size):
        best = min(by_size[size], key=lambda r: float(r["rmse"]))
        writer.writerow({key: best[key] for key in fieldnames})

queue_out = Path("/root/tabular/mcpfn/runs/objective_queues/rowcol_cdf_tail1m_summary.csv")
queue_out.parent.mkdir(parents=True, exist_ok=True)
with queue_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {best_out}")
print(f"Wrote {queue_out}")
PY
