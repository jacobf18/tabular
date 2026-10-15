#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORKDIR_ROOT="${WORKDIR_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${WORKDIR_ROOT}/tabimpute-direct-pure-rabel64-joint-skinny55-randcate25-6k-from-skinny/best_checkpoint.pth}"
DEVICE="${DEVICE:-cuda}"
WANDB_MODE="${WANDB_MODE:-online}"
DOPFN_N_MAX="${DOPFN_N_MAX:-10}"
LOW_RANK_REPS="${LOW_RANK_REPS:-3}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

run_and_eval() {
  local run_name="$1"
  local init_checkpoint="$2"
  shift 2

  local work_dir="${WORKDIR_ROOT}/${run_name}"
  mkdir -p "${work_dir}"
  echo "[$(date -Is)] START ${run_name}"
  (
    cd "${REPO_DIR}"
    env \
      RUN_NAME="${run_name}" \
      WORK_DIR="${work_dir}" \
      INIT_CHECKPOINT="${init_checkpoint}" \
      DEVICE="${DEVICE}" \
      WANDB_MODE="${WANDB_MODE}" \
      "$@" \
      bash scripts/train_direct_pure_rabel_joint_matrix_causal.sh
  ) >"${work_dir}/train.log" 2>&1

  local checkpoint="${work_dir}/best_checkpoint.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${work_dir}/checkpoint_${STEPS:-8000}.pth"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[$(date -Is)] ERROR missing checkpoint for ${run_name}" >&2
    return 1
  fi

  echo "[$(date -Is)] EVAL ${run_name} checkpoint=${checkpoint}"
  (
    cd "${REPO_DIR}"
    CHECKPOINT="${checkpoint}" \
      DEVICE="${DEVICE}" \
      DOPFN_N_MAX="${DOPFN_N_MAX}" \
      LOW_RANK_REPS="${LOW_RANK_REPS}" \
      bash scripts/eval_joint_matrix_causal_checkpoint.sh
  ) >"${work_dir}/eval.log" 2>&1
  echo "[$(date -Is)] DONE ${run_name}"
}

# Idea 1: paired counterfactual CATE queries with larger treatment effects.
run_and_eval \
  "tabimpute-direct-pure-rabel64-oos-idea1-pairedcate-8k-from-best" \
  "${BASE_CHECKPOINT}" \
  STEPS=8000 \
  WARMUP_STEPS=500 \
  STABLE_STEPS=5500 \
  DECAY_STEPS=2000 \
  BATCH_SIZE=2 \
  CAUSAL_PROB=0.70 \
  QUERY_TREATMENT_MODE=paired \
  CATE_TAU_SCALE=2.0 \
  CATE_QUERY_NOISE_SCALE=0.15 \
  SKINNY_BLOCK_PROB=0.15 \
  SKINNY_COLS_LOW=6 \
  SKINNY_COLS_HIGH=18 \
  SKINNY_RANK_LOW=2 \
  SKINNY_RANK_HIGH=8 \
  SKINNY_EXTRA_MCAR_PROB=0.0 \
  LR=1e-5 \
  MIN_LR=3e-6 \
  CHECKPOINT_EVERY=2000

# Idea 2: generic skinny held-out-column low-rank OOS tasks, broader and less sparse.
run_and_eval \
  "tabimpute-direct-pure-rabel64-oos-idea2-skinnywide-8k-from-best" \
  "${BASE_CHECKPOINT}" \
  STEPS=8000 \
  WARMUP_STEPS=500 \
  STABLE_STEPS=5500 \
  DECAY_STEPS=2000 \
  BATCH_SIZE=2 \
  CAUSAL_PROB=0.0 \
  QUERY_TREATMENT_MODE=factual \
  SKINNY_BLOCK_PROB=0.80 \
  SKINNY_COLS_LOW=8 \
  SKINNY_COLS_HIGH=32 \
  SKINNY_RANK_LOW=2 \
  SKINNY_RANK_HIGH=12 \
  SKINNY_EXTRA_MCAR_PROB=0.03 \
  LR=1e-5 \
  MIN_LR=3e-6 \
  CHECKPOINT_EVERY=2000

# Idea 3: low-LR hybrid continuation from the best of the previous quick search.
run_and_eval \
  "tabimpute-direct-pure-rabel64-oos-idea3-hybrid-lowlr-10k-from-best" \
  "${BASE_CHECKPOINT}" \
  STEPS=10000 \
  WARMUP_STEPS=500 \
  STABLE_STEPS=7500 \
  DECAY_STEPS=2000 \
  BATCH_SIZE=2 \
  CAUSAL_PROB=0.35 \
  QUERY_TREATMENT_MODE=paired \
  CATE_TAU_SCALE=1.5 \
  CATE_QUERY_NOISE_SCALE=0.25 \
  SKINNY_BLOCK_PROB=0.45 \
  SKINNY_COLS_LOW=6 \
  SKINNY_COLS_HIGH=24 \
  SKINNY_RANK_LOW=2 \
  SKINNY_RANK_HIGH=10 \
  SKINNY_EXTRA_MCAR_PROB=0.01 \
  LR=5e-6 \
  MIN_LR=2e-6 \
  CHECKPOINT_EVERY=2000

python - <<'PY'
from pathlib import Path
import pandas as pd

reports = Path("/root/matrix-completion-benchmarks/benchmarks/reports")
runs = [
    "tabimpute-direct-pure-rabel64-oos-idea1-pairedcate-8k-from-best-best_checkpoint",
    "tabimpute-direct-pure-rabel64-oos-idea2-skinnywide-8k-from-best-best_checkpoint",
    "tabimpute-direct-pure-rabel64-oos-idea3-hybrid-lowlr-10k-from-best-best_checkpoint",
]
rows = []
for tag in runs:
    summary = reports / f"dopfn_matrix_completion_joint_{tag}" / "plots" / "dopfn_matrix_completion_summary.csv"
    if not summary.exists():
        continue
    df = pd.read_csv(summary)
    joint = df[df.algorithm == "tab_impute_pure_rabel_joint_matrix_causal"].iloc[0]
    col = df[df.algorithm == "col_mean"].iloc[0]
    soft = df[df.algorithm == "soft_impute"].iloc[0]
    rows.append(
        {
            "run": tag.replace("-best_checkpoint", ""),
            "joint_pehe_mean": float(joint.pehe_mean),
            "delta_vs_colmean": float(joint.pehe_mean - col.pehe_mean),
            "delta_vs_softimpute": float(joint.pehe_mean - soft.pehe_mean),
        }
    )
out = pd.DataFrame(rows).sort_values("joint_pehe_mean")
path = Path("/root/tabular/mcpfn/src/tabimpute/workdir/joint_oos_idea_queue_summary.csv")
out.to_csv(path, index=False)
print(out.to_string(index=False))
print(f"wrote: {path}")
PY
