#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
export WANDB_MODE WANDB_PROJECT

declare -a RUNS=(
  "5e-5:lr5e5"
  "8e-5:lr8e5"
  "1e-4:lr1e4"
)

for spec in "${RUNS[@]}"; do
  lr="${spec%%:*}"
  label="${spec##*:}"
  run_name="tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-${label}"
  work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"

  if [[ -f "${work_dir}/checkpoint_160000.pth" ]]; then
    echo "Skipping completed run ${run_name}"
    continue
  fi

  mkdir -p "${work_dir}"
  echo "Starting ${run_name} with LR=${lr}"
  (
    cd "${REPO_DIR}"
    RUN_NAME="${run_name}" \
    WORK_DIR="${work_dir}" \
    LR="${lr}" \
    bash scripts/train_direct_pure_rabel_spectrum_curriculum.sh
  ) 2>&1 | tee -a "${work_dir}/train.log"
done
