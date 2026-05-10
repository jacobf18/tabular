#!/usr/bin/env bash
set -euo pipefail

# Multi-phase rectangular-axis curriculum.
#
# This starts from the 50x50 rank-10 checkpoint, then alternates row-long and
# column-long rectangular shapes. Each phase loads the checkpoint produced by
# the previous phase.
#
# Defaults:
#   phase 1: 50x50, 20x100, 100x20
#   phase 2: 50x50, 25x150, 150x25, 100x100
#   phase 3: 50x50, 32x200, 200x32, 100x100
#
# Example:
#   bash scripts/train_curriculum_rank10_rectangular_axes_phases.sh
#
# Useful overrides:
#   PHASE_STEPS=10000 BATCH_SIZE=8 bash scripts/train_curriculum_rank10_rectangular_axes_phases.sh
#   PHASE1_SHAPES="50x50,20x128,128x20" PHASES=1 bash scripts/train_curriculum_rank10_rectangular_axes_phases.sh

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export REPO_DIR

BASE_RUN_DIR="${BASE_RUN_DIR:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-50x50}"
INITIAL_CHECKPOINT_PATH="${INITIAL_CHECKPOINT_PATH:-${BASE_RUN_DIR}/checkpoint_30000.pth}"
export BASE_RUN_DIR INITIAL_CHECKPOINT_PATH

PHASE_STEPS="${PHASE_STEPS:-15000}"
PHASES="${PHASES:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-offline}"
DECODE_ONLY_MISSING="${DECODE_ONLY_MISSING:-1}"
export BATCH_SIZE CHECKPOINT_EVERY DEVICE WANDB_PROJECT WANDB_MODE DECODE_ONLY_MISSING

PHASE1_SHAPES="${PHASE1_SHAPES:-50x50,20x100,100x20}"
PHASE2_SHAPES="${PHASE2_SHAPES:-50x50,25x150,150x25,100x100}"
PHASE3_SHAPES="${PHASE3_SHAPES:-50x50,32x200,200x32,100x100}"

RUN_PREFIX="${RUN_PREFIX:-tabimpute-new-stable_v2-rank_10_10-rectangular-axes}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
export WORK_ROOT

phase_shapes() {
    case "$1" in
        1) printf '%s\n' "${PHASE1_SHAPES}" ;;
        2) printf '%s\n' "${PHASE2_SHAPES}" ;;
        3) printf '%s\n' "${PHASE3_SHAPES}" ;;
        *) echo "No default shapes configured for phase $1" >&2; exit 1 ;;
    esac
}

if (( PHASES < 1 || PHASES > 3 )); then
    echo "PHASES must be between 1 and 3; got ${PHASES}" >&2
    exit 1
fi

checkpoint_path="${INITIAL_CHECKPOINT_PATH}"

for phase in $(seq 1 "${PHASES}"); do
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "Checkpoint not found before phase ${phase}: ${checkpoint_path}" >&2
        exit 1
    fi

    start_epoch="$(python - "${checkpoint_path}" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(ckpt["epoch"]))
PY
)"
    target_epoch="$((start_epoch + PHASE_STEPS))"
    shapes="$(phase_shapes "${phase}")"
    run_name="${RUN_PREFIX}-phase${phase}-step${start_epoch}_to_${target_epoch}"
    work_dir="${WORK_ROOT}/${run_name}"

    echo "=== Phase ${phase}/${PHASES} ==="
    echo "Checkpoint: ${checkpoint_path}"
    echo "Shapes: ${shapes}"
    echo "Steps: ${start_epoch} -> ${target_epoch}"
    echo "Work dir: ${work_dir}"

    CHECKPOINT_PATH="${checkpoint_path}" \
    RUN_NAME="${run_name}" \
    WORK_DIR="${work_dir}" \
    SHAPE_SCHEDULE="${shapes}" \
    CURRICULUM_STEPS="${PHASE_STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    DEVICE="${DEVICE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    DECODE_ONLY_MISSING="${DECODE_ONLY_MISSING}" \
    bash "${REPO_DIR}/scripts/train_curriculum_rank10_rectangular_axes.sh"

    checkpoint_path="${work_dir}/checkpoint_${target_epoch}.pth"
done

echo "Rectangular-axis curriculum complete."
echo "Final checkpoint: ${checkpoint_path}"
