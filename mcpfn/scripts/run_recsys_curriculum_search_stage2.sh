#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/root/matrix-completion-benchmarks}"
COMMON="${REPO_DIR}/scripts/train_compressed_cross_rank10_recsys_curriculum_search_common.sh"
WORKDIR="${REPO_DIR}/src/tabimpute/workdir"
PREFIX="${PREFIX:-tabimpute-comp-recsys-search-stage2}"

BASELINE_NAME="tabimpute-comp-recsys-search-stage1-p1e-5-c2e-5-wd3e-2"
BASELINE_CHECKPOINT="${WORKDIR}/${BASELINE_NAME}/checkpoint_5000.pth"
SOURCE_CHECKPOINT="${WORKDIR}/tabimpute-comp-rank10-recsys-exposed-fitted_bias-broadrect100to700x400to1700-100chunks-random-k20-50k-from-broadbest/best_checkpoint.pth"

run_config() {
  local suffix="$1"
  local source_checkpoint="$2"
  local steps="$3"
  local k="$4"
  local targets_per_step="$5"
  local curriculum="$6"
  local rows_min="$7"
  local rows_max="$8"
  local cols_min="$9"
  local cols_max="${10}"
  local run_name="${PREFIX}-${suffix}"
  local checkpoint="${WORKDIR}/${run_name}/checkpoint_${steps}.pth"

  if [[ -f "${checkpoint}" ]]; then
    echo "Skipping completed ${run_name}"
    return
  fi

  local warmup stable decay
  if [[ "${steps}" -le 5000 ]]; then
    warmup=250
    stable=4250
    decay=500
  elif [[ "${steps}" -le 10000 ]]; then
    warmup=500
    stable=8000
    decay=1500
  else
    warmup=500
    stable=$((steps - 3000))
    decay=2500
  fi

  RUN_NAME="${run_name}" \
  WORK_DIR="${WORKDIR}/${run_name}" \
  COMPRESSED_CHECKPOINT="${source_checkpoint}" \
  STEPS="${steps}" \
  WARMUP_STEPS="${warmup}" \
  STABLE_STEPS="${stable}" \
  DECAY_STEPS="${decay}" \
  K_ROW="${k}" \
  K_COL="${k}" \
  PREDICTOR_LR=1e-5 \
  COMPRESSOR_LR=2e-5 \
  WEIGHT_DECAY=0.03 \
  RANDOM_TARGETS_PER_STEP="${targets_per_step}" \
  MATRIX_ROWS_MIN="${rows_min}" \
  MATRIX_ROWS_MAX="${rows_max}" \
  MATRIX_COLS_MIN="${cols_min}" \
  MATRIX_COLS_MAX="${cols_max}" \
  RECSYS_CURRICULUM="${curriculum}" \
  EXPOSURE_DENSITY_LOW=0.04 \
  EXPOSURE_DENSITY_HIGH=0.06 \
  "${COMMON}"
}

evaluate_screen() {
  local output_dir="${BENCHMARK_DIR}/tests/rec_sys/results/recsys_curriculum_search_stage2_seed101"
  cd "${BENCHMARK_DIR}"
  PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" \
  python tests/rec_sys/eval_synthetic_movielens_proxy.py \
    --num-users 943 \
    --num-items 1682 \
    --seeds 101 \
    --chunk-rows 100 \
    --chunk-cols 100 \
    --residual-scales 0.25 \
    --no-default-models \
    --skip-raw \
    --model-spec "stage1_baseline=${BASELINE_CHECKPOINT}" \
    --model-spec "k20_one=${WORKDIR}/${PREFIX}-k20-one-target-5k/checkpoint_5000.pth" \
    --model-spec "k32_one=${WORKDIR}/${PREFIX}-k32-one-target-5k/checkpoint_5000.pth" \
    --model-spec "k20_two=${WORKDIR}/${PREFIX}-k20-two-target-5k/checkpoint_5000.pth" \
    --model-spec "k20_direct=${WORKDIR}/${PREFIX}-k20-direct-target-5k/checkpoint_5000.pth" \
    --output-dir "${output_dir}"
}

echo "Stage 2A: 5k-step capacity and gradient screen"
run_config "k20-one-target-5k" "${BASELINE_CHECKPOINT}" 5000 20 1 \
  movielens_proxy_20k 150 1000 300 1800
run_config "k32-one-target-5k" "${BASELINE_CHECKPOINT}" 5000 32 1 \
  movielens_proxy_20k 150 1000 300 1800
run_config "k20-two-target-5k" "${BASELINE_CHECKPOINT}" 5000 20 2 \
  movielens_proxy_20k 150 1000 300 1800
run_config "k20-direct-target-5k" "${BASELINE_CHECKPOINT}" 5000 20 1 \
  none 800 1000 1400 1800

evaluate_screen

SCREEN_CSV="${BENCHMARK_DIR}/tests/rec_sys/results/recsys_curriculum_search_stage2_seed101/synthetic_movielens_proxy_summary.csv"
read -r WINNER BASELINE_RMSE WINNER_RMSE IMPROVEMENT < <(
  python - "${SCREEN_CSV}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
rmse = {row["method"]: float(row["mean_rmse"]) for row in rows}
baseline = rmse["stage1_baseline__bias_residual_s0.25"]
candidates = {
    key.split("__", 1)[0]: value
    for key, value in rmse.items()
    if key.startswith(("k20_one__", "k32_one__", "k20_two__", "k20_direct__"))
}
winner, winner_rmse = min(candidates.items(), key=lambda item: item[1])
print(winner, baseline, winner_rmse, baseline - winner_rmse)
PY
)

echo "Stage 2A winner=${WINNER} baseline=${BASELINE_RMSE} winner_rmse=${WINNER_RMSE} improvement=${IMPROVEMENT}"

CONTINUE="$(
  python - "${IMPROVEMENT}" <<'PY'
import sys
print("yes" if float(sys.argv[1]) >= 0.0002 else "no")
PY
)"

FINAL_NAME=""
FINAL_CHECKPOINT=""
if [[ "${CONTINUE}" == "yes" ]]; then
  case "${WINNER}" in
    k20_one)
      INITIAL="${WORKDIR}/${PREFIX}-k20-one-target-5k/checkpoint_5000.pth"
      run_config "k20-one-target-continue15k" "${INITIAL}" 15000 20 1 \
        movielens_proxy_20k 150 1000 300 1800
      FINAL_NAME="k20_one_20k"
      FINAL_CHECKPOINT="${WORKDIR}/${PREFIX}-k20-one-target-continue15k/checkpoint_15000.pth"
      ;;
    k32_one)
      INITIAL="${WORKDIR}/${PREFIX}-k32-one-target-5k/checkpoint_5000.pth"
      run_config "k32-one-target-continue15k" "${INITIAL}" 15000 32 1 \
        movielens_proxy_20k 150 1000 300 1800
      FINAL_NAME="k32_one_20k"
      FINAL_CHECKPOINT="${WORKDIR}/${PREFIX}-k32-one-target-continue15k/checkpoint_15000.pth"
      ;;
    k20_two)
      INITIAL="${WORKDIR}/${PREFIX}-k20-two-target-5k/checkpoint_5000.pth"
      run_config "k20-two-target-continue5k" "${INITIAL}" 5000 20 2 \
        movielens_proxy_20k 150 1000 300 1800
      FINAL_NAME="k20_two_10k"
      FINAL_CHECKPOINT="${WORKDIR}/${PREFIX}-k20-two-target-continue5k/checkpoint_5000.pth"
      ;;
    k20_direct)
      FINAL_NAME="k20_direct_5k"
      FINAL_CHECKPOINT="${WORKDIR}/${PREFIX}-k20-direct-target-5k/checkpoint_5000.pth"
      ;;
  esac
else
  echo "No Stage-2 configuration cleared the 0.0002 RMSE continuation gate."
fi

if [[ -n "${FINAL_CHECKPOINT}" ]]; then
  CONFIRM_DIR="${BENCHMARK_DIR}/tests/rec_sys/results/recsys_curriculum_search_stage2_confirm_2seeds"
  cd "${BENCHMARK_DIR}"
  PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}" \
  python tests/rec_sys/eval_synthetic_movielens_proxy.py \
    --num-users 943 \
    --num-items 1682 \
    --seeds 101 102 \
    --chunk-rows 100 \
    --chunk-cols 100 \
    --residual-scales 0.25 \
    --no-default-models \
    --skip-raw \
    --model-spec "source=${SOURCE_CHECKPOINT}" \
    --model-spec "stage1_baseline=${BASELINE_CHECKPOINT}" \
    --model-spec "${FINAL_NAME}=${FINAL_CHECKPOINT}" \
    --output-dir "${CONFIRM_DIR}"
  echo "Stage-2 confirmation: ${CONFIRM_DIR}/synthetic_movielens_proxy_summary.md"
fi
