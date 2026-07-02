#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/root/matrix-completion-benchmarks}"
COMMON="${REPO_DIR}/scripts/train_compressed_cross_rank10_recsys_curriculum_search_common.sh"
WORKDIR="${REPO_DIR}/src/tabimpute/workdir"
PREFIX="${PREFIX:-tabimpute-comp-recsys-search-stage1}"
STEPS="${STEPS:-5000}"

run_config() {
  local suffix="$1"
  local predictor_lr="$2"
  local compressor_lr="$3"
  local weight_decay="$4"
  local run_name="${PREFIX}-${suffix}"
  local checkpoint="${WORKDIR}/${run_name}/checkpoint_${STEPS}.pth"

  if [[ -f "${checkpoint}" ]]; then
    echo "Skipping completed ${run_name}"
    return
  fi

  RUN_NAME="${run_name}" \
  STEPS="${STEPS}" \
  PREDICTOR_LR="${predictor_lr}" \
  COMPRESSOR_LR="${compressor_lr}" \
  WEIGHT_DECAY="${weight_decay}" \
  "${COMMON}"
}

# Factorial search:
# - equal vs 2x compressor LR
# - inherited 0.03 weight decay vs no weight decay
run_config "peq1e-5-ceq1e-5-wd3e-2" 1e-5 1e-5 0.03
run_config "p1e-5-c2e-5-wd3e-2" 1e-5 2e-5 0.03
run_config "peq1e-5-ceq1e-5-wd0" 1e-5 1e-5 0.0
run_config "p1e-5-c2e-5-wd0" 1e-5 2e-5 0.0

RESULT_DIR="${BENCHMARK_DIR}/tests/rec_sys/results/recsys_curriculum_search_stage1_seed101"
SOURCE="${WORKDIR}/tabimpute-comp-rank10-recsys-exposed-fitted_bias-broadrect100to700x400to1700-100chunks-random-k20-50k-from-broadbest/best_checkpoint.pth"

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
  --model-spec "source=${SOURCE}" \
  --model-spec "equal_wd003=${WORKDIR}/${PREFIX}-peq1e-5-ceq1e-5-wd3e-2/checkpoint_${STEPS}.pth" \
  --model-spec "compressor2x_wd003=${WORKDIR}/${PREFIX}-p1e-5-c2e-5-wd3e-2/checkpoint_${STEPS}.pth" \
  --model-spec "equal_wd0=${WORKDIR}/${PREFIX}-peq1e-5-ceq1e-5-wd0/checkpoint_${STEPS}.pth" \
  --model-spec "compressor2x_wd0=${WORKDIR}/${PREFIX}-p1e-5-c2e-5-wd0/checkpoint_${STEPS}.pth" \
  --output-dir "${RESULT_DIR}"

echo "Stage-1 summary: ${RESULT_DIR}/synthetic_movielens_proxy_summary.md"
