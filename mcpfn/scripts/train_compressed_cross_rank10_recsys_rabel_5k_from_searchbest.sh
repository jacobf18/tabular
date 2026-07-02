#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/root/matrix-completion-benchmarks}"
COMMON="${REPO_DIR}/scripts/train_compressed_cross_rank10_recsys_curriculum_search_common.sh"
WORKDIR="${REPO_DIR}/src/tabimpute/workdir"

RUN_NAME="${RUN_NAME:-tabimpute-comp-recsys-rabel64-5k-from-searchbest}"
STEPS="${STEPS:-5000}"
SOURCE="${SOURCE:-${WORKDIR}/tabimpute-comp-recsys-search-stage1-p1e-5-c2e-5-wd3e-2/checkpoint_5000.pth}"
CHECKPOINT="${WORKDIR}/${RUN_NAME}/checkpoint_${STEPS}.pth"

if [[ ! -f "${SOURCE}" ]]; then
  echo "Source checkpoint not found: ${SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  RUN_NAME="${RUN_NAME}" \
  STEPS="${STEPS}" \
  WARMUP_STEPS=500 \
  STABLE_STEPS=4000 \
  DECAY_STEPS=500 \
  PREDICTOR_LR=1e-5 \
  COMPRESSOR_LR=2e-5 \
  ENCODER_LR=1e-4 \
  WEIGHT_DECAY=0.03 \
  MODEL_FAMILY=rabel \
  RABEL_NUM_KERNELS=64 \
  RABEL_SIGMA=1.0 \
  RABEL_EXPONENT_DIGITS=1 \
  RABEL_EXPONENT_EMBEDDING_SIZE=32 \
  RABEL_INITIAL_SCALE=0.05 \
  COMPRESSED_CHECKPOINT="${SOURCE}" \
  "${COMMON}"
else
  echo "Skipping completed training: ${CHECKPOINT}"
fi

RESULT_DIR="${BENCHMARK_DIR}/tests/rec_sys/results/${RUN_NAME}_seed101"
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
  --model-spec "rabel64=${CHECKPOINT}" \
  --output-dir "${RESULT_DIR}"

echo "RaBEL validation: ${RESULT_DIR}/synthetic_movielens_proxy_summary.md"
