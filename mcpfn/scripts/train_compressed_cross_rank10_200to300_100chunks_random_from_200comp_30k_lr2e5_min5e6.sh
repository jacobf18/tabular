#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-200to300-100x100chunks-random-k20-30k-from-200comp-lr2e5-min5e6}"
export PREDICTOR_LR="${PREDICTOR_LR:-2e-5}"
export COMPRESSOR_LR="${COMPRESSOR_LR:-2e-5}"
export MIN_LR_RATIO="${MIN_LR_RATIO:-0.25}"

exec "${SCRIPT_DIR}/train_compressed_cross_rank10_200to300_100chunks_random_from_200comp_30k.sh"
