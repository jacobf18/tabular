#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODE="residual"

exec "${SCRIPT_DIR}/train_compressed_cross_rank10_recsys_bias_residual_50k_from_broadbest_common.sh"
