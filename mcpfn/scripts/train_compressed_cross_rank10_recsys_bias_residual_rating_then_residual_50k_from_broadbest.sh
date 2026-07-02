#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/train_compressed_cross_rank10_recsys_bias_residual_rating_50k_from_broadbest.sh"
"${SCRIPT_DIR}/train_compressed_cross_rank10_recsys_bias_residual_residual_50k_from_broadbest.sh"
