#!/usr/bin/env bash
set -euo pipefail

# Main continuation run for the rec-sys spectrum+exposure prior.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-tabimpute-comp-rank10-recsys150to300-100x100chunks-random-k20-100k-from-200comp-lr2e5-min5e6}"
export STEPS="${STEPS:-100000}"
export WARMUP_STEPS="${WARMUP_STEPS:-5000}"
export STABLE_STEPS="${STABLE_STEPS:-70000}"
export DECAY_STEPS="${DECAY_STEPS:-25000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

exec "${SCRIPT_DIR}/train_compressed_cross_rank10_recsys_150to300_100chunks_from_200comp_10k.sh"
