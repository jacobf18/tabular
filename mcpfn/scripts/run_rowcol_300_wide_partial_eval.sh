#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-shallow1536x6-cdf2-huber50-lr5e5-cos200k-scratch"
OUTPUT_PREFIX="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-shallow1536x6-partial_eval"

cd "${REPO_DIR}"

python scripts/eval_train_level_chunks_interface.py \
  --sizes 300 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --generator gaussian_factors \
  --scale rank \
  --skip-softimpute \
  --direct-names \
  --direct-prediction-stat median \
  --extra-direct "RowCol target300 shallow1536x6 CDF2 Huber50 phase025k|${WORK_DIR}/checkpoint_phase1_25000.pth|300|rowcol_latent_attention" \
  --extra-direct "RowCol target300 shallow1536x6 CDF2 Huber50 phase050k|${WORK_DIR}/checkpoint_phase2_50000.pth|300|rowcol_latent_attention" \
  --output-prefix "${OUTPUT_PREFIX}"
