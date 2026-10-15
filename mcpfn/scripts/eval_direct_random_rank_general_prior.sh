#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python "${REPO_DIR}/scripts/eval_direct_random_rank_general_prior.py" \
  --device "${DEVICE:-cuda}" \
  --latent-rank-low "${LATENT_RANK_LOW:-1}" \
  --latent-rank-high "${LATENT_RANK_HIGH:-40}" \
  --p-missing "${P_MISSING:-0.4}" \
  --missingness-type "${MISSINGNESS_TYPE:-mcar}" \
  --batches-per-spec "${BATCHES_PER_SPEC:-8}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --include-softimpute \
  --include-extended \
  --tag "${TAG:-random_rank_general_prior}" \
  --output-dir "${OUTPUT_DIR:-${REPO_DIR}/runs/general_prior_random_rank_eval}"
