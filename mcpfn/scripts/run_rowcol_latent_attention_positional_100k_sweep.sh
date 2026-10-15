#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
DIAG_DIR="${REPO_DIR}/runs/diagnostics"
mkdir -p "${FIG_DIR}" "${DIAG_DIR}"

PHASES_JSON='[{"name":"50_square","end_step":25000,"rows_low":50,"rows_high":50,"cols_low":50,"cols_high":50,"batch_size":16},{"name":"50_to_100","end_step":50000,"rows_low":50,"rows_high":100,"cols_low":50,"cols_high":100,"batch_size":8},{"name":"100_to_200","end_step":75000,"rows_low":100,"rows_high":200,"cols_low":100,"cols_high":200,"batch_size":4},{"name":"200_to_300","end_step":100000,"rows_low":200,"rows_high":300,"cols_low":200,"cols_high":300,"batch_size":2}]'

run_variant() {
  local slug="$1"
  local label="$2"
  local overrides_json="$3"
  local run_name="tabimpute-direct-rowcol-latentattn768-pos-${slug}-rank10-p02-curriculum-100k-scratch"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"
  local final_ckpt="${work_dir}/checkpoint_100000.pth"
  local best_ckpt="${work_dir}/best_checkpoint.pth"

  mkdir -p "${work_dir}"
  if [[ ! -f "${final_ckpt}" ]]; then
    python "${REPO_DIR}/scripts/train_direct_rowcol_latent_attention_random_rank_curriculum.py" \
      --run-name "${run_name}" \
      --work-dir "${work_dir}" \
      --device "${DEVICE:-cuda}" \
      --seed "${SEED:-48}" \
      --prior-kind latent_factor \
      --missingness-type "${MISSINGNESS_TYPE:-mcar}" \
      --p-missing "${P_MISSING:-0.2}" \
      --latent-rank-low "${LATENT_RANK_LOW:-10}" \
      --latent-rank-high "${LATENT_RANK_HIGH:-10}" \
      --lr "${LR:-1.5e-4}" \
      --weight-decay "${WEIGHT_DECAY:-0.03}" \
      --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
      --warmup-steps "${WARMUP_STEPS:-5000}" \
      --stable-steps "${STABLE_STEPS:-75000}" \
      --decay-steps "${DECAY_STEPS:-20000}" \
      --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
      --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
      --phases-json "${PHASES_JSON}" \
      --model-overrides-json "${overrides_json}" \
      --wandb-project "${WANDB_PROJECT:-tabimpute}"
  fi

  python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
    --sizes 20 50 75 100 150 200 300 400 500 \
    --seeds 0 1 2 3 4 \
    --rank 10 \
    --missing-rate 0.2 \
    --generator gaussian_factors \
    --scale rank \
    --skip-softimpute \
    --direct-names \
    --extra-direct "${label} final|${final_ckpt}|500|rowcol_latent_attention" \
    --extra-direct "${label} best|${best_ckpt}|500|rowcol_latent_attention" \
    --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_pos_${slug}_eval"

  python "${REPO_DIR}/scripts/diagnose_rowcol_information_flow.py" \
    --spec "${label} final|${final_ckpt}|rowcol_latent_attention" \
    --sizes 50 300 500 \
    --seeds 0 1 2 \
    --flow-seeds 0 \
    --rank 10 \
    --missing-rate 0.2 \
    --scale rank \
    --output-prefix "${DIAG_DIR}/rowcol_information_flow_pos_${slug}_p02_rank10"
}

run_variant \
  "idxonly" \
  "RowCol pos index-only 100k" \
  '{"use_axis_identity_features":false,"use_absolute_positional_embeddings":true,"positional_encoding_kind":"sinusoidal_index","positional_damping_factor":0.05}'

run_variant \
  "normsin" \
  "RowCol pos normalized-sin 100k" \
  '{"use_axis_identity_features":false,"use_absolute_positional_embeddings":true,"positional_encoding_kind":"sinusoidal_normalized","positional_damping_factor":0.05}'

run_variant \
  "fourier" \
  "RowCol pos Fourier-MLP 100k" \
  '{"use_axis_identity_features":false,"use_absolute_positional_embeddings":true,"positional_encoding_kind":"fourier_mlp","positional_num_frequencies":32,"positional_damping_factor":0.10}'

run_variant \
  "table" \
  "RowCol pos learned-table 100k" \
  '{"use_axis_identity_features":false,"use_absolute_positional_embeddings":true,"positional_encoding_kind":"learned_table","positional_max_size":1024,"positional_damping_factor":0.05}'
