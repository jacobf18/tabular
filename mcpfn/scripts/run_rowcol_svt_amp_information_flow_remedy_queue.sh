#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-widehead4096-generalmix-30k-fromobservedattn/best_checkpoint.pth}"
STEPS="${STEPS:-12000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
METRICS_JSONL_EVERY="${METRICS_JSONL_EVERY:-25}"

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

PHASE1_END="$((STEPS / 3))"
WARMUP_STEPS="${WARMUP_STEPS:-600}"
DECAY_STEPS="${DECAY_STEPS:-$((STEPS - WARMUP_STEPS))}"

PHASES_JSON="$(printf '[
  {"name":"remedy_rect_warmup","end_step":%s,"rows_low":40,"rows_high":160,"cols_low":20,"cols_high":80,"batch_size":3},
  {"name":"remedy_mteb_scale","end_step":%s,"rows_low":120,"rows_high":320,"cols_low":35,"cols_high":100,"batch_size":2}
]' "${PHASE1_END}" "${STEPS}")"

PRIOR_OVERRIDES_JSON='{
  "generator_profile_name": "priormix_v1",
  "profile_p_missing_low": 0.10,
  "profile_p_missing_width": 0.35,
  "profile_p_missing_anchor_prob": 0.40,
  "profile_p_missing_anchor_value": 0.23,
  "latent_rank_low": 1,
  "latent_rank_high": 48,
  "gp_length_scale_low": 0.15,
  "gp_length_scale_high": 3.0,
  "profile_gp_grid_size": 256
}'

make_model_json() {
  local variant="$1"
  python - "${variant}" <<'PY'
import json
import sys

variant = sys.argv[1]
cfg = {
    "svt_steps": 0,
    "svt_init_rank": 80,
    "svt_max_rank": 100,
    "svt_init_scale": 0.02,
    "svt_threshold_fraction_init": 0.006,
    "svt_threshold_fraction_min": 0.0001,
    "svt_threshold_fraction_max": 0.02,
    "svt_threshold_mode": "inverse_size",
    "svt_threshold_size_scale_init": 1.2,
    "use_svt_feature_decoder": True,
    "svt_feature_decoder_scale_init": 0.1,
    "svt_feature_decoder_hidden_size": 768,
    "use_observed_neighbor_features": True,
    "observed_neighbor_temperature": 0.20,
    "observed_neighbor_topk": 64,
    "use_observed_attention_decoder": True,
    "observed_attention_topk": 64,
    "observed_attention_dim": 128,
    "observed_attention_temperature_init": 1.0,
    "observed_attention_scale_init": 0.05,
    "observed_attention_hidden_size": 768,
    "extra_decoder_hidden_size": 4096,
    "extra_decoder_scale_init": 0.02,
    "svt_mixture_steps": [0, 16, 32, 64],
    "svt_mixture_hidden_size": 768,
    "svt_mixture_temperature": 1.0,
    "svt_mixture_initial_logit": 0.0,
    "svt_mixture_other_logit": 0.0,
    "svt_mixture_detach_refinement": True,
    "svt_mixture_spectral_feature_count": 24,
    "svt_mixture_missingness_feature_count": 28,
    "svt_mixture_policy": "controller_plus_cv",
    "svt_mixture_cv_fraction": 0.10,
    "svt_mixture_cv_temperature": 0.01,
    "svt_mixture_cv_min_count": 32,
    "svt_mixture_include_cv_features": True,
}
if variant in {"memory_point", "memory_softmix"}:
    cfg.update(
        {
            "latent_memory_residual_start_layer": 6,
            "latent_memory_residual_scale_init": 0.05,
            "latent_memory_residual_normalize": True,
        }
    )
if variant == "memory_softmix":
    cfg.update(
        {
            "svt_mixture_temperature": 2.0,
            "svt_mixture_cv_temperature": 0.05,
        }
    )
print(json.dumps(cfg))
PY
}

run_eval() {
  local label="$1"
  local checkpoint="$2"
  local eval_log="$3"

  cd "${BENCHSELECT_DIR}"
  export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"
  BENCHSELECT_TABIMPUTE_MODEL=direct_checkpoint \
  BENCHSELECT_TABIMPUTE_LABEL="${label}" \
  BENCHSELECT_TABIMPUTE_CHECKPOINT="${checkpoint}" \
  BENCHSELECT_TABIMPUTE_DEVICE=cuda \
  BENCHSELECT_TABIMPUTE_PREDICTION_STAT=mean \
  BENCHSELECT_DATASETS=mteb \
  BENCHSELECT_METHODS=entropy,mi,random \
  BENCHSELECT_N_FOLDS=10 \
  BENCHSELECT_K_MAX=15 \
  BENCHSELECT_MAX_ROWS=4000 \
  BENCHSELECT_TEST_BATCH=200 \
    python code/eval_tabimpute.py \
    2>&1 | tee "${eval_log}"
  cd "${REPO_DIR}"
}

run_diagnostic() {
  local label="$1"
  local checkpoint="$2"
  local diag_log="$3"

  python scripts/diagnose_dwip_vs_rowcol_mteb_information_flow.py \
    --spec "dwip-v2-2|/root/tabimpute_anchor3_named/dwip_v2_t003_checkpoint_17000.pth|new_stable|${REPO_DIR}/src/tabimpute/workdir/dwip_v3r1_t003_ctl-trial_001/config.json" \
    --spec "${label}|${checkpoint}|rowcol_svt_hybrid" \
    --dataset mteb \
    --fold 0 \
    --k-selected 10 \
    --max-targets 1536 \
    --output-dir "${REPO_DIR}/runs/information_flow_diagnostics" \
    2>&1 | tee "${diag_log}"
}

run_experiment() {
  local tag="$1"
  local variant="$2"
  local lr="$3"
  local nll_weight="$4"
  local point_weight="$5"
  local cdf_weight="$6"
  local svt_entropy_bonus="$7"
  local parameter_overrides="$8"
  local seed="$9"

  local run_name="tabimpute-direct-rowcol-svthybrid768-${tag}-${STEPS}steps"
  local work_dir="${WORK_ROOT}/${run_name}"
  local final_checkpoint="${work_dir}/checkpoint_${STEPS}.pth"
  local best_checkpoint="${work_dir}/best_checkpoint.pth"
  local train_log="${QUEUE_DIR}/${tag}_train.log"
  local eval_log="${QUEUE_DIR}/${tag}_mteb_eval.log"
  local diag_log="${QUEUE_DIR}/${tag}_information_flow_diag.log"
  local model_overrides_json
  model_overrides_json="$(make_model_json "${variant}")"

  if [[ ! -f "${final_checkpoint}" ]]; then
    python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
      --run-name "${run_name}" \
      --prior-kind latent_factor \
      --generator-type generator_mix \
      --missingness-type profile_mixed \
      --prior-overrides-json "${PRIOR_OVERRIDES_JSON}" \
      --phases-json "${PHASES_JSON}" \
      --init-model-checkpoint "${INIT_CKPT}" \
      --allow-partial-init \
      --seed "${seed}" \
      --precision-mode amp_bfloat16 \
      --lr "${lr}" \
      --weight-decay 0.02 \
      --grad-clip-norm 0.5 \
      --warmup-steps "${WARMUP_STEPS}" \
      --stable-steps 0 \
      --decay-steps "${DECAY_STEPS}" \
      --min-lr-ratio 0.05 \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --metrics-jsonl-every "${METRICS_JSONL_EVERY}" \
      --nll-loss-weight "${nll_weight}" \
      --point-loss-weight "${point_weight}" \
      --point-loss-kind huber \
      --point-loss-prediction mean \
      --huber-delta 0.05 \
      --cdf-loss-weight "${cdf_weight}" \
      --distribution-loss-chunk-size 2048 \
      --svt-mixture-depth-distill-weight 0.05 \
      --svt-mixture-depth-distill-temperature 0.08 \
      --svt-mixture-cv-regret-weight 0.05 \
      --svt-mixture-entropy-bonus-weight "${svt_entropy_bonus}" \
      --parameter-value-overrides-json "${parameter_overrides}" \
      --model-overrides-json "${model_overrides_json}" \
      2>&1 | tee "${train_log}"
  else
    echo "Skipping training; found ${final_checkpoint}" | tee "${train_log}"
  fi

  local checkpoint="${best_checkpoint}"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${final_checkpoint}"
  fi
  run_eval "${tag}" "${checkpoint}" "${eval_log}"
  run_diagnostic "${tag}" "${checkpoint}" "${diag_log}"
}

run_experiment \
  "amp_precision_only" \
  "baseline" \
  "6e-6" \
  "1.0" \
  "0.2" \
  "0.05" \
  "0.0" \
  "{}" \
  "1201"

run_experiment \
  "amp_memory_obsscale_point" \
  "memory_point" \
  "6e-6" \
  "0.5" \
  "1.0" \
  "0.05" \
  "0.0" \
  "{\"observed_attention_scale\":0.18}" \
  "1202"

run_experiment \
  "amp_memory_softmix_point" \
  "memory_softmix" \
  "6e-6" \
  "0.5" \
  "1.0" \
  "0.05" \
  "0.02" \
  "{\"observed_attention_scale\":0.18}" \
  "1203"

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
paths = {
    "Gaussian": logs / "entropy_vs_mi_mteb.npz",
    "DWIP v2-2": logs / "tabimpute_dwip-v2-2_mteb.npz",
    "SVT observed": logs / "tabimpute_rowcol_svt_observedattn_generalmix_mteb.npz",
    "SVT wide": logs / "tabimpute_rowcol_svt_widehead4096_generalmix_mteb.npz",
    "AMP precision": logs / "tabimpute_amp_precision_only_mteb.npz",
    "AMP memory point": logs / "tabimpute_amp_memory_obsscale_point_mteb.npz",
    "AMP memory softmix": logs / "tabimpute_amp_memory_softmix_point_mteb.npz",
}
print("backend\tmethod\tr2@5\tr2@10\tr2@15")
for name, path in paths.items():
    if not path.exists():
        continue
    data = np.load(path, allow_pickle=True)
    for method in ("entropy", "mi", "random"):
        key = f"{method}_r2"
        if key not in data.files:
            continue
        arr = data[key]
        values = []
        for k in (5, 10, 15):
            if arr.ndim == 2 and arr.shape[1] >= k:
                values.append(f"{float(np.nanmean(arr[:, k - 1])):.4f}")
            else:
                values.append("")
        print(f"{name}\t{method}\t" + "\t".join(values))
PY
