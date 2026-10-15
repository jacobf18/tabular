#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-observedattn-generalmix-30k-fromneighbor/checkpoint_30000.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-observedattn-benchstyle-30k-fromobservedattn}"
STEPS="${STEPS:-30000}"
LR="${LR:-6e-6}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
METRICS_JSONL_EVERY="${METRICS_JSONL_EVERY:-25}"

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

WARMUP_STEPS="${WARMUP_STEPS:-1200}"
DECAY_STEPS="${DECAY_STEPS:-$((STEPS - WARMUP_STEPS))}"
PHASE1_END="$((STEPS / 3))"

PHASES_JSON="$(printf '[
  {"name":"benchstyle_sparse_query_warmup","end_step":%s,"rows_low":80,"rows_high":180,"cols_low":35,"cols_high":80,"batch_size":5},
  {"name":"benchstyle_sparse_query_mteb_scale","end_step":%s,"rows_low":160,"rows_high":340,"cols_low":45,"cols_high":90,"batch_size":3}
]' "${PHASE1_END}" "${STEPS}")"

MODEL_OVERRIDES_JSON='{
  "svt_steps": 0,
  "svt_init_rank": 80,
  "svt_max_rank": 100,
  "svt_init_scale": 0.02,
  "svt_threshold_fraction_init": 0.006,
  "svt_threshold_fraction_min": 0.0001,
  "svt_threshold_fraction_max": 0.02,
  "svt_threshold_mode": "inverse_size",
  "svt_threshold_size_scale_init": 1.2,
  "use_svt_feature_decoder": true,
  "svt_feature_decoder_scale_init": 0.1,
  "svt_feature_decoder_hidden_size": 768,
  "use_observed_neighbor_features": true,
  "observed_neighbor_temperature": 0.20,
  "observed_neighbor_topk": 64,
  "use_observed_attention_decoder": true,
  "observed_attention_topk": 64,
  "observed_attention_dim": 128,
  "observed_attention_temperature_init": 1.0,
  "observed_attention_scale_init": 0.05,
  "observed_attention_hidden_size": 768,
  "svt_mixture_steps": [0, 16, 32, 64],
  "svt_mixture_hidden_size": 768,
  "svt_mixture_temperature": 1.0,
  "svt_mixture_initial_logit": 0.0,
  "svt_mixture_other_logit": 0.0,
  "svt_mixture_detach_refinement": true,
  "svt_mixture_spectral_feature_count": 24,
  "svt_mixture_missingness_feature_count": 28,
  "svt_mixture_policy": "controller_plus_cv",
  "svt_mixture_cv_fraction": 0.10,
  "svt_mixture_cv_temperature": 0.01,
  "svt_mixture_cv_min_count": 32,
  "svt_mixture_include_cv_features": true
}'

PRIOR_OVERRIDES_JSON='{
  "generator_profile_name": "priormix_v1",
  "profile_p_missing_low": 0.10,
  "profile_p_missing_width": 0.35,
  "profile_p_missing_anchor_prob": 0.40,
  "profile_p_missing_anchor_value": 0.23,
  "latent_rank_low": 1,
  "latent_rank_high": 40,
  "gp_length_scale_low": 0.15,
  "gp_length_scale_high": 3.0,
  "profile_gp_grid_size": 256
}'

WORK_DIR="${WORK_ROOT}/${RUN_NAME}"
CHECKPOINT="${WORK_DIR}/best_checkpoint.pth"
TRAIN_LOG="${QUEUE_DIR}/rowcol_svt_observed_attention_benchstyle_train.log"
EVAL_LOG="${QUEUE_DIR}/rowcol_svt_observed_attention_benchstyle_mteb_eval.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
    --run-name "${RUN_NAME}" \
    --prior-kind latent_factor \
    --generator-type generator_mix \
    --missingness-type profile_mixed \
    --prior-overrides-json "${PRIOR_OVERRIDES_JSON}" \
    --phases-json "${PHASES_JSON}" \
    --benchmark-selection-curriculum \
    --benchmark-query-fraction-low 0.08 \
    --benchmark-query-fraction-high 0.18 \
    --benchmark-context-observed-prob-low 0.70 \
    --benchmark-context-observed-prob-high 0.90 \
    --benchmark-query-k-low 1 \
    --benchmark-query-k-high 15 \
    --benchmark-query-selected-observed-prob-low 0.65 \
    --benchmark-query-selected-observed-prob-high 0.95 \
    --init-model-checkpoint "${INIT_CKPT}" \
    --allow-partial-init \
    --seed 853 \
    --fp32-param-substrings "svt_feature_decoder_scale,observed_attention_scale,observed_attention_log_temperature,logit_sigma_unconstrained,svt_threshold_unconstrained,svt_threshold_size_scale_unconstrained" \
    --lr "${LR}" \
    --weight-decay 0.02 \
    --grad-clip-norm 0.5 \
    --warmup-steps "${WARMUP_STEPS}" \
    --stable-steps 0 \
    --decay-steps "${DECAY_STEPS}" \
    --min-lr-ratio 0.05 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --metrics-jsonl-every "${METRICS_JSONL_EVERY}" \
    --nll-loss-weight 1.0 \
    --point-loss-weight 0.2 \
    --point-loss-kind huber \
    --point-loss-prediction mean \
    --huber-delta 0.05 \
    --cdf-loss-weight 0.05 \
    --distribution-loss-chunk-size 2048 \
    --svt-mixture-depth-distill-weight 0.05 \
    --svt-mixture-depth-distill-temperature 0.05 \
    --svt-mixture-cv-regret-weight 0.05 \
    --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
    2>&1 | tee "${TRAIN_LOG}"
else
  echo "Skipping training; found ${CHECKPOINT}"
fi

cd "${BENCHSELECT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"
BENCHSELECT_TABIMPUTE_MODEL=direct_checkpoint \
BENCHSELECT_TABIMPUTE_LABEL=rowcol_svt_observedattn_benchstyle \
BENCHSELECT_TABIMPUTE_CHECKPOINT="${CHECKPOINT}" \
BENCHSELECT_TABIMPUTE_DEVICE=cuda \
BENCHSELECT_TABIMPUTE_PREDICTION_STAT=mean \
BENCHSELECT_DATASETS=mteb \
BENCHSELECT_METHODS=entropy,mi,random \
BENCHSELECT_N_FOLDS=10 \
BENCHSELECT_K_MAX=15 \
BENCHSELECT_MAX_ROWS=4000 \
BENCHSELECT_TEST_BATCH=200 \
  python code/eval_tabimpute.py \
  2>&1 | tee "${EVAL_LOG}"

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
paths = {
    "Gaussian": logs / "entropy_vs_mi_mteb.npz",
    "DWIP v2-2": logs / "tabimpute_dwip-v2-2_mteb.npz",
    "SVT observed attention": logs / "tabimpute_rowcol_svt_observedattn_generalmix_mteb.npz",
    "SVT observed attention benchstyle": logs / "tabimpute_rowcol_svt_observedattn_benchstyle_mteb.npz",
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
