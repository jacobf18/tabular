#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-latentattn768-rank10-p02-cdf2-balanced120k-lr2e5-fromlarge80k/best_checkpoint.pth}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-featuredecoder-generalmix-30k-from-rowcolattn}"
STEPS="${STEPS:-30000}"
LR="${LR:-2e-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
METRICS_JSONL_EVERY="${METRICS_JSONL_EVERY:-25}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-none}"

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${WAIT_FOR_TMUX}" != "none" ]] && tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
  echo "Waiting for active session ${WAIT_FOR_TMUX} to finish."
  while tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; do
    sleep 300
  done
fi

PHASES_JSON="$(printf '[
  {"name":"general_mix_rect_warmup","end_step":%s,"rows_low":40,"rows_high":160,"cols_low":20,"cols_high":80,"batch_size":8},
  {"name":"general_mix_mteb_scale","end_step":%s,"rows_low":120,"rows_high":320,"cols_low":35,"cols_high":100,"batch_size":4}
]' "$((STEPS / 3))" "${STEPS}")"

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
  "latent_rank_high": 32,
  "gp_length_scale_low": 0.15,
  "gp_length_scale_high": 3.0,
  "profile_gp_grid_size": 256
}'

WORK_DIR="${WORK_ROOT}/${RUN_NAME}"
CHECKPOINT="${WORK_DIR}/best_checkpoint.pth"
TRAIN_LOG="${QUEUE_DIR}/rowcol_svt_feature_decoder_general_train.log"
EVAL_LOG="${QUEUE_DIR}/rowcol_svt_feature_decoder_general_mteb_eval.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
    --run-name "${RUN_NAME}" \
    --prior-kind latent_factor \
    --generator-type generator_mix \
    --missingness-type profile_mixed \
    --prior-overrides-json "${PRIOR_OVERRIDES_JSON}" \
    --phases-json "${PHASES_JSON}" \
    --init-model-checkpoint "${INIT_CKPT}" \
    --allow-partial-init \
    --seed 809 \
    --fp32-param-substrings "svt_feature_decoder_scale,logit_sigma_unconstrained,svt_threshold_unconstrained,svt_threshold_size_scale_unconstrained" \
    --lr "${LR}" \
    --weight-decay 0.02 \
    --grad-clip-norm 0.5 \
    --warmup-steps 1800 \
    --stable-steps 0 \
    --decay-steps $((STEPS - 1800)) \
    --min-lr-ratio 0.05 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --metrics-jsonl-every "${METRICS_JSONL_EVERY}" \
    --nll-loss-weight 1.0 \
    --point-loss-weight 0.1 \
    --point-loss-kind huber \
    --point-loss-prediction mean \
    --huber-delta 0.05 \
    --cdf-loss-weight 0.05 \
    --distribution-loss-chunk-size 2048 \
    --svt-mixture-depth-distill-weight 0.10 \
    --svt-mixture-depth-distill-temperature 0.05 \
    --svt-mixture-cv-regret-weight 0.10 \
    --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
    2>&1 | tee "${TRAIN_LOG}"
else
  echo "Skipping training; found ${CHECKPOINT}"
fi

cd "${BENCHSELECT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"
BENCHSELECT_TABIMPUTE_MODEL=direct_checkpoint \
BENCHSELECT_TABIMPUTE_LABEL=rowcol_svt_featuredecoder_generalmix \
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
    "SVT residual mixed": logs / "tabimpute_rowcol_svt_dwipv2prior_residual_mixed20k_mteb.npz",
    "SVT feature decoder": logs / "tabimpute_rowcol_svt_featuredecoder_generalmix_mteb.npz",
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
