#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
INIT_CKPT="${INIT_CKPT:-${WORK_ROOT}/tabimpute-direct-rowcol-svthybrid768-learnedcv-svtmixer-cycle_round19_cvfeatures_mnar_ctrlpluscv_pred_t01_lr5e5-10k/best_checkpoint.pth}"
LR="${LR:-2e-5}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-dwipv2prior-full-20k-lr2e5-fromround19}"
STEPS="${STEPS:-20000}"
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

PHASES_JSON="$(printf '[{"name":"dwip_v2_prior_10x5_to_50x50","end_step":%s,"rows_low":10,"rows_high":50,"cols_low":5,"cols_high":50,"batch_size":16}]' "${STEPS}")"

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

WORK_DIR="${WORK_ROOT}/${RUN_NAME}"
CHECKPOINT="${WORK_DIR}/best_checkpoint.pth"

if [[ ! -f "${CHECKPOINT}" ]]; then
  python scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py \
    --run-name "${RUN_NAME}" \
    --prior-kind latent_factor \
    --missingness-type mcar \
    --p-missing 0.4 \
    --latent-rank-low 1 \
    --latent-rank-high 11 \
    --phases-json "${PHASES_JSON}" \
    --init-model-checkpoint "${INIT_CKPT}" \
    --allow-partial-init \
    --seed 701 \
    --lr "${LR}" \
    --weight-decay 0.02 \
    --grad-clip-norm 0.5 \
    --warmup-steps 1200 \
    --stable-steps 0 \
    --decay-steps $((STEPS - 1200)) \
    --min-lr-ratio 0.05 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --metrics-jsonl-every "${METRICS_JSONL_EVERY}" \
    --nll-loss-weight 1.0 \
    --point-loss-weight 0.0 \
    --cdf-loss-weight 0.0 \
    --distribution-loss-chunk-size 2048 \
    --svt-mixture-depth-distill-weight 0.25 \
    --svt-mixture-depth-distill-temperature 0.05 \
    --svt-mixture-cv-regret-weight 0.25 \
    --model-overrides-json "${MODEL_OVERRIDES_JSON}" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_svt_dwip_prior_transfer_train.log"
else
  echo "Skipping training; found ${CHECKPOINT}"
fi

cd "${BENCHSELECT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"
BENCHSELECT_TABIMPUTE_MODEL=direct_checkpoint \
BENCHSELECT_TABIMPUTE_LABEL=rowcol_svt_dwipv2prior_full20k \
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
  2>&1 | tee "${QUEUE_DIR}/rowcol_svt_dwip_prior_transfer_mteb_eval.log"

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
paths = {
    "Gaussian": logs / "entropy_vs_mi_mteb.npz",
    "DWIP v2-2": logs / "tabimpute_dwip-v2-2_mteb.npz",
    "Round19": logs / "tabimpute_rowcol_svt_round19_smoke_mteb.npz",
    "SVT DWIP prior": logs / "tabimpute_rowcol_svt_dwipv2prior_full20k_mteb.npz",
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
