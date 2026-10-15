#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

FIG_DIR="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds"
QUEUE_DIR="${REPO_DIR}/runs/objective_queues"
mkdir -p "${FIG_DIR}" "${QUEUE_DIR}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-target300-cdf2-huber50-lr2e5-cos200k-from-nlldecay200k/checkpoint_phase1_25000.pth}"
SEED="${SEED:-52}"

PHASES_300_10K='[
  {"name":"target300_005k","end_step":5000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1},
  {"name":"target300_010k","end_step":10000,"rows_low":300,"rows_high":300,"cols_low":300,"cols_high":300,"batch_size":1}
]'

run_variant() {
  local tag="$1"
  local label="$2"
  local lr="$3"
  local overrides="$4"
  local run_name="tabimpute-direct-rowcol-alshybrid768-rank10-p02-target300-sweep-${tag}"
  local work_dir="${REPO_DIR}/src/tabimpute/workdir/${run_name}"
  mkdir -p "${work_dir}"

  if [[ ! -f "${work_dir}/checkpoint_10000.pth" ]]; then
    python "${REPO_DIR}/scripts/train_direct_rowcol_als_hybrid_random_rank_curriculum.py" \
      --run-name "${run_name}" \
      --work-dir "${work_dir}" \
      --device "${DEVICE:-cuda}" \
      --seed "${SEED}" \
      --prior-kind latent_factor \
      --missingness-type mcar \
      --p-missing 0.2 \
      --latent-rank-low 10 \
      --latent-rank-high 10 \
      --weight-decay "${WEIGHT_DECAY:-0.02}" \
      --grad-clip-norm "${GRAD_CLIP_NORM:-2.0}" \
      --warmup-steps "${WARMUP_STEPS:-500}" \
      --stable-steps "${STABLE_STEPS:-0}" \
      --decay-steps "${DECAY_STEPS:-9500}" \
      --min-lr-ratio "${MIN_LR_RATIO:-0.10}" \
      --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
      --distribution-loss-chunk-size "${DISTRIBUTION_LOSS_CHUNK_SIZE:-2048}" \
      --wandb-project "${WANDB_PROJECT:-tabimpute}" \
      --lr "${lr}" \
      --phases-json "${PHASES_300_10K}" \
      --init-model-checkpoint "${INIT_CHECKPOINT}" \
      --allow-partial-init \
      --model-overrides-json "${overrides}" \
      --cdf-loss-weight 0.5 \
      --point-loss-weight 30.0 \
      --point-loss-kind huber \
      --point-loss-prediction median \
      --huber-delta 0.02 \
      2>&1 | tee "${QUEUE_DIR}/rowcol_300_als_sweep_${tag}_train.log"
  fi

  python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
    --sizes 300 \
    --seeds 0 1 2 3 4 \
    --rank 10 \
    --missing-rate 0.2 \
    --generator gaussian_factors \
    --scale rank \
    --skip-softimpute \
    --direct-names \
    --direct-prediction-stat median \
    --extra-direct "${label} phase005k|${work_dir}/checkpoint_phase1_5000.pth|300|rowcol_als_hybrid" \
    --extra-direct "${label} phase010k|${work_dir}/checkpoint_phase2_10000.pth|300|rowcol_als_hybrid" \
    --extra-direct "${label} final|${work_dir}/checkpoint_10000.pth|300|rowcol_als_hybrid" \
    --extra-direct "${label} trainloss best|${work_dir}/best_checkpoint.pth|300|rowcol_als_hybrid" \
    --output-prefix "${FIG_DIR}/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_target300-alshybrid-stability-sweep-${tag}_eval" \
    2>&1 | tee "${QUEUE_DIR}/rowcol_300_als_sweep_${tag}_eval.log"
}

BASE_SIGMA='"logit_sigma_init":0.08,"logit_sigma_min":0.008,"use_neural_residual":false'

run_variant \
  "exactsolve-ridge1e3-damp085-lr5e5" \
  "RowCol ALS sweep exact-solve ridge1e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"solve\",\"als_jitter\":0.0,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "choljit-ridge1e3-damp085-lr5e5" \
  "RowCol ALS sweep cholesky-jitter ridge1e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "choljit-ridge1e3-damp05-lr5e5" \
  "RowCol ALS sweep cholesky-jitter ridge1e-3 damp0.50 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.50,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "choljit-ridge1e3-damp07-lr5e5" \
  "RowCol ALS sweep cholesky-jitter ridge1e-3 damp0.70 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.70,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "choljit-ridge3e3-damp085-lr5e5" \
  "RowCol ALS sweep cholesky-jitter ridge3e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.003,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "choljit-ridge1e2-damp085-lr5e5" \
  "RowCol ALS sweep cholesky-jitter ridge1e-2 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.01,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "scaledridge-ridge1e3-damp085-lr5e5" \
  "RowCol ALS sweep scaled-ridge ridge1e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":true,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "clip5-ridge1e3-damp085-lr5e5" \
  "RowCol ALS sweep clip5 ridge1e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_factor_norm_clip\":5.0,\"als_init_scale\":0.035,${BASE_SIGMA}}"

run_variant \
  "broadsigma-ridge1e3-damp085-lr5e5" \
  "RowCol ALS sweep broad-sigma ridge1e-3 damp0.85 lr5e-5" \
  "5e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,\"logit_sigma_init\":0.12,\"logit_sigma_min\":0.02,\"use_neural_residual\":false}"

run_variant \
  "lowlr-choljit-ridge1e3-damp085-lr1e5" \
  "RowCol ALS sweep cholesky-jitter ridge1e-3 damp0.85 lr1e-5" \
  "1e-5" \
  "{\"als_rank\":10,\"als_steps\":4,\"als_ridge\":0.001,\"als_update_damping\":0.85,\"als_ridge_scale_by_obs\":false,\"als_solve_method\":\"cholesky\",\"als_jitter\":0.000001,\"als_init_scale\":0.035,${BASE_SIGMA}}"

python - <<'PY'
from pathlib import Path
import re
import numpy as np

queue = Path("runs/objective_queues")
rows = []
for train_log in sorted(queue.glob("rowcol_300_als_sweep_*_train.log")):
    tag = train_log.name.removeprefix("rowcol_300_als_sweep_").removesuffix("_train.log")
    text = train_log.read_text(errors="ignore").replace("\r", "\n")
    pat = re.compile(r"(\d+)/10000.*?ema=([^,\]\s]+), loss=([^,\]\s]+), phase=(\d+)")
    by_step = {}
    for m in pat.finditer(text):
        try:
            by_step[int(m.group(1))] = (float(m.group(3)), float(m.group(2)))
        except ValueError:
            pass
    if not by_step:
        continue
    losses = np.array([by_step[s][0] for s in sorted(by_step)])
    eval_log = queue / f"rowcol_300_als_sweep_{tag}_eval.log"
    rmse = ""
    if eval_log.exists():
        eval_text = eval_log.read_text(errors="ignore")
        matches = re.findall(r"final\t300\t([0-9.]+)", eval_text)
        if matches:
            rmse = matches[-1]
    rows.append(
        (
            tag,
            len(losses),
            float(np.median(losses)),
            float(np.percentile(losses, 95)),
            float(np.percentile(losses, 99)),
            float(np.max(losses)),
            float(np.mean(losses > 0)),
            float(np.mean(losses > 20)),
            rmse,
        )
    )
print("tag\tn\tmedian\tp95\tp99\tmax\tfrac_gt0\tfrac_gt20\tfinal_rmse")
for row in rows:
    print("\t".join(str(x) for x in row))
PY
