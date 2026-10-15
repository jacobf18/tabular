#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
NEIGHBOR_RUN="${NEIGHBOR_RUN:-tabimpute-direct-rowcol-svthybrid768-featuredecoder-neighbor-generalmix-30k-from-rowcolattn}"
CHECKPOINT="${CHECKPOINT:-${WORK_ROOT}/${NEIGHBOR_RUN}/best_checkpoint.pth}"

mkdir -p "${QUEUE_DIR}"
cd "${BENCHSELECT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

run_eval() {
  local label="$1"
  local anchors="$2"
  local iters="$3"
  local lambda_init="$4"
  local gamma_scale="$5"
  local log="${QUEUE_DIR}/mteb_rfm_agop_sweep_${label}.log"
  echo "=== ${label}: anchors=${anchors} iters=${iters} lambda=${lambda_init} gamma=${gamma_scale} ==="
  BENCHSELECT_TABIMPUTE_MODEL=rfm_agop_direct \
  BENCHSELECT_TABIMPUTE_LABEL="${label}" \
  BENCHSELECT_TABIMPUTE_CHECKPOINT="${CHECKPOINT}" \
  BENCHSELECT_TABIMPUTE_DEVICE=cuda \
  BENCHSELECT_RFM_NUM_ANCHORS="${anchors}" \
  BENCHSELECT_RFM_AGOP_ITERS="${iters}" \
  BENCHSELECT_RFM_LAMBDA_INIT="${lambda_init}" \
  BENCHSELECT_RFM_GAMMA_SCALE_INIT="${gamma_scale}" \
  BENCHSELECT_DATASETS=mteb \
  BENCHSELECT_METHODS=mi \
  BENCHSELECT_N_FOLDS=10 \
  BENCHSELECT_K_MAX=15 \
  BENCHSELECT_MAX_ROWS=4000 \
  BENCHSELECT_TEST_BATCH=200 \
    python code/eval_tabimpute.py \
    2>&1 | tee "${log}"
}

run_eval rowcol_svt_neighbor_rfm_agop512_i2_l1e-3_g1 512 2 0.001 1.0
run_eval rowcol_svt_neighbor_rfm_agop512_i2_l1e-2_g0p5 512 2 0.01 0.5
run_eval rowcol_svt_neighbor_rfm_agop512_i2_l1e-2_g2 512 2 0.01 2.0
run_eval rowcol_svt_neighbor_rfm_agop512_i3_l1e-2_g1 512 3 0.01 1.0
run_eval rowcol_svt_neighbor_rfm_agop1024_i2_l1e-2_g1 1024 2 0.01 1.0
run_eval rowcol_svt_neighbor_rfm_agop1024_i3_l1e-2_g1 1024 3 0.01 1.0
run_eval rowcol_svt_neighbor_rfm_agop1024_i2_l1e-3_g1 1024 2 0.001 1.0
run_eval rowcol_svt_neighbor_rfm_agop1024_i2_l1e-2_g0p5 1024 2 0.01 0.5
run_eval rowcol_svt_neighbor_rfm_agop1024_i2_l1e-2_g2 1024 2 0.01 2.0

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
rows = []
for path in logs.glob("tabimpute_rowcol_svt_neighbor_rfm_agop*_mteb.npz"):
    data = np.load(path, allow_pickle=True)
    if "mi_r2" not in data.files:
        continue
    arr = data["mi_r2"]
    vals = []
    for k in (5, 10, 15):
        vals.append(float(np.nanmean(arr[:, k - 1])) if arr.ndim == 2 and arr.shape[1] >= k else float("nan"))
    rows.append((vals[0], path.name, *vals))

rows.sort(reverse=True)
print("label\tmi_r2@5\tmi_r2@10\tmi_r2@15")
for _, name, r5, r10, r15 in rows:
    label = name.removeprefix("tabimpute_").removesuffix("_mteb.npz")
    print(f"{label}\t{r5:.4f}\t{r10:.4f}\t{r15:.4f}")
PY
