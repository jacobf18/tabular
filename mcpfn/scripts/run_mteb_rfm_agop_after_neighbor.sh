#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
NEIGHBOR_RUN="${NEIGHBOR_RUN:-tabimpute-direct-rowcol-svthybrid768-featuredecoder-neighbor-generalmix-30k-from-rowcolattn}"
CHECKPOINT="${CHECKPOINT:-${WORK_ROOT}/${NEIGHBOR_RUN}/best_checkpoint.pth}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_feature_neighbor}"
LABEL="${LABEL:-rowcol_svt_neighbor_rfm_agop512_i2}"

mkdir -p "${QUEUE_DIR}"

if [[ "${WAIT_FOR_TMUX}" != "none" ]] && tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
  echo "Waiting for active session ${WAIT_FOR_TMUX} to finish."
  while tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; do
    sleep 300
  done
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${BENCHSELECT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
BENCHSELECT_TABIMPUTE_MODEL=rfm_agop_direct \
BENCHSELECT_TABIMPUTE_LABEL="${LABEL}" \
BENCHSELECT_TABIMPUTE_CHECKPOINT="${CHECKPOINT}" \
BENCHSELECT_TABIMPUTE_DEVICE=cuda \
BENCHSELECT_RFM_NUM_ANCHORS="${BENCHSELECT_RFM_NUM_ANCHORS:-512}" \
BENCHSELECT_RFM_AGOP_ITERS="${BENCHSELECT_RFM_AGOP_ITERS:-2}" \
BENCHSELECT_RFM_LAMBDA_INIT="${BENCHSELECT_RFM_LAMBDA_INIT:-0.01}" \
BENCHSELECT_RFM_GAMMA_SCALE_INIT="${BENCHSELECT_RFM_GAMMA_SCALE_INIT:-1.0}" \
BENCHSELECT_DATASETS=mteb \
BENCHSELECT_METHODS=entropy,mi,random \
BENCHSELECT_N_FOLDS=10 \
BENCHSELECT_K_MAX=15 \
BENCHSELECT_MAX_ROWS=4000 \
BENCHSELECT_TEST_BATCH=200 \
  python code/eval_tabimpute.py \
  2>&1 | tee "${QUEUE_DIR}/mteb_rfm_agop_after_neighbor.log"

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
paths = {
    "Gaussian": logs / "entropy_vs_mi_mteb.npz",
    "DWIP v2-2": logs / "tabimpute_dwip-v2-2_mteb.npz",
    "Feature decoder": logs / "tabimpute_rowcol_svt_featuredecoder_generalmix_mteb.npz",
    "Neighbor decoder": logs / "tabimpute_rowcol_svt_featuredecoder_neighbor_generalmix_mteb.npz",
    "Neighbor RFM-AGOP": logs / "tabimpute_rowcol_svt_neighbor_rfm_agop512_i2_mteb.npz",
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
        vals = [
            f"{float(np.nanmean(arr[:, k - 1])):.4f}"
            if arr.ndim == 2 and arr.shape[1] >= k
            else ""
            for k in (5, 10, 15)
        ]
        print(f"{name}\t{method}\t" + "\t".join(vals))
PY
