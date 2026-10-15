#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/src/tabimpute/workdir}"
QUEUE_DIR="${QUEUE_DIR:-${REPO_DIR}/runs/objective_queues}"
BENCHSELECT_DIR="${BENCHSELECT_DIR:-/root/tabular/benchmark-selection}"
RUN_NAME="${RUN_NAME:-tabimpute-direct-rowcol-svthybrid768-featuredecoder-neighbor-generalmix-30k-from-rowcolattn}"
STEPS_CSV="${STEPS_CSV:-2000,10000,30000}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-rowcol_svt_feature_neighbor}"

cd "${REPO_DIR}"
mkdir -p "${QUEUE_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"

IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"
for step in "${STEPS[@]}"; do
  checkpoint="${WORK_ROOT}/${RUN_NAME}/checkpoint_${step}.pth"
  if [[ "${step}" == "10000" ]]; then
    checkpoint="${WORK_ROOT}/${RUN_NAME}/checkpoint_phase1_10000.pth"
  elif [[ "${step}" == "30000" ]]; then
    checkpoint="${WORK_ROOT}/${RUN_NAME}/best_checkpoint.pth"
  fi

  while [[ ! -f "${checkpoint}" ]]; do
    if [[ "${WAIT_FOR_TMUX}" != "none" ]] && ! tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; then
      echo "Training session ${WAIT_FOR_TMUX} ended before ${checkpoint} appeared."
      exit 1
    fi
    echo "Waiting for ${checkpoint}"
    sleep 300
  done

  label="rowcol_svt_featuredecoder_neighbor_generalmix_step${step}"
  log="${QUEUE_DIR}/rowcol_svt_feature_decoder_neighbor_mteb_step${step}.log"
  echo "Evaluating ${label}: ${checkpoint}"

  cd "${BENCHSELECT_DIR}"
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
    2>&1 | tee "${log}"
  cd "${REPO_DIR}"
done

python - <<'PY'
import numpy as np
from pathlib import Path

logs = Path("/root/tabular/benchmark-selection/logs")
paths = {
    "Gaussian": logs / "entropy_vs_mi_mteb.npz",
    "DWIP v2-2": logs / "tabimpute_dwip-v2-2_mteb.npz",
    "Feature decoder": logs / "tabimpute_rowcol_svt_featuredecoder_generalmix_mteb.npz",
    "Neighbor step2000": logs / "tabimpute_rowcol_svt_featuredecoder_neighbor_generalmix_step2000_mteb.npz",
    "Neighbor step10000": logs / "tabimpute_rowcol_svt_featuredecoder_neighbor_generalmix_step10000_mteb.npz",
    "Neighbor best": logs / "tabimpute_rowcol_svt_featuredecoder_neighbor_generalmix_step30000_mteb.npz",
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
