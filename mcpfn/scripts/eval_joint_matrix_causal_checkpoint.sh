#!/usr/bin/env bash
set -euo pipefail

MCPFN_DIR="${MCPFN_DIR:-/root/tabular/mcpfn}"
MCBENCH_DIR="${MCBENCH_DIR:-/root/matrix-completion-benchmarks}"
CHECKPOINT="${CHECKPOINT:-${MCPFN_DIR}/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-joint-spectrum-cateblock-bs2-60k/best_checkpoint.pth}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${CHECKPOINT}")")-$(basename "${CHECKPOINT}" .pth)}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${MCBENCH_DIR}/src:${MCBENCH_DIR}/submodules/Do-PFN:${MCPFN_DIR}/src:${PYTHONPATH:-}"

PARAMS_JSON="${MCBENCH_DIR}/benchmarks/configs/dopfn_tabimpute_joint_eval_${RUN_TAG}.json"
python - <<PY
import json
from pathlib import Path

base_path = Path("${MCBENCH_DIR}/benchmarks/configs/dopfn_tabimpute_model_params.json")
payload = json.loads(base_path.read_text())
payload["tab_impute_pure_rabel_joint_matrix_causal"] = {
    "checkpoint_path": "${CHECKPOINT}",
    "device": "${DEVICE}",
    "entry_wise_features": False,
    "max_num_rows": 300,
    "max_num_cols": 300,
    "max_num_chunks": 100,
    "reuse_loaded_model": True,
}
Path("${PARAMS_JSON}").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY

DOPFN_OUT="${MCBENCH_DIR}/benchmarks/reports/dopfn_matrix_completion_joint_${RUN_TAG}"
python "${MCBENCH_DIR}/scripts/run_dopfn_matrix_completion_benchmark.py" \
  --data-root "${MCBENCH_DIR}/submodules/Do-PFN/data" \
  --dopfn-root "${MCBENCH_DIR}/submodules/Do-PFN" \
  --prediction-root "${DOPFN_OUT}/predictions" \
  --algorithms \
    soft_impute \
    col_mean \
    tab_impute_pure_rabel_spectrum \
    tab_impute_pure_rabel_causal \
    tab_impute_pure_rabel_joint_matrix_causal \
  --no-semi-real \
  --include-synthetic \
  --synthetic-n-max "${DOPFN_N_MAX:-10}" \
  --algorithm-params-json "${PARAMS_JSON}"

python "${MCBENCH_DIR}/scripts/plot_dopfn_matrix_completion_results.py" \
  --data-root "${MCBENCH_DIR}/submodules/Do-PFN/data" \
  --dopfn-root "${MCBENCH_DIR}/submodules/Do-PFN" \
  --prediction-root "${DOPFN_OUT}/predictions" \
  --output-dir "${DOPFN_OUT}/plots" \
  --top-k-algorithms 12

LOW_RANK_OUT="${MCBENCH_DIR}/benchmarks/reports/low_rank_mcar_joint_${RUN_TAG}"
python "${MCBENCH_DIR}/scripts/low_rank_mcar/run_low_rank_mcar_rank_sweep_benchmark.py" \
  --n-rows "${LOW_RANK_ROWS:-50}" \
  --n-cols "${LOW_RANK_COLS:-50}" \
  --rank-min "${LOW_RANK_RANK_MIN:-5}" \
  --rank-max "${LOW_RANK_RANK_MAX:-30}" \
  --n-mcar-replicates "${LOW_RANK_REPS:-3}" \
  --missing-fraction "${LOW_RANK_MISSING:-0.4}" \
  --dgp-type "${LOW_RANK_DGP:-low_rank_gaussian}" \
  --dataset-root "${MCBENCH_DIR}/benchmarks/simulated/low_rank_mcar_joint_${RUN_TAG}" \
  --output-dir "${LOW_RANK_OUT}" \
  --algorithms \
    soft_impute \
    col_mean \
    tab_impute_pure_rabel_spectrum \
    tab_impute_pure_rabel_joint_matrix_causal \
  --algorithm-params-json-file "${PARAMS_JSON}"

echo "Do-PFN plots: ${DOPFN_OUT}/plots"
echo "Low-rank results: ${LOW_RANK_OUT}"
