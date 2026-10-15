# Matched LLM benchmark completion

This evaluation uses the existing general and synthetic recommender-specialist
checkpoints. It changes evaluation bookkeeping, not the architecture, priors,
optimization, or neural weights. There is no neural training.

## Frozen comparison

- Four released matrices: MTEB, MMLU, Merged, and BenchPress.
- Ten source-release model-row folds, seed 42; task budgets 1 through 15.
- One deterministic training-context subset per fold, capped at 3,800 rows,
  shared by all predictors. This cap affects only MMLU. Covariance, task
  selection, and metric standardization use that same subset.
- Released entropy, MI, and random selection, computed once per fold and
  stored with covariance, scaling statistics, indices, and all target masks.
- Every originally observed unselected test entry is scored. Rows without
  selected observations remain eligible. Gaussian predicts the context mean
  for these rows.
- Context and selected test inputs retain their raw values. All predictions
  and targets use the same context-column standardization and [-10, 10]
  clipping. R2 uses the context mean as its reference and is averaged over
  folds; standardized RMSE is also reported.
- MatrixFormer and MatrixFormer+ retain predictive means, FP32 weights/BF16
  autocast, and 200-row test batches. MatrixFormer+ retains the existing
  observed-reconstruction blend, fitted solely on exposed entries.
- The Gaussian comparator retains the source covariance estimator and 0.01
  ridge. Covariance estimation from context is part of this statistical
  baseline; it does not update either neural checkpoint.

These results form a new matched protocol, not an exact reproduction of the
source paper's rounded scores. Source selection can be sensitive to numerical
ties; storing the orders prevents predictors from selecting different tasks.
Previously inspected benchmark results remain descriptive evaluations.

## Run

From `/root/tabular`:

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=mcpfn/src:mcpfn/scripts
python mcpfn/scripts/evaluate_matched_llm.py --stage prepare
python mcpfn/scripts/evaluate_matched_llm.py --stage evaluate
python mcpfn/scripts/evaluate_matched_llm.py --stage audit
python mcpfn/scripts/audit_matched_llm_predictions.py mcpfn/runs/benchmark_evaluations/matched_llm_20261005
```

The default output is
`mcpfn/runs/benchmark_evaluations/matched_llm_20261005`.
Preparation freezes source code, checkpoint/data hashes, folds, context,
orders, and masks before inference. Evaluation resumes completed settings
without overwriting them; `--limit-settings 1` supports a smoke run. A single
file lock prevents concurrent writers. At least 10 GiB of free disk is retained.

Predictions are stored in standardized, unclipped coordinates with target
values and query coordinates. The audit independently reapplies the common
clipping and scoring to all 1,800 setting/fold artifacts, checks target masks,
and verifies unchanged checkpoints. Summary exports require the complete audit.

To update the paper after the audit:

```bash
cd /root/MatrixFormer
python scripts/export_matched_llm_results.py
```

The exporter preserves historical unmatched CSVs and sections separately.
Only evaluation results are exported; no model is trained by these commands.
