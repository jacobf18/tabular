# MatrixFormer+ and the two-expert router

`src/tabimpute/routing.py` implements an observed-only hard router and a fitted
convex blend of the general MatrixFormer checkpoint and the recommender
specialist. It accepts any two experts exposing
`impute(matrix, return_full=True) -> (completed, raw_predictions)`.

**MatrixFormer+** names the observed-error-fitted convex blend (`observed_blend`),
not the hard router or masked-observation control. It combines the general model
and recommender specialist without training a new checkpoint. The comparison
tables are in `MATRIXFORMER_PLUS_RESULTS.md`; archived variant keys remain unchanged.

```python
from tabimpute.routing import TwoExpertRouter

router = TwoExpertRouter(general_imputer, specialist_imputer, policy="observed_blend")
completed = router.impute(masked_matrix)
decision = router.last_result.decisions["observed"]

comparison = router.predict_all(masked_matrix)
holdout_blend = comparison.predictions["holdout_blend"]
```

## Evidence and safeguards

- `observed_hard` selects by observed-column-standardized reconstruction MSE.
- `observed_blend` minimizes the same loss over a convex two-model combination.
- `holdout_hard` and `holdout_blend` hide 10% of observations (seed 42), preserving
  at least one observation in each initially supported row/column. Experts must
  recompute their preprocessing from the reduced context. Final imputation uses
  the full original context, not the reduced matrix.
- Recommender routing uses global scaling in rating units instead of column
  scaling. Expert preprocessing is a separate concern.
- The decision applies to a whole input matrix, not individual rows/cells. An
  LLM test batch is one such input matrix.
- Original observations, input dtype and memory layout are preserved. Preserving
  layout before float32 UCI column reductions is necessary for baseline parity.
- Ties or absent evidence select the general model. Nonfinite expert outputs
  raise an error rather than silently excluding difficult evidence cells.
- The router never accepts the true values of benchmark-missing entries.

**Important:** Raw predictions on observed entries are not copied observations,
but the encoder still sees their true values. Their reconstruction loss is
in-sample evidence, not a held-out accuracy estimate. The masked-observation
control tests this distinction; it is not guaranteed to match MNAR targets or
LLM test-row sparsity either.

## Frozen evaluation

From `/root/tabular/mcpfn`:

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python scripts/evaluate_two_expert_router.py --output runs/benchmark_evaluations/NEW_DIRECTORY

PYTHONPATH=src:scripts python scripts/summarize_two_expert_router.py \
  --output runs/benchmark_evaluations/NEW_DIRECTORY

PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest tests/test_observed_router.py tests/test_router_evaluation.py \
  tests/test_streamed_inference.py -q
```

The campaign uses:

- MissBench: all 462 frozen cases, full matrix context.
- UCI: all 1,800 frozen cases, 2000×1000 windows and original standardization.
- MovieLens 100K and Netflix small: frozen outer rating splits and user-macro
  RMSE/MAE. The general expert uses global normalization and 9976×1000 streamed
  context; the specialist uses its winning two-way/user-scale preprocessing and
  2000×1000 context. This is recipe routing, not a checkpoint-only comparison.
- MTEB, MMLU, Merged and BenchPress: unchanged ten-fold LLM harness, all three
  selection methods and budgets 1–15. Predictions for all six variants share
  inference; scoring is replayed on identical masked inputs.

The full run is saved in
`runs/benchmark_evaluations/two_expert_router_v2_20261004/`. The final audit/report
command requires complete results, independently rescores saved predictions and
panels, and writes `RESULTS.md` and `final_audit.json`. Decisions, weights, timing,
predictions, fold panels, source snapshots and hashes remain on disk. New source
or protocol changes require a fresh output directory. Tabular cases can resume
only under an identical manifest; the other stages rerun in full.

The v2 campaign enables cuDNN SDPA for LLM inference, matching that historical
harness; tabular/recommender inference retains its disabled setting. An initial
LLM attempt caught this difference via baseline-parity assertions. The completed
tabular/recommender stages are linked from the first campaign, not recomputed or
silently relabeled. `--reuse-completed PATH` verifies identical frozen inputs,
inference dependencies and evaluator function ASTs before registering those links.

Observed routing costs two expert evaluations; the masked control costs four
in total, shared across all six outputs. It is not a cheap pre-inference gate.
No model training, causal-inference/R-method reruns, or paper edits are involved.
