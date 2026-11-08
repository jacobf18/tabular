# Fixed MatrixFormer evaluation on TabArena regression

Scope: all 13 downloaded TabArena regression datasets, official Lite outer
split (repeat 0, fold 0, sample 0), every held-out row. This extends the
three-task smoke test; it does not change its frozen adapter. No classification
experiments, neural training, hyperparameter search, test-driven settings,
feature selection, target clipping, bagging, or inference ensembling.

## Prediction and preprocessing

Use the final combined single MatrixFormer checkpoint, SHA256
`0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.
For a query, complete the missing final target cell of a matrix containing
at most 1023 labeled training rows plus that one query. Independent queries
are batched, with no test-to-test attention. All encoded feature columns
are retained, including the 1024 QSAR-TID-11 features.

Use the unchanged `matrixformer_supervised.py` preprocessing described in
`SUPERVISED_TABARENA.md`: training-only numerical quantile transforms to normal,
binary contrasts, nominal one-hot encoding with a 32-category cap and explicit
other/missing/unknown handling, and dtype/date fallback. Missing numerical
values remain missing for MatrixFormer. Fit transformations and target mean/SD
on all training rows, even when context is a seed-0 random subset. Standardize
training y, decode the predictive distribution mean, and invert target scaling.

Baselines: untuned RandomForestRegressor (200 trees, seed 0, four CPU workers)
and training-mean DummyRegressor, both using all labeled training rows. Use
the same fitted feature encoding, then train-fitted median imputation with
missing indicators. This is an initial adapter comparison, not a claim of
competitive performance against TabArena's tuned leaderboard methods.

## Metrics and reproducibility

Primary: RMSE in original target units. Also save MAE and R². Never average
raw RMSE across datasets; the summary averages RMSE divided by the corresponding
training-mean dummy's test RMSE, with equal dataset weight, and reports wins.
One split and seed do not establish uncertainty or statistical significance.
Checkpoint/procedure-selection dataset overlap remains to be audited.

Process smaller label-free estimated workloads first, then the wider tasks;
the suite is fixed before scoring. Preserve failures and report coverage.
Save predictions before extracting held-out labels for scoring, official row
indices, sampled context indices, fitted transforms, metric values, timings,
and source/data/checkpoint hashes. Reading the target parquet to extract train
labels does load the file; the claim is absence of test-label use in prediction,
not absence of file access.

Checkpoint query progress every 16 inference batches. Chunk boundaries are
multiples of the adapter's original batch size, so chunking never changes its
batch layout (BF16 outputs can differ slightly across different batch shapes).
Resume only with identical source/data/settings/environment hashes, verifying
saved chunk hashes. Do not silently reduce rows, columns or context on failure.
The approximate batching budget is 131072 cells, at least one matrix, at most
eight matrices. CPU/disk overhead is included in per-chunk inference timing;
baseline preprocessing is separate. These are diagnostic timings, not a
controlled runtime comparison.

## Run

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/mnt/matrixformer_scratch/tabarena_20261028/venv/bin/python
$PY mcpfn/scripts/evaluate_tabarena_regression.py
# To resume an interrupted run with exactly the same sources/environment:
$PY mcpfn/scripts/evaluate_tabarena_regression.py --resume
```

Default results directory:
`/mnt/matrixformer_scratch/tabarena_20261028/regression_lite/`.
See `RESULTS.md`, `summary.csv`, `aggregate.csv`, and `status.json` there.
