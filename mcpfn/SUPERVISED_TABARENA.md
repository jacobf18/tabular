# MatrixFormer as a supervised TabArena predictor

## Benchmark and local copy

TabArena is a **suite, not one dataset**. Its v0.1 IID suite has 51 datasets:
30 binary classification, 8 multiclass classification, and 13 regression tasks.
The pinned curated metadata specifies three outer folds and either three or
ten repeats, for **816 outer splits**. TabArena-Lite uses split 0 only: one
official train/test split per dataset, not all three folds.

- Official repository: <https://github.com/autogluon/tabarena>, downloaded at
  commit `f5f2c7ad99f2b61ace44a839aef7bf5b7dd35bc4`.
- Dataset curation repository: <https://github.com/TabArena/tabarena_dataset_curation>,
  commit `4cd5cdf0430721f239a991884624095388d337a9`.
- Paper: *TabArena: A Living Benchmark for Machine Learning on Tabular Data*,
  <https://arxiv.org/abs/2506.16791>.
- Official numerical metric defaults: **ROC-AUC** for binary classification,
  **log loss** for multiclass, and **RMSE** for regression. These are not
  interchangeable with the imputation paper's NRMSE.

Local root: `/mnt/matrixformer_scratch/tabarena_20261028/`.
`upstream/` and `dataset_curation/` contain the cloned code. `data/tasks/<task_id>/`
contains features/targets in Parquet, every official split in NPZ, and checksums.
`data/inventory.csv` records actual feature counts, missingness, and categorical
cardinalities. `data/status.json` reports download coverage and failures.
Original dataset licenses are recorded per task; the suite license does not
replace each dataset's license. No benchmark data are committed to Git.

The current upstream runner requires Python 3.11–3.13, AutoGluon nightlies and
Ray. The existing MatrixFormer environment uses Python 3.10. We do **not**
upgrade its PyTorch or replace its dependencies. An isolated scratch-directory
environment installs OpenML 0.15.1 and accesses the official data and splits;
the full AutoGluon/TabArena experiment runner is not installed by this pilot.

## Supervised prediction is target-column completion

For a test feature vector, construct

\[
M(x_*) = \begin{bmatrix}
T(X_{\mathrm{context}}) & S(y_{\mathrm{context}})\\
T(x_*) & \mathrm{NaN}
\end{bmatrix}.
\]

The frozen checkpoint predicts only the final row's target cells. `fit(X,y)`
fits preprocessing and stores labeled context; it does **not** update network
weights. Neither fitting nor prediction receives test labels. The scorer reads
test labels only after predictions are written.

Unlike matrix-completion benchmarks, a supervised benchmark **requires its
official held-out rows**. Use the exact OpenML splits, not randomly masked
target cells across the entire dataset. Training-only inner validation must
select any future settings; the outer test set is never a tuning set.

Each test row gets an independent context matrix. GPU batching stacks these
matrices on the batch axis; test rows cannot attend to each other. This is an
**inductive** predictor, not transductive batch completion. Predicting a row
alone or with unrelated test rows does not change its available information.

## Fixed preprocessing for the first pilot

All preprocessing below is fitted **only on the official training split**,
including categorical vocabularies: never use the category list attached to a
full-dataset pandas categorical as the observed training vocabulary.

- **Numerical columns:** replace nonfinite values by missing tokens; fit a
  per-column empirical-quantile-to-normal transform on finite training values
  (at most 1024 quantiles), then clip to [-5,5]. Test values beyond the training
  range saturate. Preserve NaNs rather than inserting test-derived means.
  This handles skew and disparate units but sacrifices tail extrapolation and
  may distort physical linear relationships; it is a fixed initial choice,
  not an empirically selected winner.
- **Constant/all-missing columns:** drop when they have no training information.
  For constant numeric values with varying missingness, retain the missingness
  indicator. If every column drops, retain a constant context feature.
- **Binary nominal columns:** a single -1/+1 contrast avoids doubling the
  thousands of binary features in some tasks. Missing/unseen categories map
  to NaN; with one observed value plus training missingness, encode that
  observed-versus-missing distinction. No multi-category ordinal coding.
- **Other nominal columns:** one-hot encode the 32 most frequent training
  categories; group other *seen training categories* into OTHER, and reserve
  distinct MISSING and UNKNOWN columns. This bounds very high cardinality
  without target encoding. The rare-category cap is lossy and must be reported.
- **Schema fallback:** OpenML nominal flags do not list every string column.
  Treat explicitly nominal variables as categorical even if their values look
  numeric; also infer nonnumeric/bool columns from the training frame's dtype.
  The label-free full-suite preflight exposed unflagged strings in six datasets,
  including age ranges, experience labels, and an ISO date.
- **Dates:** detect ISO dates from training values only, excluding explicitly
  nominal columns. Expand to epoch days and month/weekday sine/cosine features,
  then apply training-fitted numerical transforms. Malformed or missing test
  dates become missing, rather than changing the fitted schema.
- No test-label-based encoding, feature selection, imputation,
  normalization, or tuning. Do not apply a second observed-global normalization
  after stacking the test row: that would make the fitted transformation
  query-dependent and mix feature/target units.

## Targets and distribution outputs

**Regression:** standardize the training targets by their training mean and
population SD. Decode the full-support distribution mean at the query target
cell and invert this affine transformation. Do not clip target predictions or
normalize using test labels. RMSE is computed in original target units.

**Binary classification:** assign the two training classes targets -1 and +1.
The positive-class probability is `1 - CDF(0)`, not the rounded prediction mean.
Zero lies inside the histogram's finite borders, so its CDF includes both tail
masses correctly. The class order is saved with every prediction file.

**Multiclass:** append one -1/+1 one-vs-rest target column per training class;
mask *all* of them for the query. Read `q_k = P(Y_k > 0)` and report
`p_k = q_k / sum(q)`, with a fixed 1e-7 numerical probability floor before
normalizing. This avoids an arbitrary ordering between class labels. These
are continuous-imputer-derived class scores, **not a proven calibrated
categorical posterior**; log loss and future calibration tests are essential.

The checkpoint remains the final combined single MatrixFormer:
`/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth`.
SHA256: `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.

## Context, pilot, and honest comparison

- At most **1023 labeled training rows plus one query**; retain every encoded
  feature/target column. Use all rows when the training split fits. For larger
  tasks use a fixed seeded sample; classification preserves at least one
  example of each observed training class. Save context row indices. Do not
  silently describe this as using all available training labels in context.
- Batch independent queries with at most eight matrices and an approximate
  131072-cell batching budget (at least one matrix). Decode only query targets.
- Initial smoke test: the **three datasets in the official quickstart**,
  `blood-transfusion-service-center`, `QSAR_fish_toxicity`, and `anneal`;
  repeat 0, fold 0. They exercise binary, regression, mixed categorical,
  and multiclass paths. Use every test row in these splits. These three curated
  datasets have no explicit NaNs; missing-value behavior is covered by unit tests
  and a separate preprocessing-only preflight over all 51 datasets.
- Fixed comparisons: training-prior/mean dummy and a 200-tree RandomForest,
  same train/test splits, train-fitted feature encoding and median imputation
  for the forest. No HPO, model selection, calibration, or test-driven reruns.
- Save predictions **before** extracting test labels for scoring, probabilities,
  metrics, timings, context indices, transformations, and source/checkpoint hashes.
- This is an adapter smoke test, **not the 51-dataset TabArena leaderboard**.
  TabArena's official model protocol also includes its prescribed inner
  validation/bagging/HPO. A later full run should wrap this estimator in
  AutoGluon's `AbstractModel` (`_fit`, `_predict_proba`, regression `predict`),
  accept the runner's validation sets and resource budget, and let TabArena
  handle its standard folds and aggregation. No leaderboard rank claim from
  three single-split evaluations against two untuned baselines.

Implementation: `scripts/matrixformer_supervised.py`.
Download: `scripts/download_tabarena.py`.
Pilot: `scripts/evaluate_tabarena_smoke.py`.
