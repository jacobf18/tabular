# Five-seed MissBench-family evaluation

This is a **new, report-only development benchmark**, not a replacement for the
frozen historical tables or the running million-matrix training monitor.
Real benchmark measurements and masks never enter training gradients. No
checkpoint-selection rule is changed during the current training run.

## Data audit and independent units

- Preserve the original 12 UCI matrices, but correct `blood_transfusion` to
  `statlog_landsat_satellite`. The historical downloader requests UCI ID 146,
  which is Landsat Satellite (6,435 rows, 36 features), not Blood Transfusion
  (ID 176, 748 rows, 4 features). No dataset is silently replaced.
- MissBench's 42 historical names contain a duplicate:
  `appendicitis_test_edsa` and `appendicitis_test` agree up to positive
  column-affine scaling. Evaluate the first name once: **41 independent
  MissBench dataset units**, rather than counting the duplicate twice.
- Stored MissBench ground truths use different pattern-specific affine scales.
  Freeze one complete historical MCAR truth matrix per dataset. Verify every
  historical truth copy is equivalent up to positive column-affine scaling,
  with identical row/column order. These are stored benchmark coordinates, not
  a claim that the files contain original physical units. UCI feature/target
  inclusion remains exactly as in the historical matrices.
- Preserve historical files, predictions, tables, and paper figures unchanged.
  `historical_coordinate_audit.json`, dataset aliases, source paths, and hashes
  document the transition. The checks detect same-order affine duplicates;
  they are not a claim to detect all related datasets or overlapping samples.

## Masks: freeze before evaluating any model

All 11 families are used for both suites:

| Family | Mechanisms |
|---|---|
| MCAR | MCAR, Latent-MCAR, Clust-MCAR |
| MAR | Col-MAR, Phase-MAR |
| MNAR | Self-MNAR, NN-MNAR, Block-MNAR, Pol-MNAR, Soft-Pol-MNAR, Censor-MNAR |

- MissBench: target rate 0.4. UCI: target rates 0.1, 0.2, 0.3, 0.4, 0.5.
- Five independent mask-family seeds per dataset and stochastic mechanism.
  Each seed fixes the mechanism's random parameters and uniform draws across
  rates; rate settings are paired, not independent experimental units.
- Pol-MNAR is a deterministic central quantile interval: evaluate it **once**
  per dataset/rate, not five copies advertised as independent replications.
- This produces **2,091 MissBench + 3,060 UCI = 5,151 actual cases**. Duplicate
  stochastic masks, if any, are reported in `mask_uniqueness.csv`, not hidden.
- Same frozen masks for all methods. No seed selection or mask regeneration
  based on model scores. Degenerate masks stop preparation for inspection.

### Explicit differences from the historical mask generators

This is version `missbench_families_rate_controlled_v1`, not an exact replication
of the old single-mask experiment. Several historical generators ignored
`p_missing`; copying that argument would not create a valid rate sweep.

Stochastic logits receive a scalar intercept calibrated to the **global expected
missing fraction**, preserving their ordering. Realized fractions are recorded,
not forced by repairing masks. Complete-truth column standardization is used
**only to define value-dependent mask propensities**, never as an imputer's
fitted preprocessing. Imputers see only the newly observed cells.

- Latent-MCAR samples independent rank-1-to-5 factors and row/column biases.
  Clust-MCAR uses five row clusters, four column clusters, and random effects.
  Both are independent of measurement values.
- Col/Phase-MAR protect `max(1, floor(0.4*d))` columns. A standardized random
  linear predictor, with slope 2, determines missingness in the other columns.
  Phase-MAR shares one Bernoulli draw across a row's target columns.
- Self-MNAR uses slope 2 and an independently sampled sign per column, targeting
  all columns rather than the historical randomly selected subset.
- NN-MNAR reuses the original independent per-cell random MLP implementation
  and architecture distribution, in **at most 32-row × 16-column tiles**.
  This explicitly bounds the enormous memory cost of constructing a separate
  global-neighborhood network for every cell of a large UCI matrix. Tiling is
  a mask-definition change, not MatrixFormer inference chunking.
- Block-MNAR pools balanced contiguous approximately 3×3 blocks, runs those
  same tiled networks on the pooled matrix, and samples one Bernoulli per block.
  Rate calibration weights block areas. Neural probabilities are clipped to
  `[1e-6, 1-1e-6]` before logit calibration to handle sigmoid saturation.
- Soft-Pol-MNAR fixes exponent 2.5 and epsilon 0.05 before intercept calibration.
- Pol/Censor use strict quantile inequalities. Ties and discrete columns can
  prevent the requested fraction; the audit reports the actual rate. Censor
  chooses a tail per column/seed; its threshold strength does not vary by seed.

The local neural mechanisms must be described as the revised rate-controlled
family variants, not claimed identical to the original global neural masks.

## Methods and scoring

The manifest registers 31 methods/configurations: original combined MatrixFormer,
the fixed scientific step-3000 checkpoint, TabImpute (MissBench only), the main
Python baselines, the appendix neural/diffusion baselines, and 14 existing R
adapters. No mutable `latest` or `best` training checkpoint is evaluated.
Future checkpoints require a separately registered comparison, not silently
changing the model behind an existing result file.

All methods receive observed-only column-standardized inputs. Constant-column
scales are 1; fully unobserved column centers are 0, with an explicit zero input
fallback. This common preprocessing differs from some historical baseline
recipes, so the old baseline numbers must not be pooled into this campaign.
Algorithm randomness is fixed separately at seed 42. It is not five algorithm
retraining seeds. MICE-Forest retains five imputations and five iterations;
its existing finite-output retry disables row subsampling, not feature sampling.
Other baseline hyperparameters are the frozen existing adapter/plugin defaults.

MatrixFormer uses native orientation, predictive mean, BF16 autocast, and
maximum 2,000×1,000 context with the existing overlapping tail windows. No
train/test row split or transpose averaging is introduced. TabImpute uses the
paper's `masters/mcar/step-117000.ckpt`, not the different public
`tabimpute_001.ckpt` used in the runtime study. It retains the historical
two-head, no-EMA, median/full-BF16 recipe with two row-column permutations,
one row permutation, and one column permutation. It is not added to UCI.

Primary NRMSE is the **historical mean of column RMSE divided by held-out-truth
column SD**, including its original constant-column rules. This is not observed
SD normalization. For clarity, a separate `nrmse_observed_sd` diagnostic is also
saved. Prediction files allow rescoring without rerunning inference. Cases with
no scoreable columns are failures, not fabricated zero errors.

Average seeds within each dataset–pattern–rate, then rates within patterns, then
patterns equally within a dataset. Aggregate datasets equally. Compute 95% t
intervals across the **41 or 12 dataset means**, not masks or entries. Report
two-sided paired Wilcoxon tests on complete dataset-level overall scores and
Holm-adjust their p-values across comparisons within each suite.

Timeouts, dependency failures, invalid predictions, and coverage remain explicit.
A dataset-panel is summarized only when all its requested cases succeed.
Incomplete suite summaries are provisional, not a ranking or a final paper
result. No silently substituted baseline, successful-case-only mean, or mask
repair is used to conceal failures.

## Resource limits and operations

Large artifacts live at:

`/mnt/matrixformer_scratch/repeated_imputation_20261108/five_seed_v1`

Mask preparation is CPU-only with one PyTorch/BLAS thread and low scheduling
priority. After preparation, one CPU-method lane and one GPU-method lane run
concurrently. GPU workers have an **8 GiB PyTorch allocator cap**, wait for at
least 10 GiB free before each case, and release cached tensors between cases.
The cap does not include CUDA-context/other-library allocations; it is not a
claim of zero contention. Existing jobs are never paused or terminated.

Each method/case has a fixed 1,800-second budget. The supervisor kills only its
own worker process group on timeout, records the failure, and resumes at the
next case. R/CPU workers cannot see CUDA. BLAS/OpenMP and joblib CPU counts are
limited to one. Scratch temporary files have their own campaign directory.
Full baseline reruns are much more expensive than MatrixFormer forward passes.

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python mcpfn/scripts/run_repeated_imputation.py register
python -u mcpfn/scripts/run_repeated_imputation.py run
```

`run` is resumable and protected by an exclusive lock; do not register over an
existing output. Sources, checkpoints, masks, and cached predictions are hashed.
Do not edit registered implementation files while the campaign is running.
`--datasets`, `--methods`, and `--limit` support separate preflight directories;
they are not a way to change a registered full run.

```bash
cat /mnt/matrixformer_scratch/repeated_imputation_20261108/five_seed_v1/status.json
cat /mnt/matrixformer_scratch/repeated_imputation_20261108/five_seed_v1/preparation_status.json
tail -f /mnt/matrixformer_scratch/repeated_imputation_20261108/five_seed_v1/background.log
python mcpfn/scripts/run_repeated_imputation.py summarize
python mcpfn/scripts/run_repeated_imputation.py audit
```

Outputs include `manifest.json`, `mask_audit.csv`, `mask_uniqueness.csv`, method
logs/status files, compressed masks/predictions, and `summary/SUMMARY.md`,
`summary/summary.csv`, `summary/coverage.csv`, `summary/dataset_scores.csv`,
`summary/per_case.csv`, and `summary/paired_tests.csv`. Paper updates are deferred
until coverage is audited; current published figures are not overwritten.
