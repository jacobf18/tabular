# Cylinder-wake matrix completion

This is a zero-shot physics pilot, not a reproduction of a published numerical
leaderboard and not a new training run. It evaluates the published single-model
MatrixFormer and the existing two-specialist MatrixFormer* router.

## Existing data and implementations

- **PyDMD** supplies `CYLINDER_ALL_LOW_RES.mat` in its
  [BOP-DMD tutorial](https://github.com/PyDMD/PyDMD/blob/024464ff667e4b0226f9ecbe933bc67f3fe21e81/tutorials/user-manual1/user-manual-bopdmd.py).
  The tutorial identifies the data as two-dimensional cylinder-wake vorticity
  at Reynolds number 100 and credits the DMD book dataset. The MAT file has
  9,834 spatial locations and 151 time snapshots. The tutorial adds noise in
  its own example; **this pilot does not add that noise**.
- [fancyimpute](https://github.com/iskandr/fancyimpute) supplies `SoftImpute` and
  `IterativeSVD`. The latter is our simple iterative rank-constrained POD/SVD
  completion baseline, labeled "Iterative gappy POD". It is not an exact
  implementation of the weighted reconstruction algorithm in Venturi and
  Karniadakis.
- The physical reconstruction problem is established in Venturi and
  Karniadakis, *Gappy data and reconstruction procedures for flow past a cylinder*,
  Journal of Fluid Mechanics **519**, 315–336 (2004),
  [doi:10.1017/S0022112004001338](https://doi.org/10.1017/S0022112004001338).
- SoftImpute is described in Mazumder, Hastie, and Tibshirani, *Spectral
  Regularization Algorithms for Learning Large Incomplete Matrices*, JMLR
  **11**, 2287–2322 (2010),
  [paper](https://jmlr.org/papers/v11/mazumder10a.html).

The evaluator pins the PyDMD revision, hashes the data, checkpoint/config files,
model sources and installed baseline implementations, and records package
versions. It does not need a full PyDMD installation: the tested task is missing
entry reconstruction, not fitting a DMD dynamical model.

## Pilot protocol

- Sample 2,048 spatial locations without replacement using seed 20261027;
  sort them in source order and retain all 151 chronological snapshots.
  The resulting matrix is **2,048 × 151**. No spatial coordinates are passed
  to MatrixFormer. This is a subsampled dataset, not a full-field evaluation.
- Evaluate 20%, 50%, and 80% missingness with mask seeds 0, 1, and 2.
- Test MCAR and sensor outages. An outage removes one contiguous *circular*
  time interval per location, with a random start. Wraparound preserves
  observations in every snapshot even at 80% missingness. It is an artificial
  periodic-block mask, not a claim that the finite recording is exactly periodic.
- There is **no row-wise train/test split**. Outer hidden entries are the test
  targets; all remaining cells constitute the inference context.
- Fit a global mean and population SD to available entries only. The same
  physics normalization is used for all three frozen networks. All sampled
  rows and time columns enter each forward pass, with no context chunking or
  transpose ensemble. Decoder microbatching only limits temporary allocations.
- MatrixFormer is the published V57 checkpoint with SHA256
  `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.
  MatrixFormer* fits its convex expert weight on **raw observed-cell
  reconstructions**, not on copied observations or outer hidden labels.
  This is an adaptation of the existing router to global-normalized physics
  data, not the recommender-specific preprocessing recipe.
- Classical controls are column mean, temporal linear interpolation with
  constant endpoint extrapolation, iterative POD/SVD completion, and SoftImpute.
  Temporal interpolation knows the chronological order; POD/SVD and SoftImpute
  use the matrix only.
- Select POD rank from 2/4/8/16/32 and SoftImpute shrinkage from
  0.005/0.02/0.1 times the largest singular value of the zero-filled normalized
  observed matrix. Use 10% held-out **observations**, maintaining row/column
  support, then refit on all outer observations. Normalization is refitted for
  the inner holdout. Both iterative solvers have a 200-iteration cap and
  `1e-5` stopping tolerance; POD uses fixed rank without rank continuation.

## Metrics and predictive distributions

Point metrics use every outer hidden entry. **NRMSE = hidden-entry RMSE divided
by the population SD of the outer observed values.** This is an observed-global
normalization, not the per-column MissBench metric. RMSE, MAE, and hidden-cell
R² are also archived. The same denominator applies to all methods in a case.

For each mask, sample up to 10,000 hidden entries uniformly, independently of
prediction errors, and report:

- CRPS via numerical integration of pinball loss over 199 midpoint quantiles;
- native negative log density in the original vorticity units;
- 50%, 90%, and 95% central interval coverage and mean width.

The model's native density has uniform interior bins and half-normal tails.
The evaluator inverts those tails explicitly rather than using a bounded-bin
quantile approximation. MatrixFormer* uses a convex **mixture of the two
predictive densities**, with the same observed-error weight as its mean.
This is not the distribution of an average of independent expert samples.

For a simple probabilistic reference, each fitted low-rank baseline receives a
homoscedastic Gaussian residual model whose SD is its selected inner-validation
RMSE. This is explicitly **not** a Bayesian POD posterior. Under block outages,
the random inner holdout need not match the outer error distribution.

These are marginal cellwise distributions, not joint physically constrained
flow posteriors. Calibration is measured without post-hoc recalibration on
physics test labels. Seed SD/error bars represent mask variability on **one
flow trajectory**, not uncertainty over independently simulated flows.

## Run and inspect

From `/root/tabular`:

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest mcpfn/tests/test_cylinder_completion.py -q

PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  PYTHONWARNINGS=ignore::FutureWarning \
  python mcpfn/scripts/evaluate_cylinder_completion.py

PYTHONPATH=mcpfn/src:mcpfn/scripts \
  python mcpfn/scripts/report_cylinder_completion.py \
  mcpfn/runs/physics_cylinder_20261027/pilot
```

Results are in `/root/tabular/mcpfn/runs/physics_cylinder_20261027/pilot/`:

- `RESULTS.md`: human-readable results, caveats and provenance.
- `manifest.json`, `status.json`: frozen protocol and completion status.
- `per_case.csv`, `summary.csv`: all scores and mask-seed means/SDs.
- `dataset.npz`: sampled truth and original spatial indices.
- `cases/`, `predictions/`: masks, inner validation masks, point predictions,
  sampled distribution diagnostics, baseline choices and routing weights.
- `accuracy.pdf/png`, `calibration.pdf/png`: comparison and coverage figures.
- `audit.json`: independent checks of saved masks, scores and coverage.

## Exploratory measurement acquisition

After the completion pilot, a separately registered follow-up uses the six
80%-missing cases. The 10,000 sampled uncertainty cells become the candidate
measurement pool. **All other originally hidden cells form a fixed test set**,
which is disjoint from every acquisition policy's candidates. Each policy
reveals exactly 1,000 pool cells, restores the original observations, refits
observed-only normalization, and reruns the same single checkpoint:

- Uniform random acquisition, with a fixed seed.
- Largest native predictive standard deviation.
- Largest absolute predicted mean, to control for simply measuring the
  strongest predicted signal rather than using uncertainty.

The target-value lookup happens only after selecting the measurement indices.
All policies use the original observed SD for scoring, so denominator changes
cannot manufacture an improvement. This is one acquisition batch, not a
sequentially optimized sensor-placement algorithm. The follow-up was designed
after seeing the completion pilot and is reported as exploratory, not as a
preregistered confirmatory finding.

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/evaluate_cylinder_acquisition.py \
  mcpfn/runs/physics_cylinder_20261027/pilot --budget 1000
```

Its manifests, every selected measurement index, fixed test masks, new
predictions, audits and CSVs are saved under `pilot/acquisition/`.

The initial 256-row software smoke test is separate from the registered
2,048-row pilot. Its failed first launch and corrected launch are retained.
No checkpoint is selected, trained, or changed based on these physics results.
No paper edits or Git publication are part of this experiment.
