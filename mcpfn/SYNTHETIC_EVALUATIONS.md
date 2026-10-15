# Frozen MatrixFormer+ synthetic evaluations

This is an **evaluation-only** experiment for a synthetic-results section in
`/root/MatrixFormer`. It does not resume the recommender optimization goal.
Neither expert is trained, fine-tuned, adapted, or selected on these results.
All inputs are newly generated synthetic matrices. Classical completion solvers
fit the visible entries as part of their ordinary evaluation procedure; that is
not additional neural training.

## Fixed methods and inference

- MatrixFormer+ is the existing `observed_blend`: the general checkpoint and
  fixed recommender specialist, one analytic convex weight per matrix fitted
  from **raw observed-cell** predictions with observed-column scaling.
- General: `runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth`.
- Specialist: `runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth`.
- The runner checks both checkpoints against the established MatrixFormer+
  campaign hashes. Inference uses predictive means, ordinary observed-column
  preprocessing, full matrix context, no rating clipping, and no new router.
- Controls: the individual general and specialist experts. Baselines: column
  mean, distance-weighted K-NN, SoftImpute, and IterativeSVD.
- Default paper runs use dense CUDA inference with AMP BF16. The streamed
  backend can be registered in a separate plan if memory requires it; it is
  not an automatic fallback that silently changes inference mid-experiment.

Observed reconstruction is **in-sample evidence**, not unbiased validation.
There are no hidden targets in routing, preprocessing, or solver selection.
Models run in evaluation and inference modes with gradients disabled. State
version counters and checkpoint hashes are checked; no optimizer exists here.

## Registered evaluation grid

The protocol lives in `evaluation_protocols/matrixformer_plus_synthetic.json`.
Each case generates `L = U V^T / sqrt(rank)` with independent standard Gaussian
factors, then `X = L + noise_std * E`. Expected signal variance is one at every
rank. Stored data and predictions are float32; scoring accumulates in float64.

| Profile | Shapes | Ranks | Missing fractions | Noise SD | Replicates | Cases |
|---|---|---|---|---|---|---|
| smoke | 32 × 24 | 2, 8 | .50, .90 | 0 | 2 | 8 |
| main | 256 × 256 | 2, 8, 16, 32, 64, 128 | .10, .30, .50, .70, .80, .90, .95 | 0, .10 | 10 | 840 |
| stress | 512 × 512; 1024 × 128 and transpose; 2048 × 128 and transpose | 4, 16, 64, 128 | .50, .80, .90, .95 | 0, .10 | 5 | 800 |

Every case has seven methods. Rank 128 with a dimension of 128 is deliberately
a **full-rank limit control**, not a low-rank recovery claim. Ranks above the
paper's 2–16 low-rank training family and extreme missingness probe limits;
rectangular orientations test sensitivity to row/column geometry.

Independent RNG streams generate factors, noise, and exact-count MCAR masks.
All methods see the same matrix/mask. Missing sets are nested across rates;
noise settings share the same signal and mask. Different ranks/shapes get
different factors. No row/column support repair or dropped difficult targets
is allowed. Fresh seeds are not a claim of an entirely unseen distribution:
Gaussian factors are related to the synthetic training families.

## Baseline fairness and metrics

Nontrivial baseline parameters are chosen on a common, deterministic holdout
of up to 10% of **observed** entries, preserving existing row/column support.
Normalization is refit on the reduced context, and the chosen solver is refit
on every original observation. Fixed grids:

- K-NN neighbors: 3, 5, 10 (distance weights).
- SoftImpute shrinkage: .01, .05, .20 times the largest singular value of the
  zero-filled, observed-column-standardized context.
- IterativeSVD ranks: 2, 4, 8, 16, 32, 64, 128; the generating rank is **not**
  supplied. Effective solver rank is capped at supported minimum dimension
  minus one and recorded separately from the requested rank.
- At most 100 solver iterations; convergence tolerance `1e-4`.

If no support-preserving validation cells exist, the first registered candidate
is used and that lack of evidence is recorded. Empty rows use observed-column
means; entirely empty columns use the observed global mean. Solver warnings,
failed configurations, chosen parameters, and validation-mask hashes are saved.
There is an iteration cap, not a wall-clock timeout; full grids can be costly.

**Primary metric:** RMSE over every hidden entry of `X`, pooled within each
matrix. Report mean, sample SD, and 95% Student-t intervals across seed-level
RMSEs, separately for every shape/rank/noise/missingness condition. This is not
column-normalized NRMSE or a confidence interval over dependent individual cells.

Also archive clean-signal RMSE against `L`, MAE, supported/cold-axis RMSE,
empty-axis counts, counts of axes with fewer than `rank` observations, and
observed entries divided by `rank * (rows + cols - rank)`. The latter is a
necessary-count diagnostic, **not a sufficient recovery threshold**. For noisy
targets, independent test noise creates irreducible uncertainty; signal RMSE
is a separate diagnostic, not a replacement for the primary target.

Paired differences use only matched successful seeds (method minus
MatrixFormer+; positive favors MatrixFormer+). Failures and pending results
remain visible as coverage counts. Conditions sharing seeds are not independent
significance tests. These MCAR experiments do not establish MNAR performance.

## Commands

From `/root/tabular`:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

OUT=mcpfn/runs/benchmark_evaluations/matrixformer_plus_synthetic_smoke_20261004
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py plan \
  --output "$OUT" --profile smoke --device cpu --precision float32
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py run \
  --output "$OUT" --methods baselines
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py run \
  --output "$OUT" --methods neural --case-limit 1
python mcpfn/scripts/summarize_matrixformer_plus_synthetic.py --output "$OUT"
```

Planning requires an empty directory. Skip `plan` to resume. `--case-limit`
limits newly attempted cases in that invocation, not the registered grid.
Requesting any neural method evaluates the shared expert pair and saves all
three neural outputs. Successful results are reused; failures are retained
unless `--retry-failed` is explicitly supplied (old failure records are archived).

For the full paper evaluation, use a new directory, `--profile main`, and
`--device cuda --precision amp_bfloat16`. Repeat with `--profile stress` in a
separate directory. Plan first, then run CPU baselines and neural inference
separately if desired. Do not substitute CPU smoke scores for the paper run.

```bash
OUT=mcpfn/runs/benchmark_evaluations/matrixformer_plus_synthetic_main_20261004
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py plan \
  --output "$OUT" --profile main --device cuda --precision amp_bfloat16
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py run \
  --output "$OUT" --methods baselines
python mcpfn/scripts/evaluate_matrixformer_plus_synthetic.py run \
  --output "$OUT" --methods neural
python mcpfn/scripts/summarize_matrixformer_plus_synthetic.py --output "$OUT"
```

The runner refuses neural CUDA work if `nvidia-smi` reports another compute
process. Do not stop the separate MovieLens evaluation or overlap its GPU job.
The default free-space reserve is 20 GiB; currently available disk space is
tight. Plans print an uncompressed-array storage estimate, excluding source
snapshots/reports and compression effects. Both profiles together can exceed
available space above the reserve; review storage before launching stress.

## Artifacts and paper integration

Each output directory records immutable protocol/case lists, checkpoint/config
hashes, source snapshots including `borders.pt`, package versions, and inference
settings. Resume rejects changes; create a new plan rather than mixing runs.
Per-case `inputs.npz` stores signal/truth/context/mask. Per-method files retain
all hidden-cell predictions, RMSE, tuning/routing metadata, timings, and hashes.

The summarizer regenerates every saved matrix/mask and independently rescores
all archived predictions, then emits:

- `summary.csv`: mean/SD/CI and valid/failed/pending counts per condition/method.
- `paired_differences.csv`: paired seed-level RMSE differences versus the blend.
- `figures/`: PDF/PNG RMSE-versus-missingness panels, split by rank/shape/noise.
- `paper/synthetic_results.tex` and `paper/synthetic_tables.tex`: section and
  complete longtable; the latter requires LaTeX `longtable` and `booktabs`.
- `summary_audit.json`: reproduction checks and completeness, reported separately.

This audit reproduces metrics, not the blend weight independently from raw
observed predictions (those raw arrays are not archived). Router behavior is
covered by tests and the frozen implementation snapshot.

Only after the full GPU profile finishes, export with:

```bash
python mcpfn/scripts/summarize_matrixformer_plus_synthetic.py \
  --output "$OUT" --paper-dir /root/MatrixFormer
```

The exporter refuses smoke, CPU-diagnostic, incomplete, or existing exports.
Completed campaigns with failures retain explicit counts rather than silently
discarding them. It writes dedicated `results/synthetic_low_rank/<profile>/`
artifacts and `sections/synthetic_<profile>_results.tex`, without modifying
`main.tex` or existing results. Review findings and then include the section in
the paper; the setup itself makes no empirical performance claims.

## Focused tests

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts python -m pytest -q \
  mcpfn/tests/test_synthetic_low_rank_benchmark.py \
  mcpfn/tests/test_observed_router.py mcpfn/tests/test_router_evaluation.py
```

## Setup validation: 2026-10-04

- 47 focused tests pass, including routing, observed-only solver selection,
  immutable-plan checks, resume/retry, failure accounting, and independent scoring.
- The CPU smoke profile is complete: eight cases, all seven methods, **56/56
  successful results**, no failures. Both real frozen checkpoints were evaluated;
  their hashes remain unchanged and no neural updates occurred.
- All saved smoke predictions independently reproduce their recorded metrics.
  PDF/PNG plots were generated and inspected; the generated LaTeX section and
  full table compile together in an isolated document. Smoke scores remain
  explicitly provisional, not paper-grade evidence.
- Output: `runs/benchmark_evaluations/matrixformer_plus_synthetic_smoke_20261004/`.
- Main and stress plans are already registered under sibling directories
  `matrixformer_plus_synthetic_main_20261004/` and
  `matrixformer_plus_synthetic_stress_20261004/`. Skip `plan` and invoke `run`
  to use them. **Neither full sweep has been launched.**
- Uncompressed array estimates are 1.54 GiB for main and 5.48 GiB for stress.
  Review disk headroom above the 20 GiB reserve before proceeding; the separate
  MovieLens GPU evaluation was left untouched.
