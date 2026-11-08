# Frozen MatrixFormer synthetic capability and attention study

## Current manuscript scope

The manuscript now focuses on nearest-neighbor completion and embedding
clustering. Broad rank/noise/solver capability sweeps and attention-map
diagnostics are no longer included in the main text or appendix. Their results and figures
remain archived and reproducible; this document preserves the full study.
The main neighbor plot shows MatrixFormer, two-way K-NN, oracle neighbors,
and oracle block means at 20% and 40% observed, noise SD 0.25, including both
directions of the MatrixFormer/K-NN ordering. See `SYNTHETIC_REPRESENTATIONS.md`
for the current presentation and figure reproduction.

## Scope and identity

This evaluation uses the released single combined MatrixFormer, not the earlier
MatrixFormer* ensemble. It does not restart a training goal, tune a checkpoint,
or change the inference convention used in the main paper. The checkpoint is
`/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth`,
SHA256 `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.

Primary inference uses standard orientation, whole-matrix context, observed-only
column normalization, predictive means, FP32 parameters and BF16 autocast.
There is no transpose averaging, neural fitting, or selection of successful
examples after examining errors. Eight unit tests and a three-case pipeline
smoke run were completed before the main sweep. The smoke cases are a subset
of the fixed main grid; they are not additional independent evaluation samples.

## Fixed grid

All matrices are 64 by 64. Each condition has ten seeded matrix realizations.

| Family | Parameters | Observed fractions | Cases |
| --- | --- | --- | --- |
| Gaussian factors | ranks 2, 4, 8, 16, 32; no added noise | .2, .4, .6, .8 | 200 |
| Biclusters | four balanced groups per axis; noise SD 0, .25, 1 | .2, .4, .6, .8 | 120 |
| Independent noise | independent standard-normal entries | .2, .4, .6, .8 | 40 |

Gaussian factors generate `L = U V^T / sqrt(rank)` with unit expected signal
variance. Cluster signals are `L[i,j] = B[g[i],h[j]]`, with standard-normal
block values and independent, uniformly permuted balanced memberships on both
axes. Noise is added independently. The pure-noise control has zero clean
signal. Factor, noise and mask RNG streams are separate. Masks have exact-count
MCAR observations; masks are nested across rates and shared across noise levels
within a seed. No support repair removes difficult targets. Fresh seeds do not
make these distributions out of distribution: the training mixture contains
related factor structures. These tests do not characterize MAR/MNAR, large
matrices, arbitrary nonlinearities, or all distribution shifts.

## Comparators and oracle information

Column mean, row K-NN, column K-NN, their fixed 50/50 average, SoftImpute and
IterativeSVD reuse the established observed-only baseline machinery. Row K-NN
uses observed-column standardization and distance weights; column K-NN runs the
same procedure on the transpose, including its own observed-column scaling.
Both orientations tune neighbor counts from 3, 5, 10 and 20. SoftImpute tunes
shrinkage fractions .01, .05 and .2 of the zero-filled leading singular value;
IterativeSVD tunes ranks 2, 4, 8, 16 and 32. The existing solvers retain their
100-iteration budget and `1e-4` convergence criterion.

All baseline selection uses the same deterministic, support-preserving holdout
of up to 10% of the original observations. Normalization is refit on that
reduced context and the chosen settings are refit on all original observations.
No held-out evaluation value enters selection. Failed runs are recorded and
prevent a complete paper export rather than being silently dropped.

The oracle-neighbor estimator receives only the true latent coordinates or
group memberships, not true missing outcomes. For Gaussian factors it tunes
the same four neighbor counts using Euclidean distances in generating latent
coordinates, distance-weights observed donors, and averages row/column
predictions. For clusters it averages all observed donors of the same true
group in the target column or row, then averages the two predictions. The
oracle block mean is a separate clustered-data comparator: it averages all
observed entries belonging to the target's true row-group/column-group block.
Absent eligible donors, the corresponding observed-column mean is used.
An empty column falls back to the observed global mean. Oracle neighbors are
not a lower bound on error: local averaging has finite-sample variance and
bias even with correct neighborhoods. Independent noise has no meaningful
oracle neighborhood and no invented oracle-neighbor score is reported.

### Uncentered exact-SVD sensitivity control

Smoke inspection highlighted that subtracting observed column means can raise
the clean matrix rank by one. A separate, explicitly exploratory control
therefore runs exact truncated-SVD iterations without centering, scaling only
by the observed global standard deviation. It tunes ranks 1, 2, 4, 8, 16 and 32
on the same observed holdout, with at most 500 iterations and a relative
missing-prediction change threshold of `1e-6`, then refits. This control is
evaluated on every registered case. Its code, configuration, predictions and
scores are saved separately; original centered baselines remain unchanged.
Differences cannot be attributed to centering alone because the numerical
solver and convergence budget also differ. The main phase comparison includes
this stronger comparator, not just the more favorable original SVD baseline.

## Attention and interventions

Attention is measured at 40% observations on 60 matrices: Gaussian ranks 2 and
16, all three cluster noise levels, and independent noise, ten seeds each.
Eight missing query cells per matrix are sampled by a fixed RNG. Read-only
pre-forward hooks reconstruct query/key probabilities with the model's actual
rotary embeddings, without replacing the production attention kernel.
Every captured forward pass must reproduce its uncaptured predictions
bit-for-bit. A unit test checks reconstructed weights against explicit native
attention on a small model.

All eight layers and all sixteen heads are retained. Per-query reports include
special-token mass, self mass, total observed-donor mass, and missing-key mass
(the latter includes self mass because queried cells are missing). Same-group
mass is measured conditional on eligible observed donors and compared with
the same-group fraction of that donor pool. Top-five same-group fractions are
also archived. These are descriptive associations, not attribution proofs.

The figure's example is specified before evaluation: clustered data, noise SD
.25, 40% observations, seed zero, first sampled query. Ground-truth groups sort
the heatmaps for display only; inference sees shuffled rows/columns. Maps show
the mean across every head and layer for the example's row or column context.
Special tokens are omitted visually but their mass is not renormalized into
the displayed data-data submatrix. Numerical summaries report their mass.

For the first four sampled queries and each axis, remove three observed donors
with the highest attention averaged over all layers/heads. Compare with three
random-removal replicates matched to the chosen donors' logarithmic positional
distance bands (nearest available band if necessary), and with random
same-group donors where at least three exist. All arms remove equally many
observed cells for a given query. Oracle-removal coverage is reported by the
saved rows and can be lower when same-group support is insufficient. Random
arms may overlap the attention arm, as expected under the null. Original
normalization statistics are frozen during all removals to avoid confounding
the effect with refitted scaling. Record both prediction shifts and changes
in hidden-query squared error; increases and decreases are retained.

## Metrics, uncertainty and audit

Primary RMSE pools all hidden cells within each matrix in raw units; this is
not the column-normalized metric used for MissBench. MAE and clean-signal RMSE
are also archived. Scores are summarized independently for each generator,
rank, noise and observation-rate condition. Two-sided 95% Student-t intervals
use the ten independent matrices, not individual cells. For attention,
heads/queries are averaged within each matrix before calculating intervals;
for removals, random replicates and queries/axes are likewise averaged first.
Paired model differences and attention-minus-random removal differences use
the same matrix seeds. There is no multiple-comparison correction or claim
that all conditions are independent. No predictive-distribution calibration
claim is made by this point-prediction study.

The audit regenerates every input and mask, rescores every saved prediction,
checks all artifact hashes and unchanged checkpoint identity, recomputes
attention summaries from stored weights, and verifies donor eligibility and
intervention error arithmetic. Protocol, source snapshots, hyperparameter
trials, warnings, per-matrix predictions and diagnostics remain on disk.

## Reproduction

From `/root/tabular`:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts:mcpfn
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
OUT=/mnt/matrixformer_scratch/synthetic_capabilities_20261103
python mcpfn/scripts/run_matrixformer_synthetic_study.py plan --output "$OUT"
python mcpfn/scripts/run_matrixformer_synthetic_study.py run --output "$OUT"
python mcpfn/scripts/run_matrixformer_synthetic_study.py audit --output "$OUT"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python mcpfn/scripts/synthetic_uncentered_svd_control.py --output "$OUT"
python mcpfn/scripts/report_matrixformer_synthetic_study.py --output "$OUT" \
  --paper /root/MatrixFormerPaper
```

Use a new output directory for a new plan; omit `plan` to resume. The runner
refuses a different checkpoint/protocol/source hash or an already-occupied
GPU. A stored failure must be investigated rather than overwritten. The
report refuses partial or smoke exports. Plotting uses the paper's shared
`scripts/figure_style.py`; all experiment/evaluation/plot code lives in this
research repository. No benchmark results elsewhere in the paper are changed.

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts:mcpfn python -m pytest -q \
  mcpfn/tests/test_matrixformer_synthetic_study.py \
  mcpfn/test_attention_diagnostics.py \
  mcpfn/tests/test_synthetic_low_rank_benchmark.py
```

## Completed results: November 3, 2026

The full sweep and the SVD control completed: 360/360 cases, 3,320 saved
method results, 60 attention cases, and 2,151 donor-removal evaluations.
All inputs regenerated, all method predictions rescored, all attention
statistics recomputed, and all intervention eligibility/arithmetic checks
passed. The checkpoint is unchanged; all captured predictions are bit-identical
to the normal inference path. The focused test suite passes 41 tests.

At 40% observations, mean raw RMSE is:

| Condition | MatrixFormer | Two-way K-NN | Oracle neighbors | Uncentered SVD |
| --- | ---: | ---: | ---: | ---: |
| Gaussian rank 2 | 0.0120 | 0.3307 | 0.3119 | 0.00000923 |
| Gaussian rank 4 | 0.0479 | 0.5819 | 0.4899 | 0.0000181 |
| Gaussian rank 16 | 0.8705 | 0.9064 | 0.7574 | 0.9491 |
| Clusters, noise SD .25 | 0.3179 | 0.2838 | 0.2673 | 0.3407 |
| Independent noise | 1.0390 | 1.0165 | Not applicable | 1.0829 |

These numbers expose limitations, not universal dominance. MatrixFormer is
much better than neighbor averaging on the lowest-rank factors, but a properly
configured exact solver is substantially more accurate on sufficiently observed,
exactly low-rank matrices. Two-way K-NN and known-group oracles win on the
clustered examples. Column mean reaches 1.0237 on independent noise, better
than MatrixFormer. The portable score summaries retain all confidence intervals
and every condition, not just the examples in this table.

At the preselected noisy-cluster attention setting, layer-six row-axis
same-group donor mass exceeds chance by `0.0909 ± 0.0166`. The paired absolute
prediction-shift increase from attention-selected versus random removal is
`0.0427 ± 0.0293`, but the paired squared-error increase is
`0.0136 ± 0.0295`, with an interval containing zero. A rank-16 factor diagnostic
has a larger error-removal effect (`0.0874 ± 0.0568`); all six settings remain
reported. These associations and interventions do not prove the network is
an oracle nearest-neighbor algorithm.

The paper now contains `sections/synthetic_results.tex` and
`sections/synthetic_results_appendix.tex`, with four shared-style figures and
portable results in `/root/MatrixFormerPaper/results/synthetic_capabilities/`.
The compiled paper is `/root/MatrixFormerPaper/main.pdf`.

Full artifacts: `/mnt/matrixformer_scratch/synthetic_capabilities_20261103/`.
