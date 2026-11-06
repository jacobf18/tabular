# Retrospective embedding-geometry diagnosis (2026-11-06)

## Decision

Restore the **mean-pooled row and column** representation illustration,
not a cleaner synthetic dataset. Keep the original seed-zero matrix, noise
SD 0.25, 40% mask, checkpoint, colors/shapes, full-space clustering, and
label-blind PCA coordinates. The main figure now tests both axes. Raw cell
results, including failures, remain in the appendix alongside full profiles.

This is a retrospective diagnostic/presentation choice after the cell-level
results, not a new preregistered test or a change to benchmark predictions.

## Evidence

All 120 frozen cases and all 30,720 context/layer/axis combinations are used.
The diagnostic reads raw saved grids; it never calls the model or fits a
clustering algorithm. `FeatureEncoderStable` uses a missing-token path and
adds a missingness embedding, so observed and masked entries do not have to
share geometry even when they belong to the same row or column group.

Within each context, the between-partition sum of squared distances divided
by total scatter measures association with observation status, true group,
or their joint partition. The fractions overlap and are not additive or
causal. True memberships enter this diagnostic only. First average context
fractions within a matrix; use Student-t intervals across ten matrices.

At noise SD 0.25 and 40% observed, status explains 0.322 ± 0.015 of final
row-cell scatter within fixed columns and 0.354 ± 0.024 of column-cell
scatter within fixed rows. True groups explain 0.402 ± 0.034 and
0.355 ± 0.027; joint partitions explain 0.830 ± 0.041 and 0.813 ± 0.030.
Observation status is therefore a substantial competing source of geometry,
not simply noise that vanishes at the final layer.

The original mean-pooled row ARI improves from 0.382 ± 0.122 to
0.996 ± 0.010; column ARI improves from 0.002 ± 0.013 to 0.987 ± 0.015.
The same archived individual-cell scores are much lower (final 0.279 and
0.264). Full profiles confirm recovery without averaging across blocks
(final row/column ARI 0.992 ± 0.013 / 1.000 ± 0.000).

Reducing noise from 0.25 to zero at 40% observed does **not** consistently
help. Paired final raw-cell ARI changes by +0.022 ± 0.015 for rows and
−0.056 ± 0.016 for columns. Mean-pooled noiseless column ARI is only
0.420 ± 0.107 versus full-profile ARI 0.942 ± 0.052. High-noise sparse
matrices remain difficult even for the aggregated representations.

## Reproduction

Saved diagnostic: `/mnt/matrixformer_scratch/embedding_geometry_20261106`.
Portable paper export:
`/root/MatrixFormerPaper/results/synthetic_embedding_geometry`.

From `/root/tabular`, with a fresh directory for a new diagnostic:

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python mcpfn/scripts/diagnose_synthetic_embedding_geometry.py \
  --output /mnt/matrixformer_scratch/embedding_geometry_NEW_RUN
PYTHONPATH=mcpfn/scripts:mcpfn/src python -m pytest -q \
  mcpfn/tests/test_synthetic_embedding_geometry.py \
  mcpfn/tests/test_synthetic_cell_embeddings.py \
  mcpfn/tests/test_synthetic_cluster_representations.py
```

The script checks archived grid/input hashes and performs no refitting.
Per-context, per-matrix, and summary outputs include all three noise levels,
all four observation fractions, and both axes/layers. Noise differences are
paired by seed; conditions are not independent tests. Source/artifact hashes
and an explicit retrospective protocol are saved before/after analysis.

From `/root/MatrixFormerPaper`:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/plot_pooled_embedding_axes.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figures (PDF and PNG):

- `figures/synthetic_mean_embedding_axes`: main two-axis pooled example.
- `figures/synthetic_profile_embedding_axes`: opposite-axis-preserving control.
- `figures/synthetic_representation_recovery`: all conditions and all three
  representations at block eight, with matrix-level confidence intervals.

The plotter validates full-space ARIs from saved assignments and uses the
original PCA coordinates, without jitter, subsampling, or selecting better
axes. Run this new plotter last if also regenerating old figures, because
the historical plotters update the same manuscript figure inventory.
