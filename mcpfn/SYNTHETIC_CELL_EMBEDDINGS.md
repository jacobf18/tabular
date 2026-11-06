# Individual-cell embedding probes

This experiment tests individual cell vectors within fixed columns. The
mirrored fixed-row experiment checks column-group recovery, including the
other block axis. Following the November 6 retrospective geometry diagnosis,
these are appendix controls; the main illustration returns to mean pooling,
now for both row and column embeddings. Pooled scores must not be described
as cell-level recovery. See `SYNTHETIC_EMBEDDING_GEOMETRY.md` for the diagnosis.

## Frozen protocol

- Use the unchanged combined MatrixFormer checkpoint and standard orientation.
- Reuse all 120 clustered matrices from
  `/mnt/matrixformer_scratch/synthetic_capabilities_20261103`: 64 rows by
  64 columns, four balanced groups per axis, noise SD 0/0.25/1,
  observed fraction 0.2/0.4/0.6/0.8, ten matrix seeds per condition.
- Extract each 768-dimensional cell vector immediately after the encoder and
  after transformer block eight; exclude special rows and columns.
- Within **every column**, cluster the 64 row-cell vectors and score against
  row memberships. Within **every row**, cluster the 64 column-cell vectors
  and score against column memberships. There is no pooling.
- Fit K-means with K=4, 20 initializations, Lloyd updates, maximum 300
  iterations, and seed 20261104. Do not rescale features or use PCA before
  fitting. Membership labels are unavailable to clustering and inference.
- Primary ARI includes all observed and masked cells. Secondary masked-only
  ARI uses the same all-cell assignments, without refitting; require at least
  two masked cells and two true masked groups.
- First average the 64 slice ARIs within each matrix. Compute means and
  two-sided 95% Student-t intervals across the ten matrices per condition.
  Before/after differences are paired by matrix. Contexts are not independent
  replicates; conditions share seeds and masks.
- Verify hooked, fresh unhooked, and archived predictions agree bit-for-bit
  for every matrix; check that the model parameter version counters do not
  change. Independently rescore all saved assignments during the audit.

The full experiment has 30,720 K-means fits and 61,440 score records (two
scoring scopes). No parameters are trained or selected.

## Predetermined illustration

Reuse construction example `7f7a39e928ff6b43`: seed zero, noise SD 0.25,
1,638 of 4,096 cells observed. Before clustering, select the first
original-order column in each true column group, and the first row in each
true row group. The one-based column indices are 8, 5, 2, 1; row indices
are 2, 1, 3, 4. These are not the highest-scoring contexts.

Each panel displays all 64 cell vectors using independently fitted,
label-blind two-dimensional PCA. Colors and shapes show the memberships
being recovered. ARI is evaluated in full feature space, not the PCA plane.
No jitter is added; coincident vectors overlap. The processing arrow is not
a claim about individual point trajectories or an attention-only intervention.

## Artifacts and reproduction

The complete run is saved at
`/mnt/matrixformer_scratch/cell_embeddings_20261105`. It includes frozen
source snapshots, protocol, checkpoint and input hashes, full cell grids,
assignments, prediction-parity checks, per-slice scores, per-matrix scores,
condition summaries, paired differences, and the independent rescore audit.

Run from `/root/tabular` with a fresh output directory for a new campaign:

```bash
export PYTHONPATH=mcpfn/scripts:mcpfn/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
output=/mnt/matrixformer_scratch/cell_embeddings_NEW_RUN
python mcpfn/scripts/run_synthetic_cell_embeddings.py plan --output "$output"
python mcpfn/scripts/run_synthetic_cell_embeddings.py run --output "$output" --jobs 4
python mcpfn/scripts/run_synthetic_cell_embeddings.py audit --output "$output"
python mcpfn/scripts/report_synthetic_cell_embeddings.py \
  --source "$output" \
  --destination /root/MatrixFormerPaper/results/synthetic_cell_embeddings
python -m pytest -q mcpfn/tests/test_synthetic_cell_embeddings.py \
  mcpfn/tests/test_synthetic_cluster_representations.py
```

The paper has portable score CSVs, predetermined raw cell vectors and
assignments, provenance, and a plot manifest in
`/root/MatrixFormerPaper/results/synthetic_cell_embeddings`. Regenerate figures:

```bash
cd /root/MatrixFormerPaper
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python scripts/plot_cell_embedding_examples.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Outputs: `figures/synthetic_cell_columns.pdf` (appendix),
`figures/synthetic_cell_rows.pdf` (mirrored appendix), and
`figures/synthetic_cell_recovery.pdf` (all conditions, appendix), plus PNGs.
The old row-only pooled scatter is archived; the main figure instead uses
the two-axis `figures/synthetic_mean_embedding_axes.pdf`.

## Interpretation

At noise SD 0.25 and 40% observations, mean all-cell ARI rises from
0.075 ± 0.011 to 0.279 ± 0.012 for row groups and from 0.057 ± 0.012
to 0.264 ± 0.012 for column groups (95% Student-t intervals across ten
matrix means). Secondary masked-cell ARIs rise from zero to 0.653 ± 0.040
and 0.612 ± 0.036 using the same all-cell assignments. These are partial,
not near-perfect, recoveries. There are regressions: with no noise and
80% observed, all-cell row ARI falls from 0.507 to 0.073 and column ARI
from 0.493 to 0.046. All conditions, gains, and regressions are retained.

This tests membership recovery with a supplied group count, not discovery of
the number of groups. Raw cell embeddings can encode cell value, observation
status, and other information in addition to group identity. Pooling and
single-cell clustering answer different questions. A more visually separated
PCA projection is not evidence of perfect full-space clustering or a causal
account of imputation. The study concerns small clustered MCAR matrices;
it does not establish robustness on arbitrary real tables.
