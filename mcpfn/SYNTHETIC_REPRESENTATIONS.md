# MatrixFormer: attention examples and clustered embeddings

This follow-up uses the frozen single combined MatrixFormer checkpoint and
the clustered inputs from `SYNTHETIC_CAPABILITIES.md`. It does not train,
fine-tune, select a checkpoint, transpose, or use true groups in inference.

## Artifacts

Final campaign:
`/mnt/matrixformer_scratch/cluster_representations_actual_qk_20261103/`

- `protocol.json`, `manifest.json`, `audit.json`: registered settings, source
  and parent-input hashes, and independent rescoring.
- `cases/<id>/features.npz`: encoder and all eight block representations.
- `cases/<id>/clustering.npz`: fitted group assignments, label-blind PCA
  coordinates, and explained-variance fractions.
- `cases/<id>/metrics.csv`: each clustering result; no pooling of cells.
- `cases/<id>/attention.npz`: all-head, all-layer attention for fixed examples.
- `report/`: portable per-matrix results, confidence intervals, paired
  comparisons, and a compact fixed-example export.
- `figures/`: PDF/PNG figures and a 32-page full-head PDF gallery.
- `paper_figures/`: simplified figures, each answering one question; the
  dense diagnostic grids above are not inserted into the manuscript.

The earlier `cluster_representations_20261103` directory is **superseded**.
Its provisional attention reconstruction reran projections on smaller
BF16 batches, producing small rounding differences. The final campaign
reads the actual production Q/K outputs instead and reruns every case;
it does not silently reuse that provisional campaign's artifacts.

## Findings

All 120 cases and 5,280 clustering results passed the audit. Captured
predictions are bit-identical to unhooked and parent predictions throughout.
At 40% observed and noise SD 0.25, mean-pooled final embeddings recover row
groups with ARI **0.996 ± 0.010** and column groups with **0.987 ± 0.015**.
Observed values plus mask give **0.932 ± 0.090** and **0.713 ± 0.131**.
These are means ± two-sided 95% intervals across ten matrices, not cells.

There are important counterexamples:

- At noise SD 1 and 20% observed, final mean-pooled row/column ARI falls
  to **0.348 ± 0.168 / 0.164 ± 0.077**.
- On noiseless columns at 40% observed, mean pooling gives only
  **0.420 ± 0.107**, versus **0.942 ± 0.052** for full cell profiles.
  Completed values recover the groups perfectly. This is a pooling failure,
  not evidence that the model cannot complete those matrices.
- Recovery is **not monotone in noise** for mean-pooled representations:
  adding a little noise improves the column diagnostic. No claim that
  noise universally helps or hurts the model follows.
- Good embedding clusters do not overturn the earlier finding that ordinary
  two-way K-NN beats MatrixFormer at 40%, 60%, and 80% observed on this family's
  moderate-noise imputation RMSE; MatrixFormer has lower mean error at 20%.
  Clustering quality and imputation optimality are different questions.

The main paper now focuses on nearest-neighbor completion and embeddings
rather than attention maps or broad capability contours. The Gaussian rank, solver,
and independent-noise plots are no longer included in either the main text
or appendix; all scores, figures, and reproduction code remain archived.
`synthetic_neighbors.pdf` compares MatrixFormer, ordinary two-way K-NN,
oracle neighbors, and oracle block means at 20% and 40% observed, noise SD
0.25. It retains both directions of the MatrixFormer/K-NN ordering, with
two-sided 95% intervals across ten matrices per rate. The oracles know group
memberships, never missing targets. `neighbor_comparison_summary.csv` records
the eight unchanged condition-method summaries; the plot rejects incomplete
method/rate coverage. All four observation rates remain in the parent scores.

Attention heatmaps, layer galleries, enrichment statistics, and donor-removal
discussion are no longer included in the main text or appendix; their data,
figures, and reproduction code remain archived. The before/after embedding
example remains in the main text,
with arrows linking the same eight rows (the first eight indices) between
the encoder and final representations. All 64 rows remain visible. The
unchanged PCA fits are independent, so arrows show identity correspondence,
not common-coordinate displacement or measured intermediate trajectories.
Open markers denote the encoder; filled markers denote the final layer.
`paper_figures/embedding_identity_links.csv` records the exact row IDs and
unchanged endpoint coordinates for auditing. All example choices retain the
previously fixed seed-zero/noise-0.25/40%-observed setting. Individual heads
and both poolings remain in the archive; the pooling failure is explicitly
a discovered counterexample, not a new selection rule.

## Design and interpretation

The 120 cases cover 64-by-64 matrices, four balanced shuffled groups on
each axis, noise SD 0, 0.25, or 1, and observation fractions 0.2, 0.4,
0.6, or 0.8. There are ten matrix seeds per condition. Conditions share
latent matrices and nested masks, so the 120 cases are not 120 independent
datasets. Uncertainty is computed separately within each condition.

Each cell has a 768-dimensional representation. Read-only forward hooks
capture the encoder (layer 0) and all eight transformer blocks. We exclude
the 12 special rows and columns and construct two predefined representations
of each row and column:

1. **Mean embedding**: average the cell vectors over the opposite data axis,
   including observed and missing positions.
2. **Cell-profile embedding**: concatenate the cell vectors over the opposite
   axis. A full centered Gram eigendecomposition compresses this to at most
   64 coordinates, preserving every Euclidean pairwise distance up to
   floating-point roundoff. This is not a selected/truncated PCA projection.

Mean pooling can discard different cluster profiles with similar averages;
the profile representation retains their coordinate structure. Both are
reported at every layer. Neither representation uses true memberships.
Inference sees the original shuffled matrices.

We fit K-means with **known K=4**, 20 initializations, and one fixed random
seed. The known cluster count is privileged information supplied equally
to all representations; this does not establish automatic discovery of K.
Labels are used only afterward to compute adjusted Rand index (ARI), NMI,
and top-five-neighbor purity. ARI is one for perfect recovery and approximately
zero for chance, and can be negative. We do not select a layer by its ARI.

Comparisons use the same K-means settings:

- Observed-column-standardized, zero-filled values concatenated with the
  observation mask. One mask coordinate has unit scale; no test-tuned weight.
- Mask-only features, a negative control.
- Model-completed values, standardized using the original observed statistics.
  These retain true observed entries, not the model's estimates of them.
- Complete noisy values, an explicitly **privileged oracle** that sees hidden
  entries. This is an explanatory comparator, not an eligible imputation method.
- Twenty fixed permutations of true labels score chance agreement without
  refitting or selecting clusters.

K-means uses the full representations, with no feature-wise restandardization.
Two-component PCA is fitted without labels **only for visualization**. Colors
indicate true groups. In the detailed diagnostic grid, marker shapes indicate
independently fitted, unordered K-means groups; those IDs do not have consistent
semantics across panels. The main paper's arrow figure instead uses open versus
filled markers for encoder versus final-layer representations.

Means and two-sided 95% Student-t intervals use the ten independent matrices
within each noise/rate/axis/layer condition. The paired comparisons first
subtract the observed-input ARI within each matrix. No multiplicity correction
or confirmatory hypothesis testing is performed across the exploratory grid.

## Attention examples

Examples are fixed at seed 0, 40% observed entries, and all three noise levels.
The detailed gallery uses noise SD 0.25, as did the earlier capability study.
It contains every head of every layer on both axes, for both the first
previously sampled missing-query context and the average over all 64 data
contexts. No head, layer, seed, or query is chosen for visual clarity.

Hooks read the production Q/K projection outputs, apply the original RoPE,
and reconstruct softmax probabilities in FP32. They do not enable native
attention capture or replace the fused production kernel. Attention averaged
over data contexts excludes special contexts, but its **keys retain all
special tokens**. Displayed maps omit special keys without reallocating
their mass to data keys. Gold boundaries indicate true groups, sorted only
for display. Square-root color scaling makes low probabilities visible;
the colorbar is labeled in actual probabilities, not rescaled similarities.

The layer summary averages all heads and contexts, separately for each layer
and axis. Full-head pages retain individual heads. A band or block in these
maps is an association, not proof that the model implements nearest neighbors.
The controlled donor-removal results in `SYNTHETIC_CAPABILITIES.md` are a
separate experiment.

Every hooked forward pass must match both a fresh unhooked prediction and
the parent experiment prediction **bit-for-bit**, and parameter version
counters must remain unchanged. The audit regenerates inputs, checks saved
artifact hashes, recomputes all clustering metrics from saved assignments
and features, and checks complete layer/head coverage and attention row sums.

## Reproduction

From `/root/tabular`:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts:mcpfn
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
OUT=/mnt/matrixformer_scratch/cluster_representations_actual_qk_20261103

# Planning requires a new output directory; choose a new OUT to rerun.
python mcpfn/scripts/run_synthetic_cluster_representations.py plan --output "$OUT"
python mcpfn/scripts/run_synthetic_cluster_representations.py run --output "$OUT"
python mcpfn/scripts/run_synthetic_cluster_representations.py audit --output "$OUT"
python mcpfn/scripts/report_synthetic_cluster_representations.py --output "$OUT"
python mcpfn/scripts/plot_synthetic_paper_figures.py --output "$OUT"

python -m pytest -q mcpfn/tests/test_synthetic_cluster_representations.py
```

Full per-cell grids are not retained: pooled features and distance-preserving
profile coordinates suffice for all reported clustering analyses. Large
artifacts stay on the scratch disk; the paper receives portable summaries,
the fixed example, and figures. This is a small related-family MCAR probe,
not evidence of universal semantic clustering, calibrated uncertainty, or
out-of-distribution robustness.
