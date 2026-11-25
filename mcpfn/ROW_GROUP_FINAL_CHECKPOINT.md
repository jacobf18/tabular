# Single-matrix row-group probe with the final checkpoint

The paper's row-group figure uses the final 250,000-additional-matrix checkpoint.
The seed-zero 64-by-64 matrix, four balanced row groups, noise SD 0.25,
1,638 observed entries, normalization, clustering and plotting settings are
unchanged from the previous example. There are no column groups, repeated
simulations or confidence intervals. K-means uses four clusters in the full
768-dimensional mean-pooled row representation; PCA is for display only.

The measured ARI is -0.016150 before attention, -0.016150 at layer 2,
0.836096 at layer 4, and 1.000000 at layers 5 through 8. Hooked and unhooked
predictions are bit-identical and the model parameters are unchanged.

Full artifacts, including inputs, features, predictions and clustering
assignments, are saved in
`/mnt/matrixformer_scratch/row_group_embedding_final_seed0_20261125`.
The paper retains a compact record in `results/row_group_final/` and the
two-panel PDF and PNG in `figures/row_group_example_panels.*`.

## Reproduce

Use a new output directory; the runner refuses to overwrite an experiment.
Changing checkpoints requires both the path and its expected SHA-256, so the
historical default cannot silently substitute for the final model.

```bash
cd /root/tabular/mcpfn
export PYTHONPATH=src:scripts MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT=/path/to/new/row_group_final
CHECKPOINT=/mnt/matrixformer_scratch/scientific_prior_training_20261108/million_v1/snapshots/matrices_0250000/checkpoint.pth
python scripts/run_row_group_embedding_example.py run --output "$OUT" --device cuda \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 933850d7941afe58e5bdf838424ec7160cef9764fbcb26c6462c61c5648a3859
python scripts/run_row_group_embedding_example.py render-matrix --output "$OUT" --colorbar left
python scripts/assemble_row_representation_panels.py \
  --top "$OUT/row_group_matrix_arrows.pdf" --bottom "$OUT/row_group_embeddings.pdf" \
  --output "$OUT/row_group_example_panels" --width 396 --gap 2 --trim-inner \
  --top-content 'Construction of the single matrix with four row groups and no column groups' \
  --bottom-content 'Mean-pooled row embeddings and ARI for the same matrix at layers 0 through 8'
python -m pytest -q tests/test_row_group_embedding_example.py \
  tests/test_synthetic_embedding_filmstrip.py tests/test_row_representation_panels.py
```

`manifest.json` describes the initial inference and rendering. The subsequent
left-colorbar rendering and panel assembly have their own JSON receipts;
`publication_validation.json` records the final artifact hashes and confirms
that every input array matches the earlier fixed experiment.
