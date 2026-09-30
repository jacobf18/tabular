# Streamed rating inference

`src/tabimpute/streamed_inference.py` implements inference-only streaming for
the stable MatrixFormer checkpoint. It microbatches encoding, independent
attention sequences, feed-forward cells, and output-distribution decoding.
Attention still spans the entire other axis; this is not smaller attention
windows or a new model. Weights remain unchanged. The hidden grid stays on GPU.

`StreamedImputePFN` is a standalone adapter, so these scripts do not depend on
uncommitted research changes to the general imputation interface. It supports
one context chunk, no preprocessing ensemble, and no absolute position embeddings.
The decoder preserves the original strided layout to avoid BF16 fused-bias
rounding changes. Its cell batch size is a target: it decodes at least two rows
when available, with an overlapping tail if necessary.

## Validated results

On the H200, a 2000 × 1000 block drops from 93.48 GiB peak allocated memory
to 6.86 GiB. Maximum prediction difference against dense inference is
1.05e-6 rating units. Blocks with 4000, 8000, and all 9976 Netflix users fit
at 13.35, 26.34, and 32.75 GiB respectively, holding item width at 1000.

With the same fixed checkpoint and frozen Netflix small split:

| Normalization | Rows | RMSE | MAE |
|---|---:|---:|---:|
| Global | 2000 | 0.934119 | 0.801749 |
| Global | 9976 | 0.906991 | 0.781905 |
| Column-wise | 2000 | 0.933048 | 0.800095 |
| Column-wise | 9976 | 0.905849 | 0.779545 |

MovieLens 100K is unchanged at reported precision. Scores are means of
per-user errors, on the original rating scale, after clipping. They are not
pooled RMSE or NRMSE. This is a previously inspected split, not a blind test.
The initial streaming prototype failed its parity gate; correcting tensor
layouts resolved the discrepancy without loosening that gate. Its artifacts
remain available, alongside the successful probe and held-out scoring audits.

Artifacts are under `runs/benchmark_evaluations/` in
`matrixformer_streaming_layout_probe_20260930/` and
`matrixformer_recsys_streamed_allusers_20260930/`. Datasets, checkpoints, and
raw predictions are local artifacts, not checked into Git. The checkpoint is
the step-98000 `best_validation_checkpoint.pth` from
`tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5`,
SHA256 `1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d`.

## Run

From `mcpfn`, using the project Python environment (PyTorch with CUDA, NumPy,
SciPy, Cornac, and pytest), plus the local checkpoint and frozen splits:

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 pytest -q tests/test_streamed_inference.py tests/test_matrixformer_recsys_ratings.py tests/test_large_recsys.py
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled \
  python scripts/probe_matrixformer_streaming.py --output runs/benchmark_evaluations/probe_repeat
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled \
  python scripts/evaluate_matrixformer_recsys_ratings.py --streamed --max-num-rows 9976 \
  --output runs/benchmark_evaluations/allusers_repeat
```

## Larger benchmarks

`scripts/evaluate_large_recsys.py` evaluates MovieLens 1M and 10M, retaining
every training user and item. Both global and column-wise normalization use
training observations only. Fixed 10000 × 1000 maximum user–item windows
are selected before scoring, with overlapping-tail averaging and no model
updates. A sparse training matrix supplies each block, so the full dense
user–item prediction matrix is never allocated or saved.

The runner uses Cornac's 80/20 rating-entry split, seed 123, excluding unknown
users/items. It also fits MF, PMF, SVD, BPR, UserKNN, and ItemKNN with the
previous harness's hyperparameters. BPR is ranking-oriented; its rating errors
do not measure ranking quality. Split arrays, data/configuration hashes,
held-out predictions, query coverage, and independent rescoring audits are
saved per case. Errors remain explicit. No ranking metrics are run.

Cache `ratings.dat` from the official GroupLens `ml-1m.zip` and `ml-10m.zip`
archives under `~/.cornac/ml-1m/` and `~/.cornac/ml-10M100K/` respectively.
The runner requires these files and does not download data implicitly. Only
ratings are needed; movie metadata and tags need not be extracted.

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=disabled \
  python scripts/evaluate_large_recsys.py --datasets 1M 10M
```

The large runner can resume with the same command; it validates frozen splits
and artifact hashes, skips completed cases, and refuses changed code or
protocol. A lock prevents duplicate processes. Each dataset has eight cases;
`status.json` reports the active method and neural block progress, while
`results.csv` contains only completed outcomes. Failed cases are retained,
not automatically retried. The 10M suite can take substantially longer than
the smaller datasets. Nothing is selected by test error.
