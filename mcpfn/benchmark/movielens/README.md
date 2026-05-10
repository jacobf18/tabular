# benchmark/movielens — matrix-completion benchmark harness

Reproducible harness for the matrix-completion experiments in
*Nearest Neighbors for Matrix Estimation Interpreted as Blind Regression for
Latent Variable Model* (Lee, Li, Shah, Song; [arXiv:1705.04867](https://arxiv.org/abs/1705.04867)).

Includes the paper's User-Item Gaussian Kernel estimator (Algorithm 4),
user-user CF, item-item CF, SoftImpute, and an adapter for the local
TabImpute models so they can be evaluated on the same matrices.

## Contents

Lives at `benchmark/movielens/` (handles both MovieLens and Netflix; the
folder name reflects the primary dataset).

```
benchmark/movielens/
├── configs/
│   ├── datasets.yaml            # URLs, encodings, kaggle slugs
│   ├── experiments.yaml         # replicates, withhold fractions, lambdas, chunk sizes
│   └── tabimpute_models.json    # normalized tabimpute model manifest
├── src/
│   ├── data/                    # download/parse/subsample/chunk
│   ├── methods/                 # paper_gaussian_kernel + 3 baselines + tabimpute_adapter
│   ├── eval/                    # drivers + aggregation + smoke test
│   └── utils/                   # paths, logging, sparse I/O, splits
├── data/{movielens,netflix}/<size>/
│   ├── raw/ extracted/          # archive + extracted inputs
│   ├── processed/
│   │   ├── full/                # normalized sparse matrix + ratings table
│   │   ├── paper_subsamples/    # 100 × 2000×2000 sampled submatrices
│   │   └── chunked_50x10/       # pre-chunked full matrix (optional)
│   └── manifests/               # dataset statistics
└── outputs/{metrics,predictions,plots,logs}/
```

## Install

Requires `numpy`, `scipy`, `matplotlib`, `torch` (for the TabImpute adapter).
Optional: `pandas`/`pyarrow` for parquet output, `fancyimpute` for a
maintained SoftImpute, `PyYAML` for the yaml configs (a small fallback
parser is bundled).

```
pip install numpy scipy matplotlib pandas pyarrow torch fancyimpute pyyaml kaggle
```

Most of the harness works without a GPU. TabImpute inference will be much
faster on CUDA.

## Quick smoke test

Builds a tiny synthetic low-rank matrix under `data/movielens/1m_smoke/`,
runs all four paper baselines, aggregates, and plots — no external
downloads required.

```
cd benchmark/movielens
python src/eval/smoke_test.py --clean
```

To include the TabImpute adapter on the smoke path (must be able to load
a checkpoint — see *TabImpute checkpoints* below):

```
python src/eval/smoke_test.py --clean \
    --include-tabimpute --model dwip-v2-1 \
    --checkpoint-override /root/tabular/mcpfn/src/tabimpute/workdir/dwip_v3r1_t002_ctl-trial_001/checkpoint_17000.pth \
    --device cpu
```

## Full pipeline

1. **Download data.** MovieLens 1M is fetched automatically. Netflix Prize
   is attempted via the `kaggle` CLI (requires a configured
   `~/.kaggle/kaggle.json` and accepting the dataset terms once). If that
   fails, the raw directory gets a `MANUAL_DOWNLOAD_README.md` with
   manual instructions.

   ```
   python src/data/download_datasets.py --dataset all
   ```

2. **Parse into normalized sparse matrices.**

   ```
   python src/data/parse_movielens.py --size 1m
   python src/data/parse_netflix.py --size prize
   ```

   Writes `processed/full/matrix.npz` + `user_ids.npy` + `item_ids.npy`,
   a ratings table (`ratings.parquet` if pandas is available, else
   `ratings.npz`), and `manifests/full.json`.

3. **Build the 100 × 2000×2000 paper subsamples.**

   ```
   python src/data/build_paper_subsamples.py --dataset movielens --size 1m \
       --num-reps 100 --rows 2000 --cols 2000 --seed 123
   python src/data/build_paper_subsamples.py --dataset netflix --size prize \
       --num-reps 100 --rows 2000 --cols 2000 --seed 123
   ```

   Each replicate lives under `processed/paper_subsamples/subsample_XXXX/`
   with its own `matrix.npz`, index arrays, and `meta.json`. Sampling
   draws from a densest-K pool of users and items so the slices are not
   pathologically sparse; the derivation is documented in
   `src/data/build_paper_subsamples.py`.

4. **(Optional) Build chunked representation of the full matrix.**

   ```
   python src/data/build_chunks.py --dataset movielens --size 1m \
       --row-chunk 50 --col-chunk 10
   ```

5. **Run the paper baselines.** Caches completed (method, rep, withhold)
   triples in `outputs/metrics/<dataset>_<size>_paper.csv` — re-running is
   incremental unless `--force` is passed.

   ```
   python src/eval/run_paper_benchmarks.py --dataset movielens --size 1m
   python src/eval/run_paper_benchmarks.py --dataset netflix --size prize
   ```

   Available methods: `paper_gaussian_kernel`, `user_user_cf`,
   `item_item_cf`, `softimpute`. Pass `--methods a,b,c` to subset.

6. **Run TabImpute on the subsamples.**

   ```
   python src/eval/run_tabimpute_eval.py --dataset movielens --size 1m \
       --model dwip-v2-1 --eval-mode auto
   python src/eval/run_tabimpute_eval.py --dataset netflix  --size prize \
       --model dwip-v2-2 --eval-mode auto
   ```

   `--eval-mode` can be `full`, `chunked`, or `auto`. In `auto` the
   adapter picks `full` only when the matrix fits under
   `tabimpute.full_rows_limit` / `tabimpute.full_cols_limit` in
   `experiments.yaml` (400 × 400 by default), otherwise it uses chunked.
   It also falls back from `full` to `chunked` on CUDA OOM.

7. **Run TabImpute on the full matrix in chunked mode (extension).**

   ```
   python src/eval/run_tabimpute_eval.py --dataset movielens --size 1m \
       --model dwip-v1 --input full --eval-mode chunked \
       --row-chunk 50 --col-chunk 10
   python src/eval/run_tabimpute_eval.py --dataset netflix  --size prize \
       --model dwip-v2-1 --input full --eval-mode chunked \
       --row-chunk 50 --col-chunk 10
   ```

8. **Aggregate and plot.**

   ```
   python src/eval/aggregation.py --all
   ```

   Produces:
   - `outputs/metrics/aggregate_paper.csv` / `aggregate_tabimpute.csv`
   - `outputs/plots/<dataset>_<size>_paper_rmse.png`
   - `outputs/plots/<dataset>_<size>_tabimpute_rmse.png`
   - `outputs/plots/<dataset>_<size>_paper_plus_tabimpute.png`

## Paper method notes (Algorithm 4)

The implementation in `src/methods/paper_gaussian_kernel.py` mirrors the
real-data variant used in §5 of the paper:

* Row dissimilarity is the unbiased sample variance of
  `Z[u, Q] − Z[v, Q]` over the observed-column overlap `Q_row(u, v)`.
* Column dissimilarity is the analogous quantity for items.
* Combined dissimilarity is `D = max(s_row, s_col)`.
* Gaussian kernel weight is `exp(−lambda · D)` (this is the β=2 case; the
  general β pre-exponentiates `D` by `β/2`). `beta` is preserved as a
  parameter in case you want to sweep it.
* First-order estimate
  `Ẑ(u,i) = Σ w (Z(v,i) + Z(u,j) − Z(v,j)) / Σ w`.
* Lambda schedule per dataset × withheld fraction is in
  `experiments.yaml`. Override via
  `run_paper_benchmarks.py --lambda-override 1.2`.

**Approximation for speed.** Pairwise dissimilarities are computed in
closed form via three dense matmuls. For each withheld entry we restrict
to the top-`k_neighbors` candidate users (by row dissim) and top-`k`
candidate items (by col dissim). `k_neighbors` defaults to 100, which for
the 2000×2000 subsamples is effectively equivalent to the exact method
since the observed-row/column counts are typically in the 40–300 range.

## TabImpute checkpoints

`configs/tabimpute_models.json` is the normalized manifest. The
user-facing names used on the CLI are:

| normalized  | original_key       | architecture            | run_name              |
|-------------|--------------------|-------------------------|-----------------------|
| `dwip-v2-1` | `dwip_v2_t002`     | `model_new_stable.py`   | `dwip_v2-trial_002`   |
| `dwip-v2-2` | `dwip_v2_t003`     | `model_new_stable.py`   | `dwip_v2-trial_004`   |
| `dwip-v1`   | `dwip_v1_local14`  | `model_new.py`          | `dwip_v1-trial_003`   |

All three point at checkpoint paths under `/mnt/checkpoints_sweep/workdir/…`.
**Those directories did not exist when this harness was built.** The
adapter fails fast with a clear message in that case. To evaluate against
an alternative (e.g. a `dwip_v3r1_*` checkpoint you have locally), pass
`--checkpoint-override /path/to/checkpoint_NNNNN.pth`. The adapter will
load the training-time `config.json` from the checkpoint directory if
present.

## Chunked evaluation

TabImpute is trained on matrices roughly 10–50 rows × 5–50 cols, so
evaluating on a whole 2000×2000 (or full-dataset) matrix in one forward
pass is out of distribution and memory-expensive. Chunked mode splits the
input into non-overlapping `row_chunk × col_chunk` blocks (default
50 × 10), runs TabImpute on each independently, and stitches the
predictions back together. Overhead is controlled by the existing
chunking path inside `ImputePFN` (`max_num_rows` / `max_num_cols`).

`auto` mode picks `full` only for matrices that fit within
`full_rows_limit × full_cols_limit` (from `experiments.yaml`); otherwise
it uses `chunked`. It also catches CUDA OOM and retries in `chunked`
mode.

## Fallback hierarchy when a method cannot score an entry

1. Row mean ⊕ column mean (both observed).
2. Row mean (only rows observed).
3. Column mean (only columns observed).
4. Global observed mean.

Counts are logged as `fallback_count` in the per-run CSVs.

## Metrics and reporting

Primary metric is RMSE on withheld entries. Aggregation reports
`mean_rmse`, `std_rmse`, `n`, `ci95_low`, `ci95_high` per
(dataset, size, method, withhold). The 95% CI uses the normal
approximation `mean ± 1.96 · σ / √n`, which matches what the paper shows.

## Known blockers and caveats

* **Netflix acquisition.** Netflix has not distributed the Prize dataset
  itself since 2010. The harness tries `kaggle` first (requires auth +
  accepting terms on the Kaggle UI). If that fails, `raw/` gets a
  `MANUAL_DOWNLOAD_README.md`; the pipeline stops at step 2 until the
  user places `combined_data_*.txt` files in `raw/`.
* **TabImpute checkpoints.** The three absolute paths in the shipped
  manifest do not exist on the current machine. The three
  `dwip_v3r1_*_ctl-trial_001/checkpoint_17000.pth` files under
  `src/tabimpute/workdir/` are compatible architectural replacements
  (same `model_new_stable.py` family). Use `--checkpoint-override` until
  the real `dwip_v2-trial_00X` / `dwip_v1-trial_003` checkpoints are
  available.
* **Subsample-sampling bias.** The paper does not specify exactly which
  users/items it subsamples from the 6 000×3 700 MovieLens or 480 k×17 770
  Netflix matrix. Drawing uniformly produces near-empty 2000×2000 slices,
  so the builder biases toward the densest `oversample × sample_count`
  pool. This matches the follow-up literature's common practice but may
  differ slightly from the original paper's subsampling.
* **Paper method speed.** `k_neighbors=100` is an explicit approximation.
  At the paper's 2000×2000 scale it is equivalent to the exact method for
  the vast majority of withheld entries. For smaller matrices with very
  dense rows/columns, raise `k_neighbors`.

## Assumed TabImpute model interface

The adapter uses `tabimpute.interface.ImputePFN` from the local repo,
specifically the `entry_wise_features=False` path that takes a
`json_config` describing the model and a `checkpoint_path`, and supports
row/column chunking via `max_num_rows` / `max_num_cols`. Key assumptions:

* Checkpoints contain a top-level `model` state dict (matches `.model`
  attribute of `TabImputeModelNew` / `TabImputeModelStable`).
* A `config.json` next to the checkpoint contains a `model: { … }`
  block with the exact hyperparameters used at training time. If it is
  missing, a conservative default (`BEST_TRANSFER_MODEL_CONFIG`-style)
  is used — this may fail if the checkpoint was trained with a different
  architecture.
* `ImputePFN.impute(X)` expects missing values encoded as `NaN`,
  normalizes column-wise using `nanmean`/`nanstd`, and returns an array
  the same shape as `X`.
