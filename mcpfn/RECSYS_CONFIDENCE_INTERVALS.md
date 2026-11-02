# Frozen recommender baseline reruns and confidence intervals

The research implementation lives in `scripts/rerun_small_recsys_baselines.py`
in this repository. It preserves the MovieLens 100K / Netflix small rerun
workflow used to fill the missing MatrixFormer paper error bars. Running it
does not require the paper repository, a checkpoint, a GPU, or neural inference.

## Run from the tabular repository root

The reproduction environment uses Python 3.10, Cornac 2.3.5, and NumPy 2.2.6.
Install the optional benchmark dependencies in your environment if needed:

```bash
python -m pip install 'cornac==2.3.5' 'numpy==2.2.6'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python mcpfn/scripts/rerun_small_recsys_baselines.py
```

The default output is
`mcpfn/runs/benchmark_evaluations/small_recsys_baseline_ci/`.
Use `--output /path/to/run` to choose another directory. Completed fits resume
only when the script, package versions, configuration, thread environment,
split, and saved artifact hashes match. Rerunning then regenerates CIs without
refitting. Use a fresh output directory after changing the evaluator.

## Frozen data

The default input directory is
`mcpfn/runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919/`.
It must contain `movielens_100k/split.npz` and `netflix_small/split.npz`.
These files are copied into each output dataset directory and their SHA256
hashes are checked against the original protocol. The script preserves all
training-row ordering and internal user/item indices; it never redraws a split.
It rejects duplicate coordinates and train/test overlap.

These evaluation data are not distributed in Git. On a new machine, copy the
frozen split files and pass `--split-source /path/to/split_parent`. The existing
paper also keeps identical copies at
`results/recsys/small_baseline_rerun/{100K,netflix}/split.npz`; arrange these into
the input directory names above, or place them in the corresponding output
dataset directories before running. Existing output splits are reused and checked.

## Protocol and artifacts

- Fit SVD, PMF, UserKNN, and ItemKNN once per dataset with the original
  hyperparameters and seed 123. BPR is deliberately excluded.
- Clip predictions to the training rating range through Cornac's `rate()`.
  Independently verify both user-level and macro-averaged errors against Cornac.
- Compute RMSE **within each user first**, then average users equally; do not
  use pooled rating-level RMSE. MAE also gives users equal weight.
- Bootstrap the sorted test-user scores with replacement 1,000 times, seed 0;
  use the 2.5th/97.5th percentiles. The seed resets for each method and metric,
  giving common resampled users on each dataset. This matches the paper's
  Seaborn bootstrap but needs only NumPy. There are 942 scored users for
  MovieLens 100K and 9,202 for Netflix small.
- Intervals condition on the fitted model, fixed split, and observed items.
  They do not estimate training-seed or new-item-population uncertainty.

Each run saves `summary.csv` (16 metric/CI rows), `manifest.json`, and
`status.json`. Each dataset/method directory saves `predictions.npz`,
`per_user.npz`, and `metrics.json`, including configuration, timing, versions,
and artifact hashes. The shared `user_bootstrap_interval()` function is also
importable for other saved per-user metrics.

The original paper results and figures stay in `MatrixFormerPaper`; they are
not moved or overwritten by the default command. The optional
`--paper-root /path/to/MatrixFormerPaper --publish` mode writes into that
paper's results directory and updates only the eight small-baseline score
rows, preserving a pre-update score archive. As with standalone runs, an
existing run from a different script fails the resume check: use a fresh
paper-results directory after archiving the previous run. Then run the paper's
`scripts/accuracy_statistics.py --import-sources` and plot generators to update
the figures. The paper retains its own original evaluator snapshot so existing
result provenance remains valid.

Run focused regression tests from the tabular root:

```bash
PYTHONPATH=mcpfn/scripts python -m pytest -q mcpfn/tests/test_small_recsys_baseline_ci.py
```
