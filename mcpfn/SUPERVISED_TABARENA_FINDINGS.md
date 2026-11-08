# TabArena investigation: downloaded data and supervised MatrixFormer prototype

## Completed

- Downloaded the official TabArena source and all **51 v0.1 datasets / 816
  prescribed outer splits**. The exported datasets and split arrays use 109 MB;
  OpenML's original cache uses about 973 MB. Everything is on scratch storage.
- Implemented frozen supervised prediction as `[training X, training y]` plus
  one `[test x, missing y]` row per independent matrix. No test-to-test attention.
- Tested train-fitted numerical, categorical, unknown-category, missing-value,
  and date processing on **all 51 datasets**, without reading test labels in
  that preflight. Encoded widths range from **4 to 1774** features.
- All **12 unit tests** pass. The saved prediction audit verifies all nine
  prediction sets, 24 metric values, official splits, context membership, and
  source/data/checkpoint hashes.
- Retained the first preflight's six schema failures and original source
  snapshots. Some unflagged OpenML string columns need dtype-based categorical
  handling; `Marketing_Campaign.Dt_Customer` needs date handling. The schema
  correction leaves all three smoke-test MatrixFormer prediction arrays
  **bitwise unchanged**. No settings were chosen from test scores.

## Initial results

These are the official quickstart's three datasets, repeat 0 / fold 0. All
training rows fit the 1023-row context limit. Every held-out row is evaluated.
The forest has 200 trees with no tuning; the dummy uses training means/priors.

| Dataset | Primary metric | MatrixFormer | RandomForest | Dummy |
|---|---|---:|---:|---:|
| blood-transfusion-service-center | ROC-AUC ↑ | 0.68930 | 0.66246 | 0.50000 |
| QSAR_fish_toxicity | RMSE ↓ | 1.04328 | 0.93584 | 1.50204 |
| anneal | Log loss ↓ | 1.11797 | 0.05598 | 0.83234 |

The mechanism works, but these three results do not establish general
competitiveness. The multiclass one-vs-rest/CDF normalization is particularly
weak here: its log loss is worse than the class-prior dummy. Calibration or a
different target representation should be selected with **inner training
validation**, never by searching these outer test scores.

BF16 introduces small batch-size-dependent numerical rounding: the checked
binary probability differs by 0.00032151 between batch sizes 1 and 5. Changing
the *other queries* at fixed batch size changes the first prediction by exactly
zero. FP32 predictions agree exactly across those batch sizes in the check.
The initial stricter BF16 cross-batch-size assertion and diagnostic values are
preserved in `supervised_smoke/batch_precision_diagnostic.json`; no benchmark
inference settings were changed to improve scores.

## What is not done

- No full 51-dataset neural evaluation, 816-split evaluation, HPO, bagging,
  uncertainty-calibration study, or official leaderboard rank.
- No claim of full-matrix training context on larger datasets: the adapter uses
  a reproducible 1023-row labeled context with all encoded columns retained.
  Full-suite preprocessing success is not full-suite inference/capacity success.
- The cloned upstream runner requires Python 3.11+ and AutoGluon nightlies;
  it is **not installed** in the existing Python 3.10 model environment.
  OpenML downloading and the standalone pilot use an isolated scratch venv.
- Before paper-level generalization claims, audit overlap between TabArena
  sources and the MissBench/UCI datasets previously used for checkpoint or
  procedure selection. Synthetic-only weight training alone does not rule out
  benchmark overlap through model selection.

## Regression-only extension

The next run focuses on **all 13 regression datasets**, using each official
Lite split (repeat 0 / fold 0) and every held-out row. It retains the frozen
adapter, train-only preprocessing, 1023-row context cap, and untuned 200-tree
forest/mean-dummy comparisons. No setting is selected using these test scores.

The live results are in `regression_lite/RESULTS.md` and `summary.csv` under
the scratch root below; `status.json` records completion and failures. Raw
RMSE is in original target units. Cross-dataset summaries divide each RMSE
by that dataset's mean-dummy RMSE rather than averaging incompatible units.
Until all 13 tasks complete, any summary is explicitly provisional.

Protocol: `SUPERVISED_TABARENA_REGRESSION.md`. Run with
`scripts/evaluate_tabarena_regression.py`; its resumable chunks preserve the
original inference batch layout. Audit with
`scripts/audit_tabarena_regression.py` after completion. The expanded adapter
and regression-runner test suite contains 26 passing tests, including an
interrupted/resumed prediction-cache test and corruption detection.

## Files and commands

Root: `/mnt/matrixformer_scratch/tabarena_20261028/`.

- `data/inventory.csv`: downloaded datasets, types, missingness, task IDs.
- `data/tasks/<task_id>/`: features, target, all official splits, checksums.
- `preprocessing_preflight/inventory.csv`: train-fitted encoded dimensions.
- `supervised_smoke/RESULTS.md`, `summary.csv`, `audit.json`: results and audit.
- `supervised_smoke/<task_id>/`: predictions, fitted preprocessors, row indices.

Protocol: `SUPERVISED_TABARENA.md`. Adapter: `scripts/matrixformer_supervised.py`.
The adapter exposes `fit`, `predict`, and `predict_proba`; fitting stores context
and preprocessing without updating the checkpoint.

From `/root/tabular`, use:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/mnt/matrixformer_scratch/tabarena_20261028/venv/bin/python
$PY mcpfn/scripts/download_tabarena.py
$PY -m pytest mcpfn/tests/test_matrixformer_supervised.py -q
$PY mcpfn/scripts/audit_tabarena_smoke.py
```

The downloader reuses checksum-verified files. To deliberately rerun the fixed
pilot, pass a **new** `--output` path to `scripts/evaluate_tabarena_smoke.py`;
the existing results are protected from overwriting. The audit script checks
the canonical `supervised_smoke` directory. Future official integration should
use TabArena's `AbstractModel` wrapper and its prescribed validation/bagging
protocol, then begin with the 51-dataset Lite split before the full suite.

References: [TabArena repository](https://github.com/autogluon/tabarena),
[TabArena paper](https://arxiv.org/abs/2506.16791), and
[dataset curation](https://github.com/TabArena/tabarena_dataset_curation).
