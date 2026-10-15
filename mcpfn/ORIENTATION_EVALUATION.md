# Original, transposed and averaged inference

The orientation campaign extends the fixed MatrixFormer and MatrixFormer+
evaluations to the same transpose/average comparison used by the causal harness.
It does not train or select a checkpoint, select an orientation on test labels,
change the original missingness masks, or rerun external baselines.

## Variants

- `general` / `observed_blend`: original MatrixFormer / MatrixFormer+ predictions.
- `general_transpose` / `observed_blend_transpose`: transpose the observed matrix,
  apply the deployed recipe, and transpose predictions back before scoring.
- `general_average` / `observed_blend_average`: arithmetic mean of original and
  mapped-back transposed predictions, **not the mean of their scores**.
- Parallel `specialist*` variants are retained as diagnostic data, not standalone
  paper methods.

Preprocessing is recomputed from observed values in each orientation. This follows
`benchmark/eval_dwip_transpose.py`: transpose first, then standardize columns. It
does not normalize in the original orientation and then swap a normalized tensor.
Feature scales and the roles of rows/columns therefore matter; improvement is not
guaranteed. MatrixFormer+ fits a separate observed-only weight in each orientation;
its average combines the two final blended predictions, rather than fitting a
new blend across averaged experts. Observed reconstruction remains in-sample
evidence and is not unbiased validation.

Entirely unobserved transposed columns use zero mean and unit scale, matching
the causal harness. Their entries remain missing; no hidden target is used for
normalization. Nonempty-column normalization is exactly the original UCI rule.

## Coverage and geometry

- MissBench: all 462 frozen cases, eleven mechanisms, whole-matrix inference.
- UCI: all 1,800 cases, three mechanisms and all rates/repeats, at most 2000×1000
  windows in each orientation. Transposition can change the axis being windowed.
- LLM: all four datasets, ten folds, three selection methods and budgets 1–15.
  Transposition happens after the original harness constructs each masked test
  batch; selection, held-out targets and scoring remain unchanged.
- Ratings: MovieLens 100K/1M/10M and Netflix small, the same frozen splits and
  user-macro RMSE/MAE. General inference retains global normalization and
  9976×1000 or 10000×1000 limits; specialist inference retains 2000×1000 limits.
  In the transposed specialist recipe, row-residual scaling acts on items.
  Clipping uses the observed rating range, including 0.5–5 for MovieLens 10M.
- No additional causal or native-R baseline reruns.

Original tabular/rating predictions are reused from hash-verified artifacts.
The LLM original predictions must be recomputed to average before scoring; all
original fold panels are checked against the completed original campaign.
Only observed/query rating predictions are stored, not giant dense completions.

## Run and audit

From `/root/tabular`:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
python mcpfn/scripts/evaluate_orientation_ensemble.py
python mcpfn/scripts/summarize_orientation_ensemble.py
```

Default output:
`/root/tabular/mcpfn/runs/benchmark_evaluations/matrixformer_orientation_v2_20261004/`.
Each stage writes progress, timing, predictions or fold panels, decisions and
summary CSVs. `final_audit.json` and `RESULTS.md` are written only after the entire
campaign passes the completeness, hash, scoring and prediction-average checks.
The paper exporter refuses incomplete campaigns.

Tests:

```bash
python -m pytest -q mcpfn/tests/test_orientation_ensemble.py \
  mcpfn/tests/test_matrixformer_plus_large_recsys.py
```

The full campaign completed on 2026-10-05 and passed the independent final audit:
2,262 tabular cases, 180 LLM settings with ten folds each, and four rating datasets.
Original predictions or fold panels are preserved; prediction averages and
scores were independently checked. The full report is
`runs/benchmark_evaluations/matrixformer_orientation_v2_20261004/RESULTS.md`.
The portable paper export is `/root/MatrixFormer/results/orientation/`.

Transpose and averaging worsen overall tabular NRMSE for both models. Transpose
worsens all 180 LLM settings, while averaging improves nine and worsens 171 for
each model. MatrixFormer+ orientation averaging slightly improves RMSE on all
four rating datasets, but improves MAE only on MovieLens 100K. For MovieLens 10M,
its RMSE changes from 0.833401 to 0.831572 and MAE from 0.678188 to 0.678736.
The primary deployed recipes and their results remain unchanged.

The campaign ran in tmux session `mf-orientation` after the original MovieLens
ensemble run, avoiding GPU contention, with a 20 GiB free-space reserve and a
24-hour timeout. The small tabular pilot is retained in
`matrixformer_orientation_pilot_20261004/`; it is not the full benchmark result.
The initial `matrixformer_orientation_20261004/` attempt is retained unchanged:
it stopped when the original UCI wrapper rejected an all-missing transposed
column. The v2 campaign froze the neutral-statistics safeguard and reran all
cases; no incomplete campaign is exported as a full result.
