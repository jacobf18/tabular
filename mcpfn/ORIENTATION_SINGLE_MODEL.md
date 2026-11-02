# Combined MatrixFormer: orientation ablation

This report-only experiment evaluates standard, transposed, and prediction-average
inference for the fixed combined MatrixFormer. It does not train models, select an
orientation per test case, rerun external baselines, or modify the paper.

## Scope

- MissBench: all 462 frozen cases, 42 datasets and 11 missingness patterns.
- UCI: all 1,800 frozen cases, 12 datasets, three patterns, five rates and ten repeats.
- Ratings: MovieLens 100K, Netflix small and MovieLens 1M.
- LLM completion: MTEB, MMLU, merged and BenchPress, all 1,800 frozen fold settings
  (180 ten-fold panels covering three selection methods and budgets 1 through 15).

MovieLens 10M was initially planned but explicitly excluded by the user because
it is not used in the paper. Its inference was stopped before completion. The
original manifest, suite status and partial window accumulators remain intact;
`scope_override.json` records the request and binds it to the original manifest.
Only the three completed rating datasets are scored and reported.

Checkpoint:
`/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth`

SHA256: `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.

## Fixed inference protocol

The standard predictions are reused from the hash-verified October 20 evaluation.
Representative standard-orientation controls are rerun before proceeding; the
manual preprocessing path preserves the original floating-point reduction order
as well as its observed-only statistics, epsilon and constant-column rules.

Transposed inference uses the **same normalized values as standard inference**,
transposes that matrix without refitting normalization, maps predictions back to
the original axes, and undoes the original scaling. This avoids mixing unrelated
tabular feature units during re-normalization. It is a different protocol from
the older orientation experiment and the causal experiment, which independently
normalized columns in each orientation. These experiments must not be conflated.

MissBench uses full matrices. UCI retains its original outer column normalization
and the interface's inner normalization, both frozen on the original axes. UCI and
ratings use 2,000-row by 1,000-column overlapping-tail windows in each orientation;
transposition changes which original cells share a window, so this is not a pure
orientation-only comparison with identical window membership.

Ratings retain original observed-only two-way biases and user-residual scales,
rather than fitting new item-residual scales after transposing. Each orientation
is clipped to the frozen training rating range after overlap averaging; the
orientation average then averages these clipped predictions. Only test queries
are stored, with overlap coverage checked explicitly.

LLM evaluation preserves the original context rows, 200-row test batches,
visible features, missing-entry masks and scoring normalization. Each batch's
original observed-column statistics are frozen before transposing. Predictions
are averaged before the unchanged scorer clips standardized predictions to
[-10, 10]. R² retains its original context-centered definition.

All inference uses the same weights, predictive mean, FP32 parameters and BF16
autocast as the archived standard run. cuDNN SDPA is enabled only for the LLM
suite, matching the original suite-specific settings.

## Test and summarize

From `/root/tabular`:

```bash
PYTHONPATH=mcpfn/scripts:mcpfn/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest -q mcpfn/tests/test_single_matrixformer_orientation.py \
    mcpfn/tests/test_single_matrixformer_orientation_summary.py

PYTHONPATH=mcpfn/scripts:mcpfn/src \
  python mcpfn/scripts/summarize_single_matrixformer_orientation.py
```

The default output directory is
`/mnt/matrixformer_scratch/single_matrixformer_orientation_20261102/`.
The summarizer accepts `--output`. Do not resume the original evaluator's
`--stage run` command for this completed campaign: its immutable inference
manifest still includes the excluded MovieLens 10M dataset.
A campaign lock prevents duplicate inference runs;
completed predictions and rating-window accumulators resume only after their
hashes and manifest identity have been verified. Do not change the evaluator
or dependencies while a campaign is running.

The first preflight caught a C-/Fortran-layout floating-point discrepancy before
any predictions were accepted. Its failed manifest/log are preserved separately
in `single_matrixformer_orientation_20261102_preflight_v1/`; the successful
campaign uses the corrected, parity-tested implementation and a fresh manifest.

`status.json` and the suite-specific status/log files record progress. The
summarizer requires every case in the reporting scope to finish, independently
rescores all 4,065 saved prediction artifacts, checks all 29 bit-identical parity
controls, exact prediction averaging and standard-cache equality, then writes:

- `RESULTS.md`: full point-estimate comparisons, with all three orientations.
- `tabular/per_case.csv`, `tabular/per_dataset.csv`, `tabular/summary.csv`.
- `recsys/summary.csv` and compressed per-user error tables.
- `llm/per_fold.csv`, `llm/panels.csv`, `llm/summary.csv`.
- `audit.json`: checked artifacts, source and output hashes.

The final campaign status is `complete_revised_scope`. The interrupted original
RecSys suite status is retained in the audit, not rewritten as a completed
four-dataset run. Missing required prediction artifacts still fail the audit.

Summary CSVs include exact 95% paired percentile-bootstrap intervals for the
scores and their differences from standard orientation. Units are datasets for
tabular results (10,000 resamples), test users for ratings (1,000), and folds for
LLM results (1,000). These quantify conditional dataset/user/fold variation, not
training-seed or independent-new-benchmark uncertainty. LLM folds share context.
The report also separates all LLM settings from greedy-MI budgets 5/10/15.

## Paper export

The inference campaign remains frozen. Its subsequent paper export places all
three variants in an exploratory appendix ablation while retaining standard
orientation in the main comparisons, including the restored causal primary.
No best-performing orientation is selected from these outcomes.

```bash
python mcpfn/scripts/export_single_matrixformer_orientation_results.py
cd /root/MatrixFormerPaper
python scripts/plot_single_model_orientation_results.py
```

Portable CSVs and audit provenance are in
`/root/MatrixFormerPaper/results/single_model/orientation/`. The exporter checks
the completed audit and every copied CSV hash. All three orientations appear
in shared-style bar plots with confidence intervals; MovieLens 10M is excluded.
The causal controls remain a separate orientation-specific-normalization ablation.
