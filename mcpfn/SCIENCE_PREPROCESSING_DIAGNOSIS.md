# Scientific-panel failure diagnosis

## Bottom line

**Part of the failure is fixable preprocessing: context locality and local
column scaling. It is not a universal normalization bug.** On eight previously
unused Intel temperature days, the fix reduces combined MatrixFormer's error
by **49.8%** and MatrixFormer-Impute's by **52.8%**, improving both models on
all eight days. Temporal regression still has the lowest aggregate error.
These results support a useful preprocessing fix, not a new baseline win.

The river panels improve substantially on previously scored windows, but
remain behind temporal methods. Wastewater and ocean chemistry are not
solved by the simple preprocessing changes tested here.

No training, checkpoint, paper, or Git-history changes were made. MRI was
not included. The temporal preprocessing is an optional scientific-panel
adapter; it has not been applied to arbitrary-row MissBench/UCI data.

## What went wrong

### 1. Too broad a temporal context, plus an unhelpful scaling window

For a smooth sensor channel, reconstructing a short gap is a local problem.
The original Intel input contains a whole day: 288 five-minute rows, plus
other sensors and time features. Using the full day's observed column mean
and SD presents the model with a different scale of variation than a short
local window. The following controlled ablation separates these changes.

Values are mean normalized RMSE across the same eight original Intel days.
All neural conditions in this table use ordinary observed-column scaling,
no delay features, the same checkpoint, and the same point estimator.

| Context / preprocessing | MatrixFormer | MatrixFormer-Impute |
|---|---:|---:|
| Full day, full-day column scaling | 0.05691 | 0.05871 |
| 32 rows, retain full-day column scaling | 0.03394 | 0.03360 |
| 32 rows, local column scaling | **0.02472** | **0.02505** |
| Local column mean only, no model | 0.17998 | 0.17998 |

Both restricting the context and rescaling locally help. Local-mean
imputation does not explain the improvement. This is an output-level
ablation, not a claim that we measured attention dilution or identified
one particular internal representation as the cause.

### 2. Some validation masks poorly cover the scored targets

The original recorded-mask validation sometimes omits entire columns that
receive test-score weight. Across development and confirmation cases:

- Wastewater has one such column in each of six cases.
- Intel has 15 such column occurrences across nine cases, at most two per case.
- Susquehanna has only 14–16 validation target cells per case.

This makes preprocessing selection noisy and sometimes unrepresentative.
It is not direct test leakage. The new adapter preserves the existing
validation mask and supplements it with observed-only, short-block
holdouts until each supported target has at least
`min(12, observed_count // 4)` validation cells. Insufficiently supported
targets raise an error rather than silently skipping validation.

The fresh-day native models and classical baselines receive the same
strengthened validation masks as the local models. Therefore the fresh
local-versus-native comparison does not give only the local models better
validation coverage. Wastewater's original scores are not replaced by a
rerun using this coverage fix.

### 3. The strongest baselines match the gap geometry especially well

In the original confirmation masks, 453/664 Intel scored cells (68%) and
237/302 Potomac cells (78%) have observed immediate neighbors on both sides.
Interpolation is exceptionally well suited to smooth, isolated gaps.
Susquehanna also often loses two entire auxiliary channels, leaving much
less cross-channel information than a rich multivariate panel suggests.

Generic matrix-completion training does not guarantee the extremely small
errors of a task-specific temporal method. This remains a plausible
training/task mismatch after fixing locality. The model is not blind to
ordering: **both checkpoints use rotary row and column position encodings**,
and the sensor inputs already include time features.

## The preprocessing fix

Implementation: `scripts/science_context_preprocessing.py`.

1. Mask test values before any transformation or feature construction.
2. Choose 32, 64, or 128 rows using observed-entry validation only.
3. Optionally append target-channel offsets ±1 and ±3 rows, also chosen
   on validation. Construct every shifted copy after masking so hidden
   targets cannot reappear as features.
4. Normalize columns using observed values from the full input, then
   normalize the local window. The first transform supplies a sensible
   global-level fallback for unsupported local columns; for supported
   columns, the two affine transforms recover local centering and scaling.
5. Run the unchanged checkpoint in its standard orientation; retain the
   central 16 predictions from each sliding window and invert the scaling.
6. Restore observed entries exactly.

This is offline imputation: future observed neighbors may be used. It is
**not forecasting**, a transpose ensemble, low-rank refinement, or a new
trained model. Temporal offsets require meaningful, regularly spaced row
order. KNN receives the same local-context and delay candidates in the
fresh confirmation, plus its own neighbor-count choices.

## Fresh confirmation: eight previously unused Intel days

All remaining eligible, nonoverlapping target days were registered before
their predictions were generated. Original target rules and recorded-mask
donor pool were retained. These are new days from the **same deployment**,
not independent sites. Residual temporal dependence is possible.

Metric: per-column RMSE divided by the SD of observed values in the full
scoring window; average scored target columns, then average days equally.
**The scoring denominator is unchanged by local preprocessing.** The gains
are not an artifact of scoring against a different local SD.

Intervals below are approximate 95% Student-t confidence-interval half-widths
across eight days, not across cells. Lower is better.

| Method | Normalized RMSE ± 95% CI |
|---|---:|
| Temporal regression | **0.03102 ± 0.01799** |
| MatrixFormer-Impute, local preprocessing | 0.03296 ± 0.01820 |
| Temporal interpolation | 0.03414 ± 0.02208 |
| MatrixFormer, local preprocessing | 0.03677 ± 0.02547 |
| KNN, same local/delay candidates | 0.04168 ± 0.02549 |
| KNN, full context | 0.04452 ± 0.03011 |
| MatrixFormer-Impute, native context | 0.06982 ± 0.03985 |
| MatrixFormer, native context | 0.07320 ± 0.04311 |

Paired improvements are reference error minus local-model error, positive
favoring the local model. Each comparison uses all eight days:

| Local model | Reference | Paired improvement ± 95% CI | Days improved |
|---|---|---:|---:|
| MatrixFormer | Native MatrixFormer | **0.03643 ± 0.02252** | 8/8 |
| MatrixFormer-Impute | Native MatrixFormer-Impute | **0.03686 ± 0.02732** | 8/8 |
| MatrixFormer | Temporal regression | -0.00575 ± 0.01442 | 4/8 |
| MatrixFormer-Impute | Temporal regression | -0.00194 ± 0.00722 | 3/8 |
| MatrixFormer | Local KNN | 0.00491 ± 0.01337 | 6/8 |
| MatrixFormer-Impute | Local KNN | 0.00873 ± 0.01048 | 6/8 |

The native-to-local improvement is clear in this small study. Differences
from the strong classical baselines remain uncertain; no superiority or
equivalence claim is justified. These are exploratory, unadjusted intervals.

## Retrospective diagnosis on the original windows

These windows were already scored before the preprocessing study. Settings
are selected on validation, but this table is **not independent confirmation**.
Values are means; full intervals and per-window scores are saved on disk.

| Dataset | Windows | MatrixFormer before → local | Impute before → local | Best original baseline |
|---|---:|---:|---:|---|
| Intel temperature | 8 | 0.05800 → 0.02437 | 0.05679 → 0.02404 | Temporal regression: 0.02379 |
| Potomac nitrate | 7 | 0.59498 → 0.26322 | 0.61181 → 0.21205 | Interpolation: 0.13683 |
| Susquehanna nitrate | 4 | 1.15299 → 0.39671 | 1.17696 → 0.56238 | Temporal regression: 0.20073 |

The before/after comparison here combines context changes and improved
validation coverage. Original baselines are shown for orientation, not as
a fresh matched-validation benchmark. Fresh Intel results above provide
that matched comparison.

For ocean alkalinity, a separate fixed-128-row exploratory probe changes
MatrixFormer from 0.25950 to 0.23567, still behind CANYON-B with local
calibration at 0.19147. For wastewater, that probe worsens MatrixFormer
from 0.61909 to 0.62840; iterative forest was 0.60229. Wastewater's original
gap is small and statistically uncertain, not a decisive failure.

## What did not explain the failures

- **Wrong checkpoint / inference configuration:** both checkpoints load
  strictly with their matching configurations and run in evaluation mode.
- **Global versus column normalization:** original validation never selected
  global scaling on these cases. The useful distinction is full-window
  versus local column statistics, not simply switching to column scaling.
- **Low numerical precision:** float32 produces only small, inconsistent
  score changes relative to bfloat16, not the observed performance gap.
- **Empty columns / metadata:** removing empty and constant columns, or
  ocean station/cast/year metadata, does not provide a general solution.
- **Orientation:** transposition is usually worse. A favorable wastewater
  Impute result is retrospective and is not promoted as a selected win.
- **Posterior decoder:** mean, median, and mean excluding the two broad
  outer bins are nearly identical on Intel (combined: 0.05691, 0.05695,
  0.05691). Ocean differences are also small. The broad bins are not the
  main cause.
- **Nonlinear inverse transform:** inverse-asinh of the transformed-space
  mean is not the raw-space posterior expectation. That mathematical caveat
  exists, but affine-only ablations do not resolve the failures. Blindly
  exponentiating broad distribution tails is not a justified fix.
- **Simple ocean physical transforms:** salinity normalization and
  observed-data salinity residualization do not establish a baseline win.

## Reproducibility and integrity

All results are under:

`/mnt/matrixformer_scratch/science_panels_20261106/`

| Directory | Completed scope |
|---|---|
| `diagnosis_v1/` | 448 model/case/preprocessing prediction records; original references and validation-selected rows are additional CSV summaries |
| `context_fix_v2/` | 44 records, 22 original cases × 2 models |
| `context_confirmation_v1/` | 64 records, 8 new days × 8 methods |
| `local_scaling_ablation_v1/` | 54 records, 9 original Intel cases × 2 models × 3 conditions |
| `decoding_diagnosis_v1/` | 240 metric records, 40 cases × 2 models × 3 decoders |
| `diagnosis_report_v1/` | Consolidated audit and all fresh-day paired comparisons |

Development cases are retained in files but excluded from reported
confirmation means. Source hashes, checkpoint/configuration hashes, input
hashes, prediction integrity, selection minima, observed-entry preservation,
and recomputed metrics are checked by the report runners. Registrations
and raw prediction files are retained.

`context_fix_v1/` is an intentionally stopped prototype, excluded from every
reported result. Its newly introduced local all-missing-column fallback
used physical zero; this was corrected before registering `context_fix_v2`.
That prototype issue did **not** affect the original full-window benchmarks.

To regenerate the reports without running inference:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python mcpfn/scripts/diagnose_science_preprocessing.py report
python mcpfn/scripts/analyze_science_failure_modes.py
python mcpfn/scripts/test_science_context_fix.py report
python mcpfn/scripts/confirm_science_context.py report
python mcpfn/scripts/report_science_preprocessing_diagnosis.py
python -m pytest -q -p no:cacheprovider \
  mcpfn/tests/test_science_preprocessing_diagnosis.py \
  mcpfn/tests/test_science_panels.py \
  mcpfn/tests/test_realistic_science.py
```

The focused suite passes **27 tests**. No registered evaluator source was
changed after these experiments were scored. See
`SCIENTIFIC_PANEL_FINDINGS.md` for the earlier five-application study and
`REALISTIC_SCIENCE_FINDINGS.md` for the separate chemical-sensor lead.
