# Realistic scientific application: chemical-sensor gap reconstruction

## Bottom line

The strongest lead from the new realistic-gap search is **repairing missing
reference-analyzer measurements in a field-deployed chemical sensor array**.
Both frozen checkpoints have lower mean normalized error than every completed
classical comparator on the Italian sensor data. **MatrixFormer-Impute has
the stronger result; it must not be relabeled as combined MatrixFormer.**

Across 16 nonoverlapping, two-week reference windows not used for development,
the specialist improves mean normalized RMSE by **9.7%** over the strongest
single baseline, an iterative random-forest imputer. Combined MatrixFormer
improves by **5.5%**. On the ten additional replication windows alone, the
improvements are 8.9% and 3.4%, respectively, without changing neural settings.

The clearest analyte-specific result is NOx: specialist mean RMSE **67.7 ppb**
versus **83.9 ppb** for the strongest classical comparator, a **19.3% reduction**.
This is a promising case study, not a demonstrated universal scientific-data
advantage or a completed field deployment.

## Why this is more realistic than the SDSS stress test

The original measurements come from an Italian roadside deployment of five
metal-oxide chemical sensors, colocated certified reference analyzers, and
weather sensors in 2004-2005. The source describes cross-sensitivities and
sensor/concept drift. Its documented -200 sentinel marks missing readings;
approximately 18% of CO and 17.5% of NOx/NO2 reference readings are missing.
The archive does not identify the cause of every individual gap.

The task reconstructs CO, NOx, and NO2 when reference readings are unavailable,
using whatever other measured channels remain. Instead of choosing 80%-95%
independent random deletion, we replay the actual multichannel missingness
mask from a different recorded fortnight. This preserves consecutive-hour
gaps and their simultaneous occurrence in different instruments. Native gaps
remain missing and are never evaluation targets.

The 16 scored windows withhold 104-383 actually measured primary-target
entries out of 1,008 potential analyzer/hour slots per window (about 10%-38%).
These fractions emerge from recorded masks and reference-data availability;
they are not tuned missingness-rate parameters. This is **controlled mask
replay on real measurements**, not observed ground truth for naturally missing
readings. It does not replicate all signal-dependent failure mechanisms.

## Evaluation and controls

- Two early development windows, six initial confirmation windows, and ten
  additional nonoverlapping reference windows. Two replication candidates
  are excluded before prediction for insufficient scorable/validation cells;
  exact starts and reasons are preserved in the protocol.
- Every method receives the same local 336-hour context and clock covariates.
  This is retrospective reconstruction, not forecasting: later observed
  readings within that window are available to all methods.
- All setting selection uses a second recorded mask over observed entries
  only. Normalization is refitted after inner masking. The outer held-out
  concentrations never select preprocessing or hyperparameters.
- Frozen standard-orientation MatrixFormer checkpoints: no scientific-data
  fine-tuning, ensemble, transpose averaging, or neural-output rank refinement.
- Eleven classical comparators: temporal interpolation, KNN, extended-budget
  SoftImpute, direct random forest, ridge regression, PLS regression,
  RBF-SVR, gradient boosting, two temporally corrected regression methods,
  and iterative random forests (100 trees, up to ten imputation rounds).
- The Italian branch has **234 completed and prediction-audited method/window
  records**: 26 development records and 208 confirmation/replication records.
  All 13 methods cover the identical 16 scored confirmation windows.

## Main results

The metric is the unweighted mean over the three analytes of held-out RMSE
divided by that analyte's observed population standard deviation, followed
by an unweighted mean over windows. Entries are mean +/- approximate Student-t
95% confidence half-width across **16 time windows**, not independent pixels,
hours, instruments, or deployments. Temporal dependence can limit these intervals.

| Method | Normalized RMSE |
|---|---:|
| **MatrixFormer-Impute** | **0.4865 +/- 0.1601** |
| MatrixFormer | 0.5095 +/- 0.1665 |
| Iterative random forest | 0.5390 +/- 0.1467 |
| SoftImpute | 0.5707 +/- 0.1727 |
| KNN | 0.6117 +/- 0.1601 |
| PLS calibration | 0.6328 +/- 0.1621 |
| Ridge calibration | 0.6590 +/- 0.1780 |
| RBF-SVR | 0.6907 +/- 0.1919 |
| RBF-SVR + temporal residual | 0.7033 +/- 0.1549 |
| Random forest | 0.8167 +/- 0.1571 |
| Gradient boosting | 0.8306 +/- 0.1981 |
| Ridge + temporal residual | 0.8782 +/- 0.2022 |
| Temporal interpolation | 1.2121 +/- 0.1890 |

The specialist beats the strongest single baseline in **12/16** windows;
combined MatrixFormer does so in **11/16**. This does not mean either wins
every condition. The paired normalized-error improvement for the specialist
is **0.0525 +/- 0.0594**; for the combined model, **0.0295 +/- 0.0649**.
Both aggregate paired intervals include zero: do **not** call the overall
advantage statistically conclusive on this sample.

### Physical-unit errors

Mean of each window's RMSE, not a pooled mixture of different chemical units.
The comparator below is iterative random forest, the strongest single method
on the primary aggregate metric. Full per-analyte comparisons are archived.

| Analyte | MatrixFormer | MatrixFormer-Impute | Iterative random forest |
|---|---:|---:|---:|
| CO, mg/m^3 | 0.4804 | **0.4509** | 0.5558 |
| NOx, ppb | 74.5070 | **67.7067** | 83.8529 |
| NO2, micrograms/m^3 | 25.4620 | **24.9294** | 27.1764 |

SoftImpute is the strongest classical method specifically for NO2 (26.5759),
not iterative forest; the specialist's mean is still lower. For NOx, the
paired specialist improvement over iterative forest is **16.15 +/- 12.11 ppb**.
Unlike the aggregate interval, this analyte-specific interval excludes zero,
but NOx is one of three predeclared analytes, not a new primary endpoint.
No multiplicity-adjusted significance or independent-deployment claim is made.

## Retained negative branch: Beijing monitoring network

The second application uses measured PM2.5 at 12 Beijing stations, with other
pollutants as auxiliary observations and recorded station-gap masks. It has
two development windows, six 2015 confirmation windows, and six 2016
replication windows. Original missingness across the 72 pollutant channels
is about 2.8%, not a high independent random holdout.

On the 12 confirmation/replication windows, the predeclared primary normalized
errors are **0.17922** for MatrixFormer, **0.18477** for MatrixFormer-Impute,
and **0.17683** for temporal regression. Neither neural model wins this metric.
Physical-unit RMSE favors combined MatrixFormer (16.17 versus 19.02
micrograms/m^3), but we do not switch endpoints to declare this branch a win.

All 11 original/nonlinear-control methods are complete and prediction-audited
on the Beijing windows (154 records including development). Additional
iterative-forest/linear controls were stopped after this negative primary
screen; their completed and partial records remain. They are **not** counted
as finished methods or silently included in a complete comparison. The stop
ledger is `stopped_beijing_optional_controls.json`; all stopped processes were
owned by this experiment. No unrelated GPU or CPU jobs were stopped.

## Scope before a paper claim

The positive case has one deployment and a local 14-day information budget.
We have not beaten sensor-specific calibrations trained on months of extra
historical data, established recovery of the originally missing concentrations,
or independently validated the result at another site. The strongest next
validation is an independent deployment or a historical-calibration comparison,
not selecting an easier mask or changing the endpoint after inspecting scores.
No new scientific section has been added to the paper for this result.

## Artifacts and code

Artifact root: `/mnt/matrixformer_scratch/realistic_science_20261106/`.

- Final positive results: `reports/italian_final/`, including `summary.csv`,
  `comparisons.csv`, `per_case.csv`, `per_analyte.csv`,
  `per_analyte_summary.csv`, `audit.json`, `scope.json`, and a bar plot with CIs.
- Initial and replication reports: `reports/italian_initial_final/` and
  `reports/italian_replication_final/`.
- Negative screen: `reports/beijing_completed_methods/`, explicitly limited
  to the completed method set; the optional-control stop ledger is separate.
- Recorded-mask/reconstruction illustration:
  `figures/italian_recorded_gap_example.pdf` (PNG also saved). It shows the
  first prespecified initial confirmation window, not the best test case.
- Data and provenance: `data/manifest.json` and the downloaded source metadata.
- All protocols, source hashes, validation decisions, predictions, per-case
  worker files, and failed pre-scoring registrations remain on disk.
- Research code: `/root/tabular/mcpfn/scripts/` files matching
  `*realistic*science*`, `control_realistic_*.py`, `register_science_replication.py`,
  `run_science_control_case.py`, and `plot_realistic_outage_example.py`.
- Tests: `/root/tabular/mcpfn/tests/test_realistic_science.py`.

Validation: **32 tests pass** across the new suite and the existing measured-
physics, SDSS, numerical-control, and reporting tests. No experiment workers
remain running after the recorded stop of the optional Beijing extension.

To regenerate the pooled positive report from the two audited phase reports:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=/mnt/matrixformer_scratch/realistic_science_20261106
python mcpfn/scripts/combine_realistic_science_reports.py \
  --reports "$root/reports/italian_initial_final" \
    "$root/reports/italian_replication_final" \
  --output "$root/reports/italian_final"
```

The methodological chronology is in `REALISTIC_SCIENCE_SEARCH.md`. No commit
or push has been performed. Raw data have not been republished.

## Sources

- [UCI Air Quality data](https://archive.ics.uci.edu/dataset/360/air+quality),
  DOI: [10.24432/C59K5F](https://doi.org/10.24432/C59K5F).
- De Vito et al. (2008), *On field calibration of an electronic nose for
  benzene estimation in an urban pollution monitoring scenario*,
  DOI: [10.1016/j.snb.2007.09.060](https://doi.org/10.1016/j.snb.2007.09.060).
  The original dataset description specifies research-only use and citation.
- [UCI Beijing Multi-Site Air Quality data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data),
  DOI: [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G).
- Zhang et al. (2017), *Cautionary tales on air-quality improvement in Beijing*,
  Proceedings of the Royal Society A; the source paper is identified in the
  archived UCI metadata.
