# Realistic scientific reconstruction search

## Objective and decision rule

Find a useful measured-data task on which the frozen combined MatrixFormer
or explicitly named MatrixFormer-Impute improves over credible baselines.
MRI is excluded. High-rate independent random masking alone is not sufficient.
Preserve failed branches; do not choose masks or endpoints by test performance.
Any application chosen after screening remains exploratory until confirmation.

## First branches: measured environmental sensor outages

Authoritative UCI downloads, metadata, and SHA256s are retained under
`/mnt/matrixformer_scratch/realistic_science_20261106/data/`.

- **Italian gas-sensor field deployment**, DOI `10.24432/C59K5F`;
  De Vito et al., *On field calibration of an electronic nose for benzene
  estimation in an urban pollution monitoring scenario*, DOI
  `10.1016/j.snb.2007.09.060`. Hourly measured metal-oxide sensor responses,
  reference analyzers, and meteorology; missing readings are encoded -200.
  The repository API's generic missing-values flag is inconsistent with the
  actual file and description; the documented sentinel is treated as missing.
  Primary targets: reference CO, NOx, and NO2, not benzene (which is closely
  related to an auxiliary sensor). Exclude the 90%-missing NMHC column before
  evaluation. Remaining sensor and weather channels are auxiliary inputs.
- **Beijing 12-station monitoring network**, DOI `10.24432/C5RK5G`;
  Zhang et al., *Cautionary tales on air-quality improvement in Beijing*,
  Proceedings of the Royal Society A (2017). Hourly measured air pollutants,
  2013-2017; use all 72 pollutant/station channels. Primary targets are PM2.5
  at the 12 sites; other pollutants are auxiliary inputs. Original missingness
  is about 2.8% across these channels, not an imposed 80%-95% holdout.

## Protocol fixed before scoring

`mcpfn/scripts/evaluate_realistic_science.py` fixes two development windows
and six later confirmation windows per dataset. Each is 336 hours (14 days).
Replay an actual, distinct earlier window's missingness mask on the target
window; retain all original gaps and score only originally measured values.
Preserve multi-channel outages and consecutive missing hours. Do not enlarge
gaps to manufacture a favorable sparsity regime. A second recorded mask
provides observed-entry validation, with transforms refitted after masking.

This is **retrospective gap repair**, not forecasting. Every method may use
later observed measurements within the window. Only clock covariates (hour
sine/cosine and relative time) are added; no hidden targets, labels, external
training data, or reference imputations are supplied. Models use the full
336-row context, standard orientation, no ensemble or output refinement.

Baselines: temporal interpolation, KNN, extended-budget SoftImpute, a
100-tree random forest with missing indicators, ridge cross-channel prediction
with temporally interpolated residuals, and (for Beijing) contemporaneous
other-station medians with temporally interpolated station residuals. Settings
are selected only on the observed validation mask, identically for all methods.
This is not an exhaustive comparison with specialized pollution forecasting
networks; those solve a different task and usually use additional training.

Primary error is mean target-column RMSE divided by its observed population
standard deviation. Also report PM2.5 RMSE/MAE in micrograms/m^3 and Italian
per-analyte errors in their native units; do not average unlike physical units
into a scientific claim. Confirmation uncertainty must use time-window means,
not individual hours or pollutant/station cells as independent datasets.

Mask replay is still artificial withholding on real measurements, with
empirically observed gap geometry. It does not establish accuracy on the
originally missing readings, reproduce signal-dependent failure mechanisms,
or provide independent replicate ground truth. These limitations remain even
if the model wins. No paper section or claim is justified before the results.

The first registration (`outage_replay_v1`) failed the pre-scoring coverage
review: one Italian case had only three evaluable entries, one Beijing
development case only one, and Beijing confirmation validation only 4-5.
No predictions or scores were produced for that registration. Before any
outcomes, the revised registration replaces those donors with recorded
windows having more scorable target gaps and a broader validation mask.
The target time windows and endpoints are unchanged; masks are not enlarged.
Both registrations remain archived. The running protocol is `outage_replay_v2`.

## Baseline safeguards

The first development results favored the neural models on the Italian data,
but the effect was smaller on the first confirmation batch. Additional
baselines are retained rather than stopping at a favorable initial comparison:

- `calibration_controls_v1`: RBF-SVR (18 observed-validation settings), SVR
  plus temporal residual interpolation, and histogram gradient boosting.
  The grids were registered before reading the confirmation outcomes.
- `linear_controls_v1`: ridge regression, PLS regression (a standard
  chemometric calibration model), and an iterative random-forest imputer.
  The latter uses 100 trees and up to ten imputation rounds, not merely
  median-filling the other sensor channels. These are added after the initial
  confirmation comparison, using the identical observed-validation masks;
  no checkpoint, neural transform grid, target window, or test mask changes.

All methods currently use the same local 14-day context. This is a constrained
retrospective reconstruction comparison, not a demonstration against a
sensor-specific model calibrated on months of additional historical data.
The stronger historical-data comparison is a remaining external-validity
question, and any final claim must state this limitation.

## Additional untouched-window replication

`outage_replication_v2` freezes additional reference windows after the initial
confirmation, without changing checkpoints, preprocessing grids, endpoints,
or baseline settings. The Italian candidate list uses the remaining fixed
nonoverlapping fortnight windows not used as prior model inputs; ten qualify.
Two candidate windows are excluded before prediction because the overlap
with naturally available reference measurements leaves no scored targets
or fewer than 30 target/validation cells. Their exact starts and reasons
are in the replication protocol. `outage_replication_v1` is the aborted
pre-scoring registration that exposed this coverage problem; it has no scores.

The Beijing replication uses six 2016 windows, after the six 2015 initial
confirmation windows. Masks come from the same fixed recorded donor library.
Italian mask-donor dates can postdate the reference window; only the Boolean
mask is transferred, not future sensor values. This is retrospective mask
replay, not a forecasting claim. The repeated masks and shared deployment
mean that confidence intervals describe these time windows, not independent
sites or instruments. Neither incomplete comparisons nor per-pixel CIs
should be reported as final evidence.

`run_science_control_case.py` distributes expensive independent CPU baseline
fits across cases without altering their registered algorithms or parameters.
Per-case protocols, locks, logs, and predictions are retained. Only completed
prediction/JSON pairs are published into the main records directory.

The UCI source explicitly describes metal-oxide cross-sensitivities and
sensor/concept drift. These motivate local multivariate calibration, but the
archive does not label the cause of each gap: do not assert that every -200
entry is a diagnosed hardware failure. Its original description specifies
research-only use and requests citation of De Vito et al.; raw downloads
remain in the scratch data directory, not a newly published data package.

## Outcome

The Italian branch is complete with all 13 methods on two development and
16 nonoverlapping confirmation/replication windows. MatrixFormer-Impute has
9.7% lower mean normalized error than iterative random forest; combined
MatrixFormer has 5.5% lower error. The aggregate paired intervals still
include zero. NOx reconstruction is the clearest analyte-specific result:
67.7 ppb versus 83.9 ppb for the specialist and strongest baseline.

The Beijing branch does not win its predeclared primary normalized metric.
Its original and nonlinear-control evaluations are complete. The optional
expensive iterative-forest extension was stopped rather than spending more
compute on a branch already rejected on that endpoint; all partial artifacts
and a stop ledger remain. Physical-unit error is reported but not substituted
post hoc for the unsuccessful primary endpoint.

See `REALISTIC_SCIENCE_FINDINGS.md` for the full tables, limitations, and
final artifact paths. No new result from this search has been added to the
paper, committed, or pushed.
