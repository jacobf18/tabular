# Additional measured scientific panels

## Bottom line

Five additional real-data applications were evaluated: wastewater chemistry,
indoor sensor temperatures, nitrate in two rivers, and ocean alkalinity.
**None establishes another aggregate win over the strongest completed baseline.**
The earlier Italian chemical-sensor deployment remains the positive lead;
its aggregate advantage is not statistically conclusive either. See
`REALISTIC_SCIENCE_FINDINGS.md` for that separate experiment.

The emerging distinction is **recovering missing assays from related measurements**,
not a general advantage on all time-series data. On smooth signals with short
gaps, temporal interpolation and regression are especially strong. We should
not claim otherwise or choose a more favorable secondary endpoint afterward.

No changes to the paper, checkpoints, training, or Git history were made for
this batch. MRI remains excluded.

## Primary results

Values are mean normalized RMSE ± approximate 95% Student-t confidence-interval
half-width across the listed independent scoring units: nonoverlapping target
windows, or distinct ocean cruises. Within-site dependence can remain, so
nonoverlap should not be interpreted as a proof of independence. Lower is better.
Intervals are not computed across individual cells. For multivariate targets,
normalize each column's RMSE by its observed population SD, average columns
with scored entries, then weight windows equally.

| Application | Units | MatrixFormer | MatrixFormer-Impute | Strongest completed baseline |
|---|---:|---:|---:|---|
| Wastewater BOD/COD/suspended solids | 5 windows | 0.619 ± 0.220 | 0.704 ± 0.309 | Iterative random forest: **0.602 ± 0.271** |
| Intel indoor temperatures | 8 days | 0.0580 ± 0.0260 | 0.0568 ± 0.0226 | Temporal regression: **0.0238 ± 0.0127** |
| Potomac nitrate, native inputs | 7 windows | 0.595 ± 0.324 | 0.612 ± 0.310 | Temporal interpolation: **0.137 ± 0.057** |
| Susquehanna nitrate, native inputs | 4 windows | 1.153 ± 1.489 | 1.177 ± 1.428 | Temporal regression: **0.201 ± 0.284** |
| Atlantic ocean alkalinity | 10 cruises | 0.260 ± 0.116 | 0.244 ± 0.124 | CANYON-B + local calibration: **0.191 ± 0.121** |

The wastewater neural result is close, not a clear failure or a win. Its paired
improvement over iterative forest is -0.0168 ± 0.2165 normalized RMSE for the
combined model. The small number of windows and sparse assay holdouts make
this result particularly uncertain.

Ocean alkalinity native-unit RMSE, averaging cruises equally:

| Method | RMSE, micromol/kg ± 95% CI |
|---|---:|
| CANYON-B + local calibration | **6.240 ± 3.922** |
| Random forest | 6.406 ± 2.694 |
| Published CANYON-B | 6.443 ± 3.585 |
| RBF-SVR | 6.664 ± 3.467 |
| MatrixFormer-Impute | 7.837 ± 3.985 |
| MatrixFormer | 8.274 ± 3.701 |

Native errors are not pooled across different wastewater analytes or units.
Normalized scores must not be interpreted as a cross-domain difficulty scale.

## Follow-up: explicit time-delay inputs

After the original river screen, a separately registered preprocessing test
appended past/future nitrate observations at candidate hourly lags:
none, ±1, ±1/±3, or ±1/±6/±24. The candidate and normalization are chosen on a
second recorded validation mask, never by test performance. Every lagged copy
is constructed **after** removing both test and validation measurements,
preventing a hidden value from reappearing as a neighbor feature.

The same delay candidates are available to KNN, SoftImpute, random forest,
and ridge. Direct temporal interpolation and temporal regression remain
separate controls. No checkpoint is trained or changed; predictions use one
standard-orientation forward pass and the original target outputs, not a
transpose ensemble or rank refinement.

Previously scored river windows are labeled `exploratory` in this follow-up.
The following results use additional windows registered before delay scores:

| Application | New windows | MatrixFormer | MatrixFormer-Impute | Best baseline |
|---|---:|---:|---:|---|
| Potomac nitrate | 13 | 0.194 ± 0.080 | 0.318 ± 0.280 | Temporal interpolation: **0.0938 ± 0.0286** |
| Susquehanna nitrate | 8 | 0.465 ± 0.485 | 0.305 ± 0.186 | Temporal regression: **0.1345 ± 0.0645** |

These are different target windows from the first table; their means must
not be subtracted to claim an improvement caused by preprocessing.
On the same previously scored Potomac windows, delay candidates reduce
MatrixFormer's mean from 0.595 to 0.351 and Impute's from 0.612 to 0.185,
but both still lose to temporal interpolation at 0.137. That comparison is
exploratory. The reserved-window results confirm no temporal-baseline win.

## Data and realistic missingness

All studies replay **recorded multichannel availability masks** from separate
windows or cruises. Originally missing values remain missing and are never
scored. This is controlled withholding of measured values, not a claim to
know the true values of naturally missing entries. The recorded gaps can
reflect sampling schedules, quality control, outages, or deployment changes;
their precise causes are not inferred from an absent record alone.

### Wastewater process chemistry

- Source: [UCI Water Treatment Plant, dataset 106](https://archive.ics.uci.edu/dataset/106/water+treatment+plant),
  527 daily sampling occasions from an urban wastewater-treatment plant.
  The archived `water-treatment.names` gives the original Poch/Bejar/Cortes
  provenance and related process-diagnosis publications.
- Use 29 raw measurements. Remove **all nine derived treatment-efficiency
  columns** before masking to avoid reconstructing assays from their own
  precomputed removal percentages.
- Targets: influent and effluent biological oxygen demand, chemical oxygen
  demand, and suspended solids. Only columns with actual held-out values
  contribute to a given window's score; exact counts are retained.
- Context: 64 sampling occasions, preserving irregular calendar dates.
  Temporal controls use elapsed calendar time, not just row number.
- Roughly 2.17% of raw-panel entries are naturally absent. Recorded mask replay
  withholds about 8.4% of active target slots in confirmation windows;
  this subset is not asserted to have the archive-wide mean missingness rate.
- All 13 registered methods completed, including iterative forest, direct
  forest, PLS, ridge, SVR, KNN, low-rank and temporal controls.

### Indoor wireless sensor network

- Source: [Intel Berkeley Research Lab deployment](https://db.csail.mit.edu/labdata/labdata.html),
  54 physical Mica2Dot motes logging approximately every 31 seconds in 2004.
  The source explicitly documents missing epochs and missing/truncated motes.
- Use March 1–20, five-minute median temperatures. Exclude invalid mote IDs
  and temperatures outside a broad [-20, 60] Celsius range. These quality-rule
  exclusions also contribute to the replayed missingness and are documented.
- Do not input temperature-corrected humidity as an auxiliary target shortcut.
  Motes need 50% coverage in the first three donor days to be target candidates;
  scored channels also need sufficient local observed context.
- Each matrix has 288 rows and 54 temperature channels plus clock covariates.
  The selected early empirical masks hide about 1.35–2.87% of active target
  entries in confirmation days. No high-rate independent masking was imposed.
- Eight screening methods completed. Additional forest controls were not
  needed to establish the negative result against already stronger temporal
  methods and were not run for this dataset.

### River nutrient monitoring

- Source: approved 2019 [USGS instantaneous-value records](https://waterservices.usgs.gov/rest/IV-Service.html),
  Potomac site `01646500` and Susquehanna site `01579550`.
- Target: nitrate plus nitrite, parameter `99133`, mg/L as N. Other measured
  water-quality channels and, where available, streamflow are auxiliary.
- Exclude estimated measurements. Retain approved revised values. Do not
  duplicate a measurement by adding backup sensors or separate-depth streams.
  Concatenate nonoverlapping sonde versions with explicit method provenance.
- Aggregate to hourly means without interpolation; 336-hour local windows.
  Regions with insufficient observed targets are excluded **before scoring**
  and recorded in the protocol; long instrument-absence periods are not
  silently treated as validated imputations.
- Initial screening: eight methods. Delay follow-up: nine methods, all complete.

### Ocean biogeochemistry

- Source: [NOAA NCEI GLODAPv2.2023 Atlantic archive](https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Atlantic_Ocean.mat).
  Lauvset et al. (2024), *The annual update GLODAPv2.2023: the global interior
  ocean biogeochemical data product*, DOI
  [10.5194/essd-16-2047-2024](https://doi.org/10.5194/essd-16-2047-2024).
- Predict measured total alkalinity from location, pressure, temperature,
  salinity, oxygen and nutrients. Include sampling identifiers and decimal year.
  **No pH, DIC, fCO2, AOU or derived density inputs.**
- Require WOCE flag 2 for salinity, oxygen, nutrients and target alkalinity.
  Flag 0 interpolated/calculated values are excluded, including from truth.
- First 512 bottles per cruise ordered by station/cast/pressure. All qualifying
  2016 cruises form two development cases; ten qualifying 2017–2021 cruises
  form the scoring set. Record every coverage exclusion before predictions.
- Missing-assay donors are distinct cruises from 2013 or earlier. Recorded
  masks hide roughly 8–34% of all alkalinity slots in scoring cruises.
- Domain baseline: official [CANYON-B R implementation](https://github.com/HCBScienceProducts/CANYON-B),
  revision `a5b1efce24fcffc8b9e6dd3a0d1e54fa384a01d5`.
  Bittig et al. (2018), *An Alternative to Static Climatologies: Robust
  Estimation of Open Ocean CO2 Variables and Nutrient Concentrations From T,
  S, and O2 Data Using Bayesian Neural Networks*, DOI
  [10.3389/fmars.2018.00328](https://doi.org/10.3389/fmars.2018.00328).
  Its published alkalinity check value, 2359.331 micromol/kg, was reproduced.
- CANYON-B uses its published global training, unlike the local-data-only
  classical controls. Its extra local-calibration variant selects no change,
  an offset, or an affine correction using observed validation entries only.
  Missing predictor measurements are filled using observed-data KNN; measured
  targets being scored never enter that preprocessing.
- Twelve methods completed. Iterative forest was registered as an optional
  stronger control but not run after the existing methods established the
  negative screen; it is not counted as completed or included in summaries.

## Artifacts and reproducibility

All data, protocols, masks, predictions, tuning trials, exclusions, scores and
figures are saved under:

`/mnt/matrixformer_scratch/science_panels_20261106/`

| Path under that root | Contents |
|---|---|
| `data/manifest.json` | Download URLs and source checksums |
| `prepared_v2/` | Cleaned sensor/water panels, provenance and missingness diagnostics |
| `replay_v1/` | Initial four-panel protocols and 254 completed method/case records |
| `delays_v1/` | Leakage-safe delay protocol and 306 completed records |
| `ocean_v2/` | Measured-alkalinity protocol and 144 completed records |
| `reports/screen/` | Initial eight-method comparison on four panels |
| `reports/wastewater_full/` | Complete 13-method wastewater comparison |
| `reports/delays_replication/` | All nine methods on reserved river windows |
| `reports/ocean/` | Complete 12-method ocean screening comparison |

Every report contains `summary.csv`, `comparisons.csv`, `per_case.csv`,
`per_analyte.csv`, `native_rmse.csv`, `audit.json`, `scope.json`, and PDF/PNG
bar plots. All **704 completed records** were audited: masks exclude original
NaNs; predictions preserve observations; scores recompute from archived arrays;
tuning grids and validation minimizers match; data/protocol/prediction hashes
match. Optional unrun controls are explicitly outside each completed scope.

`prepared_v1/` is a pre-scoring metadata inspection. `ocean_v1/` was registered
but never scored and was superseded before predictions by `ocean_v2/`, which
uses actual collection dates rather than a middle-of-year approximation.
Neither abandoned preparation is treated as an experiment with results.

Code lives in the main tabular repository:

- `mcpfn/scripts/fetch_science_panels.py`
- `mcpfn/scripts/prepare_science_panels.py`
- `mcpfn/scripts/evaluate_science_panels.py`
- `mcpfn/scripts/evaluate_science_delays.py`
- `mcpfn/scripts/evaluate_ocean_panels.py`
- `mcpfn/scripts/canyon_b_bridge.R`
- `mcpfn/scripts/report_science_panels.py`
- `mcpfn/tests/test_science_panels.py`

The original realistic-science evaluators were reused without modifying their
frozen source. New registration functions require a fresh output directory;
registered source hashes intentionally reject subsequent code changes.

Example reproduction environment:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python mcpfn/scripts/fetch_science_panels.py
pytest -q mcpfn/tests/test_science_panels.py mcpfn/tests/test_realistic_science.py
```

Use the scripts' `--help` for fresh registration directories and method/case
subsets. CANYON-B requires `Rscript` and the recommended R package `mgcv`;
its upstream implementation, weights and license are retained under scratch
data, not vendored into this repository.

## Interpretation and next direction

The strongest current positive evidence remains nonlinear **cross-instrument
chemical calibration** in the Italian deployment. Simple temporal smoothness
is not enough to make MatrixFormer the right tool. Ocean chemistry was worth
checking because assays are expensive and real panel gaps exist, but its
domain-specific published estimator remains better in this screen.

Further candidates should emphasize heterogeneous, jointly informative assay
or instrument channels and compare against domain calibration methods, rather
than increase artificial missingness or remove temporal baselines to manufacture
a favorable example. Any new win found in this exploratory search needs another
untouched deployment or independent acquisition campaign before a broad claim.
