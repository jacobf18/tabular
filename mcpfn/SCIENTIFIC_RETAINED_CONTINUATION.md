# Scientific continuation with real-benchmark retention

## Active experiment

Run: `/mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2`

Continue for **3,000 additional optimizer updates** from the 600-step scientific
pilot. Keep the architecture and inference interface unchanged. Improve all
seven existing scientific applications while explicitly limiting degradation
on the older imputation, recommender, and LLM benchmarks.

No published checkpoint is overwritten. No paper changes, commits, pushes,
or model uploads are performed by this experiment.

## What the first real retention check found

The first pilot was previously checked only against synthetic retention cases.
The new real-data screen audited **572 cases** against the original combined
MatrixFormer using frozen inputs, original inference protocols, saved predictions,
and independent rescoring:

- MissBench: all 462 cases, across 42 datasets and 11 patterns.
- UCI: 36 cases, one fixed repeat at 30% missingness for each of 12 datasets
  and three patterns.
- LLMs: 72 settings, all four datasets and all three selection methods,
  folds 0/5, budgets 1/8/15.
- Recommenders: the complete MovieLens 100K and Netflix-small query sets.

The pilot **fails the stricter retention screen** despite better aggregate
imputation scores:

| Check | Original | Scientific pilot | Change |
|---|---:|---:|---:|
| MissBench, full-suite NRMSE | 1.611742 | 1.588867 | -1.42% |
| UCI, screen-subset NRMSE | 1.092214 | 1.060126 | -2.94% |
| MissBench humans_numeric, mean NRMSE | 2.541521 | 3.085846 | +21.42% |
| BenchPress, screen-subset R² | 0.276539 | 0.242015 | -0.034524 |
| Merged LLMs, screen-subset RMSE | 0.832657 | 0.853229 | +2.47% |
| Merged LLMs, screen-subset R² | 0.324607 | 0.280268 | -0.044339 |

Other checks, including the recommender checks, passed. The UCI and LLM values
above are **screen subsets**, not their full-suite paper numbers. This experiment
does not treat improved average tabular performance as permission to conceal
individual-dataset or LLM regressions.

The complete audit is in
`/mnt/matrixformer_scratch/scientific_prior_training_20261108/pilot_retention_screen_v1/`.

## New synthetic coverage

The six pilot generators remain. Four more are added in
`src/tabimpute/prior/scientific_processes.py`:

1. **Treatment processes:** flow, concentration, residence time, reaction,
   saturation, and inlet/outlet relationships with noise. This targets the
   wastewater failure without using wastewater measurements as training data.
2. **Assay covariates:** nonlinear proxy measurements, threshold effects,
   regime-dependent offsets, and ordered or unordered latent samples.
3. **Local smoothness:** shared and channel-specific smooth trajectories,
   varying knot spacing, low measurement noise, and structured gaps.
4. **Spatial profiles:** multiple profiles with depth, shared transitions,
   profile-specific offsets, and nonlinear cross-channel relationships.

Treatment-process and assay families receive an extra occurrence in the
fixed science-family schedule. Missingness cycles through isolated gaps,
bursts, network outages, scheduled assays, self-MNAR, and covariate-MAR.

A separate synthetic **score-panel replay** generator supplies dense context
rows and partially observed test rows, including varied capability factors,
benchmark difficulty, nonlinear score ceilings, and sparse visible test columns.
It is intended to protect the LLM-completion interface missing from the pilot's
rehearsal mixture.

All gradients still come from synthetic data. No measured benchmark values,
recorded masks, fitted benchmark parameters, or real target labels are used in
gradient computation. All normalization is observed-only.

## Training and rehearsal

Trainer: `scripts/train_science_retained_continuation.py`.

Each optimizer update accumulates four independent matrix losses:

| Component | Weight | Point target |
|---|---:|---|
| Scientific priors | 35% | True synthetic measurement |
| Existing imputation priors | 35% | Frozen original combined MatrixFormer |
| Existing recommender priors | 20% | Frozen original combined MatrixFormer |
| Synthetic score panels | 10% | Frozen original combined MatrixFormer |

The retention teacher is **the original combined model**, not the scientific
pilot, so rehearsal does not lock in the pilot's new errors. Scientific point
targets remain ground truth rather than the parent's failed scientific predictions.

Scientific loss: NLL + 4 × Huber(mean, truth) + 0.05 × CDF MSE.
Replay loss: 8 × Huber(mean, frozen parent mean) + 0.05 × true CDF MSE,
plus true NLL weighted 0.2 for imputation/score panels and 0.1 for ratings.

AdamW uses peak LR `5e-6`, warmup, cosine decay, gradient clipping `0.5`,
FP32 parameters/moments, BF16 autocast, and gradient-safe streamed attention.
The full model is trained. The GPU-memory preflight passed on all four domains
with finite gradients/parameters, **4.01 GiB peak allocation**, and valid queries
in all 58 synthetic validation cases. The disposable preflight update is not
used to initialize training.

## Monitoring and selection

Synthetic diagnostics run every 100 updates: 40 scientific cases and six cases
each for imputation, recommenders, and score panels. These diagnostics cannot
override a failed real-benchmark check.

At initialization and every 500 updates, immutable checkpoint snapshots receive:

- The **72 existing scientific windows/cruises**, with the original native
  full-window inference, masks, validation-based normalization choices, and
  all cached controls. The local-context adapter is not used.
- The **572-case real retention screen** described above. At step zero, the
  already-completed pilot screen is reused only after verifying identical
  initial model parameters and its registered gate fingerprint.

Retention limits, fixed before this continuation:

- At most **2% higher** MissBench/UCI overall NRMSE, each recommender RMSE/MAE,
  and each LLM dataset's standardized RMSE.
- At most **5% higher** tabular NRMSE for any missingness pattern.
- At most **10% higher** tabular NRMSE for any individual dataset.
- At most **0.01 lower R²** for any LLM dataset.

The primary candidate must improve mean error on **every one of the seven
scientific applications** relative to the original combined model, and pass
every retention-screen check. Among eligible candidates, select the earliest
strict minimum geometric mean of the seven scientific error ratios.
This is an improvement gate against MatrixFormer, **not a claim that the
candidate beats the strongest classical baseline on every application**.

All old historical scientific `confirmation` cohorts are now explicitly
development cohorts because their results have already guided model design.
Their labels are retained only for comparability with the earlier tables.
The real legacy screens also participate in selection and are not untouched tests.

After training, the selected candidate must pass a **full legacy evaluation**:
2,262 MissBench/UCI cases, 1,800 LLM settings, and complete MovieLens 100K,
Netflix-small, and MovieLens 1M splits: **4,065 evaluation records**. MovieLens
10M and MRI remain excluded. Only a candidate satisfying the seven-application
science gate and this full retention gate produces `recommended_checkpoint.pth`.

## Saved artifacts

- `status.json`, `worker.pid`, `launch.json`: actual worker state.
- `config.json`, `sources/`: registered recipe, hashes, source snapshots.
- `training.jsonl`, `synthetic_*.json`: learning curves and synthetic cases.
- `snapshots/step_*/science/`: saved predictions and paired scientific summaries.
- `snapshots/step_*/retention/`: old-benchmark predictions and gate checks.
- `monitoring.jsonl`: every candidate's eligibility, including failures.
- `latest_checkpoint.pth`: resumable optimizer/model/RNG state.
- `best_retention_checkpoint.pth`: best screen-safe diagnostic candidate;
  it need not improve every scientific dataset.
- `best_validation_checkpoint.pth`: candidate passing all seven science
  comparisons and the screen. Until one exists, this explicitly contains
  the unchanged original model, tagged as fallback step `-1`.
- `final_full_retention/`: complete old-benchmark evaluation.
- `recommended_checkpoint.pth`: created only after the full acceptance rule.

Resume only after confirming the existing worker has stopped:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python mcpfn/scripts/train_science_retained_continuation.py train --resume
```

The run is additive: the 600-step pilot, its frozen source files, published
checkpoints, and all unsuccessful candidates remain available for comparison.
