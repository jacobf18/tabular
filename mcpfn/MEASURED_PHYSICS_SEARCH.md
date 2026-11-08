# Measured-physics search

The goal is a reproducible advantage of MatrixFormer or MatrixFormer-Impute
on an actual measured physics dataset, not a random synthetic matrix or an
artificially favorable simulation. MRI remains excluded. The cylinder and
quantum pilot failures remain archived; the cylinder paper section is now
an inactive draft.

## Initial candidates, before inspecting completion scores

1. **Measured hyperspectral spectroscopy:** Indian Pines (AVIRIS) for
   development; Salinas (AVIRIS) and Pavia University (ROSIS) reserved for
   confirmation. The HybridSN authors' public repository distributes the
   standard corrected scenes. Class labels are neither downloaded nor used.
2. **Measured photoemission spectroscopy:** the PyARPES authors' example
   photoemission datasets. Inspect dimensional metadata to define a
   physically meaningful completion task before computing benchmark scores.
   The source documentation notes that examples may have been binned or
   smoothed for distribution; do not describe them as untouched detector data.

Downloads, repository revisions and SHA256 hashes are saved under
`/mnt/matrixformer_scratch/measured_physics_20261106/data`.

## Evaluation rules

- No physics-specific neural training, new synthetic noise, or ground-truth
  labels as features. Artificial withholding of measured entries is allowed,
  as in ordinary retrospective completion benchmarks.
- Freeze matrices/regions, missingness conditions, seeds, transforms and
  baseline grids before scoring an iteration. Keep failed iterations.
- Tune only on temporarily hidden observations. Refit preprocessing on that
  reduced context; never select settings using outer hidden targets.
- Report the combined and imputation-specialist checkpoints separately.
- Include strong application-relevant controls, not just an untuned low-rank
  method. For hyperspectral data, include spectral interpolation and tensor
  completion as well as matrix-completion and nearest-neighbor controls.
- Reserve independent datasets or measurements for confirmation after the
  pipeline is frozen. A discovery win alone does not qualify as a paper claim.
- Primary score is hidden-entry RMSE divided by observed-global population
  SD. Also retain MAE and appropriate physical secondary metrics. Do not
  pool missing cells to manufacture narrow confidence intervals.
- The initial screening rates are 50%, 80%, and 95% missing entries. Report
  the entire grid. Sparse acquisition (80% and 95%) is a predeclared secondary
  regime, not a subset to choose retrospectively.
- A useful win should be at least 10% lower RMSE than the strongest evaluated
  baseline overall or in that predeclared sparse regime, and should reproduce
  on independent measured data. This is a working success criterion, not a
  promise that the existing checkpoints will satisfy it.

The experiment and negative-result ledger will be extended here as runs finish.

## Registered development screens (6 November 2026)

- `discovery_v1`: geometrically central 32 x 32 patch of Indian Pines, with
  all 200 retained bands; central photon-energy slice of the PyARPES measured
  `photon_energy.nc` (160 angles x 212 binding energies). Rates 50/80/95%,
  mask seed 710. Baselines: band interpolation, normalized Gaussian
  interpolation, distance-weighted KNN, SoftImpute, sum-of-unfolding-nuclear-
  norms completion (HaLRTC), and tensor nuclear norm completion (TNN, 3D
  only). Tuning uses inner observed-entry holdouts, including the networks'
  global/column/row/square-root preprocessing. Source and inputs are hashed.
  One duplicate neural worker was inadvertently launched and finished;
  both used exactly the same deterministic inputs/settings. Neither was a
  different candidate or selected by test performance.
- `sdss_discovery_v1`: measured SDSS galaxy spectra from astroML's
  `spec4000.npz`. Crucially, use only `spectra`, the original quality `mask`,
  and wavelength coordinates; never use the supplied PCA reconstruction,
  coefficients, eigenvectors, norms, or class labels. These are calibrated,
  rest-frame rebinned measurements, not new synthetic spectra. Select
  every second wavelength bin from indices 100:900 and retain objects with
  >=90% valid bins; take the first 256 eligible objects in seed-860 random
  order. Reserve the next 256 objects for confirmation (same survey, not
  a new instrument). Rates 50/80/95%, seed 710. Keep negative measured flux.
  All methods tune on observed validation only; include flux normalization,
  signed asinh, standardization, KNN, spectral interpolation/smoothing,
  SoftImpute and iterative PCA. Never evaluate on source-quality-masked bins.

For these astrophysical data, rank-restricted iterative PCA is a domain-
appropriate baseline; the astroML distribution itself illustrates gap filling
with iterative PCA. Its provided fit is intentionally not reused because it
was fitted before the new masks.

Implementations and full prediction artifacts live in this repository and
`/mnt/matrixformer_scratch/measured_physics_20261106`, respectively. Unit tests
check observed-value preservation, proximal operators, measured masks,
disjoint object splits and validation masking. The report script recomputes
scores from saved predictions and checks that settings minimize observed
validation error. Confidence intervals must use seed-level means (conditional
on the fixed measured scenes), not individual pixels as independent datasets.

## Frozen hyperspectral confirmation

Before inspecting any Salinas or Pavia completion scores, freeze the identical
`evaluate_measured_physics.py` pipeline and all its grids on the geometrically
central 32 x 32 patches of both scenes, all retained bands, rates 50/80/95%,
and new mask seeds 711/712/713. Run all eight methods, not just the baselines
that lost on Indian Pines. Each scene/seed has a separate immutable protocol
under `confirmation_{salinas,pavia}_s{711,712,713}` for parallel execution.
Report both networks separately and all rates, even if the proposed sparse-
acquisition advantage does not reproduce. These are different measured scenes;
Pavia also uses a different instrument (ROSIS instead of AVIRIS).

**Input-validation failure, before confirmation scoring:** Salinas has 61
small negative calibrated values (minimum -7) in the fixed 208,896-entry
patch. The first evaluator correctly rejected it before writing any cases:
its square-root intensity transform requires nonnegative values. Do not
silently discard these measurements or replace their targets. Pavia has
finite nonnegative data and proceeds under the unchanged frozen pipeline.
Salinas needs an explicitly documented signed-intensity protocol before it
can be evaluated. Its empty output directory records the failed attempt.

## Frozen SDSS confirmation and solver checks

The first SDSS discovery mask at 95% missingness gives 0.1240 for
MatrixFormer-Impute and 0.1288 for MatrixFormer versus 0.1509 for the initial
SoftImpute implementation. This is exploratory, not yet a confirmation.
Freeze the same signed-flux pipeline on the reserved next 256 galaxy spectra,
all three rates and new seeds 711/712/713, before inspecting their outcomes.
Run all seven methods and retain every condition. In parallel, rerun the
entire original SoftImpute validation grid with a 3,000-iteration limit and
stricter fixed-point and iterate tolerances (1e-7), because some sparse-case
fits reached the original iteration cap. The Gram-eigendecomposition
proximal operator is tested against full SVD; this changes numerical cost,
not the nuclear-norm objective. Do not count an underconverged baseline as
evidence of a neural advantage.

Additional baseline safeguards: tensor controls extend ADMM initialization
to 1e-4/1e-3/1e-2 and 500 iterations, because the initial best setting was
on the lower grid boundary. The SDSS PCA control repeats the original
rank/transform grid with exact projections, gradual rank increase and up to
2,000 iterations. Retain the original capped fits as well. All extensions
are selected on the same observed-only validation entries; no test errors
select their settings. Report scripts withhold comparisons for incomplete
runs and recompute every score from archived predictions.

## Source references

- Stansbury and Lanzara, *PyARPES: An analysis framework for multimodal
  angle-resolved photoemission spectroscopies* (2020),
  DOI: `10.1016/j.softx.2020.100472`; data from `chstan/pyarpes-datasets`.
- Roy et al., *HybridSN: Exploring 3D-2D CNN Feature Hierarchy for
  Hyperspectral Image Classification* (2020). The authors' `HybridSN`
  repository mirrors the standard AVIRIS Indian Pines/Salinas and ROSIS
  Pavia University measured scenes; classification labels are not used.
- York et al., *The Sloan Digital Sky Survey: Technical Summary* (2000),
  DOI: `10.1086/301513`; Abazajian et al., *The Seventh Data Release of the
  Sloan Digital Sky Survey* (2009), DOI: `10.1088/0067-0049/182/2/543`.
- VanderPlas et al., *Introduction to astroML: Machine learning for
  astrophysics* (2012), DOI: `10.1109/CIDU.2012.6382200`. The source loader
  points to DR7 spectral products; the preparation code retains the measured
  rebinned spectra separately from PCA fits and flags unreliable bins.
- Liu et al., *Tensor Completion for Estimating Missing Values in Visual
  Data* (2013), DOI: `10.1109/TPAMI.2012.39` (HaLRTC).
- Canyi Lu's `LibADMM` supplies reference `lrtc_snn.m`, `lrtc_tnn.m`, and
  `prox_tnn.m`; our tensor proximal operator and update equations were
  checked against those public implementations. No MATLAB code was executed.

## Confirmed outcome

The measured hyperspectral branch meets the working success criterion:
combined MatrixFormer improves over the best single classical comparator
by 11.8% overall on development Indian Pines and approximately 20% on the
reserved Pavia scene. The sparse-regime advantage also reproduces. All
99 Pavia method-case records, including extended SoftImpute and tensor
controls, are complete and prediction-audited. The strongest classical
method at 95% missingness is Gaussian interpolation, not SoftImpute; the
combined model still improves over it by approximately 21%. At 50%, the
classical method wins. See `MEASURED_PHYSICS_FINDINGS.md` for full details.

The paper now contains a measured-optical-spectroscopy subsection, a
matching-style bar plot with mask-seed confidence intervals, an appendix,
citations, and exported audit/protocol files. Its main plot uses the
combined checkpoint only. The old cylinder draft remains inactive. The
PDF compiles successfully and the new research tests pass (21 tests).
No commit or push has been performed.

The separate SDSS branch is now complete: all 81 confirmation records and
27 development records, including longer-budget controls, pass the saved-
prediction audit. Final reports are `reports/sdss_confirmation_final` and
`reports/sdss_discovery_final` under the experiment root. The specialist's
95%-missing confirmation error is 14.6% lower than extended-budget
SoftImpute; the combined checkpoint is approximately tied with that
baseline overall. Retain both original and extended PCA results because
longer optimization is not consistently better. See `ASTRONOMY_FINDINGS.md`
for the complete results and limitations. Do not silently substitute the
specialist for the combined checkpoint or interpret noisy-flux completion
as validated denoising. No astronomy section has been added to the paper.
