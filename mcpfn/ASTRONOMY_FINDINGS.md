# Astronomy: completing real SDSS galaxy spectra

## Finding

Real astronomical spectra are a natural matrix-completion application:
rows are objects, columns are aligned wavelengths, and entries are measured
fluxes. **MatrixFormer-Impute has a promising sparse-data advantage in this
experiment; the combined MatrixFormer does not establish a clear win.**
No astronomy-specific training, neural-output rank refinement, ensemble,
transpose averaging, or synthetic spectra are used. MRI is excluded.

At 95% missingness, the specialist's mean normalized RMSE is **0.12139**
versus **0.14212** for extended-budget SoftImpute, a **14.6% reduction**.
Across all three missingness rates, the reduction is 7.4%. These are
comparisons with the strongest single evaluated classical method by mean
error, not claims of superiority over all astronomical reconstruction methods.

## Data and frozen evaluation

- Actual SDSS DR7 calibrated, rest-frame rebinned galaxy measurements from
  astroML's `spec4000.npz`; source revisions and SHA256s are archived.
- Use only the measured `spectra`, original quality `mask`, and wavelength
  coordinates. Never use supplied PCA fits, coefficients, eigenvectors,
  source normalization factors, or class labels.
- Select wavelength indices `100:900:2` (400 columns). Require at least 90%
  valid bins. Shuffle eligible objects with seed 860; use 256 development
  objects and the next, disjoint 256 objects for confirmation.
- Development mask seed 710; confirmation seeds 711, 712, 713. Artificially
  withhold 50%, 80%, or 95% of valid measured entries. Originally flagged
  bins are neither observations nor evaluation targets. Keep negative flux.
- Select transforms and method settings using a 10% holdout of the remaining
  observed entries only. Refit normalization after this inner masking.
  Candidates include global, column, and row standardization, observed
  row-flux normalization, and a signed asinh transform.
- Metric: held-out flux RMSE divided by the population standard deviation
  of globally pooled observed flux. This is not the columnwise MissBench
  normalization. Per-object flux-normalized errors are also archived.
- This is confirmation on different objects from the same survey, not a
  second survey or independent instrument.

All **81 confirmation records** and **27 development records** are complete.
The audit recomputes scores from saved predictions, checks preservation of
observed entries and exclusion of quality-masked targets, and checks that
each selected setting minimizes observed-validation error.

## Confirmation results

Normalized RMSE, lower is better. Entries are mean +/- Student-t 95%
confidence half-width across **three mask seeds**, conditional on the fixed
256 objects. For Overall, average the three rates within each seed first.
Neither pixels nor the nine rate-seed combinations are independent datasets.

| Method | 50% missing | 80% missing | 95% missing | Overall |
|---|---:|---:|---:|---:|
| MatrixFormer | 0.08505 +/- 0.00594 | 0.09881 +/- 0.01075 | 0.14031 +/- 0.05911 | 0.10806 +/- 0.02224 |
| MatrixFormer-Impute | 0.08279 +/- 0.00442 | **0.09664 +/- 0.01478** | **0.12139 +/- 0.00626** | **0.10027 +/- 0.00710** |
| SoftImpute, original budget | **0.08233 +/- 0.00369** | 0.10047 +/- 0.01228 | 0.14384 +/- 0.01065 | 0.10888 +/- 0.00320 |
| SoftImpute, extended budget | 0.08233 +/- 0.00368 | 0.10048 +/- 0.01228 | 0.14212 +/- 0.00409 | 0.10831 +/- 0.00186 |
| Iterative PCA | 0.10293 +/- 0.02556 | 0.10083 +/- 0.00071 | 0.21292 +/- 0.02661 | 0.13890 +/- 0.01294 |
| Iterative PCA, extended | 0.09323 +/- 0.01391 | 0.11251 +/- 0.04970 | 0.21413 +/- 0.35808 | 0.13996 +/- 0.10645 |
| KNN | 0.09859 +/- 0.00833 | 0.11815 +/- 0.00401 | 0.22208 +/- 0.00750 | 0.14628 +/- 0.00414 |
| Spectral interpolation | 0.12707 +/- 0.00438 | 0.14266 +/- 0.00156 | 0.17219 +/- 0.01844 | 0.14731 +/- 0.00669 |
| Spectral smoothing | 0.12588 +/- 0.00516 | 0.13717 +/- 0.00165 | 0.26796 +/- 0.03158 | 0.17700 +/- 0.01007 |

Bold indicates the lowest unrounded mean, not statistical significance.
At 50%, the two SoftImpute means differ by less than 0.000001.

The extended SoftImpute check repeats the original validation grid with up
to 3,000 iterations and tighter tolerances. Some fits reach the cap; the
CSV label `SoftImpute (converged)` does **not** guarantee convergence.
Extended PCA repeats its grid with exact projections, gradual rank increase,
and up to 2,000 iterations. It is unstable in some cases; both original and
extended versions remain in the report and figure, rather than replacing
the original with a weaker comparator.

In the predeclared sparse regime (80%/95%), the specialist's error is
0.10901 versus 0.12130 for extended-budget SoftImpute (10.1% lower).
Even against a per-case oracle choosing the best test-scoring classical
method, the specialist's reduction is 4.8% overall and 6.7% in the sparse
regime. This oracle is a conservative reference, not a deployable selection
rule. The combined model does not beat that oracle.

## Realism check: missingness rates and geometry

The 50%/80%/95% independent random holdouts are stress tests, not a measured
model of ordinary SDSS data loss. The positive 95% result alone does not
establish usefulness for astronomical data reduction.

An audit of the distributed archive finds **15.52175% flagged bins** across
all 4,000 x 1,000 entries (median per-object fraction 15.2%; range
5.9%-38.9%). Our cropped, quality-selected confirmation subset has only
**5.27246% originally flagged bins**, with median 5.0% and maximum 10.0%
per object. The latter maximum follows from the evaluation's eligibility
rule, so this subset is not representative of the complete survey.

These are astroML preparation masks, **not raw SDSS instrument bitmasks or
a survey-wide prevalence estimate**. The pinned preparation calls
`compute_mask(0.5, 5)`, which flags noise spikes, invalid/large errors and
zero flux, then expands the affected regions. Coverage boundaries can
therefore contribute. These structured masks are different from uniformly
discarding independent valid bins, and low S/N is not itself missingness.

A practically motivated follow-up should use actual survey quality masks
and exposure uncertainties. Two appropriate tasks are repairing localized
bad-pixel/sky-residual gaps and reconstructing unreliable wavelength regions
in low-S/N exposures. Use an independent repeat exposure as a reference
where available. Replaying measured masks on high-quality spectra is a
useful controlled fallback, but remains an artificial-masking experiment.
Preserve the observed-wavelength geometry before any rest-frame alignment;
do not arbitrarily position gaps or choose rates for favorable model scores.
Tune on development objects with the same corruption mechanism, and retain
negative results on reserved objects. Evaluate spectral-line fluxes and
ratios as well as pixel error. No such realistic-mask or repeat-exposure
evaluation has been run yet.

## What this says about noisy astronomy data

The measurements are real and noisy, but the targets are those same noisy
measurements at withheld entries. **Lower completion RMSE does not establish
denoising of a latent noise-free spectrum.** The astroML archive does not
include the per-pixel inverse variances needed for a proper heteroscedastic
noise model. Its prepared spectra also do not directly test raw detector
artifacts or wavelength/redshift alignment.

The strongest next experiment is repeated SDSS or DESI spectroscopy:

1. Use real low-S/N individual exposures with flux, inverse variance, masks,
   and resolution information; freeze object splits, S/N groups, wavelength
   alignment, missingness, transforms, and endpoints before confirmation.
2. Construct a higher-S/N reference from independent repeats, **excluding
   the input exposure from the reference coadd** to avoid shared-noise leakage.
   The reference is still noisy, not exact ground truth.
3. Compare with inverse-variance-weighted PCA/probabilistic factor models,
   tuned matrix completion, and an appropriate spectral smoother or GP.
4. Evaluate reconstruction and emission/absorption-line preservation, plus
   CRPS/log score and interval coverage after accounting for measurement
   noise. Do not identify the model's missing-value distribution with a
   calibrated noisy-observation likelihood without testing that assumption.

Official DESI documentation confirms that `spectra` files contain non-coadded
individual exposures and FLUX, IVAR, MASK, and RESOLUTION HDUs. SDSS also
documents inverse variance and masks. These products make a noise-aware
follow-up feasible; they have **not** been downloaded or evaluated here.

## Files and reproduction

Experiment root: `/mnt/matrixformer_scratch/measured_physics_20261106/`.

- Data provenance: `data/sdss/manifest.json`.
- Audited confirmation: `reports/sdss_confirmation_final/` (`scores.csv`,
  `by_rate.csv`, `overall.csv`, `comparisons.csv`, and `audit.json`).
- Audited development: `reports/sdss_discovery_final/`.
- Figures: `figures/sdss_all_rates.pdf` and `figures/sdss_sparse.pdf`, with PNGs.
- Each original/control directory retains its protocol, inputs, per-method
  validation records, scores, and prediction arrays.
- Code is in `/root/tabular/mcpfn/scripts/`: `fetch_measured_sdss.py`,
  `evaluate_sdss_completion.py`, `refine_measured_softimpute.py`,
  `refine_sdss_pca.py`, `report_measured_physics.py`, `plot_measured_physics.py`.

To audit and regenerate the confirmation plots without rerunning inference:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=/mnt/matrixformer_scratch/measured_physics_20261106
python mcpfn/scripts/report_measured_physics.py \
  --runs "$root"/sdss_confirmation_s{711,712,713} \
    "$root"/controls_sdss_confirmation_s{711,712,713} \
    "$root"/pca_sdss_confirmation_s{711,712,713} \
  --output "$root/reports/sdss_confirmation_final"
python mcpfn/scripts/plot_measured_physics.py \
  --report "$root/reports/sdss_confirmation_final" --dataset sdss \
  --output "$root/figures"
```

No astronomy section has been added to the paper; no commit or push has
been performed for these experiments.

## Sources

- York et al. (2000), *The Sloan Digital Sky Survey: Technical Summary*,
  DOI: [10.1086/301513](https://doi.org/10.1086/301513).
- Abazajian et al. (2009), *The Seventh Data Release of the Sloan Digital
  Sky Survey*, DOI: [10.1088/0067-0049/182/2/543](https://doi.org/10.1088/0067-0049/182/2/543).
- VanderPlas et al. (2012), *Introduction to astroML: Machine learning for
  astrophysics*, DOI: [10.1109/CIDU.2012.6382200](https://doi.org/10.1109/CIDU.2012.6382200).
- [Pinned astroML preparation code](https://github.com/astroML/astroML/blob/d7485a8acd8d01d927fc381e3a58653e6cd47865/examples/datasets/compute_sdss_pca.py).
- [SDSS spectral data documentation](https://www.sdss4.org/dr17/spectro/spectro_basics/).
- [SDSS spectroscopic caveats](https://www.sdss4.org/dr17/spectro/caveats/),
  including night-sky emission, sky subtraction, and clipped emission lines.
- [Pinned astroML mask implementation](https://github.com/astroML/astroML/blob/d7485a8acd8d01d927fc381e3a58653e6cd47865/astroML/datasets/tools/sdss_fits.py).
- [DESI individual-exposure spectra data model](https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/healpix/SURVEY/PROGRAM/PIXGROUP/PIXNUM/spectra-SURVEY-PROGRAM-PIXNUM.html).
