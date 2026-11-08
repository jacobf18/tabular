# A real measured-data advantage: sparse optical spectroscopy

## Conclusion

The frozen **combined MatrixFormer** has a reproducible advantage in
retrospective completion of real airborne hyperspectral measurements.
Indian Pines (AVIRIS) was the development scene; the reserved Pavia
University scene (ROSIS, a different instrument) confirms the advantage
with three new mask seeds. **No physics-specific training, transpose
averaging, ensemble, or low-rank refinement of the neural output is used.**
MRI is excluded.

This is not a synthetic physics simulation: the target intensities are
measured, publicly distributed sensor observations. Only the withholding
of measured entries is artificial. The claim is limited to these fixed
patches and this retrospective missingness protocol, not all hyperspectral
images, new real-time acquisitions, or every specialized restoration method.

## Reserved Pavia results

The fixed central 32 x 32 crop has 103 measured bands: a 1,024 x 103
matrix. Results are means across three independent mask seeds. NRMSE is
hidden-entry RMSE divided by the population standard deviation of the
globally pooled observed values; lower is better.

| Missing entries | Combined MatrixFormer | MatrixFormer-Impute | Best displayed classical baseline |
|---|---:|---:|---:|
| 50% | 0.06017 | 0.06005 | **0.05038**, SoftImpute |
| 80% | **0.07187** | 0.07379 | 0.07560, SoftImpute |
| 95% | 0.23537 | **0.22731** | 0.29881, Gaussian interpolation |

The displayed SoftImpute values use the extended numerical controls. The
original shorter fits and all other methods remain in the CSVs. At 95%
missingness, the combined model and specialist reduce error by about
**21% and 24%**, respectively, versus the strongest classical method at
that rate. At 50%, SoftImpute is better; this tradeoff is explicitly shown.

Across all three rates, combined MatrixFormer gives **0.12247 +/- 0.00657**
versus **0.15325 +/- 0.01299** for extended-budget SoftImpute, the best
single classical method overall. In the predeclared 80%/95% sparse regime,
the values are **0.15362 +/- 0.00923** versus **0.20468 +/- 0.01955**.
These are Student-t 95% confidence half-widths across **three seed means**,
averaging rates within a seed first. They are conditional on the fixed
scene, not intervals across 103 wavelengths, 1,024 pixels, or nine
independent images.

The combined model also beats the per-case oracle that is allowed to
choose the lowest test error among all evaluated classical configurations:
13.5% lower overall and 17.9% lower in the sparse regime. That oracle is a
conservative comparison only, never a deployable setting-selection rule.

## Development, controls, and retained failures

- Indian Pines: combined MatrixFormer has overall NRMSE **0.05630** versus
  **0.06384** for the best single classical comparator (11.8% lower), and
  **0.06156** versus **0.07331** in the sparse regime (16.0% lower). This is
  one exploratory mask seed, not a confidence interval or independent
  confirmation.
- Pavia confirmation: all **99 saved method-case records** are complete
  and audited, including the original and extended numerical controls.
- Baselines include spectral interpolation, normalized spatial/spectral
  Gaussian interpolation, KNN, SoftImpute, HaLRTC, and TNN. Tensor methods
  retain the original spatial-spatial-spectral cube.
- Preprocessing, shrinkage, neighbors and bandwidths are selected
  using temporarily hidden **observed entries only**. Normalization is
  refitted after this inner masking; original hidden targets are untouched.
- SoftImpute repeats the entire grid with up to 3,000 iterations and
  tighter tolerances. Some fits hit the cap, so the artifact's machine
  label `SoftImpute (converged)` means an extended convergence check, not a
  claim of exact optimization. Selected Pavia fixed-point residuals are
  below 1e-6; retaining both numerical budgets does not remove the win.
- Tensor controls extend ADMM initializations toward reference defaults
  and raise the iteration budget to 500. The result survives these checks.
- The PyARPES photon-energy development screen **does not** establish a
  useful overall advantage: combined-model NRMSE is 0.18057 versus
  0.13347 for Gaussian interpolation. It remains in the results ledger.
- Salinas is **not evaluated**: the initial nonnegative-intensity loader
  rejected small negative calibrated measurements before scoring. Those
  values were not deleted, changed into targets, or silently clipped.
- The previous cylinder and quantum attempts remain archived. They are
  not the evidence for this new measured-data claim.

SDSS galaxy spectra are a separate measured-data search branch, now complete
with all 81 confirmation records audited, including extended numerical
controls. On 256 reserved objects and three new mask seeds, at 95% missingness
MatrixFormer-Impute gives 0.12139 versus 0.14212 for extended-budget
SoftImpute (14.6% lower). The combined model gives 0.14031 and does not
establish a clear advantage. This is completion of noisy measured flux,
not a validated denoising result. See `ASTRONOMY_FINDINGS.md`. These results
are not required for or substituted into the confirmed Pavia claim.

## Saved artifacts and reproduction

Research guide with sources and the pre-result design history:
`/root/tabular/mcpfn/MEASURED_PHYSICS_SEARCH.md`.

All downloads, immutable protocols, masks, prediction arrays, validation
decisions, timings, hashes and reports:
`/mnt/matrixformer_scratch/measured_physics_20261106/`.

Main audited report:
`/mnt/matrixformer_scratch/measured_physics_20261106/reports/pavia_final/`.

Figures including both frozen checkpoints:
`/mnt/matrixformer_scratch/measured_physics_20261106/figures/pavia_all_rates.pdf`
and `pavia_sparse.pdf` in the same directory (PNG copies also exist).

The paper's main figure uses only the combined model, retaining the
specialist in the discussion and supplementary artifacts. The new section
and appendix are `sections/measured_spectroscopy_results.tex` and
`sections/measured_spectroscopy_appendix.tex` in `/root/MatrixFormerPaper`.
The weak cylinder draft remains inactive. No changes have been committed
or pushed for this request.

All experiment, audit, plotting and export code is retained in the main
research repository, not only in a temporary notebook:

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python mcpfn/scripts/fetch_measured_physics.py \
  --output /mnt/matrixformer_scratch/measured_physics_20261106/data

python mcpfn/scripts/evaluate_measured_physics.py register \
  --output /mnt/matrixformer_scratch/NEW_RUN \
  --datasets pavia --seeds 711 712 713
python mcpfn/scripts/evaluate_measured_physics.py run \
  --output /mnt/matrixformer_scratch/NEW_RUN

python mcpfn/scripts/refine_measured_softimpute.py \
  --parent /mnt/matrixformer_scratch/NEW_RUN \
  --output /mnt/matrixformer_scratch/NEW_SOFT_CONTROLS
python mcpfn/scripts/refine_measured_tensors.py \
  --parent /mnt/matrixformer_scratch/NEW_RUN \
  --output /mnt/matrixformer_scratch/NEW_TENSOR_CONTROLS

python mcpfn/scripts/report_measured_physics.py \
  --runs /mnt/matrixformer_scratch/NEW_RUN \
    /mnt/matrixformer_scratch/NEW_SOFT_CONTROLS \
    /mnt/matrixformer_scratch/NEW_TENSOR_CONTROLS \
  --output /mnt/matrixformer_scratch/NEW_REPORT
python mcpfn/scripts/plot_measured_physics.py \
  --report /mnt/matrixformer_scratch/NEW_REPORT --dataset pavia \
  --output /mnt/matrixformer_scratch/NEW_FIGURES

python mcpfn/scripts/export_measured_physics_paper.py \
  --report /mnt/matrixformer_scratch/measured_physics_20261106/reports/pavia_final
```
