# Phantom MRI: raw k-space versus the lifted S-matrix

The old MRI spectrum script survives at
`/root/matrix-completion-benchmarks/scripts/plot_phantom_mri_spectra.py`, but
neither its input matrices nor its results were found. This experiment
**recreates a standard synthetic phantom**, rather than claiming to reuse
those unavailable matrices. This protocol is fixed before inspecting scores.

## Sources

- The modified Shepp–Logan phantom from
  [scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.shepp_logan_phantom),
  pinned to 0.25.2, supplies the image amplitude.
- Haldar, *Low-Rank Modeling of Local k-Space Neighborhoods (LORAKS) for
  Constrained MRI*, IEEE TMI 33(3), 668–681 (2014),
  [doi:10.1109/TMI.2013.2293974](https://doi.org/10.1109/TMI.2013.2293974).
- [PyMRItools LORAKS operators](https://github.com/schmidt-jo/PyMRItools/blob/dd50a29f8f256420cd97be7808a61fbfc61d1d26/pymritools/recon/loraks/operators.py)
  provide the reference S-matrix block convention. Our NumPy lifting and
  overlap-averaging inverse are independently implemented from this linear
  mapping. The baseline is **LORAKS-style alternating rank projection**, not
  a run of the full upstream AC-LORAKS optimizer or a published-score replication.

## Data and acquisition

- One **64 × 64**, single-coil, noiseless complex phantom. Resize the standard
  phantom with anti-aliasing; multiply its amplitude by `exp(i * phase)`, with
  `phase = 0.6*x + 0.4*y + 0.3*x*y + 0.2*(x*x - y*y)` and coordinates
  in `[-1, 1]`. This fixed smooth phase avoids assuming a purely real image
  with exact conjugate-symmetric k-space.
- Use centered, orthonormal 2D FFTs. There is no row-wise training/test split,
  fitting to full-data statistics, neural training, or checkpoint selection.
- Two acquisition patterns: random **k-space points** with an 8 × 8 fully
  sampled calibration square, and random **Cartesian phase-encode lines**
  with eight fully sampled central lines.
- Missing fractions are 50% and 75% (nominal acceleration 2× and 4×), each
  with mask seeds 0, 1, 2: **12 cases**. Exact budgets include calibration;
  the 4× samples are nested in the 2× samples. No hidden measurements are
  added to repair an empty row or column.

## Lifting and leakage prevention

Use circular radius-two neighborhoods (13 offsets). For each valid symmetric
center `c` and offset `b`, form `P[c,b] = k[c+b]` and `M[c,b] = k[-c+b]`.
The real, tall S-matrix is

```text
S(k) = [ Re(P-M)  -Im(P-M) ]
       [ Im(P+M)   Re(P+M) ].
```

This is the transpose of the reference implementation's wide S-matrix;
transposition leaves its singular values unchanged. For size 64 and radius 2
there are 59² valid centers, giving a **6,962 × 26** matrix. The excluded
Nyquist row/column and twelve extreme corner points are not represented in
this valid circular-neighborhood lift; uncovered missing frequencies remain
zero, and their count is recorded.

**Mask k-space first, then lift.** A lifted entry is supplied only when both
of its contributing k-space samples were acquired. All repeated appearances
of a missing measurement therefore remain hidden. Independently masking
entries of the complete S-matrix would leak measurements and is not used.
The resulting lifted missing fraction can be much larger than the acquisition
missing fraction. The direct S-input adapter cannot expose a partially known
sum/difference when only one contributor was measured; the iterative physics
baseline can use that measurement through data-consistency constraints.

To invert a filled S-matrix, recover its proposed P/M neighborhoods from the
four quadrants and average all proposals for each original k-space location.
Restore measured k-space values exactly. This projection reconciles duplicate
predictions without using hidden truth. Metrics score each original frequency
or image pixel once, not each repeated lifted entry.

## Methods

1. **Zero-fill**: keep acquired k-space and set the rest to zero.
2. **Structured low-rank S**: alternate truncated SVD of S(k), overlap-averaging
   inversion, and exact measured-data consistency. Start at zero-fill; use
   100 iterations and relative update tolerance `1e-5`. Choose rank from
   4/8/12/18/24 using 10% held-out noncalibration observations, then refit using
   all acquired measurements. Hold out whole lines for Cartesian acquisition
   and individual frequencies for random-point acquisition. Never choose rank
   from the complete phantom's spectrum or outer reconstruction error.
3. **MatrixFormer raw**: concatenate real and imaginary k-space along columns
   into a 64 × 128 real matrix, complete with the final combined V57 checkpoint,
   and restore acquired values.
4. **MatrixFormer S**: complete the 6,962 × 26 lifted input using the same frozen
   checkpoint, restore known S entries, invert the lift, and restore acquired
   k-space values.

Both neural paths use a global mean and population SD computed only from
their actual observed input entries. They use full matrix context, not row
chunking, transposition averaging, or normalization from the full phantom.
Only the final decoder is microbatched. The raw-matrix model has no explicit
frequency coordinates; the lifted representation encodes neighborhood relations.

## Scores, spectra, and limitations

- Complex-image relative L2 error, `||image_hat-image||_F / ||image||_F`, is the
  primary **NRMSE**. This is not the tabular benchmark's SD-normalized metric.
- Also report magnitude-image NRMSE, magnitude SSIM (ground-truth magnitude
  range), PSNR, hidden complex k-space RMSE/relative error, and maximum error
  on acquired measurements. Metrics require exact data consistency.
- Reuse the old spectrum script on saved full phantom k-space and S-matrix.
  Those ground-truth spectra are diagnostic only, never hyperparameter inputs.
  Save reconstructed spectra, image/error comparisons, and mask plots too.
- This is a point-reconstruction pilot, not an uncertainty-calibration study.
  Native coefficient distributions are not automatically an image posterior.
- Three masks on one small phantom are not three subjects or a clinical MRI
  benchmark. No claims about noisy, multi-coil, in-vivo, or fastMRI performance.

## Reproduce

```bash
python -m pip install 'scikit-image==0.25.2'
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest mcpfn/tests/test_mri_completion.py -q
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/evaluate_mri_completion.py
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/report_mri_completion.py \
  mcpfn/runs/physics_mri_20261027/pilot
```

All results are saved under
`/root/tabular/mcpfn/runs/physics_mri_20261027/pilot/`, including `RESULTS.md`,
source/checkpoint hashes, input matrices, predictions, masks, rank-selection
trials, spectra, figures, CSV scores, and an independent artifact audit.
