# Acquired USC phantom: full-resolution multi-coil MRI pilot

This protocol is fixed before running reconstruction comparisons. It is separate
from the earlier **synthetic**, single-coil pilot in `PHYSICS_MRI.md`.

## Source and scope

- User-supplied `/root/phantom_data.mat`, obtained from USC's MRI group:
  <https://mr.usc.edu/download/data/>. The public page identifies the data as
  acquired physical-phantom MRI and specifies CC BY-NC-SA 3.0. The supplied MAT
  file contains no license document; no redistribution of the data is implied.
- SHA256: `8137c274b6e2cc2829c97321e270e1933197bde70873e03739b116950b8c1184`.
- MATLAB variable `kData`: **256 x 340 x 16**, complex single precision. All
  samples are finite; the summed coil-energy maximum is at array index
  `(128, 170)`, consistent with centered k-space.
- Use **all 16 coils, native resolution**, no resizing, spatial crop, coil
  selection, coil compression, denoising, or model fine-tuning. The fully sampled
  acquisition is the reference, not a noiseless ground-truth object.
- One physical object. Three masks per setting measure masking variability,
  not subject-to-subject generalization or clinical utility.

## Acquisition and representations

- Centered orthonormal 2D FFT, along spatial axes only. Root-sum-of-squares
  (RSS) magnitude combines coil images without estimated sensitivities.
- Random point masks with a central 8 x 8 calibration square, and Cartesian
  row-line masks with eight central calibration lines. Mask all coils together.
- Missing fractions 0.50 and 0.75; seeds 0, 1, 2: **12 fixed cases**. Budgets
  are exact. Nested rates use the same random ordering. No row train/test split.
- Radius-two circular neighborhoods (13 offsets); valid symmetric centers
  give **168170 x 416** real S-matrices. Stack all coils' neighborhood columns
  in each S block. For each center, `P = k[c+b]`, `M = k[-c+b]`, and
  `S = [[Re(P-M), -Im(P-M)], [Im(P+M), Re(P+M)]]`.
- Mask k-space **before** lifting. An S cell is observed only if both contributing
  frequencies are acquired. No independent lifted-cell masks or target leakage.
- The raw real-valued representation is **256 x 10880**: flatten spatial-column
  and coil axes, then concatenate real and imaginary parts.
- Invert S by averaging overlapping proposals, restore every measurement
  exactly, and leave unrepresented/unacquired boundary frequencies at zero.
  This valid-center map excludes the two Nyquist edges and some corner samples.

## Frozen methods

1. Zero-filled inverse FFT.
2. Joint multi-coil **structured low-rank S projection**: alternate rank
   truncation, inverse averaging, and measured-data consistency. Select rank
   from **8, 16, 32, 64, 128** using 10% held-out noncalibration observations
   (whole lines for line masks; every coil shares the holdout). Select by complex
   k-space RMSE, then refit all acquired data. Maximum 100 iterations per fit,
   relative-update tolerance 1e-5. Double-precision CUDA linear algebra. This is
   LORAKS-style alternating projection, **not the original optimized LORAKS
   solver or a replication of its published performance**.
3. Frozen **MatrixFormer raw** and **MatrixFormer S**, same combined single
   checkpoint used for the other physics pilots. Expected checkpoint SHA256:
   `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.
   Predict distribution means; this pilot does not assess uncertainty coverage.

Neural normalization uses one observed-only global mean and population SD per
representation, **not** statistics of the reference or of individual chunks.
Use at most **1024 rows x 1024 columns** per forward pass; no transposition,
router, ensemble, or overlapping-tile averaging. When splitting S rows, keep
both real-linear row blocks for each center together. When splitting raw
columns, keep each complex sample's real/imaginary columns together. Split axes
use a fixed, target-independent permutation seeded at 20261027 (rows) and
20261028 (columns). Unsplit axes retain natural order. Thus S uses at most
1024 x 416 and raw at most 256 x 1024; this is **chunked**, not full-matrix
attention. Decode missing entries in microbatches of 1024. Completely unobserved
tiles, if any, receive the observed-global mean and are counted explicitly.

The physics baseline uses the complete joint matrix. It can exploit partially
known sums/differences via measured-data constraints; the neural completion
adapter only marks fully known lifted cells. This is a limitation of the adapter,
not a claim that all acquired information is unavailable to MRI reconstruction.

## Metrics, provenance, and reporting

- Primary: **RSS-image NRMSE**, `||RSS(pred)-RSS(reference)||_2 / ||RSS(reference)||_2`.
  Also report RSS SSIM/PSNR, complex multi-coil relative L2 (equal to full k-space
  relative L2 by Parseval), hidden k-space RMSE/relative L2, and exact data
  consistency. This NRMSE is **not** tabular SD-normalized RMSE.
- Save the input copy/checksum, protocol/source/checkpoint hashes, masks,
  validation masks, every reconstruction, timing, selected ranks, convergence
  traces, errors/failures, individual metrics, and aggregate mean/SD/count.
- Audit saved reconstructions and all reported metrics independently of the
  inference loop. Plot RSS images/errors and raw/lifted singular-value spectra.
  Spectra from Gram eigendecomposition clip negative roundoff eigenvalues to
  zero; the extreme numerical tail is not interpreted as exact rank.
- No selection of favorable coils, cases, ranks from hidden targets, or results.
  Failure records remain on disk; missing cases are not silently omitted.

Reference: J. P. Haldar, “Low-Rank Modeling of Local k-Space Neighborhoods
(LORAKS) for Constrained MRI,” IEEE TMI 33(3), 668–681 (2014),
<https://doi.org/10.1109/TMI.2013.2293974>.

Run from `/root/tabular`:

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/evaluate_real_mri_completion.py \
  --input /root/phantom_data.mat \
  --output /mnt/matrixformer_scratch/physics_mri_real_20261027/full_resolution
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/report_real_mri_completion.py \
  /mnt/matrixformer_scratch/physics_mri_real_20261027/full_resolution
```
