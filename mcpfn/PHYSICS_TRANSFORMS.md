# Observed-only physics adaptation

This exploratory experiment evaluates the frozen **MatrixFormer-Impute**
checkpoint on cylinder flow and simulated quantum tomography. MRI is excluded.
It does not modify the original pilots, train a model, or update the paper.

## Motivation and candidates

The original specialist averages 0.16392 cylinder NRMSE versus 0.25519 for
SoftImpute, but loses on several easier cases. Flow has spatially varying
offsets/amplitudes and smooth, approximately low-rank temporal dynamics.
We test observed-only row centering, row standardization, and time-lag
augmentation. A second stage optionally alternates rank projection and
observation restoration, initialized by the neural completion. A row-mean
initialized control separates the effect of the network from that of the
rank-constrained solver.

On tomography, the original specialist averages 0.42029 fidelity versus
0.80179 for PSD least squares. A low-rank density matrix need not be low-rank
after folding its Pauli coefficients into a matrix. We test removing the
deterministic identity from neural normalization and averaging three tensor
unfoldings. The physical refinement solves

\[
\min_{\rho\succeq0,\ \mathrm{Tr}(\rho)=1}
\frac{1}{|O|}\sum_{P\in O}(\mathrm{Tr}(\rho P)-y_P)^2
+\frac{\lambda}{|H|}\sum_{P\in H}(\mathrm{Tr}(\rho P)-\widehat c_P)^2.
\]

Here \(O\) excludes identity, \(H\) contains unmeasured Pauli observables,
and \(\widehat c\) is the clipped frozen neural prediction. The zero-prior
control replaces it with zero. At zero regularization, the network has no
effect; any gain there is a physical-solver gain, not a neural gain.

## Safeguards

- All 18 cases per domain are retained, with the original masks and shot noise.
- Candidate grids and source/input/checkpoint hashes are frozen before inference.
- Choose transforms, ranks and regularization using only temporarily hidden
  **observed** entries. Never choose using test RMSE or fidelity.
- Refit preprocessing and inference on all observations after selection.
- Mask validation cells before centering, standardization, or lag duplication.
- Retain fixed-transform results as well as selected hybrids and no-neural controls.
- Do not call nonlinear-refinement predictions native calibrated distributions.
- Treat this as exploratory: original benchmark scores motivated the candidates.
  Random-cell validation may not represent long outages; quantum validation has
  only 5 or 12 noisy coefficients per case.

## Reproduce

From `/root/tabular`:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
output=/mnt/matrixformer_scratch/physics_transforms_NEW_RUN
python mcpfn/scripts/evaluate_physics_transforms.py plan --output "$output"
python mcpfn/scripts/evaluate_physics_transforms.py run --output "$output"
python mcpfn/scripts/report_physics_transforms.py "$output"
python -m pytest -q mcpfn/tests/test_physics_transforms.py
```

The runner can resume a frozen campaign, verifying completed artifact hashes.
Use `--benchmarks cylinder` or `--benchmarks quantum` to execute one domain;
the final report requires all 36 cases. No MRI command exists in this runner.

Current output: `/mnt/matrixformer_scratch/physics_transforms_20261106`.
`RESULTS.md` contains the full comparison. `cases/` stores predictions,
validation masks, all candidate validation scores, selections and timings.
`selected_settings.csv` and the per-case/by-condition CSVs retain the details.
`audit.json` checks original input hashes, re-scores saved predictions, verifies
the validation masks, and checks that every selected setting minimizes the
corresponding observed-validation objective.

## Quantum moment-matrix follow-up

The separate, fixed-recipe follow-up uses the Pauli moment matrix

\[
G_{ab}=\mathrm{Tr}(\rho P_aP_b)
=\eta_{ab}c_{a\oplus b},\qquad
\mathcal R(G)=\begin{pmatrix}\Re G&-\Im G\\\Im G&\Re G\end{pmatrix}.
\]

For four qubits, \(G\) is 256 by 256 and its realification is 512 by 512.
The product phases \(\eta\) and indices are fixed Pauli algebra, not learned
from ground truth. For noiseless physical coefficients, \(G\) is PSD and
has rank \(16\,\mathrm{rank}(\rho)\), so its realification has rank 32 for
a pure state or 64 for a rank-two state. This contrasts with the original
16-by-16 folding, which need not be low-rank even for pure states.

Every unavailable coefficient is masked in **all** of its repeated entries.
Algebraically zero real/imaginary entries are known zeros, not observations
of hidden targets. Decode predictions by averaging signed repeated estimates,
restore measured coefficients, and apply the original PSD density projection.
Also evaluate the same observed-validation-selected physical refinement.
Noisy measured moment matrices are not necessarily PSD or exactly low-rank.

```bash
output=/mnt/matrixformer_scratch/quantum_moments_NEW_RUN
python mcpfn/scripts/evaluate_quantum_moments.py plan --output "$output"
python mcpfn/scripts/evaluate_quantum_moments.py run --output "$output"
python -m pytest -q mcpfn/tests/test_quantum_moments.py
```

Current output: `/mnt/matrixformer_scratch/quantum_moments_20261106`.
Its 18 cases and validation masks match the first quantum adaptation experiment.
This is another exploratory hypothesis, not an independent confirmation set.

## Matched SoftImpute controls and paper checkpoint

`evaluate_cylinder_refinement_controls.py` runs the same refinement after
SoftImpute and evaluates the combined MatrixFormer checkpoint used in the
paper. The original specialist results remain unchanged and are not silently
relabeled as combined-model results.

The direct SoftImpute control keeps its original observed-validation-selected
shrinkage, then selects rank or no refinement. The matched control gives
SoftImpute all four neural input transforms and the same rank grid, plus its
three shrinkage choices. The combined model has the same observed-validation
selection rule as the specialist. The reporter also derives a direct
global-input refinement from the combined model's already evaluated rank grid.
All settings are selected without outer hidden targets.

Gappy POD is the traditional flow-reconstruction baseline family, established
by Everson and Sirovich (1995), DOI `10.1364/JOSAA.12.001657`, and studied for
cylinder flow by Venturi and Karniadakis (2004), DOI
`10.1017/S0022112004001338`. Our archived POD/SVD comparator is a simpler
iterative-SVD implementation, not a reproduction of their weighted modal fits.
SoftImpute is the general-purpose matrix-completion comparator. Its shrinkage
bias can be reduced by hard-rank refinement too; refinement gains are not
automatically evidence of a uniquely neural advantage.

```bash
output=/mnt/matrixformer_scratch/cylinder_refinement_controls_NEW_RUN
python mcpfn/scripts/evaluate_cylinder_refinement_controls.py plan --output "$output"
python mcpfn/scripts/evaluate_cylinder_refinement_controls.py run --output "$output" --method softimpute
python mcpfn/scripts/evaluate_cylinder_refinement_controls.py run --output "$output" --method matrixformer
python mcpfn/scripts/report_cylinder_refinement_controls.py "$output"
python /root/MatrixFormerPaper/scripts/plot_cylinder_refinement.py --source "$output"
```

Current output: `/mnt/matrixformer_scratch/cylinder_refinement_controls_20261106`.
An explicitly recorded amendment corrects only the replay assertion's tolerance
for the original float32 prediction archives; fitting and selection are unchanged.
The initial protocol and both source snapshots are retained.

The paper plots average fixed missingness rates within each seed before forming
95% Student-t intervals across three seed means. They do not treat entries or
seed-condition pairs as independent flow datasets. Condition-level plots and
all trial choices are retained; paper figures use the combined model and the
shared manuscript figure style.
