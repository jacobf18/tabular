# Quantum tomography pilot

This is a frozen-model, simulated **compressed-Pauli quantum state tomography**
experiment, not a quantum-hardware result or a reproduction of a published score.
The protocol below is written before evaluating the neural checkpoints.

## Existing code and references

- [Qiskit](https://github.com/Qiskit/qiskit), pinned to **1.4.5**, supplies Pauli
  operators, random quantum states, and state fidelity.
- [Qiskit Experiments' tomography tutorial](https://qiskit-community.github.io/qiskit-experiments/manuals/verification/state_tomography.html)
  describes full state tomography and physicality-constrained fitting. We do
  **not** use its fitter or claim to reproduce its local-setting count protocol.
- Gross, Liu, Flammia, Becker, and Eisert, *Quantum State Tomography via
  Compressed Sensing*, Physical Review Letters **105**, 150401 (2010),
  [arXiv:0909.3304](https://arxiv.org/abs/0909.3304),
  [DOI:10.1103/PhysRevLett.105.150401](https://doi.org/10.1103/PhysRevLett.105.150401).
  This motivates randomly subsampled Pauli measurements; the PSD least-squares
  control below is not an implementation of that paper's recovery algorithm.

## Protocol

- Four qubits: a complex **16 × 16 density matrix** and 256 real tensor-product
  Pauli expectations. Fold the coefficient tensor into a **16 × 16 real matrix**
  across the fixed two-qubit/two-qubit bipartition, using lexicographic `IXYZ`
  labels. A low-rank density matrix need not give a low-rank coefficient matrix.
- Three state families, each with three instances: Qiskit Haar pure states,
  rank-two Hilbert-Schmidt mixed states, and ground states of the open-boundary
  transverse-field Ising Hamiltonian
  `H = -sum Z_i Z_(i+1) - h sum X_i`, with `h = 0.5, 1.0, 1.5`.
- Randomly omit 50% or 80% of the **255 nonidentity observables**. The 50% case
  measures 128 nonidentity observables (49.80% actually missing); the 80% case
  measures 51. Identity is exactly one, always supplied, and costs no shots.
  The smaller observed set is a subset of the larger set for each instance.
- Use **2,000 independent shots per measured Pauli string**, sampling the
  number of +1 parity outcomes from `Binomial(shots, (1 + expectation)/2)`.
  These are independent Pauli parity experiments, not reuse of complete local
  measurement-setting outcomes. No row/column support repairs are made.
- All 18 cases are fixed before viewing outcomes. There is no row-wise split,
  checkpoint tuning, transpose averaging, or quantum-specific neural training.
- MatrixFormer uses the final combined V57 checkpoint. MatrixFormer* uses the
  existing two-specialist convex observed-reconstruction router. All models
  receive the same noisy, observed-only globally standardized coefficient
  matrix, including the known identity. Full context is used without chunking.
  Their checkpoint, config, and source hashes are recorded in `manifest.json`.

## Reconstruction and controls

For an n-qubit density matrix, `c_P = Tr(rho P)` and
`rho = 2^(-n) sum_P c_P P`.

For MatrixFormer, MatrixFormer*, and zero-fill, copy the measured coefficients
back into the completed array, clip the filled coefficients to `[-1, 1]`, invert
the Pauli transform, and project onto PSD trace-one matrices. Projection uses
the Euclidean projection of the Hermitian eigenvalues onto the probability
simplex, not merely clipping and renormalizing negative eigenvalues. It can
change measured coefficients; the observed-fit RMSE is saved and reported.

The primary physics control solves
`min sum_observed (Tr(rho P) - measured_P)^2`, subject to `rho >= 0` and
`Tr(rho) = 1`, with **CVXPY/SCS**, tolerance `1e-6`, and at most 50,000
iterations. This is unweighted physical least squares, not maximum likelihood
or Qiskit's fitter. Under incomplete measurements a minimizer may not be
unique. We add no hidden-label tuning, rank oracle, or nuclear-norm penalty
(the nuclear norm is already one under these constraints). Its solution is
projected for numerical cleanup; solver status/residuals are archived.

## Metrics and uncertainty

- Squared Uhlmann fidelity (Qiskit's convention; larger is better), trace
  distance, and hidden **true-expectation** RMSE after density projection.
- Pre-projection hidden RMSE, raw neural hidden RMSE, projection size,
  observed-fit RMSE, physical validity, and inference/solver timing are also
  saved. Truth is used only for data simulation and scoring, never fitting.
- Native neural CRPS and 50/90/95% interval coverage are evaluated on every
  hidden true Pauli expectation, before clipping or physical projection. The
  full-support distribution includes half-normal tails; the ensemble is a
  mixture of predictive densities. These are **marginal coefficient intervals,
  not a joint posterior over physical density matrices**. They are neither
  projected nor calibrated using the test targets. Interval widths and the
  fraction of intervals extending beyond `[-1, 1]` are recorded.
- State families and missingness levels receive equal weight. The same nine
  states are reused across missingness rates, so the 18 cases are not 18
  independent physical systems. This small pilot cannot establish broad
  quantum-tomography performance or an uncertainty advantage.

## Run

```bash
python -m pip install 'qiskit==1.4.5'
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest mcpfn/tests/test_quantum_tomography.py -q
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/evaluate_quantum_tomography.py
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python mcpfn/scripts/report_quantum_tomography.py \
  mcpfn/runs/physics_quantum_20261027/pilot
```

Artifacts: `manifest.json`, per-state truth, per-case masks, measured counts,
raw/completed predictions, reconstructed density matrices, uncertainty arrays,
solver diagnostics, `per_case.csv`, `summary.csv`, and `RESULTS.md` under
`/root/tabular/mcpfn/runs/physics_quantum_20261027/pilot/`.
