# MatrixFormer-Impute on the frozen physics pilots

This follow-up evaluates the imputation-specialized checkpoint on the same
two retained physics pilots (cylinder flow and quantum tomography) as the combined model. It changes **only the neural
checkpoint**, not the data, masks, measurement noise, normalization, matrix
representation, context sizes, physical constraints, or classical baselines.

Checkpoint:
`/root/tabular/mcpfn/runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth`

SHA256:
`c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9`

The comparator is the final combined V57 MatrixFormer, SHA256
`0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.
There is no physics-specific training, checkpoint selection, transpose
averaging, or expert routing in the new specialist runs. The original
ensemble remains in the archived baseline comparisons where available.

## Matched experiments

| Benchmark | Original cases | Context | Primary metric |
| --- | ---: | --- | --- |
| Cylinder wake | 18 | Full 2,048 × 151 | Hidden-entry RMSE / observed-global SD |
| Simulated four-qubit tomography | 18 | Full 16 × 16 Pauli coefficient array | Squared state fidelity |

There are **36 retained cases and 36 new method/case scores** in total.
MRI was cancelled and excluded at the user's request. The original 60-case
protocol and already saved artifacts remain on disk as history; they are not
included in the active comparison. See `scope_change.json` in the run directory.

Every representation uses its original observed-only global normalization.
Quantum coefficient completion, clipping, and PSD trace-one projection match the
original implementation. This is simulated tomography, not quantum hardware.

Cylinder and quantum runs also save the same native distribution metrics:
CRPS, NLL, 50/90/95% interval coverages and widths. Cylinder reuses the exact
original sampled hidden indices; quantum uses every hidden Pauli coefficient.
These are raw coefficient distributions, not distributions over projected
density matrices.

## Safeguards and uncertainty

- Read the existing dataset copies, masks, measured shot counts, and cases;
  do not regenerate or select favorable inputs.
- Check original artifact hashes and freeze source, checkpoint, config, and
  input hashes before inference. Store source snapshots with the new run.
- Replay the combined model on the first lexicographic case of each retained
  benchmark before evaluating the specialist.
- Keep model parameters unchanged and use evaluation/inference mode.
- Recompute all metrics from saved specialist predictions and verify the
  comparison includes every retained case.
- Keep all original results intact. Baselines are reused, not refitted.
- Summaries retain mean ± **sample SD**, matching the original pilots; these
  are not confidence intervals over independent flows or experimental subjects.
  Paired differences compare the specialist and combined model on identical
  cases. No average is taken across the incompatible domain-specific metrics.

## Saved results

Run directory:
`/mnt/matrixformer_scratch/physics_impute_20261106`

- `RESULTS.md`: cylinder and quantum comparison tables and protocol caveats.
- `protocol.json`, `source_snapshot/`: exact cases, hashes, and frozen code.
- `scope_change.json`: the user-requested MRI exclusion and cancellation record.
- `comparison_per_case.csv`: all retained original and specialist method scores.
- `paired_summary.csv`: specialist versus combined on the same cases.
- `audit.json`: independent saved-prediction rescoring and replay results.
- `status.json`, `run.log`: progress and inference logs.
- Each benchmark subdirectory contains `cases/` predictions and metrics,
  `combined_replay.json`, `per_case.csv`, `comparison_per_case.csv`, and
  `summary.csv`; quantum also includes `overall_by_missingness.csv`.

## Reproduce

From `/root/tabular`, use a fresh directory for a new campaign:

```bash
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
output=/mnt/matrixformer_scratch/physics_impute_NEW_RUN
python mcpfn/scripts/evaluate_physics_impute.py plan --output "$output"
python mcpfn/scripts/evaluate_physics_impute.py run --output "$output" --benchmarks cylinder quantum
python mcpfn/scripts/report_physics_impute.py "$output"
```

The runner can resume its own completed cases, checking their hashes. Always
pass `--benchmarks cylinder quantum` to avoid executing the excluded suites;
the original frozen planner still records all four suites for provenance.
The final report requires only the complete 36-case cylinder/quantum campaign.
No parent result is overwritten. The original datasets/checkpoints must be
available at the paths recorded in the protocol.

Run GPU tests when the inference process has released its CUDA allocator:

```bash
python -m pytest -q mcpfn/tests/test_physics_impute.py \
  mcpfn/tests/test_cylinder_completion.py \
  mcpfn/tests/test_quantum_tomography.py
```

The new distribution test compares the single-checkpoint implementation
directly against the original native-uncertainty helper. Related tests check
masking, observed-only normalization, physical projection, observation
preservation, and matched-case reporting.
