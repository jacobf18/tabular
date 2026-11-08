# Scientific-prior continuation of MatrixFormer

## Aim

Train a stronger single MatrixFormer rather than finding a favorable application
for the frozen model. This first continuation expands the synthetic library
using failure modes found in the scientific-panel experiments. The architecture,
5,000-bin distribution head, checkpoint format, and inference interface stay
unchanged.

Run directory:

`/mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v1`

This is a 600-update pilot, not a finished model or an asserted benchmark win.
The authoritative live state is `status.json`; `training.log` and
`validation.jsonl` contain the learning curves. The first 50-update validation
produced a small eligible improvement, but that alone says nothing about wins
on real datasets.

## New reusable generators

Implementation: `src/tabimpute/prior/scientific_panels.py`.

| Family | What varies | Motivation |
|---|---|---|
| Smooth sensors | Multiscale smooth signals, correlated channels, channel-specific deviations, noise | Short gaps in temperatures and monitoring networks |
| Dynamic factors | Persistent stochastic states, nonlinear observation functions, irregular time intervals | Environmental dynamics not captured by static low-rank matrices |
| Transport | Pulses with sensor-dependent delays, attenuation, nonlinear concentration transforms | Transported signals and river measurements |
| Nonlinear calibration | Nonlinear proxies, direct assay channels, drift, ordered or unordered samples | Recovering expensive assays from related instruments |
| Regime panels | Grouped channels, shared trajectories, abrupt or gradual changes | Panel structure and nonstationary deployments |
| Chemical mixtures | Random mixture fractions, endmember signatures, weak interactions | Cross-analyte relationships in chemical measurements |

Noise levels, offsets, scales, ranks, loading matrices, and the availability
of clock covariates vary between episodes. Training matrices have 64–384 rows
and 8–64 measurement columns; some include three additional clock columns.

Missingness includes isolated gaps, channel bursts, correlated network outages,
sparsely scheduled assays, self-MNAR, and covariate-dependent MAR. Each generated
measurement channel retains at least four observed entries. Normalization uses
only those observed entries, never hidden targets. Clock features remain observed.

These are generic synthetic families, **not fitted simulators of the benchmark
datasets**. No benchmark arrays, labels, recorded masks, fitted physical
parameters, or empirical test errors enter gradient computation.

Two additional families, damped oscillators and spatial fields, are explicitly
excluded from training and checkpoint selection. They provide a small synthetic
transfer check after the selected checkpoint is frozen. They do not replace
testing on genuinely new real domains.

## Training recipe

Trainer: `scripts/train_scientific_matrixformer.py`.

- Initialize from the published combined MatrixFormer checkpoint:
  `/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth`.
- Train the full network, not just its decoder.
- Every optimizer update contains one scientific episode and one old-prior
  episode, with equal matrix-loss weights. The old episode alternates between
  existing imputation and recommender generators: approximately 50% science,
  25% old imputation, 25% old recommender training.
- Scientific episodes use **true synthetic measurements** as point targets.
  Do not teach the student to reproduce the parent's scientific mistakes.
- Old-prior episodes use the frozen combined parent's mean for the point
  target, while retaining likelihood/CDF supervision from synthetic truth.
  This is rehearsal intended to limit forgetting, not a guarantee of retention.
- The loss remains NLL + 2 × Huber(mean, point target) + 0.05 × CDF MSE.
  The recommender NLL coefficient remains 0.1.
- AdamW, peak learning rate `1e-5`, 30-step warmup, cosine decay,
  weight decay `0.01`, gradient clipping `0.5`.
- FP32 parameters and optimizer state, BF16 autocast, gradient-safe streamed
  axial attention and decoder microbatching.

The capacity preflight performs real forward, backward, and optimizer updates
on all three training domains, then discards the updated model. Training starts
again from the untouched parent. The preflight used **3.12 GiB peak allocated
GPU memory**, with finite gradients and parameters. Device memory is capped
at 16%; other GPU processes are not stopped or modified.

## Selection and benchmark checks

Synthetic validation occurs at step zero and every 50 updates:

- 24 fixed scientific cases, four per training family.
- Four old imputation cases and four old recommender cases.
- Score: geometric mean of the six family-level RMSE ratios relative to
  the unchanged parent.
- A checkpoint is eligible only if each old domain's synthetic RMSE is at
  most 3% worse than its initial value. Select the earliest strict minimum;
  step zero is eligible, so a failed run does not silently replace the parent.

The real-data evaluator is `scripts/evaluate_scientific_continuation.py`.
It waits for training completion, snapshots the synthetic-selected checkpoint,
then evaluates **72 existing windows/cruises across seven applications**:

1. Wastewater chemistry.
2. Intel indoor temperatures.
3. Potomac nitrate.
4. Susquehanna nitrate.
5. Ocean alkalinity.
6. Italian chemical sensors.
7. Beijing pollution monitoring.

It uses the original masks and full-window inference, without the new local
preprocessing adapter, so this directly tests whether training improved the
model. Each candidate's normalization is selected on the original observed-only
validation mask, using the original transform grid. Cached parent, specialized
model, and classical predictions are audited against the same inputs and reused.
Comparators saved in float32 are checked within their quantization error bound,
not required to reproduce an unavailable float64 array bit-for-bit.

All these real applications have already been examined. They are now development
benchmarks, even where old metadata says `confirmation`. No new-domain claim is
made from improving on them. Full MissBench, UCI, recommender, and LLM reruns
remain necessary before promoting a candidate as a replacement; the synthetic
rehearsal checks alone do not establish real-benchmark retention.

## Artifacts and commands

The run retains registered source copies and SHA256 fingerprints, the parent
fingerprint, fixed synthetic validation inputs, every validation report,
per-update training logs, an atomic resumable checkpoint, and the best eligible
checkpoint. The old published model is never overwritten.

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Inspect the active run; do not start a duplicate worker.
cat /mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v1/status.json
tail -n 3 /mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v1/validation.jsonl

# Only if the existing worker has stopped and a resumable checkpoint exists:
python mcpfn/scripts/train_scientific_matrixformer.py train --resume

# A fresh run requires its own output directory and registration.
python mcpfn/scripts/train_scientific_matrixformer.py register --output /mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v2
python mcpfn/scripts/train_scientific_matrixformer.py preflight --output /mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v2
python mcpfn/scripts/train_scientific_matrixformer.py train --output /mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v2

# The launched evaluation worker already handles this after completion.
# Use a separate output directory if requesting another evaluation.
python mcpfn/scripts/evaluate_scientific_continuation.py --wait-for-completion

python -m pytest -q -p no:cacheprovider mcpfn/tests/test_scientific_panels_prior.py
```

The new prior/training helper suite has 45 passing tests. It covers every
training-family/mask combination, observation-only normalization, deterministic
generation, reserved-family exclusion, finite distribution-loss gradients,
shape boundaries, and checkpoint-retention eligibility.

No paper changes, commits, pushes, or Hugging Face uploads are part of this pilot.
