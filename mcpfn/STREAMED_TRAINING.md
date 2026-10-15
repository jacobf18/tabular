# Streamed training

## Large-matrix continuation from the selected 98k model

`scripts/train_streamed_large_matrices.py` adds a capacity-gated, resumable
continuation runner. It starts from the **98,000-step** synthetic-validation
winner, not the original 17k anchor or the terminal 128k model:

```text
src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/best_validation_checkpoint.pth
SHA256: 1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d
```

The architecture and the two frozen teachers are unchanged. Both teachers use
streamed inference over the complete matrix. The student uses differentiable
streaming and activation checkpointing, including **decoder plus full loss**:
true-target NLL + 2 × teacher-mean Huber (delta 1) + 0.05 × true-target CDF MSE.
Auxiliary teacher targets are detached and sliced with the same target indices
at every decoder microbatch. CDF reductions now stay FP32 rather than rounding
the old trainer's final CDF scalar to BF16; this is the same mathematical
objective, not bitwise reproduction of the old mixed-precision training loop.

The selected model previously trained on spectral matrices of 80–320 × 35–100,
with 10–50 × 5–50 latent rehearsal. Earlier rectangular curricula expanded
dimensions gradually and reduced matrix batch size. This continuation uses:

| New updates | Spectral rows | Spectral columns |
|---|---:|---:|
| 1–500 | 256–512 | 128–256 |
| 501–1,500 | 512–1,024 | 256–512 |
| 1,501–3,000 | 768–1,536 | 384–768 |
| 3,001–5,000 | 1,024–2,000 | 512–1,000 |

Matrices are transposed with probability 0.5, so both rectangular orientations
are trained and capacity-tested. The largest planned matrix contains **2 million
cells, 62.5×** the previous 320 × 100 training maximum. This is a curriculum
ceiling, not a claim that every update uses that size or that it is the absolute
hardware limit. Full row/column attention context is never windowed or truncated.

The eight-component prior, spectral geometries, rank ranges, MCAR/profile-mixed
missingness, sparse-support augmentation, and observed-support variance
shrinkage (pseudo-count 1) come from the previous teacher-mean recipe. The
25% small latent rehearsal remains small. Additionally, 20% of spectral draws
retain the old dimensions: in expectation **60% new-size spectral, 15% old-size
spectral, 25% small latent**. Each update accumulates two independently sampled
matrix-mean losses serially, equally weighted; larger matrices do not silently
dominate the loss through their target count.

Optimization uses a **fresh AdamW** state, FP32 weights/moments, BF16 autocast,
weight decay 0.02, gradient clipping at 0.5, 200-update linear warmup to 5e-6,
and cosine decay to 2.5e-7 at update 5,000. This is a new fine-tuning schedule,
not a resume of the old 128k optimizer/scheduler. Resume within this new run
restores Adam state, model RNG, and update index; synthetic samples are keyed
by update/microstep. There is no automatic OOM fallback, reduced-context
retry, or skipped bad batch that would silently alter the trajectory.

### Admission, selection, and checkpoints

- Preflight must complete two accumulated teacher/student forward/backward
  passes and one AdamW update at every phase maximum, in both orientations.
  The two microsteps use 60% and 99% missing cells, respectively, exercising
  both ordinary supervision and the much larger near-all-target decoder path.
  The probe verifies finite gradients/parameters and
  real parameter updates, and never saves its modified models.
- A fixed synthetic panel is evaluated at step zero, every 250 updates, and
  phase ends. It covers eight old-size/latent components and six spectral
  components at each of 512 × 256 and the final maximum, with fixed disjoint
  validation seeds, both orientations, and sparse-query cases.
- Selection minimizes the equal average of retention and large-matrix macro
  RMSE ratios relative to step zero, subject to retention RMSE remaining at
  most 1.03× its baseline. Ties retain the earlier checkpoint. Step zero is
  eligible, so failure to improve cannot silently replace the parent winner.
  No real benchmark labels enter training or selection.
- Only `best_validation_checkpoint.pth` (model-only) and
  `latest_checkpoint.pth` (resumable optimizer/RNG state) are retained. Latest
  is written every 100 updates or 600 seconds, at phase ends, and after a
  graceful stop. Writes use atomic replacement; SIGTERM/SIGINT finishes the
  current update and saves. A crash resumes the last complete saved update.
- `config.json`, source snapshots/hashes, parent/teacher hashes, per-update
  JSONL, validation JSONL, `preflight.json`, and `status.json` record provenance
  and progress. Training/resume rejects changed registered sources or inputs.
  A per-run lock prevents duplicate writers. Launch requires 23 GiB free;
  checkpoint guards preserve a 20 GiB reserve plus atomic-save headroom.

From `mcpfn`:

```bash
export PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUN=runs/streamed_training/large_matrices_98k_20260930
python scripts/train_streamed_large_matrices.py --mode preflight --output "$RUN" --wait-for-idle
python scripts/train_streamed_large_matrices.py --mode train --output "$RUN" --wait-for-idle
# After a graceful stop or crash, retain the frozen configuration:
python scripts/train_streamed_large_matrices.py --mode train --output "$RUN" --resume --wait-for-idle
```

`--max-updates N` pauses after N additional updates without changing the frozen
schedule. `--curriculum path.json`, `--warmup-steps`, and `--memory-fraction`
configure a **new preflight**, not an existing training trajectory. For a
deliberately shared-GPU capacity probe, `--allow-busy-gpu` bypasses the idle
wait while retaining the allocator cap; no other process is stopped.
The default allocator cap is 70% of GPU memory. An initial 60%-cap probe failed
during the 2,000 × 1,000 backward pass despite remaining physical GPU memory;
that failed probe is retained under `large_matrices_98k_20260930_cap60_probe`.
An intermediate 65%-cap probe was stopped and superseded by the stricter
60%/99%-missingness preflight, not admitted as a completed capacity result.

`tests/test_streamed_large_matrix_training.py` verifies full-objective
loss/gradient agreement with dense execution, detached/aligned teacher targets,
uneven decoder chunks, matrix-balanced accumulation, curriculum boundaries,
seeded sampling/rehearsal, selection, atomic-save failure recovery, and an
uninterrupted-versus-resumed training trajectory. The real 8-layer checkpoint
also completed one CUDA update, saved its full Adam/RNG checkpoint, and resumed
for a second update in `runs/streamed_training/large_matrices_resume_smoke_20260930/`.
That disposable smoke run does not supply the long run's weights.

### September 30 capacity result and live run

The complete stricter preflight **passed all nine cases** on the H200. Each
case starts from the same 98k weights and fresh Adam state, accumulates two
matrices (60% and 99% missing), includes both frozen teachers, and completes a
finite optimizer update to all 238 parameter tensors. Maximum allocations
across the two orientations were:

| Matrix ceiling | Full-objective training peak GiB |
|---|---:|
| 512 × 256 | 7.072 |
| 1,024 × 512 | 23.786 |
| 1,536 × 768 | 51.354 |
| 2,000 × 1,000 | **85.747** |

The two-million-cell cases took 165–169 seconds per two-matrix accumulated
update **while sharing the GPU with an existing evaluation**. These are
capacity measurements, not controlled throughput or convergence results.
These fresh-optimizer peaks are not steady-state peak guarantees: subsequent
backward passes also retain Adam's moments. The 70% allocator cap leaves
headroom beyond the measured peaks.
All parent and teacher hashes still match. **94 tests pass** across streamed
training/inference, the new curriculum/resume runner, and the existing
prior/optimizer regression suites.

The admitted 5,000-update run is
`runs/streamed_training/large_matrices_98k_20260930/`. Its `preflight.json`,
`verification.json`, `environment.json`, `config.json`, and source snapshots
record admission evidence. `training.pid`, `train.log`, `metrics.jsonl`, and
`status.json` expose live progress. It uses a fresh student initialized from
the 98k parent, not any capacity-probe or resume-smoke weights. The existing
MovieLens evaluation is not stopped; the training process retains the tested
70% allocator cap. A passed capacity probe establishes feasibility, **not**
that the final model improves; fixed synthetic validation determines whether
any new checkpoint should replace the step-zero model within this run.

The September 30 cleanup manifest, `runs/checkpoint_cleanup_20260930.json`,
records 20 removed historical phase-end snapshots (25.11 GiB). Their normalized
ZIP records were SHA256-identical to the retained numbered checkpoints.
All DWIP runs, best checkpoints, latest optimizer checkpoints, teachers, and
the 98k parent were preserved. Disposable pip cache removal separately freed
approximately 4.1 GiB.

## Original experimental smoke test

`src/tabimpute/streamed_training.py` is a gradient-safe prototype, separate from
the inference implementation and the existing training/evaluation runners.
It does not overwrite hidden activations in place or disable gradients.
It microbatches encoding, independent axial attention sequences, feed-forward
cells, and masked decoding plus loss. Non-reentrant activation checkpointing
at block and microbatch boundaries recomputes intermediates during backward.
The complete attended sequence is retained, and all selected targets contribute
to one correctly weighted mean loss, including uneven final microbatches.

The loss callable must return one loss per selected cell. Absolute positional
embeddings and attention capture are unsupported. Full hidden grids and their
gradients still occupy memory. Drop-path acts on whole matrix samples, not
individual attention microbatches. Dropout draws can differ from dense execution;
checkpoint recomputation preserves the chunked execution's RNG state.

## Small-matrix result

On the H200, the actual paper checkpoint (8 layers, embedding width 768, 5,000
output bins) successfully completes forward, backward, and one AdamW update on
a synthetic **64 × 32** matrix. There is one matrix per batch and approximately
60% of cells are masked. The objective is masked bar-distribution NLL, not the
complete teacher-distillation training recipe. Every case starts from the same
checkpoint on a disposable model copy; no updated checkpoint is saved.

After one warmup per execution/precision configuration:

| Precision | Execution | Forward/backward peak GiB | Entire first-step peak GiB |
|---|---|---:|---:|
| FP32 | Ordinary dense training | 2.622 | 2.622 |
| FP32 | Existing block checkpointing | 0.936 | 1.306 |
| FP32 | Checkpointing + microbatching | 0.729 | 1.300 |
| BF16 autocast | Ordinary dense training | 1.994 | 1.994 |
| BF16 autocast | Existing block checkpointing | 0.867 | 1.312 |
| BF16 autocast | Checkpointing + microbatching | 0.712 | 1.305 |

The loss matches dense execution at the recorded precision in both FP32 and
BF16. The relative L2 difference across parameter gradients is **4.30e-5** in
FP32 and **0.00459** (0.459%) in BF16. All 238 parameter tensors have finite
gradients and update successfully. BF16 is approximate agreement, not bitwise
gradient equivalence. The original checkpoint's file hash is unchanged.

These are process-local PyTorch peak allocations. The full-step column includes
the first allocation of AdamW state; it is not a steady-state multi-step peak.
Fixed parameter/gradient/optimizer memory dominates this tiny example, so most
of the total-memory reduction comes from existing block checkpointing. Additional
microbatching reduces backward memory by about 18% in BF16 versus that control,
but barely changes the full first-step peak here. It also incurs recomputation
and small-kernel overhead. The GPU is shared with a recommender evaluation, so
recorded timings are not a controlled throughput comparison.

This establishes training feasibility on a small matrix, **not** the largest
trainable matrix, long-run convergence, or parity of the full distillation recipe.
The checkpoint and running recommender evaluation are unchanged.

## Artifacts and reproduction

The primary local artifact directory is
`runs/benchmark_evaluations/matrixformer_streamed_training_smoke_warmed_20260930/`.
It contains `RESULTS.md`, full-precision `results.json`, `status.json`, and a
manifest with checkpoint/configuration/source hashes and source snapshots.
The initial unwarmed smoke test is retained separately under
`matrixformer_streamed_training_smoke_20260930/` in the same parent directory.

From `mcpfn`, with the local paper checkpoint and a CUDA-enabled PyTorch:

```bash
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  pytest -q tests/test_streamed_training.py
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python scripts/probe_matrixformer_streamed_training.py \
  --rows 64 --columns 32 --output runs/benchmark_evaluations/training_smoke_repeat
```

The unit tests cover dense loss/gradient/SGD-update agreement, uneven chunks,
multiple matrices per batch, singleton axes, full attention sequence lengths,
checkpoint RNG preservation with dropout enabled, and invalid loss/mask inputs.
The probe additionally exercises the full checkpoint in FP32 and BF16 with AdamW.
