# Million-matrix scientific continuation

Prospective experiment, November 8, 2026. Train the existing single-network
MatrixFormer on **one million additional usable synthetic matrices**. No paper
edits, model uploads, or changes to previous experiments are part of this run.

Run directory:
`/mnt/matrixformer_scratch/scientific_prior_training_20261108/million_v1/`

## Launch validation

- 151 automated tests passed across the new trainer/selection/accounting tests
  and existing science, UCI, area-prior, and structured-missingness tests.
- All 84 synthetic validation cases and all 48 structured component/mask
  combinations produced valid episodes in the GPU preflight.
- Seven forward/backward/AdamW checks passed, covering all four domains,
  structured neural missingness, medium matrices, and a 2,000 × 1,000 matrix.
  The largest check peaked at **85.78 GiB** and completed in **28.41 seconds**.
- All 155 registered source hashes were verified unchanged after launch.
- The detached training worker launched successfully; `worker.pid` and
  `launch.json` identify it. Consult `status.json` for current progress rather
  than treating this launch record as evidence that training is complete.

## Initialization and counting

Initialize from the actual step-3,000 scientific checkpoint, whose full UCI
NRMSE is 0.967735 versus 1.017006 for the original combined model:
`/mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/snapshots/step_3000/checkpoint.pth`.

Each of 250,000 optimizer updates accumulates **four separate matrices**, one
per domain. Each input matrix has batch size one; attention/decoder microbatches
and multiple losses do not increase the matrix count. Domain loss weights are
35% science, 35% imputation, 20% ratings, and 10% score panels. Draw counts are
250,000 per domain, not these loss-weight percentages.

Only explicitly degenerate structured masks may be retried, deterministically
up to 16 times. Every rejected seed is logged. Status separately tracks usable
matrices, generated draws, rejected draws, and observed/query cells. Resumes
continue checkpointed counters; session IDs distinguish replay after a crash.

## Priors and sizes

- All ten existing scientific training families, cycling six missingness
  patterns. Damped oscillators and spatial fields remain excluded.
- Imputation alternates all eight original spectral/latent components and all
  48 combinations of four generators and twelve structured missingness families.
  Both original and area-scaled spectral amplitudes are included.
- The existing noisy, sparse, user-scaled synthetic rating prior.
- Synthetic capability/score panels with dense context and sparse test rows.

Requested shapes are 80% small (up to 256 × 64), 18% medium (up to 768 × 128),
and 2% large. Large draws include 1,024 × 256 and rare 1,536 × 512 / 2,000 × 1,000
contexts (each of the latter has 0.1% requested probability). Native spectral
inputs can also transpose during training. Latent and neural-mask priors retain
their intrinsically small shapes; structured spectral shapes use their original
240 × 80 / 1,024 × 256 choices. Score panels cap at 512 × 128. These are not
claims of exact realized proportions: actual dimensions and cell counts are logged.

All gradients use synthetic measurements. Real measurements, recorded masks,
labels, and fitted benchmark parameters never enter training data.

## Loss and optimization

Keep true-target NLL, Huber, and CDF supervision in **every** domain. Huber
weights are 4 for science, 2 for imputation, and 1 for ratings/score panels; CDF
weight is 0.05. Rating NLL has weight 0.2; other NLL weights are 1.

The frozen **original combined MatrixFormer** supplies a separate Huber
regularizer, not the sole point target. Its weight decreases over the first
250,000 matrices from 2 to 0.5 for imputation and 8 to 4 for ratings/score panels.
Science has no teacher-matching term.

Full-network AdamW, FP32 weights/moments, BF16 autocast, clip 0.5, and streamed
gradient-safe attention. Warm up for 1,000 updates to LR 5e-6, hold through 50,000
updates, then cosine-decay to 5e-7 at update 250,000. Architecture and benchmark
inference remain unchanged. Large-context capacity is tested before launch.

## Validation and checkpoint selection

- Fresh, fixed 84-case synthetic development panel every 1,000 updates. Preserve
  a synthetic-best diagnostic checkpoint; do not interpret it as a real-data win.
- Real monitoring at **25k, 100k, 250k, 500k, and 1M matrices**: all 1,800 UCI
  cases, all 462 MissBench cases, 72 scientific windows/cruises, 72 LLM settings,
  and full MovieLens100K/Netflix-small query sets.
- Keep `best_uci_checkpoint.pth` independently. Keep `best_joint_checkpoint.pth`
  by the geometric mean of original-relative UCI and scientific aggregate error
  ratios, requiring both to be no worse than the scientific warm start.
- Individual MissBench regressions are **not** vetoes. Joint selection permits
  up to 10% aggregate MissBench regression versus the original combined model.
- `best_balanced_checkpoint.pth` additionally requires original-relative LLM
  RMSE regression ≤2%, R² drop ≤0.01, and recommender RMSE/MAE regression ≤2%.
  There is no fake fallback balanced checkpoint if nothing qualifies.
- At completion, run the full 4,065-record legacy evaluation, including all LLM
  settings and MovieLens1M, on the balanced best (or joint best if unavailable).
  Only a newly trained checkpoint passing the new selection/retention criteria
  produces `recommended_checkpoint.pth`. Historical strict gate outputs remain
  diagnostic and are never overwritten or quietly reinterpreted.

The initial best UCI/joint files contain the explicitly tagged warm-start model
at step zero until an improvement is measured. Already inspected real benchmarks
are **development monitors**, not untouched test evidence. No MRI or MovieLens10M.

## Operations

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python mcpfn/scripts/train_million_matrix_continuation.py register
python -u mcpfn/scripts/train_million_matrix_continuation.py preflight
python -u mcpfn/scripts/train_million_matrix_continuation.py train
```

Resume only after verifying the previous worker has stopped:

```bash
python -u mcpfn/scripts/train_million_matrix_continuation.py train --resume
```

Registration freezes source hashes, benchmark protocols, warm-start evidence,
and checkpoint identities. Source snapshots and all large artifacts live on
scratch. Do not edit registered sources during training. An exclusive file lock
prevents duplicate workers. SIGTERM/SIGINT save resumable optimizer/RNG state at
the next safe boundary; periodic saves occur every 250 updates or five minutes.

Inspect `status.json`, `training.log`, `training.jsonl`, `monitoring.jsonl`,
`best_checkpoints.json`, and `preflight.json`. `snapshots/` retains all five real
monitoring checkpoints and predictions, including unsuccessful ones.
