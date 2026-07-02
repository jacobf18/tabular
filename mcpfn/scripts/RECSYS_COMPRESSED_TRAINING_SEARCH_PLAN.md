# Compressed Rec-Sys Training Search

## Objective

Improve exposed-holdout RMSE on the fixed synthetic MovieLens proxy without
changing the matrix-completion loss. Every configuration is capped at 20,000
optimizer steps and evaluated through `CompressedCrossImputePFN`.

## Stage 1: Optimization Screen

Run four 5,000-step continuations from the same exposed-fitted source model.
Use the four-phase matrix-size and exposure curriculum, `K=20`, one random
target chunk per optimizer step, and compare:

| Config | Predictor LR | Compressor LR | Weight decay |
|---|---:|---:|---:|
| equal, regularized | 1e-5 | 1e-5 | 0.03 |
| compressor 2x, regularized | 1e-5 | 2e-5 | 0.03 |
| equal, no decay | 1e-5 | 1e-5 | 0 |
| compressor 2x, no decay | 1e-5 | 2e-5 | 0 |

Select using paired seed-101 RMSE. Confirm the top two on seeds 101 and 102.
Treat differences below 0.0002 RMSE as ties.

## Stage 2: Capacity And Gradient Screen

Using the best Stage-1 optimizer:

1. Train `K=20`, one target per step for up to 20k steps.
2. Train `K=32`, one target per step for up to 20k steps. Warm-start all
   existing weights and initialize additional learned queries from the mean
   pretrained query.
3. Train `K=20`, two target chunks averaged per optimizer step for 10k steps.
   This matches roughly 20k single-target forward/backward evaluations.
4. Train `K=20` directly on target geometry (`800-1000 x 1400-1800`,
   exposure density `0.04-0.06`) for 5k steps to test whether early curriculum
   phases are consuming useful optimization budget.

Validate at 5k-step boundaries. Stop a configuration when it is worse than its
initial checkpoint on both paired seeds or has not improved by 0.0002 RMSE.

## Stage 3: Confirmation

Evaluate the best Stage-2 checkpoint and the source checkpoint on at least
three paired full-size seeds. Report:

- exposed-holdout RMSE and MAE
- paired per-seed deltas
- bias-only and rank-10 SVD baselines
- peak memory and wall-clock time per optimizer step

Do not promote a model unless its mean RMSE improves by at least 0.001 or it
wins consistently across all confirmation seeds.

## Storage

Search runs save model-only final checkpoints, disable sampled-batch "best"
checkpoint writes, and omit intermediate checkpoints. This keeps each
configuration resumability-limited but prevents the search from exhausting the
current filesystem.
