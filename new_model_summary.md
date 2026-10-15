# Development Summary: The New DWIP Model

## Overview

The new model is **the original DWIP v2-2 architecture, fine-tuned with a broader synthetic prior, a more robust mean-prediction objective, training-time teacher supervision, and a longer learning-rate schedule**. It is not a larger architecture or an inference-time ensemble.

The final training run completed 128,000 updates. The deployed checkpoint is from step 98,000, selected using fixed synthetic validation rather than public benchmark scores.

## 1. Broaden the Synthetic Training Prior

The final training mixture contains:

| Family | Mixture share | Main characteristics |
|---|---:|---|
| Low-rank spectral matrices | 25% | Rank 2–16 |
| Slowly decaying spectral tails | 25% | Higher-rank matrices with substantial tail information |
| Steeply decaying spectral tails | 25% | Higher-rank matrices with more concentrated leading components |
| Small latent-factor matrices | 25% | Rank 1–11; small matrices resembling the original latent-factor family |

The spectral prior varies both **singular values and singular-vector structure**, mixing random orthogonal, concentrated, smooth, and offset-bearing geometries. Each family includes MCAR and structured `profile_mixed` missingness.

The 25% latent-factor component was added as **rehearsal**: spectral-only fine-tuning improved many synthetic groups but degraded performance on small latent-factor cases. This is family-level rehearsal, not an exact replay of the original model's historical training data.

## 2. Improve Sparse-Observation Training

Training includes episodes where query rows or columns have very few observed entries.

Diagnostics showed that standardizing these sparsely observed columns could produce extreme normalized targets. We therefore added **observed-only variance shrinkage toward pooled within-column variance**, with pseudo-count 1.

This stabilizes the training augmentation; it does **not** change the public inference normalization.

## 3. Make the Objective Better Match Point-Prediction Quality

The early spectral recipe combined negative log-likelihood (NLL), mean-prediction MSE, and cumulative-distribution-function (CDF) alignment. Follow-ups increased the point-loss weight from **0.2 to 2**, then replaced MSE with **Huber loss, delta 1**.

The motivation was that a small number of extreme residuals dominated MSE. Huber limits their individual influence while retaining a quadratic penalty near the target.

Next, two synthetic-validation-selected models showed complementary prediction errors:

- A standard-prior Huber model, selected at step 54,000.
- A continuous-spectrum-prior Huber model, selected at step 56,000.

Their averaged predictions were useful, so we trained **one student** to match that average:

```text
Teacher target = detached equal average of the two frozen teachers' predictive means

Loss = NLL(student distribution, true target)
     + 2 × Huber_delta_1(student mean, teacher target)
     + 0.05 × CDF alignment(student distribution, true target)
```

**Only the point-loss target comes from the teachers.** NLL and CDF alignment still use the true synthetic values. The teachers are frozen, see the same masked input as the student, and are absent at deployment.

The final student's data prior retains the separate slow-tail and steep-tail components; it does not adopt the continuous-spectrum teacher's prior modification.

## 4. Use Controlled AdamW Fine-Tuning

The optimizer remains **AdamW**, not a new optimization algorithm. The important changes concern the objective, numerical precision, and training schedule.

| Setting | Final value |
|---|---|
| Initialization | Original DWIP 17k checkpoint; fresh optimizer |
| Peak learning rate | `1e-5` |
| Weight decay | `0.02` |
| Gradient-norm clipping | `0.5` |
| Precision | FP32 weights and Adam state; BF16 computation |
| Batch size | Two matrices |
| Schedule | 600-step warmup, then cosine decay to 5% of peak |
| Duration | 128,000 updates |

The preceding teacher-mean model was still improving at its **64k endpoint**, motivating the longer schedule. The 128k experiment restarted from the original anchor—it did not resume the 64k student's weights or optimizer.

The prior, teachers, objective, seeds, warmup, and peak/floor learning rates remained unchanged. Stretching the cosine schedule changes intermediate learning rates, so this was a longer training-and-schedule experiment, not simply additional updates at matched learning rates.

## 5. Select Using Synthetic Validation, Not Benchmark Scores

After completing all 128k updates, **step 98k** had the best fixed synthetic macro RMSE: **0.958693**, versus **0.969887** for the preceding 64k teacher-mean model.

Selection used the earliest strict minimum across the 65 fixed synthetic validations, including step zero. Public benchmark scores did not select the checkpoint.

Rejected alternatives included parameter averaging and a distribution-KL distillation pilot. The latter improved density metrics but worsened mean-prediction errors, so it was not taken to a full training run.

## Outcome and Limitations

The resulting single model improves MTEB MI R² at budget 5 from **0.690752 to 0.765919**, and improves **156 of 180** benchmark-selection mean settings over original DWIP, with essentially unchanged inference cost.

Gains are not universal: some settings regress, BenchPress MI at budget 5 is worse than the preceding 64k teacher-mean model, and the later recommender benchmarks show weak top-10 ranking performance. The final horizon experiment uses one training seed; these results are not a claim of statistical significance or universal superiority.

## Model Identity

- Training run: `dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918`.
- Selected checkpoint step: **98,000**.
- Checkpoint: `/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/best_validation_checkpoint.pth`.
- SHA256: `1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d`.

The expanded technical write-up and source references are in `mcpfn/DWIP_NEW_MODEL_SUMMARY.md`, relative to `/root/tabular`.
