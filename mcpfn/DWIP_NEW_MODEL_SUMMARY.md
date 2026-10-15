# How the new DWIP model was developed

Written September 20, 2026, from the saved experiment registrations, training configuration, source code, and completed evaluations.

## Executive summary

The new model is a fine-tuned **DWIP v2-2**, not a new architecture. The development path combined a broader synthetic matrix prior, more stable sparse-support normalization, a robust loss aimed at the predictive mean, and training-time distillation from two complementary models. A final controlled experiment extended the same teacher-mean recipe from a 64,000-step to a 128,000-step cosine schedule. The deployed checkpoint is **step 98,000**, selected by fixed synthetic validation after the full run completed.

The optimizer is still AdamW. The important optimization changes are the objective, numerics, and learning-rate horizon—not a new optimizer algorithm. Deployment uses one model, with no teachers or ensemble.

## 1. Start from the original model, retaining its architecture

The student starts from `/root/tabimpute_anchor3_named/dwip_v2_t003_checkpoint_17000.pth`, the original DWIP v2-2 anchor. The architecture stays unchanged: eight layers, embedding width 768, 16 attention heads, feed-forward width 1,536, and a 5,000-bin predictive-distribution output.

The final run loads the original model weights with a fresh optimizer. It does **not** resume the 64k student's weights or optimizer, and it does not initialize from a weight average of the teachers. Earlier experiments informed the recipe and supplied teachers; they are not one uninterrupted chain of student checkpoint continuations.

## 2. Build a broader synthetic prior

The adopted prior is an eight-component, equal-probability mixture. Six components are spectral matrices and two are small latent-factor matrices, giving **75% spectral training and 25% latent-family rehearsal**.

| Matrix family | Total mixture mass | Main configuration |
|---|---:|---|
| Low-rank spectral | 25% | Rank 2–16; tail exponent 0.2–2.0 |
| Slow-tail spectral | 25% | Rank 2–160, capped by matrix dimensions; exponent 0–0.5 |
| Steep-tail spectral | 25% | Rank 2–160, capped by matrix dimensions; exponent 1.0–2.5 |
| Small latent-factor rehearsal | 25% | Rank 1–11; 10–50 rows and 5–50 columns |

Each family has an MCAR and a `profile_mixed` component. Thus the base missingness-family mixture is 50/50, before the additional sparse-support augmentation.

The spectral generator constructs matrices from singular vectors and a sampled singular-value spectrum. It varies the leading components, power-law tail, taper, overall scale, and small additive noise. It also mixes four singular-vector geometries: random orthogonal/Haar, coherent or unevenly concentrated, smooth, and offset-bearing. This broadens the structural assumptions beyond a single rank or isotropic random-factor distribution.

Spectral training matrices have 80–320 rows and 35–100 columns, with batch size two. The small latent components have explicit shape overrides. Their inclusion was motivated by independent synthetic retention diagnostics: spectral-only fine-tunes improved most groups but worsened the small latent/MCAR group by roughly 3–5%. The 25% rehearsal mass was a fixed intervention, not a benchmark-fitted mixture-weight search. This is family-level rehearsal using explicit latent-generator settings, not a claim to reproduce the anchor's historical training stream exactly.

**Important distinction:** one teacher uses a continuous high-rank exponent range of 0–2.5. The final student's own data prior retains the separate slow-tail and steep-tail ranges above; it does not adopt that teacher's continuous-prior modification.

## 3. Train for sparse observations without unstable rescaling

The `broad_support` augmentation retains the original missingness episode 25% of the time. Otherwise it further restricts observations for a subset of query rows or, with random transposition, query columns. Query subsets span approximately 10–40% of the relevant axis; allowed support fractions are sampled log-uniformly between 1% and 60%. The scored targets in these augmented episodes are missing query cells.

Early diagnostics showed that ordinary column standardization after aggressive masking could create extremely large normalized targets when little support remained. The adopted training augmentation therefore shrinks each column's estimated variance toward pooled within-column variance, with **pseudo-count 1**. The added transform uses only retained observations and transforms inputs and targets consistently.

This is a training-time stabilization, not a replacement for the public inference normalization. Its observed-only property applies to this added transform; it is not a blanket claim about all upstream generator preprocessing.

## 4. Change the objective to emphasize reliable predictive means

The initial spectral DWIP recipe used true-target negative log-likelihood (NLL), a predictive-mean MSE term weighted 0.2, and a CDF-alignment term weighted 0.05. Controlled follow-ups increased the point-loss coefficient to 2 and then replaced MSE with **Huber loss, delta 1**, keeping that coefficient at 2.

The motivation was residual concentration rather than a demonstrated optimizer bug: in sampled training streams, roughly the largest 1% of batches accounted for about 45% of summed MSE. Huber loss limits the influence of large individual residuals, which global gradient clipping alone does not do. Its benefit was tested, not assumed; Huber is not identical to MSE and can change the fitted mean.

The next useful observation was that two models made complementary prediction errors on synthetic holdouts:

- The standard-prior Huber model, selected at step 54,000.
- The continuous-spectrum-prior Huber model, selected at step 56,000.

The pair was fixed from synthetic validation rankings, without a pair or mixture-weight search. Their equal prediction mean improved both independent synthetic macro-RMSE summaries over either constituent. Weight averaging was not an adequate substitute. Instead of deploying both models, the next experiment trained one student to match their mean.

For a missing cell, let `y` be the true synthetic target, `p_student` the predicted distribution, `mu_student` its mean, and `mu_teacher` the detached equal mean of the two frozen teachers' predictive means. The final objective is:

```text
mu_teacher = stop_gradient((mu_teacher_1 + mu_teacher_2) / 2)

loss = NLL(p_student, y)
       + 2 * Huber_delta_1(mu_student, mu_teacher)
       + 0.05 * CDF_alignment(p_student, y)
```

Only the point-loss target changes to the teacher mean. **NLL and CDF alignment still use the true synthetic target.** CDF alignment is the mean squared difference between the predicted binwise cumulative distribution and the target step function. The student therefore continues to learn a predictive distribution, rather than becoming only a point regressor or only copying teacher distributions.

Both teachers see the same masked input as the student, remain frozen and in evaluation mode, and receive no gradients. Their means are averaged in FP64 and rounded to FP32. They are external to the student's saved state and are absent at inference.

## 5. Keep AdamW, but use a controlled, stable training procedure

| Setting | Final recipe |
|---|---|
| Initialization | Original 17k DWIP anchor; fresh optimizer |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| Peak learning rate | 1e-5 |
| Weight decay | 0.02 |
| Global gradient-norm limit | 0.5 |
| Precision | FP32 weights and Adam moments; BF16 autocast |
| Batch size | Two matrices per update |
| Schedule | 600-step linear warmup, then 127,400-step cosine decay |
| Learning-rate floor | 5% of peak, or 5e-7 |
| Training duration | 128,000 updates; one stationary mixture, not a staged curriculum |
| Model/data seeds | 1201 / 1150924 |
| Validation | Fixed synthetic panel every 2,000 steps, including step zero |

The mean-teacher recipe was first completed at 64k. Its best fixed-validation score occurred at the final step, motivating a longer horizon. The 128k trial changed only the phase endpoint and cosine-decay duration, apart from output paths. It restarted from the original anchor with the same prior, teachers, seeds, objective, warmup, and peak/floor rates. The saved shared data prefix matches the 64k run.

This is **not** simply appending 64k updates to the old run: the post-warmup learning rates differ because the cosine is stretched over the longer horizon. It therefore tests the longer training-and-schedule recipe, not more steps at matched intermediate learning rates.

The completed training took approximately 6.31 hours on an H200; the training/audit/evaluation pipeline took approximately 6.62 hours.

## 6. Reject unsupported alternatives and select on synthetic validation

Several explored directions did not become part of the final recipe. Gradient-pooling diagnostics did not justify adopting accumulation. Parameter averaging did not deliver the useful prediction-averaging behavior. A preregistered 1,024-update paired pilot replacing mean-Huber supervision with distribution-KL distillation improved NLL/KL but worsened both true-target and teacher-mean MSE; it failed its admission gate and was not taken to a full training run.

The final checkpoint selection rule is the **earliest strict minimum in fixed synthetic macro RMSE**, not public benchmark performance. All 128k updates completed, and all 65 fixed validations were considered. Step 98k achieved 0.9586929621, versus 0.9698871111 for the preceding 64k teacher-mean winner. Independent synthetic panels, retention tests, public benchmarks, and runtime measurements were report-only for checkpoint selection.

The model is:

```text
mcpfn/src/tabimpute/workdir/
  tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/
  best_validation_checkpoint.pth

SHA256: 1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d
```

## 7. What the resulting model achieves—and does not establish

| Benchmark-selection dataset | Original DWIP MI R² at budget 5 | New model |
|---|---:|---:|
| MTEB | 0.690752 | 0.765919 |
| MMLU | 0.876944 | 0.901758 |
| Merged | 0.286567 | 0.372824 |
| BenchPress | 0.095801 | 0.167798 |

Across the four datasets, three acquisition methods, and budgets 1–15, the new model improves 156/180 mean settings over the original. Both independent synthetic macro-RMSE summaries and all eight retention-group summaries improve over the original. Public inference runtime and peak memory are essentially unchanged because the deployed architecture and model count are unchanged.

Later frozen-model evaluations also improve aggregate NRMSE over original DWIP v2-2 on MissBench (1.733265 vs 1.778209) and the UCI imputation suite (1.009580 vs 1.196231). These suites use their own saved evaluation protocols and metric definitions.

The gains are not universal: 24/180 public mean settings regress versus the original, and BenchPress MI@5 is worse than the preceding 64k teacher-mean model. Recommender-system transfer is weaker: MovieLens 100K/Netflix-small RMSE is 0.925936/0.948745, worse than saved historical MF results, with weak top-10 ranking. The experiment establishes useful measured improvements on the evaluated suites, not statistical significance, universal superiority, or a proven causal explanation for every change. The final horizon trial uses one training seed.

## Evidence locations

Paths below are relative to `/root/tabular` unless absolute:

- Final configuration: `mcpfn/src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/config.json`.
- Final assessment and audits: `mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918/ASSESSMENT.md` and `goal_completion_review.json` in that directory.
- Student/teacher loss and optimizer: `mcpfn/scripts/dwip_teacher_mean_training.py`; horizon-only recipe: `mcpfn/scripts/dwip_teacher_mean_horizon.py`.
- Prior and support transforms: `mcpfn/scripts/run_dwip_anchor_mix.py`, `mcpfn/scripts/general_spectral_geometry.py`, `mcpfn/scripts/spectral_support_normalization.py`, and `mcpfn/scripts/rowcol_fixed_validation.py`.
- Huber motivation: `mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_point2_64k_20260916/next_robust_objective_proposal.json`.
- Teacher complementarity: `mcpfn/runs/objective_queues/dwip_top2_independent_complementarity_20260917/REPORT.md`.
- Rejected KL pilot: `mcpfn/runs/objective_queues/dwip_teacher_distribution_adam_pilot_20260918/REPORT.md`.
- Imputation transfer: `mcpfn/runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_missbench_uci_20260918/NRMSE_COMPARISON.md`.
- Recommender transfer: `mcpfn/runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919/RESULTS.md`.
