# Autoresearch timeline: DWIP v2-2 to the current MatrixFormer

**Reconstructed:** October 2, 2026. **Dates:** UTC.

This is a chronological account of the research goals, completed training
trials, diagnostics, rejected alternatives, and checkpoint decisions that led
from the original DWIP v2-2 to the current general-purpose MatrixFormer. It
covers all **16 completed DWIP training trials in the September 15–18 campaign**,
the intervening side branches, and the September 30 large-matrix continuation.
Repeated monitoring turns and preflight retries are summarized rather than
treated as new model experiments.

## 1. Executive summary

The winning approach was **not a new network architecture**. We returned to
DWIP v2-2 and improved its synthetic training recipe:

1. Broaden the prior to include spectral tails and varied singular-vector geometry.
2. Stabilize sparse-observation training with observed-only variance shrinkage.
3. Mix in small latent-factor matrices to retain the older model's capabilities.
4. Strengthen predictive-mean supervision, then replace MSE with Huber loss.
5. Use two complementary synthetic-trained models as frozen teachers, while
   retaining true-target NLL and CDF supervision.
6. Train a fresh student on a longer cosine schedule; select the **98k checkpoint**
   from a completed 128k-update run using synthetic validation.
7. Continue those selected weights with streamed, activation-checkpointed
   training on matrices up to **2,000 × 1,000**, selecting update **4,000** of
   the new 5,000-update run.

The current objective is:

```text
teacher_target = stop_gradient((teacher_1_mean + teacher_2_mean) / 2)

loss = NLL(student_distribution, true_synthetic_target)
     + 2.0 × Huber_delta_1(student_mean, teacher_target)
     + 0.05 × CDF_MSE(student_distribution, true_synthetic_target)
```

The two teachers are used **only during training**. Deployment uses one model,
not a two-model ensemble. All training in this general-purpose lineage is synthetic.

### The three checkpoints that must not be confused

| Role | Checkpoint | Meaning |
|---|---|---|
| Original anchor | `/root/tabimpute_anchor3_named/dwip_v2_t003_checkpoint_17000.pth` | The unchanged DWIP v2-2 starting model. |
| September 18 parent | [`best_validation_checkpoint.pth`][parent-checkpoint] | Step 98,000, selected after completing the 128k teacher-mean run. |
| Current general model | [`best_validation_checkpoint.pth`][current-checkpoint] | September 30 continuation update 4,000; parent-relative cumulative step 102,000. |

The 98k and 102k counters are **not total optimization since original
pretraining**: 102k means 98k selected parent updates plus 4k continuation
updates. We did not continue from the terminal 128k weights, and we must not
add together the steps of all experimental trials as one training trajectory.

## 2. Goal and session timeline

| Date | Goal or decision | What happened |
|---|---|---|
| July–August 2026 | Explore scalable matrix-completion architectures. | The earlier compression/RowCol-SVT work investigated information flow, priors, objectives, and optimization. These are precursor experiments, not the final checkpoint's direct weight ancestry. |
| August 22–23 | Recover DWIP v2's successful prior and improve generalization. | The discussion identified DWIP as a strong existing architecture and established the numerical MTEB Gaussian-reference target, MI R²@5 **above 0.7620**. |
| September 1–9 | Diagnose before changing architecture again. | Precision, gradients, head behavior, attention, point supervision, and sparse-support normalization were investigated. Generic spectral coverage became a central hypothesis. |
| September 10 | Explicit autoresearch goal for the scalable RowCol/cell-query model. | Improve on general spectral priors, without MTEB-specific training or benchmark-based checkpoint selection; retain broad accuracy and practical scaling, with runs below 200k steps and approximately 24 hours. |
| September 10–15 | Controlled RowCol spectral/optimization campaign. | Tested query freezing, axis-balanced attention, learning rates, variance shrinkage, broader spectral geometry, longer training, and optimizer-history retention. Useful training ideas emerged, but the architecture remained too slow. |
| September 15, 22:26 | **User-directed pivot back to DWIP v2-2.** | Stop the slower architecture campaign; train the existing DWIP checkpoint with the newest generic priors. Keep the DWIP architecture unchanged. |
| September 16–17 | Resume the DWIP goal in a new session. | Trials 1–15 below investigate schedule, point loss, prior coverage, rehearsal, robust loss, and teacher supervision. |
| September 17–18 | Resume again; diagnose incomplete teacher compression. | Fresh synthetic replication and a paired KL pilot precede the longer teacher-mean trial. |
| September 18 | **Generalization goal achieved on the registered suite.** | Trial 16 selects 98k, reaches MTEB MI R²@5 **0.765919**, and passes the recorded broad-accuracy/runtime assessment. |
| September 20–23 | Separate recommender-specific goals. | A synthetic-only MF-comparison recipe is established; a harder BPR-ranking-plus-rating goal remains unmet and is stopped by the user. Neither branch supplies the current general checkpoint. |
| September 30, early UTC | Larger inference contexts and streamed training. | Sequential execution of independent attention batches and checkpointed backward passes make much larger full-context matrices practical. |
| September 30, approximately 05:42–22:28 | **Large-matrix continuation.** | Complete 5,000 updates from the selected 98k parent; select update 4,000 under synthetic large-matrix and retention criteria. |
| September 30–October 1 | Evaluate the new checkpoint. | Rerun recommender, MissBench, UCI, and LLM-score-completion evaluations. UCI context-window experiments and runtime figures are inference/evaluation work, not new training checkpoints. |

**Why the architecture pivot mattered.** The archived RowCol profile measured
approximately **25.303 s** at 1,000 × 1,000 with its 128-query chunks, versus
approximately **1.376 s** for DWIP in the preceding matched comparison. Raising
the query chunk to 2,048 still took 22.800 s. Repeated SVD and per-target
query/context reconstruction dominated the former architecture. We retained
the useful prior/training lessons, not that architecture. These historical
timings are separate from the later legacy-TabImpute runtime figure.

Sources: [research ledger][ledger], [DWIP handoff][handoff],
[runtime diagnosis][runtime-diagnosis], and the session index in Section 11.

## 3. Complete direct DWIP trial ledger: September 15–18

All rows below are **completed training runs**, not partial checkpoints promoted
while training was still in progress. All use the same DWIP architecture.
They start from the **original 17k anchor with fresh optimizer state**, rather
than being chained weight continuations. Later trials inherit a *recipe*, not
the preceding student's trained weights. Teachers are an explicitly identified
exception: their frozen weights supervise the student.

The fixed-synthetic score is the common checkpoint-selection metric: lower
macro RMSE is better. MTEB MI R²@5 is report-only: higher is better. Dates
indicate the selected checkpoint/completed-run period, not necessarily the
date embedded in the directory name. The 72-case fixed cache includes monitor
cases; the selection macro excludes monitor groups.

| Run | Date (2026) | Main experimental factor | Completed updates | Selected update | Fixed synthetic RMSE | MTEB MI R²@5 |
|---|---|---|---:|---:|---:|---:|
| Original | Before campaign | Unchanged DWIP v2-2 anchor | — | Original 17k | — | 0.690752 |
| [R01][r01] | Sep 15–16 | New generic spectral/geometry prior; NLL + 0.2 MSE + 0.05 CDF | 24,000 | 24,000 | 0.991710 | 0.747885 |
| [R02][r02] | Sep 16 | Extend the original spectral recipe to a 48k cosine horizon | 48,000 | 46,000 | 0.977651 | 0.758460 |
| [R03][r03] | Sep 16 | Increase mean-MSE weight from 0.2 to 2, at 24k | 24,000 | 18,000 | 0.991865 | 0.743503 |
| [R04][r04] | Sep 16 | Continuous high-rank decay-exponent coverage, at 24k | 24,000 | 18,000 | 0.989745 | 0.748397 |
| [R05][r05] | Sep 16 | One second-data-seed replication of the 48k spectral recipe | 48,000 | 48,000 | 0.977188 | 0.755065 |
| [R06][r06] | Sep 16 | Complete the loss-weight × horizon control: MSE weight 2 at 48k | 48,000 | 44,000 | 0.980024 | 0.759423 |
| [R07][r07] | Sep 16 | Continuous-spectrum prior at the longer 48k horizon | 48,000 | 46,000 | 0.975054 | 0.753381 |
| [R08][r08] | Sep 16 | Second-data-seed replication of MSE weight 2 at 48k | 48,000 | 38,000 | 0.981743 | 0.754637 |
| [R09][r09] | Sep 16 | Add 25% small latent-family rehearsal to the spectral mixture | 48,000 | 38,000 | 0.985150 | 0.748549 |
| [R10][r10] | Sep 16 | Extend the rehearsal recipe from 48k to 64k | 64,000 | 50,000 | 0.977794 | 0.751384 |
| [R11][r11] | Sep 16 | Increase mean-MSE weight to 2 within the 64k rehearsal recipe | 64,000 | 62,000 | 0.978528 | 0.759329 |
| [R12][r12] | Sep 16–17 | Replace MSE with Huber, delta 1; retain point weight 2 | 64,000 | 54,000 | 0.972307 | 0.757873 |
| [R13][r13] | Sep 17 | Huber curvature/weight control: point coefficient 2 → 4 | 64,000 | 54,000 | 0.975741 | 0.762280 |
| [R14][r14] | Sep 17 | Huber weight 2 with continuous-spectrum prior | 64,000 | 56,000 | 0.972494 | 0.749244 |
| [R15][r15] | Sep 17 | Huber target becomes the equal mean of frozen R12/R14 teachers | 64,000 | 64,000 | 0.969887 | 0.753972 |
| [R16][r16] | Sep 18 | Same teacher-mean recipe; stretch the cosine horizon to 128k | 128,000 | **98,000** | **0.958693** | **0.765919** |

**Important selection example:** R13 already exceeded 0.7620 on MTEB, but its
fixed synthetic RMSE was worse than R12's. It was not promoted merely because
the public benchmark looked better. Conversely, R15 improved the selector while
its MTEB score fell. The sequence was not a monotonic benchmark hill-climb.

The [final inventory][inventory] contains these 16 raw training selections plus
four rejected parameter averages. Each run directory links to its manifest,
selection evidence, and `dwip1e5_mteb.json`; these are the sources of the table.

## 4. How the synthetic prior evolved

### 4.1 Spectral coverage and geometry

The transferred spectral recipe varies more than matrix rank. It varies the
singular-value decay, head/tail structure, scale, and singular-vector geometry.
The geometry generator mixes:

- **Haar/random orthogonal** bases.
- **Coherent** bases with concentrated row leverage.
- **Smooth** bases from latent spatial features.
- **Offset-bearing** bases with approximately constant directions.

The early continuous-spectrum experiments replace separated high-rank exponent
ranges with a continuous uniform range over **[0, 2.5]**. This fills the gap
between slow and steep spectra. It produces a useful second teacher, but does
**not** replace the final student's separated slow/steep mixture.

### 4.2 Sparse-support stabilization

Training includes episodes with very few observations in a query row or column.
Diagnostics found that standardizing these columns could create extreme
normalized targets. The adopted training augmentation shrinks each column's
observed variance toward **pooled within-column variance**, using pseudo-count
**1**. Means and variances depend only on the remaining observed entries.

This is a numerical/data-augmentation change, not evidence that low training
loss guarantees generalization. It did **not** silently change the public
inference normalization used in the original goal comparisons.

### 4.3 Why add latent-family rehearsal?

Spectral-only fine-tuning improved many spectral problems but hurt small
latent-factor cases. R09 therefore added a fixed **25% latent-family share**;
R10 tested whether the resulting mixture needed a longer schedule.

The final September 18 mixture has eight equal-weight components, grouped as:

| Family | Total share | Rank / decay coverage | Missingness components |
|---|---:|---|---|
| Low-rank spectral | 25% | Rank 2–16 | MCAR and `profile_mixed` |
| Slow spectral tails | 25% | Rank 2–160; exponent [0, 0.5] | MCAR and `profile_mixed` |
| Steep spectral tails | 25% | Rank 2–160; exponent [1, 2.5] | MCAR and `profile_mixed` |
| Small latent factors | 25% | Rank 1–11; mixed latent distributions | MCAR and `profile_mixed` |

Ranks are subject to feasible matrix dimensions. Spectral training shapes are
**80–320 rows × 35–100 columns**; latent rehearsal uses **10–50 × 5–50**.
Each of the eight family/missingness components carries 12.5% nominal mass.

This is **family-level rehearsal**, not a replay buffer of the original model's
historical examples. Likewise, the training mixture's missingness components
must not be confused with every named MissBench evaluation mechanism.

Sources: [R16 manifest][recipe], [geometry implementation][geometry-source],
[variance-shrinkage implementation][shrinkage-source].

## 5. How the loss and optimization evolved

### 5.1 NLL alone does not directly optimize a useful predictive mean

The initial spectral DWIP objective was:

```text
NLL(true target) + 0.2 × MSE(student mean, true target)
                 + 0.05 × CDF_MSE(true target)
```

NLL supervises the predictive distribution; the explicit point term targets
the predictive mean used for RMSE/R². The CDF term compares the predicted
cumulative probabilities against the target's step function at the output
bucket means, averaging squared differences across buckets.

R03/R06 and later R11 increased the point coefficient to **2.0**. Controlled
seed and horizon comparisons showed that the effect depended on the rest of
the recipe; a stronger point term was not sufficient by itself.

### 5.2 Why Huber rather than more MSE or just gradient clipping?

Trainer-shaped diagnostics found no evidence of a wholesale Adam sign bug or
gradient collapse. They did find substantial residual concentration: roughly
the largest **1% of sampled training batches accounted for 45% of summed MSE**.

Huber with delta 1 is quadratic near zero and linear in the tails. Its
per-residual derivative is bounded, unlike MSE's. Global gradient clipping caps
the combined update but does not remove the disproportionate influence of a
few large residuals before aggregation.

R12 therefore tested Huber at the existing coefficient 2. R13 doubled it to 4
as a local-curvature control: PyTorch Huber's quadratic branch is `0.5 * error²`,
so Huber and MSE at the same coefficient are not curvature-matched. R13's
synthetic result was worse, so coefficient 2 remained in the selected recipe.
Huber is not an unbiased guarantee of minimum RMSE; the benefit was empirical.

### 5.3 The optimizer remained AdamW

| Setting | September 18 teacher-mean recipe |
|---|---|
| Initialization | Original DWIP v2-2 anchor; fresh optimizer |
| Optimizer | AdamW |
| Peak learning rate | `1e-5` |
| Warmup | 600 updates |
| Decay | Cosine to 5% of peak (`5e-7`) |
| Weight decay | 0.02 |
| Gradient-norm clip | 0.5 |
| Precision | FP32 parameters/Adam moments; BF16 computation |
| Matrix batch size | 2 |
| R15 / R16 horizon | 64k / 128k updates |
| Architecture | 8 layers, embedding 768, 16 heads, MLP 1536, 12 CLS tokens, 5,000 output bins |

The 128k experiment **restarted from the original anchor**, not R15's 64k
weights. It preserved teachers, priors, seeds, warmup, and peak/floor rates,
but stretched cosine decay from 63,400 to 127,400 updates. The shared data
prefix was verified. The intermediate learning rates intentionally differ:
this tests a longer **training-and-schedule recipe**, not additional steps at
matched learning rates.

Sources: [training diagnostics and trial rationale][handoff],
[earlier recipe summary][older-summary], [R16 assessment][assessment].

## 6. How student–teacher training entered the model

### September 17: parameter averaging failed, prediction averaging helped

The two best fixed-synthetic raw checkpoints were R12/54k and R14/56k. They
share the model seed but differ in prior; they are **not independent-seed
replicas**. The pair and equal weighting were fixed without a pair/weight search.

| Variant | Fixed synthetic macro RMSE |
|---|---:|
| Standard-prior Huber teacher, R12/54k | 0.972307 |
| Continuous-spectrum Huber teacher, R14/56k | 0.972494 |
| Equal average of their parameters | 0.975629 |
| Equal average of their predictions | 0.962126 |

The parameter average was rejected. The prediction mean also improved both
independent synthetic macro scores: **0.997187** spectral and **7.335792**
general, compared with the better constituent scores of 1.004653 and 7.340092.
This motivated compressing the useful prediction average into a **single student**.

### September 17: teacher-mean Huber student, R15

For the same masked synthetic input, run the two frozen teachers, detach their
predictive means, and average them equally. **Only the Huber point target is
replaced.** NLL and CDF continue to use the true generated values.

R15 improved fixed RMSE to **0.969887**, selected at its 64k endpoint. However,
MTEB MI R²@5 fell to **0.753972**, and an independent spectral panel regressed
relative to R12. This was a promising synthetic-selected student, not yet a
completed generalization goal.

### September 17–18: diagnose before changing the recipe again

- A **208-case cached diagnosis** decomposed teacher error and student
  compression error. Matching the teacher mean was incomplete; a fixed-panel
  gain did not establish faithful compression.
- A **544-case fresh synthetic replication**, from 272 underlying matrix
  draws, found that the old spectral regression did **not** reproduce, while
  incomplete teacher compression did. This argued against a prior designed to
  repair one previously bad case.
- A distribution-distillation alternative replaced the point auxiliary with
  teacher-mixture KL. After feasibility/gradient checks, a paired **1,024-update
  Adam pilot** tested it from the original initialization on identical batches.

| Pilot endpoint metric | Mean-Huber control | Distribution-KL candidate |
|---|---:|---:|
| True-target MSE | 1.312036 | 1.505570 |
| Teacher-mean MSE | 0.222778 | 0.431129 |
| True-target NLL | 1.047144 | 0.944853 |
| Teacher-distribution KL | 0.521403 | 0.408727 |

KL improved density metrics but worsened both mean-error admission criteria.
The proposed full KL run was rejected; no pilot checkpoint was saved or
promoted. Better likelihood was not equivalent to better point prediction.

### September 18: extend the successful objective, R16

R15's best fixed validation was still at its final update. This supplied the
synthetic learning-curve rationale for one 128k horizon trial with the same
teacher-mean objective. It finished all 128k updates in **6.31 hours**; the
full training/audit/evaluation pipeline took **6.62 hours**.

The earliest strict minimum across **65 fixed validations**, including step
zero, selected **98k / RMSE 0.958693**. Public results did not determine the
checkpoint. The final goal assessment records completion after checking
synthetic performance, transfer, retention, runtime, and artifact integrity.

Sources: [parameter-average rejection][top2-average],
[independent complementarity][complementarity], [cached compression diagnosis][compression],
[fresh replication][fresh], [KL pilot][kl-pilot], [final assessment][assessment].

## 7. Rejected alternatives and side branches

These are included to avoid presenting only a hindsight-selected success path.

| Date | Alternative | Outcome / disposition |
|---|---|---|
| Sep 10–15 | Slower RowCol-SVT/cell-query architecture | Prior and optimization lessons transferred; architecture abandoned at the user's request. |
| Sep 16 | Average R02 trajectory snapshots at 24k/36k/48k | Fixed RMSE 0.982796; rejected. |
| Sep 16 | Average the two original-loss 48k data-seed models | Fixed RMSE 0.977461; failed the requirement to beat both constituents. |
| Sep 16 | Average the two point-weight-2 48k data-seed models | Fixed RMSE 0.980772; failed the same requirement. |
| Sep 17 | Average R12/R14 model parameters | Fixed RMSE 0.975629; rejected despite their useful prediction complementarity. |
| Sep 17 | Use the continuous-spectrum recipe as the final student prior | Not adopted; that recipe contributed a frozen teacher instead. |
| Sep 18 | Replace mean-Huber with distribution-mixture KL | Paired 1,024-update pilot failed mean-error gates; no full training launched. |
| Sep 20–21 | Recommender-specific synthetic priors, rating-risk objectives, and normalization | A separate synthetic-only recipe beat the fixed MF reference in 14/14 reported comparisons with 2,000 × 1,000 windows. It was not the current general-purpose checkpoint. |
| Sep 21–23 | Beat BPR on ranking while retaining strict MF rating wins | Persistent ranking/rating trade-offs; the joint goal was not achieved and was stopped on September 23. |

**Recommender provenance caveat.** The recommender branch explicitly retracted
an earlier result that used real-rating neural updates; those runs and their
descendants were disallowed. Its later synthetic-only MF result is separately
audited, uses specialized context normalization and previously inspected
benchmarks, and must not be confused with the general model's ordinary
global/column-normalized recommender scores. The stronger BPR goal remained
unmet. None of these recommender-specific checkpoints is an ancestor of the
September 30 general-model continuation.

Sources: [20-candidate inventory][inventory], [corrected MF result][mf-result],
[BPR goal history, including its final stop record][bpr-goal].

## 8. September 30: train the selected 98k model on larger matrices

### Memory adaptations, not a new architecture

The existing row/column attention already treats independent rows or columns
as an attention batch. Processing those independent batches sequentially limits
peak memory without shortening the attended row or column. Differentiable
streaming and activation checkpointing extend this to backward passes;
decoder-plus-loss microbatching avoids materializing all output-bin logits.

Both frozen teachers also process the complete matrix through streamed
inference. This is distinct from inference **context windowing**, which makes
separate predictions on smaller submatrices and does change available context.

An earlier small smoke test used NLL alone to verify streaming. The real
large-matrix run **does not** use that simplified objective: it preserves
NLL + 2 × teacher-mean Huber + 0.05 × true-target CDF.

### Curriculum and retained mixture

| Continuation updates | Spectral row range | Spectral column range |
|---|---:|---:|
| 1–500 | 256–512 | 128–256 |
| 501–1,500 | 512–1,024 | 256–512 |
| 1,501–3,000 | 768–1,536 | 384–768 |
| 3,001–5,000 | 1,024–2,000 | 512–1,000 |

Matrices are transposed with probability 0.5, so both orientations occur.
The two-million-cell ceiling is **62.5×** the old 320 × 100 spectral ceiling;
it is not the size of every example or a claim about the machine's absolute limit.

The original 25% small latent rehearsal is retained. Additionally, 20% of
spectral draws use the old dimensions. Expected episode shares are therefore:

- **60%** new-size spectral matrices.
- **15%** old-size spectral matrices.
- **25%** small latent-factor matrices.

The model accumulates two independently sampled matrix-mean losses serially.
Matrices are equally weighted rather than letting larger matrices dominate
through their number of missing targets.

### Optimization and selection

| Setting | Large-matrix continuation |
|---|---|
| Starting weights | Selected September 18 **98k** checkpoint |
| Optimizer | Fresh AdamW; not the old 128k optimizer state |
| Schedule | 200-update warmup to `5e-6`; cosine to `2.5e-7` at update 5,000 |
| Weight decay / gradient clip | 0.02 / 0.5 |
| Effective batch | Two serial matrix microsteps per update |
| Precision | FP32 weights/moments, BF16 autocast, FP32 per-cell loss reductions |
| Validation | Every 250 updates, including step zero: 21 validations |
| Selector | Mean of large/retention baseline-relative macro RMSEs |
| Retention guard | Retention RMSE at most 1.03 × parent baseline |
| Completed / selected updates | 5,000 / **4,000** |
| Observed training time / peak allocation | Approximately **16 h 46 min / 85.51 GiB** |

The FP32 CDF reduction preserves the mathematical objective but is not bitwise
identical to the older trainer's final BF16-rounded CDF scalar. The synthetic
validation suite and selector also change for this new size objective; its
scores must not be compared directly with the earlier 0.958693 selector score.

| Synthetic population | 98k parent | Selected continuation, update 4k | Change |
|---|---:|---:|---:|
| Large-matrix macro RMSE | 0.378201 | **0.358148** | **−5.30%** |
| Small-matrix retention macro RMSE | **0.776874** | 0.780783 | +0.50%; passes 3% guard |

Update 5k has slightly worse large/retention scores, so update 4k remains
selected. The same original R12/R14 teachers remain frozen: they were not
replaced by the 98k parent or recommender-specific models.

Sources: [streamed-training design][streamed-doc], [frozen run configuration][large-config],
[validation trajectory][large-validation], [terminal status][large-status],
[full-objective implementation][large-objective].

## 9. What improved, and what did not

### September 18: outcome of the original generalization goal

| LLM-score-completion dataset | Original DWIP MI R²@5 | Selected 98k parent |
|---|---:|---:|
| MTEB | 0.690752 | **0.765919** |
| MMLU | 0.876944 | **0.901758** |
| Merged | 0.286567 | **0.372824** |
| BenchPress | 0.095801 | **0.167798** |

The 98k parent improves **156/180** public mean settings versus the original,
both independent synthetic macro RMSEs, and all eight retention-group macros.
Its 1,000 × 1,000 latency is essentially unchanged from original DWIP.
Nevertheless, 24/180 public settings regress; BenchPress MI@5 is lower than
R15's 0.185053. This is the registered goal's measured success, not uniform
dominance or a statistical-significance claim.

### September 30–October 1: outcome of larger-matrix continuation

- **Recommenders:** with the preselected global-normalization protocol,
  RMSE and MAE improve on all four tested datasets. MovieLens 10M RMSE changes
  from **0.884089 to 0.866433**. This does not establish a win over MF/SVD.
- **MissBench:** at 40% missingness, eight of eleven pattern-level NRMSEs
  improve. MCAR changes from **0.923143 to 0.893521**; three patterns regress.
- **UCI:** full-matrix NRMSE becomes slightly worse in all three main patterns.
  A subsequent **2,000 × 1,000 context-window** experiment improves the new
  model relative to its full-matrix evaluation. MCAR becomes **0.697792**
  versus the parent's **0.698053**; MAR and MNAR remain slightly worse than
  the parent. Context-window changes are not new training.
- **LLM score completion:** MI R²@5 changes from **0.7659 to 0.7639** on
  MTEB, **0.9018 to 0.8992** on MMLU, **0.3728 to 0.4073** on Merged, and
  **0.1678 to 0.1827** on BenchPress. The continuation is not uniformly better.
- **Causal inference:** the September 30 tabular rerun explicitly excluded
  causal benchmarks; it does not establish a current-checkpoint causal gain.

These benchmark evaluations are report-only and involve previously inspected
data. Across-trial research decisions have been informed by reported results,
so synthetic checkpoint selection is not equivalent to a wholly blind research
process. The training chronology supports the adopted recipe; it does not
prove a uniquely causal contribution from every ingredient.

Sources: [98k final assessment][assessment], [recommender comparison][recsys-eval],
[MissBench/UCI comparison][tabular-eval], [UCI context-window comparison][uci-chunks],
[new-checkpoint LLM comparison][llm-eval].

## 10. Checkpoint lineage and reproducibility

```text
Original DWIP v2-2, 17k anchor
  ├─ R01–R14: controlled prior / horizon / point-loss experiments
  │    ├─ R12 selected 54k: standard-prior Huber teacher ───┐
  │    └─ R14 selected 56k: continuous-prior Huber teacher ─┤ frozen targets
  ├─ R15: fresh student, 64k teacher-mean schedule ◀────────┤
  └─ R16: fresh student, 128k teacher-mean schedule ◀───────┘
       └─ select 98k weights, not the terminal 128k weights
            └─ September 30: 5k large-matrix continuation
                 └─ select continuation update 4k [CURRENT]

The recommender-specific MF/BPR branch is separate and is not used above.
```

| Checkpoint | Recorded SHA-256 |
|---|---|
| Original DWIP v2-2 | `36f5a0db09a4226329f8d1874ea4faeff19c3f82913a0d365fa7e4171acf8b07` |
| R12/54k teacher | `2e5af70eed550575a9a3fb035eb69ca781a457085b4ab95cde098d66749ccdb9` |
| R14/56k teacher | `9193b08a96c0d11c9ad3d893b1f0b1b26dc8ab9cb7b4afb9e8168712039b517b` |
| R16/98k parent | `1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d` |
| Current continuation/4k | `c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9` |

The directory links below retain full run identities. The inventory/manifests
are authoritative for selected checkpoints and recipes; terminal assessments
are authoritative for completed scientific outcomes. Historical status prose
can still say “active” because it records an earlier stage. An older session's
blocked state does not undo the later session's completed September 18 assessment.

## 11. Source and session index

### Goal/session history

| Session | Conversation ID | Relevance |
|---|---|---|
| Scalable Architecture | `019fa127-1734-7612-9aef-e0b073bd79d8` | Precursor work, September 10 goal, September 15 pivot back to DWIP. |
| Resume Scalable Architecture chat | `01a0a7bc-0503-71c3-a68f-25c913668f16` | Direct DWIP campaign through teacher-mean development and diagnostics. |
| Resume the previous session | `01a0afb2-ecba-75f1-bd0a-59725c6dd222` | September 17 resumption, 128k outcome, later evaluations and memory work. |
| Beat MF baseline scores | `01a0bc81-5ae1-71f1-ad6a-f1a0b75ce2dc` | Separate recommender/MF/BPR branch and September 23 stop. |
| Train on larger matrices | `01a0f0ba-2b74-7ef2-a83b-cbd66285eab6` | September 30 curriculum, preserved composite loss, completed 5k run. |

The timeline was reconstructed from those conversation histories, saved goal
records, training manifests, the final checkpoint inventory, metric JSON/CSV
files, and completion assessments. No training or benchmark was rerun to write it.
Relative links work from this file in the repository; a standalone downloaded
copy retains the narrative and numbers, but its linked artifacts remain local.

[ledger]: mcpfn/runs/information_flow_diagnostics/RESEARCH_LEDGER.md
[handoff]: mcpfn/runs/information_flow_diagnostics/SCALABLE_ARCHITECTURE_HANDOFF_20260916.md
[runtime-diagnosis]: mcpfn/runs/benchmark_evaluations/geometry_runtime_profile/REPORT.md
[older-summary]: mcpfn/DWIP_NEW_MODEL_SUMMARY.md
[r01]: mcpfn/runs/objective_queues/dwip_spectral_20260915/
[r02]: mcpfn/runs/objective_queues/dwip_spectral_horizon48k_20260916/
[r03]: mcpfn/runs/objective_queues/dwip_spectral_point2_20260916/
[r04]: mcpfn/runs/objective_queues/dwip_spectral_continuous_20260916/
[r05]: mcpfn/runs/objective_queues/dwip_spectral_seed2_48k_20260916/
[r06]: mcpfn/runs/objective_queues/dwip_spectral_point2_horizon48k_20260916/
[r07]: mcpfn/runs/objective_queues/dwip_spectral_continuous_horizon48k_20260916/
[r08]: mcpfn/runs/objective_queues/dwip_spectral_point2_seed2_48k_20260916/
[r09]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_48k_20260916/
[r10]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_64k_20260916/
[r11]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_point2_64k_20260916/
[r12]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_huber1_64k_20260916/
[r13]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_huber1_point4_64k_20260917/
[r14]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_huber1_continuous_64k_20260917/
[r15]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_64k_20260917/
[r16]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918/
[inventory]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918/final_saved_model_inventory.json
[recipe]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918/manifest.json
[assessment]: mcpfn/runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918/ASSESSMENT.md
[geometry-source]: mcpfn/scripts/general_spectral_geometry.py
[shrinkage-source]: mcpfn/scripts/spectral_support_normalization.py
[top2-average]: mcpfn/runs/objective_queues/dwip_top2_fixed_parameter_average_20260917/REPORT.md
[complementarity]: mcpfn/runs/objective_queues/dwip_top2_independent_complementarity_20260917/REPORT.md
[compression]: mcpfn/runs/objective_queues/dwip_teacher_compression_gap_20260917/REPORT.md
[fresh]: mcpfn/runs/objective_queues/dwip_fresh_compression_replication_20260917/REPORT.md
[kl-pilot]: mcpfn/runs/objective_queues/dwip_teacher_distribution_adam_pilot_20260918/REPORT.md
[mf-result]: mcpfn/DWIP_RECSYS_RESULT.md
[bpr-goal]: mcpfn/DWIP_BPR_GOAL.md
[streamed-doc]: mcpfn/STREAMED_TRAINING.md
[large-config]: mcpfn/runs/streamed_training/large_matrices_98k_20260930/config.json
[large-validation]: mcpfn/runs/streamed_training/large_matrices_98k_20260930/validation.jsonl
[large-status]: mcpfn/runs/streamed_training/large_matrices_98k_20260930/status.json
[large-objective]: mcpfn/scripts/streamed_large_matrix_training.py
[recsys-eval]: mcpfn/runs/benchmark_evaluations/matrixformer_large_trained_recsys_20260930/RESULTS.md
[tabular-eval]: mcpfn/runs/benchmark_evaluations/matrixformer_large_trained_missbench_uci_20260930/NRMSE_COMPARISON.md
[uci-chunks]: mcpfn/runs/benchmark_evaluations/matrixformer_uci_chunks2000_20260930/RESULTS.md
[llm-eval]: mcpfn/runs/benchmark_evaluations/matrixformer_large_trained_llm_20261001/LLM_COMPARISON.md
[parent-checkpoint]: mcpfn/src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/best_validation_checkpoint.pth
[current-checkpoint]: mcpfn/runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth
