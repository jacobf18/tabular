# DWIP v2-2 synthetic-only ranking goal

Registered 2026-09-21, after completion of the MF comparison. Status: active;
no claim of beating BPR yet.

## Required outcome

On both MovieLens 100K and Netflix small, strictly exceed reproduced BPR
NDCG@10 and Recall@10 while keeping MAE and RMSE strictly below the frozen MF
baseline. These are eight required comparisons. Report all seven existing
metrics, including AUC, MAP and Precision@10, without treating those three as
additional completion criteria.

## Fixed constraints

- Unchanged DWIP v2-2 network and state shapes; only synthetic priors and
  optimization, including synthetic objectives, sampling and checkpoint
  selection, may change.
- All neural gradients, initialization ancestors and any teachers must be
  synthetic-only. Never use real ratings or a real-trained baseline as a
  gradient target or teacher. Existing retracted runs remain ineligible.
- Continue from the audited synthetic MF winner, SHA256
  `1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770`.
- Freeze inference: 2000 user by 1000 item windows, predictive mean, two-way
  bias centering and shrunk user scaling, clipping to [1,5], unchanged public
  overlap aggregation and observed-rating preservation. No new scoring head,
  score blending, ensembles or fitted MF/BPR factors in DWIP inference.
- Real data is confined to evaluation context and inner-validation selection.
  Conventional BPR/MF fitting is isolated evaluation-reference computation,
  never neural training.
- Preserve frozen outer 80/20 seed-123 splits and seed-20260920 inner holdouts,
  relevance >=4, user-macro metrics and the existing Cornac candidate protocol.
  The outer benchmark has been observed before; it is not a blind test.
- First select on full inner NDCG/Recall subject to strict inner MAE/RMSE
  guardrails. Require a registered full inner gate before outer confirmation.
  Do not weaken the comparator or alter the evaluation protocol to win.

## Initial campaign

1. Reproduce and freeze the existing BPR configuration (10 factors, 200
   iterations, learning rate .001, regularization .01, seed 123), recording
   split, source and factor hashes and independently checking ranking metrics.
2. Reproduce the clean starting checkpoint through the fixed public interface.
3. Improve synthetic within-user positive/negative coverage and rank-focused
   objectives; retain synthetic rating-risk auxiliaries and rating guardrails.
4. Test exposure/preference prior variations only after controlled optimization
   experiments. Record failures and step-zero fallback honestly.
5. Confirm a qualifying candidate with independently audited outer metrics and
   checkpoint ancestry. Until all eight comparisons pass, keep the goal active.

Artifacts: `runs/recsys_bpr_goal_20260921/`. Previous MF-goal artifacts remain
unchanged. Training is serial on the H200. Preserve historical files and use
bounded checkpoint storage with a free-space reserve.

## Registered reference and first experiment

The reproduced BPR reference matches the historical four-decimal results.
Every ranking metric and every evaluated user's ranking metrics match the
independent NumPy audit. Its frozen score-file SHA256 is
`7b85b00b793fbcebfdc7cac2bb16fa3a656c47ffbd9e1979a1f6e48a1030b96d`.

| Partition | Dataset | BPR NDCG@10 | BPR Recall@10 | MF MAE | MF RMSE |
|---|---|---:|---:|---:|---:|
| Inner | MovieLens | 0.080900487 | 0.104120153 | 0.757859740 | 0.889766278 |
| Inner | Netflix | 0.085404527 | 0.123053922 | 0.737268413 | 0.837008864 |
| Outer | MovieLens | 0.149992753 | 0.119526998 | 0.743013001 | 0.899831176 |
| Outer | Netflix | 0.128125743 | 0.135255380 | 0.735776529 | 0.862116873 |

The original synthetic winner reproduces all seven inner metrics with maximum
absolute discrepancy 2.23e-15 through the unchanged public inference path.
`fixed_protocol.json` binds network/inference sources and original lineage;
`starting_checkpoint_reproduction.json` records that control.

First trial: `matched_bpr_weight1_200`, seed 61, 200 AdamW updates, LR 1e-5,
EMA .95, full inner evaluation every 50 updates. Its prior retains the previous
12288x4096 parents, 1000x1000 crops, rating/noise distributions and normalization.
The query sampler now supplies up to 16 unexposed negatives for each synthetic
user with a held-out positive. User-macro BPR loss has weight 1, versus the
previous .05 auxiliary; the existing expected user-RMSE/NLL/MAE objective is
retained at weight 1. No hard-negative truncation in this first ablation.
Step zero remains eligible, and any rating-regressing checkpoint cannot replace
a feasible checkpoint regardless of its ranking improvements.

Initial focused validation: 202 tests passed. Synthetic provenance is checked
before every gradient update; public real-data evaluation runs under no-grad.
No new DWIP outer evaluation is authorized until all eight inner comparisons
pass and a checkpoint-bound gate is written.

### First inner signal and registered follow-up

At update 50, matched-negative ranking training improves Netflix inner
NDCG@10 from .020080 to .038601 and Recall@10 from .026574 to .054999, but
RMSE regresses from .832037 to .847466 (MF .837009). MovieLens RMSE also
loses its MF guardrail. The checkpoint is therefore rejected, not a new winner.

Follow-up `matched_bpr_anchor5_200` uses the same initial weights, seed 61,
episodes, ranking weight, LR, update count and evaluation schedule, adding
weight 5 on user-macro squared deviation from the clean starting model's
predictions **on synthetic held-out rating queries only**. This teacher's
complete ancestry is synthetic; it is frozen, and receives no real ratings.
It does not participate in inference. This tests whether ranking learning can
be separated from rating-calibration drift without changing the architecture
or prior's rating distribution. It runs only after the first trial finishes.

The matched sampler preserves synthetic context, rating queries, normalization
and rating targets bitwise under the same seed. Objective gradients and the
synthetic-teacher anchor are covered by the now-205 passing focused tests.

The first 200-update trial is complete. Every trained checkpoint loses at least
one rating guardrail, so selection correctly retains step zero. The final
update's inner Netflix NDCG/Recall are .037426/.052773, but MAE/RMSE are
.747871/.855327; it is not eligible. Its campaign audit verifies all archived
sources, synthetic provenance, unchanged 254-tensor/77,998,688-element state,
unchanged inference and correct fallback selection. No new outer model result
was evaluated. The anchored follow-up is now running serially.

A tested, not-yet-run sampler option permits full-catalog synthetic negatives
for a bounded uniform subset of positive users. This enables hard-negative
top-K optimization without decoding the entire million-cell crop. The default
does not change the first two experiments. Focused tests now total 206 passing.

### Anchored inner improvement and top-K experiment

At anchored update 50, all four inner MF rating guardrails pass. MovieLens
MAE/RMSE are .745063/.877967 and NDCG/Recall are .056762/.073938. Netflix
MAE/RMSE are .732039/.833393 and NDCG/Recall are .023041/.031285. This is an
eligible inner improvement over the starting model, not a BPR win and not an
outer result. Worst BPR target ratio improves from 4.630658 to 3.933313.

Registered next: `block_hard10_anchor5_200`, same original synthetic starting
weights, seed 61, LR 1e-5, 200 updates, ranking weight 1 and anchor weight 5.
For 64 uniformly selected synthetic positive users per episode, decode all
eligible unexposed items in the existing 1000-item window, then optimize against
the ten highest-scoring negatives per user. Rating queries and their prior are
unchanged. Run a synthetic-only memory/gradient preflight first, serially after
the current anchored trial; require <128 GiB peak allocation and a frozen
synthetic-only teacher. This targets top-of-list mistakes rather than mostly
easy pairwise comparisons. No new outer evaluation is authorized.

An optional catalog-mixing prior is implemented and tested but not enabled in
these three trials. It preserves the first-observation user window while
sampling unique columns from the full synthetic parent catalog, retaining a
known held-out query anchor. It can test the hypothesis that comparisons within
separate item windows fail to teach globally comparable recommendation scores.
This changes synthetic episodes only, never the real-data inference windows.
Current focused tests: 208 passed.

### Inner popularity diagnostic

Two no-neural-training diagnostic controls use observed-context item counts
only. Both are independently checked against Cornac, and neither replaces the
frozen BPR goal comparator or supplies neural targets. On Netflix inner data,
positive-interaction popularity achieves NDCG/Recall .092448/.129089, versus
BPR .085405/.123054; all-interaction popularity achieves .084075/.123483.
MovieLens all-interaction popularity achieves .081163/.106886 versus BPR
.080900/.104120. Thus much of this particular BPR target is explainable by
popularity rather than demanding a more complex personalized interaction model.

This motivates synthetic sampling/optimization that teaches exposure popularity
and comparisons across item windows, while retaining rating-risk constraints.
No popularity score is blended into DWIP inference. A synthetic-only 80-episode
catalog audit confirms broader item-support variation under catalog sampling
(mean log-support standard deviation 1.120 versus 1.000 for window sampling);
that is a sampler property, not evidence of improved real-data metrics.

### Completed anchored trial

`matched_bpr_anchor5_200` selects EMA update 200. Its SHA256 is
`3df48f3de6f3173d5ed8189c812a72b0d0832335778cbb01377451368d47b031`.
The campaign audit verifies 238 changed weight tensors, the unchanged
254-tensor/77,998,688-element architecture, synthetic-only teacher and student
ancestry, archived training sources and reproduced inner checkpoint selection.

| Dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens inner | .745646 | .879335 | .060032 | .075922 |
| Netflix inner | .731494 | .833475 | .024560 | .033037 |

All four inner rating guardrails pass; none of the four BPR ranking targets
passes. There is no new outer claim. Netflix inner NDCG/Recall improve about
22%/24% over the starting checkpoint, but are still far from the BPR targets.

The hard-negative preflight passes at 84.925 GiB peak allocated memory with
52,665 decoded synthetic negatives, finite gradients and a frozen synthetic
teacher. `block_hard10_anchor5_200` is now training serially. The full objective
remains active and unmet; improving a subset of the targets is not completion.

### Expected-relevance prior and optimization

At hard-negative update 50, all rating guardrails pass, but Netflix inner
NDCG/Recall are only .020669/.026837; hard negatives inside a single item window
have not yet closed the gap. The trial still completes its registered schedule.

Registered follow-up: `expected_relevance_catalog50_anchor5_200`, initialized
from whichever completed anchored/hard-negative trial has the better eligible
inner selection score. Keep the original clean MF winner as the frozen
synthetic-input rating teacher. Use 200 updates, LR 5e-6, seed 61, EMA .95,
ranking weight 1, anchor weight 5, listwise temperature .5, and full known-unseen
candidate catalogs for 64 synthetic positive users per episode. Half the
episodes use global synthetic item sampling and half retain contiguous windows.
Inference remains unchanged. Require a serial synthetic gradient/memory
preflight before training.

The new ranking target is computed entirely from synthetic generating latents:
`exposure_probability * holdout_fraction * P(rating >= 4) /
(1 - exposure_probability * (1 - holdout_fraction))`.
This is the conditional probability of a held-out positive given absence from
context, under the synthetic generator. A user-macro listwise loss matches its
normalized candidate distribution. It can supervise candidate relevance
without treating every unexposed synthetic item as a certain dislike; no real
model, popularity statistic or real rating supplies these targets.

Tests verify the probability formula analytically and with synthetic Monte
Carlo, listwise loss/gradients, complete unseen catalogs, and rejection of real
gradient data. A six-case archived-source equivalence audit verifies all 13
existing episode tensor fields are bitwise unchanged when the new target is
disabled. There are 216 passing focused tests. An optional negative-only
pairwise gradient is also tested, but not enabled in this registered trial.

### Hard-negative conclusion and next trial launch

The hard-negative run completes and selects update 50, SHA256
`6b3406e5167df3f51cc5f6952b02ee4eb394dca1e19f8b2299806c489d1cb7d2`.
Its complete synthetic/provenance/architecture/selection audit passes, but its
worst ranking ratio 4.585185 is worse than the anchored run's 3.724728. Later
hard-negative updates improve MovieLens ranking while degrading Netflix, so
this trial does not replace the best eligible anchored checkpoint.

The expected-relevance run therefore starts from the audited anchored update
200. Its two-mode preflight exercises both contiguous and globally sampled
synthetic item contexts; both have finite losses/gradients, the teacher remains
frozen, no real data is loaded, and peak allocation is 84.996 GiB. The trial is
now training serially. Current focused tests: 217 passed. There are still no
new DWIP outer evaluations and no BPR-goal completion claim.

### Registered diagnostic re-evaluation

After the expected-relevance trial completes, re-evaluate the best completed
eligible candidate from the anchored, hard-negative and expected-relevance
trials on full inner data, breaking identical-score ties in favor of the earlier
trial. Cross-check all metrics and per-user ranking against Cornac. Collect
read-only top-10 item-popularity, sparse-item, score-ceiling, observed-low-item
and original-item-window diagnostics, including sparse-user strata. These do
not alter scores or candidate filtering and are not neural training data. The
new profile's Recall@10 must exactly reproduce the existing ranking evaluator
within 1e-12. No outer evaluation is authorized unless the full existing inner
gate passes all eight required comparisons.

### Expected-relevance inner signal and optimization safeguard

At update 100, expected-relevance training improves MovieLens inner NDCG/Recall
to .077497/.091323 and Netflix to .047660/.066428. MovieLens still passes its
rating guardrails, but Netflix MAE/RMSE .741084/.843395 lose to MF, so this
checkpoint is rejected. This improves the ranking/error trade-off over the
earlier unanchored ranking run, but does not satisfy the goal.

A tested optimizer option now protects the synthetic rating objective to first
order. It differentiates the synthetic rating-risk plus synthetic-teacher
anchor loss, executes the normal AdamW step, and projects the **actual parameter
update** if its directional derivative would increase that protected loss.
Using the actual update accounts for Adam's momentum, adaptive scaling and
weight decay. The projection uses only synthetic gradients and leaves the
network/inference unchanged. This is a local optimization safeguard, not a
guarantee of real-data rating metrics; all existing evaluation gates remain.
Tests cover conflicting and nonconflicting updates, momentum, finite-gradient
rejection, and two backward passes through the synthetic objective. Current
focused tests: 224 passed. This option has not yet been enabled in a training
trial; first inspect the completed trial and the registered ranking profile.

### Ranking-profile finding and registered projected recovery

The full best-candidate inner re-evaluation reproduces all metrics and Cornac
per-user ranking. Its gate correctly fails the four BPR targets. Netflix top-10
items average 501 context interactions, versus 1038 for held-out positives;
only 63% of recommendations come from the first 1000 original items, versus
86% of positives. About 17% of top-10 scores clip at 5 (26% for sparse users).
Observed low-rated context items are negligible in the recommendations. This
supports further exposure/popularity learning and calibration, not changing
candidate filtering or inserting a popularity score at inference.

The expected-relevance trial finishes with no eligible trained checkpoint:
selection retains its initial anchored weights. Its saved last EMA is nevertheless
synthetic-only and useful as an optimization starting point, not as a final
candidate. At update 200, MovieLens NDCG/Recall are .080699/.095425 and Netflix
.049483/.068048, but Netflix MAE/RMSE .740030/.842824 lose the MF guardrails.

Registered recovery: `projected_relevance_catalog50_anchor5_200`, starting from
that saved synthetic last EMA, with the original MF-winning synthetic model
remaining the frozen rating teacher. Keep the same expected-relevance objective,
50% catalog mixture, temperature .5, 64-user candidate slates and objective
weights; use seed 62, LR 5e-6, 200 updates, evaluation every 50, and actual-AdamW
update projection against the synthetic protected rating gradient. First profile
the rejected last EMA on inner data, then require both synthetic preflight modes
to pass memory, finite-gradient, teacher-freezing and projection checks. No real
data enters either gradient. A recovery candidate must still pass the original
eight-metric gate; an infeasible starting checkpoint does not weaken selection.

### Projected recovery preflight and live verification

Both preflight modes pass with 89.142 GiB peak allocation, finite losses and
gradients, unchanged architecture and a frozen original synthetic teacher.
The live recovery run is verified at update 30. Projection activates at logged
updates 20 and 30 and makes the protected directional derivatives negative;
all logged first-order checks pass. This establishes optimizer behavior only,
not rating recovery or a BPR win. The best completed eligible candidate remains
the anchored update 200; no new outer DWIP evaluation has been run.

### Clipping-aware optimization hypothesis

The frozen evaluator clips scores to [1,5], but the existing synthetic ranking
loss optimizes unbounded reconstructed means. Thus an ordering between scores
above five can improve training loss while remaining a tie in evaluation. The
rejected expected-relevance profile still has 12.56% ceiling scores in Netflix
top tens, versus 17.2% for the anchored candidate. This is a measured mismatch,
not proof that correcting it will close the BPR gap.

An optional `--ranking-boundary-weight` now makes synthetic ranking use the
same [1,5] forward clipping and adds a user-macro squared penalty outside
[1.05,4.95]. The small interior margin supplies inward gradients to resolve
saturation instead of stranding scores behind a zero clamp derivative. Only
users with ranking candidates contribute to this penalty. Rating risk, teacher
targets, architecture and public inference are unchanged. The default zero
weight preserves the original objective. Tests cover exact clipping, inward
gradients, user weighting, protected-loss decomposition and refusal of real
training episodes; all 233 focused tests pass. This option is not enabled in
the already-running projected recovery trial.

After inspecting the completed projected recovery, the next controlled
ablation may repeat expected-relevance training from the anchored update 200
with its original seed 61, LR 5e-6, 200 updates, anchor weight 5, catalog mixture
50%, temperature .5 and 64-user slates, changing only the new boundary weight
from zero to one. It must first pass both synthetic preflight modes. This
registration does not authorize an outer evaluation or replace the inner gate.

### Projected recovery terminated at its numerical safeguard

The projected trial exits after logged update 170, before update 200, because
three floating-point correction attempts do not satisfy the protected
directional-derivative tolerance. Its last completed validation is update 150:
MovieLens MAE/RMSE .748458/.882479 and NDCG/Recall .081387/.097626; Netflix
.738106/.840775 and .046529/.064030. Netflix still fails both MF rating
guardrails. The trial is explicitly marked failed, not completed; its stored
checkpoint is ineligible under the existing completed-run selection gate.
The best completed eligible checkpoint remains the anchored update 200.

Proceed with the registered clipping-aware ablation from the anchored model,
without the projection option. No model from the failed run is used. Before
future projected training, add a tested conservative rollback if FP32
quantization prevents a first-order correction; do not loosen the tolerance.

The fallback is implemented and tested on an explicit FP32 quantization stall:
if three finite corrections still violate the unchanged 1e-7 tolerance, restore
the original protected parameter tensors exactly and report the rollback and
pre-rollback derivative. The resulting protected derivative is zero. Parameters
with no protected gradient remain free to update; optimizer moments advance.
Nonfinite updates still fail closed. This conservative fallback is not active
in the clipping-aware ablation and does not retroactively complete the failed
trial or make its checkpoint eligible.

The clipping-aware preflight passes both context modes at 84.996 GiB peak
allocation, with finite gradients, synthetic-only targets and a frozen teacher.
Its controlled 200-update run starts from the eligible anchored checkpoint.

### Prepared top-k expected-gain objective

A second optional synthetic objective targets ordering at the cutoff rather
than fitting an entire probability distribution. For each user, sort current
candidate scores and compare every top-10 candidate with lower-ranked
candidates, counting each pair once. Weight a pair by the absolute difference
in its two synthetic relevance probabilities and the change in truncated
logarithmic rank discount. Normalize pair weights per user and macro-average
the logistic losses. This is a top-k expected-gain pairwise surrogate, **not**
an exact expected-NDCG objective or an unbiased estimator of real NDCG.

Unlike sampled binary relevance, the targets use the synthetic generating
probabilities, so an unexposed but likely relevant item is not automatically a
negative. A low temperature can leave correctly ordered, separated scores
nearly unchanged instead of forcing them to match a calibrated listwise
distribution. It does not add an inference head or change prediction scores.
Unit tests verify exact pair enumeration, gain/discount weights, user macro
averaging, missing candidates, gradients, no force for equal probabilities and
synthetic-only supervision. This is prepared, not yet enabled in a GPU trial;
first inspect the clipping-aware ablation.

At update 50, the clipping-aware ablation remains nearly identical to the
unbounded control: Netflix NDCG/Recall .042332/.060364 versus .042319/.060140,
with MAE/RMSE .739488/.841915 still above MF. This is not sufficient evidence
of a useful improvement. All 243 focused tests pass, including the prepared
top-k objective and conservative projected-step rollback. Future rollback
events are logged even between normal ten-update logging intervals.

### Registered top-k trial, conditional on the clipping control

If the completed clipping-aware control does not pass the inner gate, run
`topk_relevance_t01_weight025_anchor5_200` from the best eligible anchored update
200, after both synthetic preflight modes pass. Use the top-10 expected-gain
pairwise objective, temperature .1, ranking weight .25, boundary weight 1,
teacher weight 5, seed 61, LR 5e-6, 200 updates and validation every 50. Retain
the existing 50% catalog mixture, 64-user slates and original frozen teacher.
Do not use projection or a failed/infeasible checkpoint as initialization.
The low temperature reduces pressure on already-correct margins; the ranking
weight compensates for its larger local logistic gradients. These are
synthetic optimization choices, not targets or teachers derived from real
ratings. No fresh outer evaluation is authorized without all eight inner wins.

A CPU comparison against the archived pre-change objective also reproduces
default pairwise and expected-listwise loss values **and gradients bitwise**
on a synthetic test episode. The new optional objectives do not silently alter
historical/default behavior.

### Completed clipping control: not adopted

The clipping-aware run completes all 200 updates. Its final EMA has MovieLens
MAE/RMSE .750122/.884510 and NDCG/Recall .081028/.095990; Netflix
.740137/.842836 and .048713/.066895. Netflix again fails both MF rating
guardrails, so the unchanged selector correctly chooses update zero. The
complete ancestry, source, architecture and selection audit passes for that
selected checkpoint (SHA256
`94b6c008f9b04ee3e9532f3ecae9a6f2b4bb839e833e0ba494d8f0c3a75eaa17`),
but this is only a copy of the earlier anchored weights, not a new winner.
The control provides no material benefit over the unbounded objective.

Proceed with the registered top-k expected-gain trial. Keep the best completed
eligible anchored checkpoint and all frozen references unchanged. The goal
remains unmet; there have still been no new DWIP outer evaluations.

The top-k preflight passes both context modes with 84.996 GiB peak memory,
unchanged architecture and a frozen synthetic teacher. Losses and gradients
are finite; the 200-update trial is now running serially. Its ranking targets
are synthetic probabilities, never real interactions or fitted baseline scores.

### Top-k inner progress and cross-window calibration evidence

At update 100 the top-k run preserves all four MF rating wins and modestly
improves ranking: MovieLens MAE/RMSE .745208/.877167, NDCG/Recall
.064564/.080186; Netflix .731935/.832806 and .025439/.033657. It does not beat
BPR. Its Netflix first-1000-item share among top-ten recommendations falls
from .631 to .511, while the corresponding held-out-positive share is .863.
Mean recommended-item support stays near 474, versus 1038 for positives.

These read-only diagnostics motivate a synthetic training change: compare
scores **across separately encoded item windows from the same synthetic
parent**. A within-window pairwise or listwise objective cannot constrain a
user-wise offset between different windows. Joint-window training can do so
without changing inference or injecting popularity scores.

The optional `--training-item-windows 2` samples one 1000-user by 2000-item
synthetic crop with shared parent normalization, encodes its two 1000-item
windows independently with the unchanged network, restores query order and
computes one joint ranking objective over both outputs. The original synthetic
teacher processes the same windows. Both graphs contribute gradients to the
same weights. Full user catalogs are mandatory, and the user/item axes cannot
be swapped on alternating updates. The default single-window path is
unchanged. Unit tests verify ordering, shared gradients, empty/tail windows,
axis preservation and cross-window offset gradients. No GPU trial of this
option has run yet; it requires a two-mode memory preflight after the current
top-k run completes.

### Completed top-k run: new best eligible inner candidate

The top-k run completes and selects EMA update 200, SHA256
`bfc16db24510f96279d84c872a2ad2ef2774bb1959207ed80d54e193321ba957`.
Its complete architecture, source, ancestry and selection audit passes;
238 weight tensors change relative to the original MF-winning model.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7452113422 | .8771496720 | .0652522636 | .0825513083 |
| Netflix | .7318185999 | .8325599211 | .0256082566 | .0338232536 |

All four MF rating guardrails pass; all four BPR targets still fail. The
selection key improves from `[0,3.724728,2.480281]` to
`[0,3.638146,2.368568]`. This is a modest inner improvement, not goal completion.

### Registered joint-window trial

First independently re-evaluate the new best checkpoint on full inner data,
cross-checking all Cornac metrics and per-user rankings and collecting the
unchanged ranking profile. Then require a joint-window preflight to pass both
prior modes. If successful, train `joint_windows_topk_anchor5_200` from this
selected checkpoint using two separately encoded 1000-item windows per
synthetic episode, 1000 shared users, a complete 2000-item candidate slate for
64 users, and shared parent normalization. Keep the same top-10 pairwise
objective, temperature .1, ranking weight .25, boundary weight 1, original
synthetic teacher weight 5, catalog mixture .5, LR 5e-6 and EMA .95. Use seed
63, 200 updates and validation every 50. No projection is enabled. If preflight
fails, do not launch training. This trial changes only synthetic context
sampling and training-time loss coupling, not public imputation or the model.

Full inner re-evaluation reproduces the top-k checkpoint's seven metrics and
every per-user ranking against Cornac. Its BPR gate correctly remains false.
The joint-window preflight passes both contiguous and catalog-sampled contexts
with 114.043 GiB peak allocation, finite gradients, unchanged architecture and
the frozen original synthetic teacher. The new run is verified through update
10 with roughly 119,000 additional ranking queries in that episode. All 252
focused tests pass. No outer evaluation has been authorized or run.

At joint-window update 50, all four rating guardrails still pass. Netflix
NDCG/Recall improve to .029211/.039582, with MAE/RMSE .731378/.832970.
Its top-ten first-1000-item share rises from .493 to .603 and recommended-item
support from 473 to 553, consistent with improved cross-window calibration.
MovieLens NDCG/Recall are .063732/.078661 with MAE/RMSE .744919/.878433.
The interim selection key improves to `[0,3.108803,2.156388]`, but all four BPR
targets still fail. This remains an ongoing trial, not a completed winner.

### Completed joint-window result and registered pressure increase

The joint-window trial completes 200 updates and selects EMA update 150,
SHA256 `03ce2d149e330003ba75452974d1fc4ad5a2a4a76dde14bf8dd45f66248a8e86`.
The architecture, source, synthetic ancestry and selection audit passes.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7449550855 | .8784151549 | .0612985725 | .0760189038 |
| Netflix | .7310620214 | .8328093943 | .0301432071 | .0407525484 |

All four MF guardrails pass, but all four BPR targets still fail. The fixed
selection key improves to `[0,3.019539,2.135568]`; this becomes the best
completed eligible inner candidate. Netflix Recall improves about 20.5% over
the preceding top-k checkpoint, at the cost of lower MovieLens ranking. It
does not constitute an outer win or completion of the BPR goal.

After a full independent inner reproduction, register
`joint_windows_topk_weight1_anchor5_200` from this selected checkpoint. Keep
the joint-window prior, original frozen synthetic teacher, temperature .1,
boundary weight 1, teacher weight 5 and all other settings. Increase only the
ranking weight from .25 to 1 for stronger ordering pressure; use seed 64, LR
5e-6, 200 updates and validation every 50. Require both preflight modes to
pass before training. Save only the best selected checkpoint to preserve disk
headroom. All rating guardrails and the eight-target inner gate remain fixed.

The selected joint-window model's full inner metrics and every per-user
ranking reproduce against Cornac; its BPR gate remains false. The stronger
ranking preflight passes both prior modes at 113.845 GiB peak, and the new
training process starts serially.

### Prepared one-sided gradient fallback

The existing negative-only option is extended to the top-k expected-gain
pairwise loss. It preserves each pair's forward loss but detaches its
higher-relevance synthetic score, so only the lower-relevance side receives
the ranking gradient. This may reduce unwanted upward pressure on likely
rating queries, but does **not** guarantee rating preservation. It is a
semi-gradient update and needs the absolute rating objective/teacher anchor
to counter global downward drift. Synthetic-only targets and all selection
guardrails still apply. Tests verify identical forward losses, nonnegative
direct score gradients, no direct gradient on the highest-relevance score,
finite complete-objective gradients and rejection of real supervision. All
255 focused tests pass. This option is **not enabled** in the active stronger
ranking trial; first inspect that trial's evidence before choosing a fallback.

### Prepared synthetic item-marginal relevance prior

The remaining Netflix gap is large, while the already-recorded positive-item
popularity diagnostic exceeds the frozen BPR reference on inner data. Those
real diagnostic scores/counts remain evaluation-only; they are not passed to
training. A new optional synthetic target tests whether a simpler shared
exposure/quality signal transfers better than personalized latent relevance:

`q[item] = mean_synthetic_users(exposure_probability[user,item] *
P(synthetic_rating[user,item] >= 4 | generating_latents))`.

The complete synthetic parent supplies the generating probabilities, including
user-dependent Gaussian noise and uniform-rating contamination. This is an
**item-marginal positive-interaction probability**, not a user-conditional
held-out probability. It is shared across synthetic users; the rating-risk
objective and original synthetic rating teacher remain personalized. No real
popularity statistic, real teacher, BPR factor, or new inference component is
introduced. The optional `--ranking-relevance-target item_marginal_positive`
is not enabled in the active weight-1 trial.

Tests verify the analytic probability against the full ordinal distribution
and synthetic Monte Carlo, uniform/no-exposure limits, identical non-ranking
episode tensors, and rejection of real supervision. A separate three-context
comparison reproduces all 14 default episode tensors bitwise against the
active run's archived pre-change prior. Frozen architecture/inference
fingerprints also remain unchanged. All 261 focused tests pass. Choose any
next trial only after inspecting the current stronger-ranking evidence.

At stronger-ranking update 50, all four inner MF guardrails pass. MovieLens
MAE/RMSE are .746169/.880897 and NDCG/Recall .069063/.084694. Netflix
MAE/RMSE are .733119/.835395 and NDCG/Recall .046038/.062662. Netflix Recall
improves 53.8% over the preceding selected joint-window checkpoint, and the
interim selection key improves to `[0,1.963776,1.554905]`. BPR still wins all
four ranking comparisons, and Netflix's RMSE margin over MF is only .00161.
Do not promote this ongoing-run checkpoint before completion and audit.

### Reference concentration audit

A CPU-only replay verifies the saved BPR factor hashes and reproduces all
reference ranking metrics, then profiles BPR and both recorded popularity
controls under the unchanged candidate protocol. All three references draw
their top tens entirely from the first 1000 items on both inner datasets.
Netflix's mean recommended-item context support is 2480 for BPR and 2375 for
positive popularity, versus 837 for DWIP at stronger-ranking update 50.
Thus matching the average support of *all held-out positives* (1038) would not
by itself match the references' much sharper top-ten concentration.

The references also spend about 11–16% of recommendation slots on observed
low ratings under the frozen positive-only exclusion rule. This is a protocol
diagnostic, **not** authorization to change filtering. DWIP continues to
preserve observed ratings and use its original inference recipe. The read-only
profiles are in `inner_reference_ranking_profiles.json`; arbitrary reference
score scales are not described as rating-ceiling saturation. No real reference
scores/counts enter any neural target or gradient.

At stronger-ranking update 100, all four MF guardrails still pass and the
interim key improves to `[0,1.894600,1.486295]`. Netflix NDCG/Recall are
.048119/.064950, but RMSE .835888 leaves only .00112 below MF. The new
synthetic item-marginal prior is motivated by the remaining concentration
gap; it has not yet been used in GPU training.

### Centered one-sided optimizer fallback

Raw one-sided gradients have a nonzero sum within each user's slate even
though pairwise forward losses are unchanged by a common score shift. This
can create unnecessary downward rating drift, especially at temperature .1.
An optional `--center-negative-only-ranking` removes that common component:
subtract a zero-valued user-mean term in the backward pass before detaching
the higher-relevance side. The pairwise forward loss remains bitwise unchanged;
the score gradient equals the raw one-sided gradient minus its per-user mean.
Compensation is spread across the slate rather than concentrated on the
higher-relevance member of each pair. Clipping, the boundary penalty and the
absolute rating/teacher objectives still apply, so this is not a guarantee of
real rating accuracy. Existing inner/outer gates are unchanged.

Tests verify exact forward equivalence, per-user zero-sum pairwise score
gradients, preservation of false-positive lowering, and finite synthetic
complete-objective gradients. All 264 focused tests pass. Both one-sided
options remain disabled in the active trial. They are prepared alternatives,
not evidence that the goal has been achieved.

### Completed stronger-ranking trial

The weight-1 joint-window run completes and selects EMA update 100,
SHA256 `084101fb6714baefa55e42c833ce531c6b7d34c391991aa612eccc920afb81b6`.
Its architecture, source, synthetic ancestry and selection audit passes.
It becomes the best completed eligible inner candidate, with fixed key
`[0,1.8946004263,1.4862953483]`.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7460800270 | .8816093614 | .0734125906 | .0887106881 |
| Netflix | .7332755655 | .8358880462 | .0481185116 | .0649498019 |

All four MF rating wins remain, but all four BPR targets still fail. No outer
evaluation is authorized. Higher pressure materially improves both datasets'
ranking over the preceding joint-window candidate, but further same-recipe
updates plateau and Netflix's RMSE margin is narrow.

### Registered item-marginal prior trial

First reproduce the selected weight-1 checkpoint's full inner metrics and
per-user rankings against Cornac. Then require both synthetic preflight modes
for `joint_windows_marginal_positive_weight1_200`. Start from that selected
checkpoint, retain the two separately encoded item windows and all optimization
settings, and change only the relevance target from personalized conditional
probability to the synthetic item-marginal positive-interaction probability.
Use seed 65, LR 5e-6, 200 updates, validation every 50, ranking weight 1,
temperature .1, boundary weight 1, teacher weight 5, catalog fraction .5,
2000 candidates for 64 users, and EMA .95. Keep both one-sided-gradient options
and projection disabled for this prior comparison. Save only the selected
best checkpoint. Rating targets, the original frozen synthetic teacher,
architecture, inference and all gates are unchanged.

The weight-1 selected checkpoint's full inner metrics and every per-user
ranking reproduce against Cornac, with the BPR gate correctly remaining false.
The item-marginal preflight passes both context modes at 113.884 GiB peak,
with finite gradients, unchanged architecture and the original teacher frozen.
Training is verified through update 20: logged targets are
`item_marginal_positive`, gradient data are synthetic, and both one-sided
options are false. The active run's archived source hashes also verify.
The full BPR+MF goal remains active and unmet; no new outer evaluation runs.

### Parent-positive ranking-user sampling control

The item-marginal trial remains active through update 170. Its validations at
50, 100 and 150 all preserve the four MF rating guardrails, but none improves
the initialized conditional-target checkpoint's registered selection key.
Do not promote an ongoing run or change its registered 200-update schedule.

Inspection finds another synthetic sampling mismatch: ranking users were
required to have a sampled positive **inside the crop, after the rating-query
cap**. Real ranking evaluation includes eligible users with positives anywhere
in the full catalog, including windows containing no local positive or context.
The optional `--ranking-user-scope parent_positive` now selects users with at
least one known-user/known-item positive holdout in the generated parent and
usable unseen candidates in the local window. Selection is independent of
local positives and the rating-query cap, with uniform subsampling to the
existing user budget. Only expected-relevance `user_catalog` training permits
this scope; sampled-pairwise/global modes reject it. All eligibility labels
come from generated synthetic ratings. Context cropping itself is unchanged.

The default `local_positive` behavior reproduces all 14 episode tensors
bitwise against the archived strongest trial, for IID, contiguous and global
catalog contexts. Frozen architecture/inference fingerprints also match.
All 283 focused tests pass, including explicit-mask validation, full-catalog
coverage, inclusion of locally empty users, query-cap independence, unchanged
non-ranking tensors, CLI forwarding and rejection of real gradient data.
New diagnostics count selected users lacking local positives, sampled positives
or local observed context.

After the current trial finishes and is audited, register
`joint_windows_parent_positive_weight1_200` from the best completed conditional
checkpoint (weight-1 EMA update 100). Change only ranking-user eligibility,
retain conditional relevance targets, and use seed 66, LR 5e-6, 200 updates,
validation every 50, ranking weight 1, temperature .1, boundary weight 1,
teacher weight 5, catalog fraction .5, two 1000-item windows, 2000 candidates
for 64 users and EMA .95. Keep projection and both one-sided options disabled.
Require both synthetic GPU preflight modes before training; save best only.
The original synthetic teacher, rating objectives, architecture, inference,
baseline hashes and eight-comparison selection gate remain unchanged. This
registration is not evidence of a win; no new outer evaluation is authorized.

### Completed item-marginal trial

The marginal-target trial completes all 200 updates and selects update 200,
SHA256 `c8623c17c20dc30cc0f0f000bdfceecc7c54eb7d8b53f65dfd02f5c1a6106330`.
Its synthetic ancestry, teacher, source snapshots, unchanged model/inference
and fixed inner selection all pass audit. The registered worst-ratio-first
key improves narrowly to `[0,1.8799088710,1.5625114939]`; therefore this is the
new best completed candidate **under that fixed rule**, although three of the
four ranking targets regress and its mean ratio is worse. Do not change the
selection rule retrospectively or describe this as a general ranking gain.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7465560991 | .8812015458 | .0669468527 | .0778092893 |
| Netflix | .7333714891 | .8353574815 | .0468338768 | .0654573869 |

All four MF rating guardrails pass, but none of the four BPR targets passes.
No independent full inner/Cornac rerun is claimed for this newly selected
checkpoint, and no outer evaluation runs. The next parent-positive control
still starts from the registered **conditional-target** update 100, so that
its only prior change is user eligibility rather than a simultaneous target
change. Its two-mode GPU preflight is now running on the otherwise idle GPU.

The parent-positive preflight passes both context modes at 114.021 GiB peak.
It includes 6/5 selected users without a sampled positive in the contiguous/
catalog modes and 4 catalog users without any local holdout positive; loss
and gradients are finite, the original teacher is frozen, and no real data
is loaded. The registered 200-update parent-positive run launches from the
conditional-target update 100 after this preflight, with one GPU job only.

The new run's archived source hashes and frozen inference fingerprints verify.
Its initial inner evaluation reproduces the conditional parent, and logged
updates through 20 use only synthetic targets. Updates 1/10/20 include 18/12/5
ranking users with no sampled local positive; update 20 includes 4 with no
local holdout positive at all. Projection and one-sided gradients stay off.
The goal remains active, with no outer evaluation and no BPR win claimed.

At parent-positive update 50, all four MF rating guardrails pass, with
MovieLens MAE/RMSE .745565/.880109 and Netflix .732265/.834570. Ranking has
not improved: MovieLens NDCG/Recall .071636/.085862 and Netflix
.044763/.060338 yield key `[0,2.0393936211,1.5723240222]`, worse than the
initialization. Continue the registered 200 updates; do not promote this
interim checkpoint or run an outer evaluation.

At parent-positive update 100, all MF rating guardrails remain intact and
the selection key improves to `[0,1.8564686397,1.4794227593]`, better than
the best completed candidate. MovieLens MAE/RMSE are .745457/.879745 and
NDCG/Recall .073381/.086612; Netflix MAE/RMSE are .733241/.835251 and
NDCG/Recall .048619/.066284. Netflix gains over the conditional initialization,
but both MovieLens ranking targets remain slightly lower. The goal still has
zero of four BPR wins. This is an interim result, not an eligible promotion.
Logged update 90 exercises 28 ranking users without a local positive,
including one with no local observed context, with finite training loss.

### Registered centered one-sided optimization control

After the parent-positive run completes and its selected checkpoint passes
the ancestry/architecture/source/selection audit, use that selected checkpoint
for `joint_windows_parent_centered_negative_weight1_200`. The prior, conditional
targets, parent-positive user scope and frozen original synthetic teacher stay
unchanged. Enable the already tested `--negative-only-ranking` together with
`--center-negative-only-ranking`: lower the lower-relevance side of each
synthetic top-k pair, remove its per-user common score-gradient component,
and retain the existing absolute rating, boundary and teacher objectives.
The forward pairwise objective is unchanged; this is an optimization control,
not a new inference scorer or a guarantee of rating preservation.

Use seed 67, LR 5e-6, 200 updates, validation every 50, ranking weight 1,
temperature .1, boundary weight 1, teacher weight 5, catalog fraction .5,
two 1000-item windows, 2000 candidates for 64 users and EMA .95. Leave
projection disabled and save only the best checkpoint. Require both GPU
preflight modes and continue to select with the same eight-target guardrails.
Do not run this control if its parent is incomplete or fails its audit.

The expanded integration tests verify that full, raw one-sided and centered
one-sided objectives all propagate finite, nonzero ranking gradients to
parent-eligible users with no local rating query and reject real gradient data.
All 285 focused tests pass.

At parent-positive update 150, the interim key improves again to
`[0,1.7718463845,1.4581315188]`. MovieLens MAE/RMSE are .745643/.880229 and
NDCG/Recall .070878/.084879; Netflix MAE/RMSE are .734129/.836256 and
NDCG/Recall .050458/.069450. All MF guardrails pass, but Netflix RMSE is only
.000753 below MF and MovieLens ranking regresses relative to update 100.
None of the BPR targets passes. Keep the original selection rule and finish
the registered run before any promotion or next GPU experiment.

### Completed parent-positive sampling trial

The user-sampling control completes 200 updates and selects update 200,
SHA256 `de5c403fc84c26ed4b39cb23ee0dee2330b0ed1a71c41216d1ff5f96d63106a2`.
Its source, synthetic ancestry, frozen teacher, architecture/inference and
fixed inner-selection audit all pass. It becomes the best completed eligible
candidate with key `[0,1.7144481087,1.4222269052]`.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7451454266 | .8787848935 | .0722290770 | .0858506220 |
| Netflix | .7349591338 | .8365075591 | .0520252003 | .0717746555 |

Relative to its conditional-target initialization, Netflix NDCG/Recall improve
8.12%/10.51%, while MovieLens ranking is modestly lower. Compared with the
previous globally selected marginal-target candidate, all four ranking metrics
improve. All MF rating guardrails still pass, but the Netflix RMSE margin is
only .000501. None of the BPR targets passes; no full independent inner/Cornac
rerun or new outer evaluation is claimed for this checkpoint.

The registered centered one-sided optimization control now starts its two-mode
GPU preflight from this audited selected checkpoint, preserving the entire
prior, synthetic teacher, architecture, inference and fixed selection protocol.

The centered one-sided preflight passes both context modes at 114.023 GiB
peak, with finite loss/gradients, unchanged architecture, the original teacher
frozen and no real data loaded. Both one-sided and centering flags are verified
active. The registered 200-update control is launched on the otherwise idle
GPU; its archived source hashes, initialization and unchanged inference
fingerprints verify. All 285 focused tests pass; the BPR+MF goal stays unmet.

The new run's initial inner key reproduces its selected parent exactly.
Updates 1 and 10 are verified finite and synthetic-only, with conditional
targets, parent-positive user sampling, and both one-sided/centering flags
active. No new outer evaluation has run.

### Synthetic window-coverage diagnosis and prior control

Exact query-anchor marginals for three generated CPU parents show that the
current contiguous 2000-item crop chooses its first window with probabilities
.97956, .96104 and .97758. These are synthetic prior diagnostics, not real-data
targets or benchmark scores (`synthetic_window_anchor_diagnostic.json`). The
contiguous component consequently sees the rare last catalog band only about
1.9–3.7% of the time. The existing 50% catalog-mixture component is separate.

An optional `--paired-context-windows` changes **only the contiguous component**
of two-window training. Generate the same synthetic parent and first-observation
ordering, retain the same query-anchored row crop, select a 1000-item primary
window containing the held-out item anchor, and pair it with a uniformly sampled
nonoverlapping 1000-item window. The anchor retains a rating query; the partner
need not contain any local positive. Both windows are independently encoded by
the unchanged model and jointly ranked for the existing parent-positive users.
This permits popular and rare windows to share a ranking slate, rather than
training two rare windows in isolation with no popular-item comparison.

On the same three synthetic parents, analytic mean item-inclusion probability
in positions 3000–4095 rises from .0194–.0370 to .4573–.4589. First-1000-item
coverage stays .8773–.9186. These numbers describe only the contiguous mixture
component, not the full training mixture or future model performance. The
calculation is saved in `synthetic_paired_window_coverage.json`. It uses no real
data, teacher or statistic. Overlapping tail windows cannot be selected together.

The new default-off prior passes 303 focused tests: deterministic disjoint
windows, retained query anchors, distant-partner coverage, tail overlap
exclusion, unchanged parent-user normalization, the unmodified catalog-mixture
branch, and finite synthetic gradients. All 14 default episode tensors match
the active run's archived prior bitwise in all three original context modes;
architecture/inference fingerprints still match. No paired-window GPU preflight
or training has run yet; the active centered-gradient trial is not changed.

At centered update 50, all four MF rating guardrails pass, but ranking regresses
sharply: MovieLens NDCG/Recall .061051/.074336 and Netflix .028543/.036786.
Its selection key is `[0,3.3451679609,2.2657896615]`, so no promotion occurs.
Continue the registered 200-update run rather than treating this as completion
or restarting it. The current source edits cannot alter its already imported,
archived training implementation.

After the active trial terminates and its outcome is audited, require both GPU
preflight modes for `joint_windows_paired_coverage_weight1_200`. Start from the
audited parent-positive update 200, SHA256
`de5c403fc84c26ed4b39cb23ee0dee2330b0ed1a71c41216d1ff5f96d63106a2`, and enable
only the paired-context option relative to that parent's original two-sided
training procedure. Keep both one-sided options and projection disabled. Use
seed 68, LR 5e-6, 200 updates, validation every 50, ranking weight 1,
temperature .1, boundary weight 1, teacher weight 5, catalog fraction .5,
2000 candidates for 64 parent-positive users, two 1000-item windows and EMA .95.
Retain conditional relevance targets, the original synthetic rating teacher,
rating objectives and all eight comparison gates. Save best only. This isolates
the coverage correction instead of combining it with the currently regressing
optimizer control. No new outer evaluation is authorized.

At centered update 100, the key worsens to `[0,3.9733655484,2.5667322140]`.
MovieLens MAE/RMSE are .748096/.882217 and NDCG/Recall .060975/.072029;
Netflix MAE/RMSE are .733291/.835842 and NDCG/Recall .024254/.030970.
All MF rating guards pass but no ranking promotion occurs. The read-only
Netflix profile shows mean top-10 item support 385 and first-1000 share .3984,
with .1265 of top-10 scores at the clipping ceiling. These are diagnostics,
not training targets or an inference adjustment. The centered trial remains
live at update 100; keep the incumbent and finish the registered schedule.

### Completed centered-gradient control

The centered one-sided run completes all 200 updates and passes the synthetic
ancestry, source, architecture/inference and selection audit. It selects
**update zero**, SHA256
`c7f40881a499e5e877da206c0bc722e1d19949f7caa4a54b2d3ea716a152e5a0`.
All 254 selected model state tensors exactly equal its parent checkpoint;
no centered-gradient update is promoted. The incumbent remains the original
parent-positive update 200 with key `[0,1.7144481087,1.4222269052]`.

The rejected final update has key `[0,3.9701463779,2.5634488897]`. MovieLens
MAE/RMSE are .748465/.883282 and NDCG/Recall .060173/.069815; Netflix
MAE/RMSE are .733631/.836785 and NDCG/Recall .024771/.030995. All MF rating
guards pass, but all four ranking targets regress. The control is completed
with a negative result, not a failed process or a goal win.

With the GPU verified idle, the registered paired-coverage control starts its
two-mode synthetic GPU preflight from the unchanged incumbent. It restores
the incumbent's original two-sided optimizer settings and changes only the
contiguous-window prior; both one-sided options and projection remain off.

The paired-coverage preflight passes both `first_observation_paired` and the
unchanged `first_observation_catalog` mode at 113.890 GiB peak. Gradients are
finite, the original teacher is frozen, architecture is unchanged, and no real
data is loaded. An additional synthetic GPU episode inspection confirms the
registered geometry and records per-window support/query counts in
`paired_coverage_gpu_prior.json`; this is not a model evaluation or score gain.

The registered paired-coverage run launches with one GPU job only. Its archived
source hashes, initial checkpoint hash, prior settings, disabled one-sided and
projection flags, and unchanged architecture/inference fingerprints verify.
The initial inner key exactly reproduces the incumbent. All 303 focused tests
pass; no outer evaluation runs and the full BPR+MF goal remains unmet.

Training logs verify both context variants: updates 1/10/20 use the original
catalog mode, while 30/40 use paired 1000-item windows. All logged gradients
remain synthetic, conditional targets and parent-positive user eligibility
are unchanged, and both one-sided options stay disabled.

At paired-coverage update 50, all four MF rating guardrails hold, but the key
`[0,1.9027396063,1.5147599660]` does not improve the incumbent. MovieLens
MAE/RMSE are .745021/.879365 and NDCG/Recall .070818/.085456; Netflix
MAE/RMSE are .733373/.835364 and NDCG/Recall .047565/.064672. Do not promote
this interim checkpoint or infer an evaluation gain from improved synthetic
coverage. Continue the registered 200-update schedule; the goal stays active.

At paired update 100 the key is `[0,1.8713421179,1.4845871892]`, and at
150 it is `[0,1.8682336669,1.4850895552]`. Both retain every MF rating win,
but neither improves the incumbent. Update 150 MovieLens MAE/RMSE are
.744730/.878321 with NDCG/Recall .073387/.085908; Netflix MAE/RMSE are
.733252/.835012 with NDCG/Recall .048587/.065866. No BPR target passes.

### Preparing inference-matched training geometry

The current joint-window prior encodes 1000 users per window, while frozen
public inference uses up to 2000. A model with row/column attention and fixed
CLS tokens is not guaranteed to be invariant to that change. Matching 2000
users in both jointly ranked windows would double retained activation graphs.
An optional `--recompute-window-gradients` now builds detached per-window
logits one graph at a time, computes the original coupled loss once, and
replays each window with its stored CPU/CUDA RNG state to accumulate parameter
gradients. Each replay must reproduce its original logits exactly before
backpropagation. The loss, teacher and architecture are not replaced.

`--training-rows` enables an explicit synthetic row-count override. Both new
options default off/unchanged. Replay rejects reused plans, missing or invalid
logit gradients, changed training mode, mismatched replay logits, mutable
running-stat modules and combination with protected-update projection. Global
RNG progression is preserved. The original window-forward function and frozen
architecture/inference fingerprints are unchanged; the active run still uses
its archived pre-change implementation.

All 319 CPU tests pass. They cover coupled cross-window losses, stochastic
dropout, activation checkpointing, original query order, empty and partial
windows, parameter gradients, an FP32 AdamW update and RNG preservation.
BF16 logits remain exact; joint versus sequential parameter-gradient sums can
differ by ordinary mixed-precision accumulation rounding. The CPU BF16 test
requires gradient relative L2 error below .02 and cosine above .999; this is
not a claim of bitwise-identical BF16 parameter updates.

`verify_dwip_bpr_window_replay.py` checks actual DWIP CUDA logits, joint-loss
gradients, RNG, synthetic teacher and architecture on a small synthetic paired
episode. These checks and the full-size preflight have now completed below;
no metric improvement is claimed from implementation tests.

### Completed paired coverage and full-size replay validation

The paired-coverage trial finishes all 200 updates and passes its campaign
audit, but selects update zero. Final update 200 has selection key
`[0,1.8174110226498188,1.4575556554963374]`: MovieLens NDCG/Recall
.073706/.088061 improve while Netflix .049286/.067708 regress. All four
MF guardrails still pass. Do not change the fixed selection to promote this
rejected update. The incumbent remains parent-positive update 200, SHA256
`de5c403fc84c26ed4b39cb23ee0dee2330b0ed1a71c41216d1ff5f96d63106a2`.

Actual CUDA replay, including optional CPU saved-activation offload, passes:
logits are identical, joint loss is 4.2146377563, gradient relative L2 error is
.0016890421 and gradient cosine is .9999985743. RNG progression is preserved,
the synthetic teacher is frozen, and architecture/inference fingerprints are
unchanged. This uses only generated data and no optimizer update. BF16
gradient summation is numerically close, not bitwise-identical. Sources are
archived with `window_replay_cuda_offload_parity.json`.

The first 2000-user/two-window preflight without CPU offload runs out of GPU
memory during replay backward; it does not train or update a checkpoint.
With `--offload-window-activations`, both original contiguous and catalog
modes pass: peak GPU allocation 125.391 GiB (<128), host peak RSS 77.354 GiB
(<160), finite gradients, and two verified gradient replays in each mode.
Teacher/architecture/synthetic-only checks pass. All 327 focused tests pass.

### Registered inference-matched geometry control

Before launching `full_context_replay_weight1_200`, fix its configuration:
start from the incumbent above; original synthetic MF-winning frozen teacher;
seed 69; 200 updates; validation every 50; LR 5e-6 and EMA .95. Change training
to **2000 users per 1000-item window**, two windows with shared normalization
and joint loss, using gradient replay and CPU activation offload. Preserve the
incumbent's original 50% contiguous / 50% catalog prior mixture; paired-window
sampling is **off**. Retain conditional relevance, parent-positive users,
64 ranking users, up to 2000 negatives/user, top-k expected pairwise objective,
temperature .1, ranking/boundary weights 1, and synthetic teacher weight 5.
Both one-sided gradient options and protected-update projection remain off.
Save only the best checkpoint. The two-mode preflight is
`full_context_replay_offload_preflight.json`.

This is a controlled training-geometry experiment, not an inference change.
Real data remain evaluation-only, all gradients/targets/teachers synthetic.
Keep the frozen selection and all eight inner comparisons, followed by an
independent inner/Cornac audit, before any new outer evaluation. No outer DWIP
evaluation has run during this campaign; the goal is still unmet.

The inference-matched trial is now running (PID 1474387). Its update-zero
metrics reproduce the incumbent exactly; all 15 archived training sources and
frozen inference fingerprints verify. Updates 1 and 10 use only synthetic
targets, have finite gradients and perform two verified replay backwards.
It uses the archived implementation from launch, not subsequent optional
optimizer controls.

### Prepared exposure-embedding optimization control

An optional `--trainable-scope exposure_embeddings` freezes all weights except
the existing mask token, missingness embedding, and row/column CLS embeddings.
This tests whether adapting the representation of exposure counts without
changing the value encoder, attention weights or decoder can reduce rating
drift during ranking optimization. It does not add, remove or replace any
network state. Freezing parameters does **not** guarantee rating retention;
the unchanged MF guardrails and BPR criteria still decide eligibility.

The default remains full-network optimization, including the live geometry
trial. Restricted optimizers receive only the four selected parameter tensors.
Their manifests record exact names/counts, and campaign audits require every
other saved state tensor to equal its synthetic initialization exactly. Small
DWIP tests cover ordinary and replay gradients, activation checkpointing,
AdamW, EMA and candidate selection, plus full-scope update equivalence. No
restricted-scope GPU trial is registered or launched yet; require the usual
synthetic-only preflight and fixed registration before using it.

All 335 focused tests pass, including eight new parameter-scope tests. The
actual DWIP configuration would expose 20,736 trainable elements in four
existing tensors, leaving the rest of the 77,998,688-element state unchanged.

### Inference-matched update 50

The live full-context trial retains all four MF rating wins, but does not
improve the incumbent at its first validation. Its selection key is
`[0,1.7926410701701756,1.4668977186014693]`.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7447862340 | .8782464149 | .0713971283 | .0850313992 |
| Netflix | .7335512002 | .8351786215 | .0497303115 | .0686439267 |

No checkpoint is promoted and no BPR target passes. All logged updates through
50 remain synthetic with finite gradients and two exact-logit replays. Host
RSS is about 78.1 GiB with no observed accumulation problem. Continue the
registered 200-update schedule before deciding whether geometry helps. The
exposure-embedding optimizer remains a tested, unlaunched alternative; the
live full-network run is unchanged. No outer evaluation runs.

### Exposure-embedding step-scale diagnostic and registered control

A read-only CPU diagnostic verifies the synthetic provenance of the incumbent
and original MF-winning teacher, then compares their four existing exposure
parameter tensors. Relative to the teacher, RMS changes are 9.424e-5 for the
column CLS embeddings, 7.902e-5 for row CLS embeddings, 7.656e-5 for missingness
embeddings and 5.230e-5 for the mask token. The embeddings themselves have RMS
about .973–1.016; the mask token has RMS .009281. Thus full-network fine-tuning
has barely moved these parameters. This is a step-scale diagnostic, not proof
that their adaptation will improve ranking. No model forward, optimizer
update or real-data evaluation is involved. The report is
`exposure_embedding_step_scale.json`.

Register `exposure_embeddings_weight1_200` as the next controlled experiment,
conditional on the full-context run ending and the eight-target goal remaining
unmet. First run both synthetic preflight modes on an otherwise idle GPU.
Initialize from parent-positive update 200, SHA256
`de5c403fc84c26ed4b39cb23ee0dee2330b0ed1a71c41216d1ff5f96d63106a2`,
regardless of the full-context trial's result, to compare against its original
recipe. Use 1000 users and two 1000-item windows, original 50% contiguous /
50% catalog sampling, conditional relevance and parent-positive ranking users.
Train only the existing four exposure tensors (20,736 elements), leaving all
other state exactly unchanged. Use seed 70, 200 updates, validation every 50,
LR .001 with the existing warmup/cosine schedule, EMA .95, ranking weight 1,
temperature .1, boundary weight 1, and original synthetic teacher weight 5.
Use 64 ranking users and 2000 candidates/user. Retain the point-rating and NLL
objectives. Replay, offload, paired coverage, projection and one-sided-gradient
options are all off. Save only the selected best checkpoint.

The larger LR is a restricted-parameter optimization control, not a claim of
isolating the scope from step size. Preflight must verify finite gradients,
unchanged architecture, the exact trainable scope, absent frozen-parameter
gradients, a frozen synthetic teacher and the existing memory limits. Completed
checkpoint audit must compare all non-exposure state against initialization
bitwise. Preserve the fixed selection rule: if the current full-context trial
produces a better eligible incumbent, this control must beat that new incumbent
to be promoted. No outer evaluation is authorized by this registration.

### Inference-matched update 100

The full-context trial's next validation yields key
`[0,1.7167236100725063,1.4165990754832403]`. All four MF rating guards pass,
and both NDCGs improve over initialization, but both Recalls are slightly
lower. The worst-ratio-first selector therefore correctly retains update zero,
despite the better mean ranking ratio; do not change the selection rule.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7447223646 | .8765351206 | .0735603213 | .0854030940 |
| Netflix | .7345536034 | .8353546479 | .0523720560 | .0716795188 |

The registered trial remains live, with finite synthetic gradients and two
verified replays per logged update through 100. No BPR comparison passes and
no outer evaluation is run. Complete the fixed 200-update schedule before
auditing the selected checkpoint or starting the restricted-parameter control.

### Inference-matched update 150

The next validation improves the fixed selection key to
`[0,1.6789072698933616,1.3918027467042546]`. All four ranking scores now exceed
initialization, while every MF rating guardrail remains intact. The trainer
saves this interim best checkpoint, but no completed-candidate promotion is
made before the registered run and campaign audit finish.

| Inner dataset | MAE | RMSE | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| MovieLens | .7454336093 | .8778931685 | .0742269866 | .0861022590 |
| Netflix | .7341354103 | .8352923942 | .0537427597 | .0732940553 |

None of the BPR targets passes. Continue to update 200 under the original
schedule; no outer evaluation or change to the registered follow-up control.

### Completed inference-matched geometry control

`full_context_replay_weight1_200` finishes all 200 updates (8931.6 seconds),
selecting update 200 with SHA256
`5dd2dd7d7da3fd80b55bb76275e162f2c1fcb4d05886d747a0c3a3fa59815a93`.
The campaign audit passes: all archived sources and synthetic ancestors
verify, 254 state tensors / 77,998,688 elements remain unchanged in structure,
238 weight tensors differ from the original MF teacher, and frozen inference
and checkpoint selection reproduce. Every logged update is synthetic and
uses two verified gradient replays. Structural audit is not goal completion.

The fixed key `[0,1.5920447872448065,1.3476380406553525]` makes this the new
best completed eligible candidate. All four ranking targets improve over
initialization; Netflix NDCG/Recall rise approximately 9.6%/7.7%. MAE rises
slightly on both datasets, while RMSE improves; all four MF guards still pass.

| Inner dataset | MAE | RMSE | AUC | MAP | NDCG@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | .7452273608 | .8778968120 | .8219838754 | .0599859332 | .0738821795 | .0458435208 | .0863629651 |
| Netflix | .7350180488 | .8361825233 | .8298024317 | .0396397452 | .0570160597 | .0284568543 | .0772930029 |

All four BPR targets still fail. Register a read-only, independent inner/Cornac
reproduction in `inner_profile_full_context_selected` before the next GPU
preflight. The exposure-embedding control keeps its previously fixed
parent-positive initialization and must outperform this newer incumbent for
promotion. No outer evaluation is run or authorized.

The completed full-context candidate's independent inner evaluation now
reproduces every aggregate metric and every per-user ranking against Cornac.
Both datasets preserve observed ratings, have finite predictions and run with
gradients disabled. Its saved `inner_gate.json` is correctly **false**, because
all four BPR targets still fail. This is not an outer evaluation.

Both synthetic context modes then pass the exposure-embedding preflight:
91.647 GiB peak GPU allocation and 2.168 GiB host peak RSS. The exact registered
scope contains four trainable tensors / 20,736 elements; the other 234 model
parameters have no gradients. Teacher/architecture/synthetic-only checks pass,
with finite gradient norms .019466 and .020468. Replay, offload, paired coverage,
projection and one-sided options are verified off.

Launch the unchanged registered `exposure_embeddings_weight1_200` control on
the otherwise idle GPU. Its initialization remains the older parent-positive
checkpoint as pre-registered; full-context update 200 remains the incumbent
to beat. All neural targets and teachers are synthetic, and the frozen
architecture, inference and eight-target gate remain unchanged.

The restricted control is verified live through update 20 (PID 1485968).
Its step-zero inner metrics reproduce the pre-registered parent-positive
initialization exactly, all 15 archived sources verify, and the frozen
inference signature is unchanged. The manifest records exactly the four
exposure tensors as trainable with all disabled options matching registration.
Logged updates 1/10/20 have finite synthetic losses and gradients. This is
training progress, not a checkpoint promotion or a completed frozen-state audit.
The independently reproduced full-context update 200 remains the incumbent.

### Exposure-only updates 50 and 100

Update 50 has key `[0,1.7140108100858928,1.421579587929355]`, barely better
than its older parent-positive initialization, but well behind the full-context
incumbent. Update 100 regresses to
`[0,1.7383705418815876,1.4409999811943357]`. Both retain all four MF guards.
At 100, MovieLens MAE/RMSE are .745554/.879479 and NDCG/Recall .071539/.085377;
Netflix is .734580/.836322 and .050981/.070787. No BPR target passes and no
global promotion is made. Finish the registered 200 updates before auditing.

### Registered shared-case synthetic diagnostic

After the exposure-only run is terminal and audited, compare its selected
checkpoint, the parent-positive initialization and the full-context winner on
six identical synthetic cases: base seeds 901/902/903, each in original
contiguous and catalog modes. Use the full-context training configuration
(2000 users, two 1000-item windows), shared synthetic teacher and original
objective. Check the reserved seeds against recorded ancestral training seeds.
This comparison asks whether the remaining problem appears within the
synthetic objective too, rather than selecting another prior change solely
from repeated real inner scores.

`diagnose_dwip_bpr_synthetic_generalization.py` loads no real datasets, uses no
optimizer, disables every model's gradients, and shares each exact episode
across candidates. It fingerprints episode tensors before/after scoring and
model state before/after the complete diagnostic. Source snapshots, checkpoint
provenance and the frozen architecture/inference signature are recorded.
Report original synthetic loss components, expected discounted-gain efficiency,
expected positive probability mass captured and saturation. These are ratios
of expected **synthetic** utility on training-style unseen slates, not exact
expectations of NDCG/Recall, Cornac metrics, or replacements for the eight
real-benchmark comparisons. No real-data training, outer evaluation or model
update is permitted in this diagnostic.

All 351 focused tests pass, including 16 diagnostic tests covering ideal and
inverted rankings, clipping/ties, sparse slates, user eligibility, input
fingerprints and rejection of real-labelled or invalid episodes. Require the
actual shared-case diagnostic before deciding whether to continue full-context
optimization or revise the synthetic prior. Output directory:
`synthetic_generalization_full_context`.

At exposure-only update 150, key `[0,1.756365358989216,1.4518087460552027]`
regresses again despite retaining every MF rating win. MovieLens NDCG/Recall
are .071675/.085194 and Netflix .050238/.070062; the local selected checkpoint
remains update 50. Finish the schedule, audit the restricted state, and then
run the registered synthetic diagnostic under the same exclusive campaign
lock. No independent GPU job is launched while training is live.

### Completed exposure-only control

The exposure-only control completes 200 updates and selects step 50, SHA256
`82d823cd9e5ec22fb4a951aa1295ab4260a9daa63d35f3f05acfed6e8eca467b`.
Its campaign audit passes, including exact equality of all **250 non-exposure
state tensors** to initialization. Key `[0,1.7140108100858928,1.421579587929355]`
does not beat the full-context incumbent; no promotion is made. Rejected final
update 200 has key `[0,1.7673507797629966,1.4602973629306253]`, with every MF
guard passing but ranking worse. Restricting these four tensors at this step
size is not sufficient to close the ranking gap.

### Shared synthetic observability result

The six-case diagnostic completes with all candidate/teacher states unchanged,
no gradients and no real datasets loaded. Its results are mixed: the full-context
winner does not consistently improve the original synthetic ranking objective.
An attempted post-hoc count reference correctly stops at an exact-input-hash
guard. A fresh two-draw probe finds identical masks, queries, ordinal targets,
latent relevance and support counts, but GPU bias normalization differs by
about 1e-6. The original diagnostic and failed log are preserved; no claim of
bitwise cross-regeneration reproducibility is made.

To avoid that ambiguity, rerun the registered six cases with all neural and
count references evaluated on each **same generated episode**. Sources and
within-run input/state fingerprints are archived in
`synthetic_generalization_with_context_references`. The resulting count-only
summary is `synthetic_observability_reference.json`; it reads stored references
instead of regenerating any data or changing inference.

| Synthetic scorer | Expected-DCG efficiency | Positive-mass capture ratio |
|---|---:|---:|
| Parent-positive checkpoint | .252134 | .044492 |
| Full-context incumbent | .251348 | .045504 |
| Exposure-only selected checkpoint | .249096 | .044028 |
| Visible-context item counts | .686570 | .124016 |
| Global-parent item counts | .697459 | .124862 |

Visible counts use only the encoded mask and outperform the neural incumbent
on every one of these synthetic cases; global-parent counts add little. The
gap therefore cannot mainly be attributed to unavailable local count information
in these cases. It does **not** prove that count-level ranking is achievable
while retaining rating accuracy. These ratios are not real NDCG/Recall, cannot
be compared numerically to the real BPR targets, and do not satisfy any goal
comparison. Neither count reference is blended into public inference.

### Registered visible-frequency distillation

An optional synthetic ranking target `visible_item_frequency` now equals
`observed_item_count / encoded_row_count` from the generated episode's own
visible mask. It removes latent target uncertainty while teaching an available
statistic. It is an exposure-frequency proxy, **not** a positive-holdout
probability. Ranking distillation still optimizes the existing neural mean;
there is no new head, real-data teacher or inference-time count correction.
Original rating targets, ordinal probabilities, exposure draws and query
sampling are unchanged. The objective verifies exact mask/target agreement on
every such episode before accepting the loss. Conditional targets remain the
default. All 366 focused tests pass; six default-prior CPU cases exactly match
archived tensors and metadata, and frozen inference fingerprints remain intact.

Before any training, register `full_context_visible_frequency_weight1_200`:
initialize from full-context update 200, SHA256
`5dd2dd7d7da3fd80b55bb76275e162f2c1fcb4d05886d747a0c3a3fa59815a93`;
seed 71; 200 updates; validation every 50; LR 2e-6 and EMA .95. Use full-network
optimization, 2000 users, two separately encoded 1000-item windows, shared
normalization and joint loss, with replay and CPU activation offload. Preserve
the 50% contiguous / 50% catalog mixture and parent-positive selection of 64
ranking users with up to 2000 candidates. Set only the ranking teacher to
visible frequency; keep temperature .1, ranking/boundary weights 1 and original
synthetic MF teacher weight 5. Preserve the existing rating/NLL objectives.
Projection, one-sided gradients and paired coverage stay off; save best only.

The lower LR is a deliberate optimization safeguard, so this is not described
as a pure target-only ablation. Require both synthetic preflight modes with
exact visible-teacher verification, finite gradients, a frozen teacher,
unchanged architecture and the existing memory limits before launching. The
MF/BPR selection and eight-target gate remain unchanged. No outer evaluation
has run, and BPR remains unbeaten.

The visible-frequency preflight passes both context modes: peak GPU allocation
125.579 GiB and host RSS 77.363 GiB. It verifies all 238 model parameters are
trainable, both gradient norms are finite (.825945/.773329), each ranking
teacher exactly matches its generated visible mask, and both windows replay
identical logits. The original synthetic rating teacher is frozen, no real
data is loaded, and architecture/state shapes remain unchanged. Launch the
registered 200-update run on the otherwise idle GPU; no configuration changes
are made after preflight.

The visible-frequency run is live (PID 1496364). Its initial inner metrics
reproduce the full-context incumbent exactly, all 15 archived sources verify,
and frozen inference and synthetic initialization/teacher provenance are
unchanged. The first update has finite loss/gradients, exact visible-mask
teacher verification and two successful gradient replays. This is not a
completed checkpoint promotion; the full-context incumbent remains selected.

### Synthetic easy-tail loss-scaling diagnostic

A separate CPU-only calculation uses fabricated scores/probabilities and the
actual current ranking loss, without loading a neural model or real data and
without any optimizer update. Ten incorrectly top-ranked items have synthetic
frequency .02, ten missed best items have .20, and added easy-tail items have
.0005 with scores well below both groups. Top-10 expected utility stays at
.1 of ideal throughout. Adding 1980 easy candidates to the same twenty hard
items reduces the loss from 1.410302 to .062820 and the hard-item score-gradient
norm from 3.563917 to .158749: a **22.45-fold dilution** caused by normalizing
over all pair-weight mass. This is a surrogate-scaling property, not a neural
parameter-gradient measurement, real benchmark result or implementation-error
claim. The report is `synthetic_easy_tail_dilution.json`.

This motivates a later controlled objective that compares the union of model
top-K and teacher top-K candidates, rather than allowing thousands of easy
tails to dilute corrections. It is not implemented or registered yet. The
currently running visible-frequency trial retains its original loss and full
200-update schedule, so its outcome remains interpretable. Any later objective
change still needs synthetic tests, memory preflight and all unchanged goal
gates; neither diagnostic establishes a BPR win.

Visible-frequency update 10 is now verified: loss 1.562502, ranking term
.560987 and gradient norm 1.440564 are finite. Both logged updates retain the
synthetic-only, exact visible-teacher and two-window replay checks. The first
new inner validation remains scheduled at update 50; no early promotion or
outer evaluation is made.

### Optional top-k union objective prepared (not yet trained)

`--ranking-candidate-scope topk_union` now restricts the existing expected
pairwise objective to the model-top-K / teacher-top-K union. Full-catalog
score ordering, model-position discounts, user-macro averaging, rating losses,
synthetic rating-teacher anchor, and public inference remain unchanged. The
default is `all`, preserving the live visible-frequency run and older recipes.
The union is a training-only loss restriction, not an inference-time filter.

Teacher-cutoff ties share the remaining slots fractionally rather than selecting
arbitrary item indices. Zero-relevance ties cannot reintroduce the catalog
tail. Current model-top-K entries always retain weight one; the resulting
effective candidate weight is at most 2K per user. One-sided gradients remain
supported, but centered one-sided gradients are rejected for union mode because
their full-catalog centering would move excluded candidates. Other ranking
objectives reject the union flag before loading a model.

All **421** focused recommendation tests pass, including tail-invariant loss
and correction gradients, hand-enumerated pair weights, fractional cutoff
ties, sparse and degenerate slates, equal user weighting, finite-difference
gradients, synthetic-only/visible-mask guards, rating-anchor preservation,
cross-window corrections and exact-logit gradient replay with offload.
The test log is `/tmp/dwip_bpr_tests_20260921_topk_union.log`.

`topk_union_default_equivalence.json` verifies 24 synthetic CPU cases against
the live trial's archived objective: default loss, diagnostics, protected
components and logit gradients match exactly. All 15 archived live-run sources
still match their registered hashes, and the frozen inference signature is
unchanged. The running Python process retains its imported original objective;
its registration, archived sources and schedule were not edited or restarted.

`synthetic_easy_tail_union_comparison.json` executes both loss scopes on the
same fabricated-score counterexample. With 20, 40, 200 or 2000 candidates, union
loss remains 1.410302 and hard-item score-gradient norm remains 3.563917; excluded
tails receive zero ranking gradient. The unrestricted loss still dilutes the
same correction 22.45-fold. This confirms the intended loss-scaling property
only, not a neural parameter-gradient benefit, rating guardrail or BPR win.

No union GPU experiment is registered or launched yet. Finish and audit the
current visible-frequency trial before any further GPU preflight. Any union
trial requires an explicit registration and the same synthetic-only provenance,
unchanged model/inference, independent inner verification and eight-target gate.
No new outer evaluation is authorized by these CPU tests.

At 16:43 UTC the existing trial is live through logged update 20: loss 1.089724,
ranking term .081981 and gradient norm 1.082489, with the exact visible-mask
teacher and two window replays verified. Its first new inner validation is still
scheduled at update 50. The eligible incumbent and unmet goal remain unchanged.

### Visible-frequency inner validation 50 (not eligible)

At 17:04 UTC, update 50 completes the first planned inner evaluation:

| Inner metric | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7454909103 | .7362063745 |
| RMSE | .8782109984 | .8371266709 |
| AUC | .8249852703 | .8368861773 |
| MAP | .0596694828 | .0394175532 |
| NDCG@10 | .0726208483 | .0563910710 |
| Precision@10 | .0452322738 | .0281895143 |
| Recall@10 | .0852438452 | .0769931282 |

The fixed selection key is `[1, 1.0001407480730007, 1.3620501927932]`, so this
checkpoint is correctly rejected. Netflix RMSE exceeds MF by .0001178074;
the other three rating comparisons still pass, but no BPR comparison passes.
Top-10 ranking is slightly worse than initialization despite higher AUC. This
is consistent with, but does not prove, a mismatch between easy-pair progress
and top-K ranking. It motivates testing the prepared union control after the
current registered trial finishes, not changing its live loss or schedule.

All logged updates through 50 retain synthetic-only gradient data, exact
visible-frequency teacher verification and two verified window replays. The
best eligible completed checkpoint remains full-context update 200 with all
four MF wins; no checkpoint promotion or new outer evaluation occurs here.

### Visible-frequency validation 100 and registered union control

Update 100 is again rejected: key
`[1, 1.0005039149027575, 1.350056453005105]`. MovieLens MAE/RMSE are
.7458983721/.8787746755, with NDCG@10/Recall@10 .0728727489/.0842039977;
Netflix is .7363662121/.8374306448, with .0578266729/.0780483702. Netflix ranking
is slightly better than initialization, but its RMSE remains above MF and no
BPR comparison passes. The current registered 200-update run remains live and
unchanged, and the selected checkpoint is still initialization.

Before any union preflight/training, register the matched control
`full_context_visible_union_weight1_200` in `topk_union_registration.json`.
Use the **same** full-context initialization SHA, seed 71, LR 2e-6, 200 updates,
validation interval 50, EMA .95, full scope, two replayed/offloaded 2000-user
by 1000-item windows, original context mixture and visible-frequency teacher.
Keep ranking/boundary weights 1, temperature .1, the original synthetic rating
teacher weight 5, and all rating/NLL settings unchanged. The only training
argument change is `ranking_candidate_scope=topk_union`; no projection,
one-sided gradients, paired coverage, inference changes or last-checkpoint
saving. Equal seeds do not imply bitwise GPU normalization reproducibility.

Finish and audit the existing run first. Then require both full-size synthetic
preflight modes, exact visible-mask teachers and two verified gradient replays,
unchanged architecture/inference and source/argument receipts before launch.
The fixed selection gate, independent reproduction requirement and prohibition
on premature outer evaluation remain unchanged. Registration is not a result.

### Visible-frequency run completed and audited; no promotion

The original 200-update run completes in 8,908 seconds. Update 150 still fails
Netflix RMSE, with key `[1, 1.000549429883837, 1.3184321644439925]`; its final
update is also rejected:

| Inner metric at update 200 | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7454456078 | .7362581102 |
| RMSE | .8777534345 | .8370879990 |
| AUC | .8235734680 | .8352990425 |
| MAP | .0607840515 | .0403232302 |
| NDCG@10 | .0748961335 | .0578851815 |
| Precision@10 | .0459657702 | .0287093420 |
| Recall@10 | .0870284213 | .0775418839 |

The final key is `[1, 1.0000945454997907, 1.3347272783995048]`; Netflix RMSE
exceeds MF by .0000791354 and no BPR target passes. Selection correctly retains
**update 0**, SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
`campaign_audit.json` passes, and `selected_initialization_identity.json`
independently verifies all 254 selected state tensors exactly equal the
full-context initialization. The different file SHA reflects checkpoint
serialization/step metadata, not changed selected weights. All 21 logged
updates through 200 retain synthetic-only gradients, exact visible-mask
teachers and two verified gradient replays. No checkpoint is promoted and no
outer evaluation runs.

The registered union control's 34 preserved training arguments and initial SHA
match the comparison config (`topk_union_recipe_comparison.json`). After the
completed audit and confirmed release of the GPU, the two-mode full-size union
preflight starts. Its 15 source snapshots are archived separately before
execution. No union neural training or real-data evaluation starts before that
preflight passes and its source/argument receipts verify.

Both union preflight modes pass: GPU peak **125.578814 GiB**, host RSS
**77.356300 GiB**, frozen original synthetic teacher, exact visible-mask targets
and two exact-logit gradient replays each. Contiguous/catalog losses are
5.791916/5.122069, with pre-clipping parameter-gradient norms 72.188812/55.347721;
effective union candidate weights average 18.75/17.671875 per user. These
gradient norms are substantially larger than the unrestricted preflight's
.825945/.773329, as expected when removing the demonstrated dilution. This is
not evidence that rating accuracy is protected; unchanged clipping and all
four strict MF comparisons remain mandatory. No ranking-weight rescaling is
made after observing the preflight.

`topk_union_preflight_receipt.json` verifies all 60 nonadministrative,
non-candidate-scope arguments exactly match the completed trial, all 15 current
and archived sources agree, and the frozen public inference signature is
unchanged. With those checks passed, `full_context_visible_union_weight1_200`
starts (PID 1508990; session 42231). It initializes from the registered
full-context checkpoint and will run the original 200-update, every-50 inner
validation schedule. The goal is still unmet; no new outer evaluation runs.

The union launch receipt passes: step-zero inner metrics and selection key
reproduce the completed comparison trial exactly; initial checkpoint and frozen
synthetic teacher match; all 15 archived/preflight/current sources agree; and
the frozen inference signature is intact. The sole training argument change
is candidate scope (apart from the trial name). First update loss is 8.789435,
ranking term 7.542850 and pre-clipping gradient norm 39.004711, all finite.
Effective candidate weight is 20 per user, with synthetic-only gradient data,
exact visible-mask teacher verification and two gradient replays confirmed.
`launch_receipt.json` archives these checks. PID 1508990 remains live; the first
post-training inner validation is scheduled at update 50. Disk free space is
8.7 GiB after saving initialization; retain the 8 GiB reserve and save-best-only
policy. This launch is progress, not evidence of a BPR win.

### Union validation 50: three BPR wins, no rating wins

The first post-training inner validation gives a substantial ranking gain, but
the complete goal still fails:

| Inner metric at union update 50 | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7597892250 | .7655009828 |
| RMSE | .8979045296 | .8696702266 |
| AUC | .8633547295 | .8951634882 |
| MAP | .0831461710 | .0655009529 |
| NDCG@10 | .0982330905 | .0864615598 |
| Precision@10 | .0599022005 | .0450764889 |
| Recall@10 | .1093279417 | .1144444613 |

Both MovieLens ranking targets and Netflix NDCG@10 beat the frozen BPR inner
reference. Netflix Recall@10 still trails .1230539223. All four rating metrics
are worse than their MF references, so the fixed key
```
[1, 1.0390215258825606, 0.9597311555997348]
```
correctly rejects this checkpoint. No checkpoint is promoted and no outer
evaluation runs; the best eligible completed model remains the full-context
incumbent. These are previously used inner-validation data, not blind testing.

Synthetic matched-seed evidence also shows the trade-off. At update 50, point
loss is .828451 versus .718476 for the unrestricted control; synthetic teacher
anchor error is .203247 versus .005929. This is not merely a change in how
the ranking loss is normalized. It does not by itself establish the direction
of the real-data rating bias, because signed query errors are not archived by
the live evaluator.

Register a conditional one-sided control before any new preflight/training:
`full_context_visible_union_negative_only_200`, recorded in
`negative_only_union_registration.json`. Preserve the current union trial's
initialization, seed71, complete prior and optimizer recipe, and change only
`negative_only_ranking=true`. This detaches each pair's more-relevant score and
suppresses its less-relevant score; ranking-loss values at fixed scores do not
change. It may avoid unnecessary boosting of frequent items, but rating
retention remains unproven. Existing union/one-sided CPU tests cover this path.
Do not enable full-catalog gradient centering, which would move excluded tails.

Finish and audit the current200-update run first. If its selected checkpoint
passes the full gate, independently verify it instead of launching the new
control. Otherwise require both synthetic GPU preflight modes, matched-source
and single-argument receipts, and all unchanged gates before a new launch.

### Union validation 100: all BPR wins, still no rating wins

All four BPR comparisons now pass on the existing inner split:

| Inner metric at union update 100 | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7736000003 | .7910141234 |
| RMSE | .9107121303 | .8958899033 |
| AUC | .8700483923 | .9084293451 |
| MAP | .0870752394 | .0777526961 |
| NDCG@10 | .1040191953 | .1002497114 |
| Precision@10 | .0625916870 | .0494281895 |
| Recall@10 | .1168404059 | .1322880141 |

The ranking improvement does not satisfy the goal: all four MF rating
comparisons still fail, and the fixed key
`[1, 1.072898430324246, 0.8627480321832717]` rejects these weights. The selected
model remains initialization. No promotion, independent final certification,
or new outer evaluation occurs; the original 200-update schedule continues.

### Optional validation-detail archival for the next control

The next registered one-sided control now additionally enables an explicitly
administrative `--save-validation-details` flag. This passes a distinct output
directory to the **existing** no-gradient inner evaluator at each already
scheduled validation (including initialization), retaining query predictions,
ranking arrays, profiles and read-only audit receipts. It adds no evaluation
point, metric, inference transform, fit or gradient, and the ranking candidate
protocol is unchanged. No real-data calibration target or fitted offset may
enter training or public inference. The purpose is to measure signed rating
errors rather than speculate about their direction if rating retention fails.

This flag defaults off; the live trial keeps its imported original script and
unchanged registration. The next trial's only optimization change remains
`negative_only_ranking=true`; its registration separately records the archival
flag as an administrative difference. All **430** focused tests pass, including
no-gradient evaluation, restored gradient context, unchanged RNG, identical
metric returns, legacy-default compatibility and per-validation output paths.
`validation_details_safety.json` confirms all 15 archived live-run sources
still match, the only workspace source change is the training wrapper, and the
frozen architecture/public inference signature is unchanged. The active run
still must finish and pass audit before any next GPU preflight.

### Union run completed and audited; no promotion

The 200-update union trial finishes in 8,907.75 seconds. Update 150 retains all
four BPR wins but loses all four MF comparisons (key
`[1, 1.077145975117706, 0.8998246864102802]`). At update 200, Netflix Recall@10
slips below BPR again and rating retention still fails:

| Inner metric at union update 200 | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7793916642 | .7928631683 |
| RMSE | .9124705651 | .8956697345 |
| AUC | .8572117387 | .9006666519 |
| MAP | .0832059994 | .0689280460 |
| NDCG@10 | .1000331835 | .0911181571 |
| Precision@10 | .0572127139 | .0428932125 |
| Recall@10 | .1148017242 | .1218161333 |

Final key `[1, 1.075406397350141, 0.9157870628780489]` is rejected. Selection
correctly retains update 0, SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`, with all
254 state tensors verified exactly equal to the initial full-context checkpoint.
The completed campaign audit passes and all 21 logged updates retain the
synthetic-only, exact visible-teacher and two-replay checks. No trained weights
are promoted; the temporary inner ranking wins do not satisfy the joint goal.
The best eligible completed checkpoint and zero-new-outer-evaluation policy
remain unchanged.

After the audit and confirmed GPU release, the registered one-sided control
starts its two-mode full-size synthetic preflight (PID 1520148; session 36060).
Its source snapshots include the tested optional validation-detail wrapper.
The completed selected candidate's eight-target gate remains false, so the
conditional registration permits this preflight rather than final verification.
No new neural training starts until preflight and source/argument receipts pass.

### One-sided preflight superseded before training: common-offset diagnosis

The uncentered one-sided preflight passes both modes (GPU peak125.578814 GiB;
parameter-gradient norms50.835365/42.924152). Its sources/arguments and synthetic
checks pass, but **no training is launched**. A new CPU-only counterexample
(`synthetic_union_offset_direction.json`) shows why an additional safeguard is
needed: a uniform -.25 score shift leaves the union ranking loss unchanged,
yet uncentered one-sided gradients give the common offset derivative +7.444944.
This can push every score downward without changing their order. It is a
synthetic score-space risk, not a measured neural/real-data bias. Preserve the
original registration, preflight and supersession receipt unchanged.

The existing optional per-user gradient centering now also supports union mode.
It removes the common-offset direction while retaining top-K-union **pair
weights**. It intentionally distributes compensating gradients across all valid
queries, including excluded tails; no zero-tail-gradient guarantee is claimed
for this new combination. With2000 candidates the counterexample's hard-item
gradient norm is2.659964 versus2.670311 uncentered, rather than the22-fold
all-pair dilution. This is training-gradient centering only: forward loss,
scores and public inference are not centered or calibrated.

All450 focused tests pass. Coverage includes the offset null direction,
user-local sparse-query centering, explicit compensating tail gradients,
protected rating/teacher components and window replay/offload. A pure-ranking
toy exposed AdamW amplification of tiny floating-point cancellation in its
unconstrained offset; optimizer-parity coverage now uses a protected point term
as the actual trainer does, without loosening tolerances. Score-space centering
is separately checked directly. All36 previously supported synthetic objective
cases reproduce archived losses, diagnostics, protected components and gradients
exactly (`centered_union_default_equivalence.json`). Frozen inference is intact.

Register `full_context_visible_union_centered_negative_200` before preflight
or training in `centered_negative_union_registration.json`. It supersedes the
uncentered control before training and preserves the symmetric-union recipe,
initialization and seed71, changing only the two optimization flags enabling
one-sided gradients and their centering. Validation-detail archival is separately
registered as administrative. Require both fresh synthetic preflights, explicit
argument/source receipts, the unchanged200-update schedule and all eight goal
comparisons. No benchmark win or rating retention is inferred from this fix.

The centered control passes both fresh synthetic preflight modes: peak GPU
125.578814 GiB, host RSS77.357929 GiB, losses5.794093/5.116689 and finite
pre-clipping gradient norms47.200520/37.383839. Both verify the exact visible-mask
teacher, frozen original synthetic rating teacher and two exact-logit gradient
replays. The explicit gradient-center scope is all valid queries per user.
`centered_negative_union_preflight_receipt.json` verifies all59 remaining
arguments, all15 current/archived sources and the frozen inference signature.
Cross-process synthetic GPU normalization is not claimed bitwise reproducible.

With those checks passed, the registered centered control starts its200-update
training run (PID1521040; session33336), with validation every50 and the new
read-only validation-detail archives. Its initialization, synthetic targets and
teacher are unchanged. The uncentered one-sided variant remains preflight-only;
its history is preserved and no uncentered training directory is created.
All goal requirements remain active; no new outer evaluation is authorized.

The centered launch receipt now passes. Initial inner metrics and every saved
query-prediction array reproduce the independently verified full-context
incumbent exactly. Rating metrics recompute from those archives, and evaluator
receipts confirm gradients disabled. The descriptive initial user-macro mean
errors are +.079704 (MovieLens) and +.036423 (Netflix); these are not fitted or
applied corrections and never enter optimizer inputs. Both extra optimization
flags, the administrative archive flag, identical synthetic initialization and
teacher, all15 current/preflight/training source hashes and frozen inference
verify. First update loss8.789434, ranking7.542850 and pre-clipping gradient
norm12.834167 are finite, with candidate weight20, explicit per-user gradient
centering, exact visible-mask teacher and two verified replays. PID1521040 is
live; no post-training inner validation exists yet. Free disk space is8.4 GiB
after initialization and detail archival; preserve the8GiB reserve and
save-best-only policy. The goal remains unmet.

### Centered-negative validation 50 and read-only error diagnosis

Update50 fails: MovieLens MAE/RMSE .7465938225/.8795828230 retain its MF wins,
but Netflix .7419827263/.8430343134 loses both. NDCG@10/Recall@10 are
.0760802949/.0883836295 for MovieLens and .0358076454/.0483959414 for Netflix;
no BPR comparison passes. The fixed key
`[1, 1.0071987885858604, 1.792286538947916]` rejects the candidate. Continue the
registered200-update schedule without changing its live loss or promoting it.

`inner_error_diagnostic_50.json` compares already-saved inner predictions
with initialization; it loads no neural model and performs no fit or update.
MovieLens user-macro mean error rises from+.079704 to+.102917; Netflix rises
from+.036423 to+.071313. Mean per-user prediction-change RMS is .092843/.095565.
Netflix's average user MSE increase .013301 decomposes into .007741 increased
squared user bias and .005560 increased within-user centered error. This is a
descriptive identity, not a calibrated score or replacement for the RMSE gate.

The Netflix ranking profile also deteriorates: mean recommended item support
falls1004.70→536.08, the share of recommendations with support≤10 rises
.008481→.074959, and first1000-item recommendation share falls.749027→.420734.
This is consistent with the explicitly introduced tail-compensation gradients
being unhelpful; it does not establish a causal attribution by itself. The local
score-gradient common-offset property did not prevent shared neural parameters
from increasing real-data prediction bias. No offsets are fitted or applied,
and no real evaluation labels enter any optimizer.

Before another preflight/training, conditionally register
`full_context_visible_union_anchor50_200` in `union_anchor50_registration.json`.
Return to the successful-ranking **ordinary two-sided** union recipe and its
same initialization/seed71; change only the synthetic rating-teacher anchor
weight5→50. Keep centering and one-sided gradients off. Preserve all other prior,
loss, optimizer, schedule and inference settings, with validation-detail saving
separately administrative. This tests genuine rating regularization rather
than a surrogate gradient redistribution; ranking retention is not assumed.
Finish/audit the current run first, and verify its selected candidate instead
if it passes all eight comparisons. Otherwise require fresh synthetic preflights,
source/argument receipts and sufficient8GiB-reserve checkpoint headroom before
launch. All baseline references and goal requirements remain unchanged.

The update50 diagnostic was renamed from `validation_50_error_diagnostic.json`
to `inner_error_diagnostic_50.json` to avoid collision with canonical validation
file discovery. Its contents and SHA256 are unchanged; the rename receipt is
`diagnostic_artifact_rename.json`. Training, evaluation and audit criteria are
unchanged, and the live process was not restarted.

### Checkpoint headroom without removing artifacts

The next trial would have lacked safe temporary-checkpoint headroom above the
8GiB reserve. The two completed visible-frequency and symmetric-union selected
step0 checkpoints are byte-identical (SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`). After checking
completion, audits, exact byte equality and synthetic provenance, their storage
was deduplicated with an atomic hard link. Both original paths and every byte
remain available; all logs, sidecars, configs and source snapshots retain their
hashes. No live checkpoint, baseline or model behavior changes. Free space rises
8.322→8.613GiB. The receipt is `completed_checkpoint_storage_deduplication_1.json`.
Completed linked files remain immutable; the existing checkpoint saver uses
atomic replacement and does not modify linked peers in place.

### Centered control terminated for futility; integrity audited

After rejected validation100, the centered-negative exploratory trial was
deliberately terminated at 2026-09-21T23:06:04.772887Z. Its original200-update
schedule was not completed. The last logged update and completed validation
are100; additional unlogged optimizer updates are possible, so no exact total
is claimed. At validation100, MovieLens MAE/RMSE/NDCG@10/Recall@10 are
.754890350/.890849731/.062358855/.080678825; Netflix values are
.753164434/.856702764/.011779950/.019064685. Only one MF guard and zero BPR
targets pass. The stop is an explicit exploratory deviation, not a timeout.

`full_context_visible_union_centered_negative_200/termination_audit.json`
passes integrity, not completed-trial certification: all15 archived/current
sources, frozen inference/baselines, synthetic initialization/teacher lineage,
all11 logged synthetic updates, visible-mask teachers and two exact-logit replays
verify. Canonical0/50/100 selection keys reproduce. The selected step0 checkpoint
has SHA256`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`,
and all254 tensors exactly equal the registered initialization. It is ineligible
for promotion; no completed campaign audit is fabricated. The process is gone
and GPU released. Best eligible completed weights remain unchanged.

The preflight/training prerequisite for the registered stronger-anchor control
was amended before launch to accept this audited termination. Original
registration remains in`union_anchor50_registration_before_futility_stop.json`.
No optimization recipe, baseline or goal criterion changes. Proceed to fresh
synthetic preflight with ordinary two-sided union and teacher anchor50, not the
failed centered surrogate. No new outer evaluation is authorized.

The stronger-anchor control passes both fresh synthetic preflight modes:
GPU peak125.578814GiB and host RSS77.364017GiB, losses5.961480/5.358147,
finite gradient norms63.063396/46.658249. Both verify exact visible-mask
teachers, the frozen original synthetic rating teacher and two exact-logit
replays. `union_anchor50_preflight_receipt.json` verifies all60 preserved
arguments, all15 source snapshots, synthetic lineage and frozen inference.
No real data is loaded by the preflight. Relative to the ordinary union recipe,
the only optimization change is teacher-anchor weight5→50; both one-sided flags
are false. Validation-detail archival is separately administrative.

After its terminal audit, the centered trial's step0 checkpoint is verified
byte-identical and hard-linked to the completed union step0 blob. Both paths,
bytes and all other artifact hashes remain intact. No live checkpoint is touched;
`terminated_checkpoint_storage_deduplication_1.json` records the operation.
Free space8.899GiB at preflight receipt accounts for the next0.291GiB checkpoint
and a temporary replacement while retaining the8GiB reserve.

The registered stronger-anchor200-update training run now starts (tool session
37386), with inner validation every50. Initialization remains the independently
reproduced full-context update200 checkpoint, never the terminated trial. All
neural optimizer inputs, targets and teacher ancestry are synthetic-only. The
four MF wins of the best completed checkpoint remain intact; no BPR goal win,
new outer evaluation or trained-checkpoint promotion is claimed at launch.

Launch receipt confirms the training configuration matches both preflight and
registration, with identical15 source snapshots and frozen inference. PID1523757
is the sole GPU training process. The initial inner evaluation exactly reproduces
the saved reference. The first logged update verifies synthetic gradient data,
exact visible-mask teacher, two window replays, ordinary two-sided gradients and
anchor50 (loss9.816903, finite pre-clipping gradient norm69.012924). These are
startup integrity checks, not trained-checkpoint evaluation or goal completion.

### Optional finite-gap frequency-choice preparation during anchor50 training

The stronger-anchor process remains live and unchanged. Its update10 log is
synthetic, with exact visible teacher and two replay checks, loss9.125389 and
teacher-anchor discrepancy.011108. There is not yet a trained inner validation;
the initial four MF wins and zero BPR wins are not new results.

A CPU-only fabricated-score diagnostic identifies a narrower optimization risk:
within the retained top-K set, the hard sign surrogate still widens a correctly
ordered finite gap. For frequencies.05/.005 and temperature.1, at gap.230259
its derivative is-.909091. A soft frequency-choice target has zero derivative
there and reverses over-separation. The first attempted top-K=1 counterexample
correctly failed its nonzero-gradient assertion: a correct one-item union
excludes the other item and has exactly zero loss/gradient. The repaired
top-K=2 diagnostic explicitly checks that exclusion control and limits the claim
to pairs within the retained union. No benchmark failure is attributed to this
toy alone.

The diagnostic also exposes a tradeoff: for close rates.05/.049, soft choices
can remain misordered under anchoring even though hard choices reverse them.
This is not proof that soft ranking will outperform the current trial.
`synthetic_finite_margin_diagnostic.json` preserves both the evidence and limits.

Prepare a separate, currently unused `recsys_soft_choice_objective.py`: same
full-catalog ordering, top-K discounts, fractional teacher ties and union pair
weights, but pair label probability is `(frequency_i+s)/(frequency_i+frequency_j+2s)`.
The registered synthetic half-count gives`s=.5/2000=.00025`. It provides finite
targets even at zero frequency; this is an exposure-choice proxy, not a claimed
posterior for real positive relevance. No real-data fit or inference transform
is involved. Both gradient directions remain active.

All140 focused CPU tests pass, covering finite optima, corrective/reverse
gradients, exact pair enumeration, common-offset invariance, sparse user-macro
aggregation, zero mass, cutoff ties, zero excluded-tail gradients and undiluted
20→2000-candidate gradients. Existing/recomputed window paths preserve logits,
gradients, RNG and AdamW steps with the soft loss and a protected point term.
The preparation receipt/snapshot archives the tested sources and log. All15
active training sources and frozen inference still match their registered hashes.

`soft_choice_union_registration.json` conditionally registers the same-seed
soft-choice control against the immutable current anchor50 configuration,
preserving initialization and all other prior/optimization settings. It does
not authorize stopping or altering the current run. The helper is not wired
into the trainer and no soft-choice GPU preflight/training has started. Finish
and audit the current trial first; if its selected candidate passes all eight
comparisons, independently verify it instead. Otherwise require preserved
legacy outputs, integration tests, fresh synthetic preflights, exact receipts
and storage headroom before any optional fallback. All original goal and outer
evaluation gates remain unchanged.

Before any optional trainer integration, freeze108 CPU synthetic objective cases
in`soft_choice_legacy_reference.pth` (SHA256
`090ac1cbc7bfd5ae2f861ccf62e9528f39a7bce4b5c01e3453b220266229c9b4`). The portfolio
covers three normalization modes, conditional/visible teachers, ordinary and
one-sided pairwise, expected-listwise, all/union top-K with ordinary/one-sided/
centered gradients, and anchor0/50. Inputs are saved so later comparisons do not
depend on episode regeneration. Synthetic expected-rating vectors act as the
toy teacher; no neural model, optimizer, real data or GPU is loaded.

The initial independent process reloads the9.14MB artifact and exactly reproduces
all108 losses, diagnostics, protected/ranking components and their logit
gradients. All15 live training sources and frozen inference remain unchanged.
This is a future legacy-regression reference, not a full-size neural preflight
or benchmark result. The source and manifest are preserved, and optional loss
integration remains deferred until the current training process terminates.

### Read-only saved-error diagnostics prepared for the scheduled validations

Add `diagnose_dwip_bpr_saved_errors.py` to compare existing inner artifacts,
without loading model weights, making new predictions, fitting calibration,
or changing training. It verifies identical query identities/targets, reproduces
saved macro MAE/RMSE and ranking averages, checks no-gradient evaluation receipts,
and hashes input artifacts before/after. It reports macro signed error, prediction
drift, rating-error decomposition and the already-saved ranking profile.
The decomposition is of macro user MSE, not squared macro RMSE. Diagnostic output
names beginning`validation_<digit>` are rejected to avoid canonical-file collision.

All9 focused tests pass. The previously saved centered-negative update50
diagnosis is independently reproduced within1e-12, with its original bytes/SHA
unchanged. The active anchor50 initial artifacts likewise reproduce and show
zero between-snapshot drift. This is artifact verification only, not a new
evaluation or independent Cornac gate. Tested sources, helper and logs are
archived in`saved_error_diagnostic_source_snapshot`; the receipt is
`saved_error_diagnostic_receipt.json`. All15 active training sources and frozen
inference remain unchanged. Run this utility on later scheduled inner artifacts
when available; no real-data statistic becomes a fitted correction or gradient.

### Stronger-anchor inner validation50: partial improvement, not eligible

The scheduled anchor50 validation completes with these inner metrics:

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7474002706 | .7429377470 |
| RMSE | .8801829604 | .8433629305 |
| AUC | .8378769615 | .8522515017 |
| MAP | .0692558566 | .0465015319 |
| NDCG@10 | .0826913938 | .0649665476 |
| Precision@10 | .0507334963 | .0332541215 |
| Recall@10 | .0906642655 | .0845998325 |

Only the two MovieLens MF rating guards and MovieLens BPR NDCG target pass:
2/4 rating guards,1/4 BPR targets. Netflix still loses both rating guards and
both ranking targets; MovieLens Recall also remains below BPR. Key
`[1,1.007689646645643,1.2239725142868623]` is rejected. The selected step0
checkpoint retains SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802` and is the
unchanged initialization, not a promoted trained candidate.

Compared with ordinary union anchor5 at the same scheduled update, anchor50
reduces Netflix MAE .765501→.742938 and RMSE .869670→.843363, but also reduces
NDCG .086462→.064967 and Recall .114444→.084600. Stronger protection therefore
shows the expected tradeoff, not a joint win. Same registered seed/recipe does
not imply bitwise cross-process GPU episode identity.

The saved-error utility reproduces all saved macro metrics without new inference.
MovieLens macro signed error changes+.079704→+.094755; Netflix+.036423→+.042496.
Netflix macro user-MSE increases.011863, split into+.006949 user-bias-squared
and+.004914 within-user error. The small overall signed shift does not explain
away the regression. Netflix mean recommended-item support rises1004.70→1038.96,
while support≤10 recommendation share rises.008481→.018016. These remain
descriptive statistics only; no fitted/applied corrections or optimizer inputs.

`inner_validation_50_observation.json` verifies the selection key and comparisons,
all15 training sources, frozen inference/baselines, unchanged selected checkpoint
and all six logged synthetic updates with visible-teacher/two-replay checks.
It is a scheduled-observation integrity receipt, not completed-trial certification.
The process remains live and its200-update schedule is unchanged. Continue to
validation100; watcher session92580 observes the same PID1523757. The prior
validation50 watcher completed normally. No new outer evaluation, final Cornac
verification or soft-choice training is authorized by this partial result.

### Optional soft-choice integration staged in isolation, not deployed

While the registered anchor50 trial continues, prepare copied training sources
under`soft_choice_integration_staging/`; none of the15 live production sources
is edited. The staged change adds the optional objective/half-count argument,
enables synthetic relevance targets for that objective, dispatches the guarded
soft-choice helper, and includes the helper in future16-source archival.
Visible-mask teacher equality, synthetic metadata, detached teachers, ordinary
two-sided union and a positive finite pseudo-count are enforced. Smoothing uses
encoded rows (`.5/2000` in the registered recipe), not the parent-row population.

All172 combined CPU tests pass, including23 staging tests. Both context modes
and odd/even two-window draws exactly preserve all prior episode tensors and
metadata relative to hard-union sampling. Rating/teacher protected components
and gradients remain exact; all108 frozen legacy objective cases reproduce
every loss, diagnostic, component and logit gradient bitwise. CLI dispatch and
synthetic-only refusal paths are tested without loading DWIP weights or using
the GPU. Symbolic invalid-metadata tests do not load real training data.

The staging receipt verifies that only the expected prior/objective/argument
functions differ in the copies, the frozen normalization AST remains exact,
and all15 active sources/inference hashes remain unchanged. Tested source copies,
helper, log, planned diff and receipt are retained in the staging directory.
This is not production deployment, GPU preflight or soft-choice training.

The conditional registration is unchanged: finish/audit the current200-update
trial first; if its selected candidate passes all eight comparisons, independently
verify it instead. Otherwise deployment must use`apply_patch`, reproduce the
legacy/integration tests on actual production sources, and pass new full-size
synthetic preflights with exact source/argument receipts before optional training.
Do not execute a training job directly from the staging copies.

### Stronger-anchor inner validation100: ranking gains still violate rating guards

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7500294897 | .7467425810 |
| RMSE | .8820827233 | .8466429753 |
| AUC | .8362673139 | .8575082930 |
| MAP | .0706745352 | .0479875686 |
| NDCG@10 | .0859853687 | .0669326202 |
| Precision@10 | .0514669927 | .0336699837 |
| Recall@10 | .0937439516 | .0863449440 |

The pass counts remain2/4 MF guards and1/4 BPR targets: only MovieLens MAE,
RMSE and NDCG pass. Both datasets' rating errors worsen from update50 while
their NDCG/Recall improve modestly. Netflix remains below both BPR targets
and above both MF error limits. Key`[1,1.01285036420488,1.188167753692487]`
is rejected. The selected step0 blob remains unchanged at SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.

Saved-prediction diagnostics reproduce the metrics without new inference.
Relative to initialization, macro signed errors actually decline:
MovieLens+.079704→+.075123 and Netflix+.036423→+.027368. Nevertheless,
macro user-MSE rises.006461/.017159. MovieLens bias-squared changes-.003053
but within-user error rises+.009514; Netflix components both worsen
(+.009400 bias-squared,+.007759 within-user). These results do not support
describing the rating regression as merely a common score offset. No fitted
or applied correction is introduced.

The update100 observation receipt verifies canonical0/50/100 selection keys,
all15 active/archived sources, frozen inference/baselines, unchanged selected
checkpoint and all11 logged synthetic updates with exact visible-teacher and
two-replay checks. This is not completed-trial certification. Continue the
unchanged200-update schedule to validation150 (watcher45648, same PID1523757).
The update100 watcher completed normally. The fallback remains staged only;
no new outer evaluation or independent final verification is authorized.
### Stronger-anchor inner validation150: no joint win

The scheduled150-update observation still passes only2/4 MF guards and1/4
BPR targets (MovieLens MAE, RMSE and NDCG). MovieLens MAE/RMSE are
.7509000042/.8827349570, NDCG/Recall .0861373474/.0940951517; Netflix
.7478381177/.8476598722 and .0673789647/.0875642779. Key
`[1,1.014336304335647,1.1796419354892715]` is rejected. Both datasets' rating
errors worsen from100 while ranking improves slightly. No candidate is promoted.

The selected step0 blob remains `f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`;
all254 tensors equal initialization. The observation receipt verifies all15
live/archived sources, frozen inference/baselines, exact keys for0/50/100/150,
and all16 logged synthetic updates through150 with exact visible teachers and
two replay windows. This is not a completed-trial audit.

Read-only saved-error diagnosis reproduces metrics without inference, training
or calibration. Relative to initialization, macro user-MSE increases
.00707515/.01909513 despite mean prediction shifts -.01270831/-.01179164.
The registered200-update run remains live (PID1523757); watcher37437 waits for
terminal status, not an early stop. The optional soft-choice fallback remains
staged only. Zero new outer evaluations; the goal remains active and unmet.

### Stronger-anchor completed; finite-margin optimization deployed

The anchor50 run completes all200 registered updates. Final MovieLens
MAE/RMSE .7509912633/.8824560874 and NDCG/Recall .0860887851/.0953455057;
Netflix .7470247144/.8466848488 and .0644787783/.0833949649. Every trained
validation passes only2 MF guards and1 BPR target. The selected checkpoint is
step0, not an improved candidate. The campaign audit and completion receipt
verify all254 initialization tensors, all21 logged synthetic updates with
exact visible teachers/two replays, archived sources and frozen inference.
The best eligible completed candidate remains full-context update200. No
outer evaluation runs. Only after this audit, byte-identical terminal step0
storage is deduplicated while preserving all paths and other artifact hashes.

The registered conditional soft-choice integration is now deployed via
`apply_patch`, exactly matching the three tested staging sources. Architecture,
inference, baseline references and the other12 archived training sources remain
unchanged. The new helper is included in the future16-source snapshot.
All519 production CPU tests pass, including22 integration cases explicitly
bound to actual production modules. The obsolete staging-only source invariant
is deselected and replaced by exact production/staging/unchanged-source checks.
All108 archived legacy losses, components, diagnostics and gradients remain
bitwise exact. Soft-choice half-count remains fixed at.5/encoded rows.

An initial receipt command incorrectly indexed an absolute-path source manifest
using relative paths. Its sequential shell continued into a preflight after
the verifier raised KeyError. That preflight was stopped, its log preserved,
and it is explicitly ineligible; no training trial or checkpoint save occurred.
This operational error is recorded in`soft_choice_aborted_preflight_attempt.json`.
Corrected verification resolves source paths, runs separately with fail-fast
shell semantics, and archives16 sources before a fresh registered preflight.
The fresh preflight is now running (PID1538919, tool45463). Training requires
its two full-size synthetic modes and exact source/argument receipt to pass.
The goal remains active; deployment/preflight is not evidence of a metric win.

### Registered soft-choice preflight passes; synthetic training starts

The fresh two-mode2000-row/two1000-item-window preflight passes with no real
data loaded. Losses are4.742096/4.540956 and gradient norms43.048317/33.906273.
Both modes verify exact synthetic visible frequencies, fixed.5/2000 smoothing,
frozen rating teacher, and two replay windows. GPU peak125.578814GiB and host
peak77.307289GiB remain within limits. The receipt verifies all16 sources and
61 preserved arguments against the immutable registration/comparison config.
Only ranking objective and the fixed half-count differ from anchor50.

With8.8715GiB free, the initial plus temporary checkpoint budget preserves
the8GiB reserve. The registered200-update synthetic training starts at
PID1539266/tool71482; first-update verification observes it via tool2204.
Initialization, teacher, seed71, LR2e-6, full scope, EMA.95, validation50 and
frozen public inference remain unchanged. No outer evaluation is authorized.

The first soft-choice update now passes its receipt: all16 current/training/
preflight source hashes match, training arguments differ from preflight only
by action/output, and initial inner metrics exactly reproduce the eligible
full-context checkpoint. First loss8.819296, gradient norm66.539528; exact
synthetic visible teacher, finite-choice target and two replays are verified.
The step0 serialization remains `f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
Watcher39931 observes validation50 without stopping or restarting training.

### Soft-choice inner validation50: less rating damage, still rejected

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7470403447 | .7416109203 |
| RMSE | .8795709585 | .8419997058 |
| AUC | .8352623785 | .8487439953 |
| MAP | .0671533218 | .0451673921 |
| NDCG@10 | .0817804608 | .0637524988 |
| Precision@10 | .0503667482 | .0325709194 |
| Recall@10 | .0919421911 | .0833769497 |

Only MovieLens MAE, RMSE and NDCG pass:2/4 MF guards and1/4 BPR targets.
Key`[1,1.0059627114223924,1.2342982805379654]` is rejected. Compared with
hard-union anchor50 at update50, Netflix MAE/RMSE improve
.742938/.843363→.741611/.842000, but NDCG/Recall decline slightly. This is
not a joint win. Selected step0 retains SHA`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.

The observation receipt verifies all16 live/archived sources, frozen inference
and baselines, canonical0/50 keys, unchanged selected bytes, and all six logged
synthetic updates with exact visible teachers, fixed half-count and two replays.
Saved-error diagnosis reproduces metrics without inference or calibration;
macro user-MSE rises.00336057/.00952929 relative to initialization. This is
an observation, not completed-trial certification. Training remains live at
PID1539266/tool71482 with its registered200-update schedule unchanged.
Watcher50491 targets validation100. Zero new outer evaluations; goal unmet.

### Conditional matched-teacher control prepared without touching the live run

The current first-update loss is computed before any optimizer step. Its
synthetic rating-teacher macro MSE is.0228212643, contributing1.1410632 at
anchor50, versus a rating loss1.1324382. The frozen teacher is the older
original synthetic checkpoint, not the current eligible full-context
initialization. Thus stronger anchoring also pulls back toward older weights;
it is not a neutral trust region around the current MF-winning function.

Register a conditional control changing only`teacher_checkpoint` to the same
synthetic checkpoint used for initialization, keeping weight50 and the entire
current soft-choice recipe. No training source, architecture, inference or
current schedule is changed. The registration is immutable and records the
comparison config hash and the synthetic first-update evidence. This is a
hypothesis, not evidence of a benchmark improvement; preserving synthetic
ratings may still fail to transfer or may suppress ranking gains.

All65 focused CPU tests pass, including8 new matched-teacher cases across both
context modes. Fabricated logits on synthetic episodes show exactly zero
initial anchor, unchanged total/protected losses and total gradients, then
restoring rating-only gradients after perturbation. No DWIP weights are loaded
for these tests. Actual BF16/window/query-batching teacher equality remains
untested and must satisfy predeclared anchor macro MSE<=.0001 in both fresh
full-size modes before any future training. All16 current sources remain exact.

The active200-update soft-choice run must finish and be audited first; if it
passes all eight comparisons, prioritize independent inner/Cornac reproduction
instead. Otherwise the optional matched-teacher trial still needs its own
source/argument receipts, synthetic lineage and full-size preflights. It has
not started. The current watcher/PID are unchanged, and no outer evaluation runs.

### Soft-choice inner validation100: joint target still fails

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7489675875 | .7443553611 |
| RMSE | .8807212748 | .8441407431 |
| AUC | .8329874425 | .8529128184 |
| MAP | .0669086712 | .0456958593 |
| NDCG@10 | .0814943786 | .0638194657 |
| Precision@10 | .0491442543 | .0318728650 |
| Recall@10 | .0898994866 | .0834032629 |

Only MovieLens MAE, RMSE and NDCG pass, again2/4 rating guards and1/4 BPR
targets. Key`[1,1.00961243907097,1.2411315414140378]` is rejected. Rating
errors worsen from50 on both datasets; MovieLens NDCG/Recall decline and
Netflix's are nearly flat. Selected step0 bytes remain unchanged, with no
trained candidate promoted.

The observation receipt verifies all16 source pairs, frozen inference/baselines,
canonical0/50/100 keys, frozen synthetic teacher provenance, and all11 logged
synthetic soft-choice updates with exact visible targets and two replays.
Saved-error diagnosis reproduces the metrics with no new inference/calibration;
relative to initialization macro user-MSE rises.00358387/.01229781 despite
mean prediction shifts -.00768174/-.01032715. This remains partial-run evidence.

The registered200-update schedule continues at PID1539266/tool71482.
Watcher22903 now observes validation150. The matched-teacher control remains
conditionally prepared, not launched; finish/audit the current trial first.
Zero new outer evaluations and no completed goal claim.

### Soft-choice inner validation150 audited; waiting for registered completion

MovieLens MAE/RMSE are.7497116014/.8812109078 and NDCG/Recall
.0811308145/.0902367568; Netflix.7452366015/.8449297544 and
.0643303297/.0841864785. Only MovieLens MAE, RMSE and NDCG pass, still
2/4 MF guards and1/4 BPR targets. Key
`[1,1.0108077167000495,1.2350731199620162]` is rejected. Rating errors
worsen from100 on both datasets; Netflix ranking improves slightly. No
trained checkpoint is promoted, and the selected step0 bytes remain exact.

The observation receipt rechecks all16 sources, frozen inference/baselines,
canonical0/50/100/150 keys and all16 logged synthetic updates with fixed
half-count, exact visible teachers and two replays. Saved-error metrics
reproduce without inference/calibration. Macro user-MSE rises
.00395177/.01385223 versus initialization despite negative prediction shifts.

PID1539266 continues the registered200-update schedule. Watcher13028 awaits
terminal status, not an early stop. The conditional matched-teacher control
remains unlaunched; no new outer evaluation or completed-goal claim.

### Soft-choice200 completes without promotion; matched-teacher preflight starts

The registered soft-choice run finishes all200 updates. Final MovieLens
MAE/RMSE .7497773163/.8810235149 and NDCG/Recall .0826060739/.0909986128;
Netflix .7445778906/.8442204488 and .0617876655/.0811937903. All four trained
validations pass2 MF guards and1 BPR target, never the joint goal. Step0 is
selected, with all254 tensors exactly equal to the eligible initialization.

The campaign audit and completion receipt pass: all16 source pairs, frozen
architecture/inference/baselines, exact selection keys and all21 synthetic
logged updates with fixed half-count, visible teachers and two replays.
Saved200 metrics reproduce without new inference or calibration. No candidate
is promoted and no outer evaluation runs. Only after terminal audit, its
byte-identical step0 storage is deduplicated, preserving all paths and other
artifact hashes; free space rises to8.864GiB.

The conditional matched-teacher gate is now satisfied. The new preflight
(PID1558699/tool48491) changes only the rating teacher to the exact synthetic
initial checkpoint; no production source changes are needed. All16 sources
are archived before starting. The predeclared initial-anchor macro MSE<=.0001
requirement remains strict in both full-size modes, alongside finite gradients,
frozen teacher, exact visible frequencies, fixed smoothing and two replays.
No matched-teacher training has started; its argument/source receipt must pass
first. The goal remains active and unmet.

### Matched-teacher preflight passes; registered synthetic training starts

Both full-size modes satisfy the predeclared initial-anchor limit. Their
macro teacher-anchor MSEs are5.9073e-15/5.6830e-15, versus the limit.0001.
The anchor is therefore effectively neutral at initialization on these probes,
unlike the older-teacher control. Losses4.551483/4.280774 and gradient norms
53.127834/43.288879 are finite. Frozen synthetic teacher, visible-mask equality,
fixed.5/2000 smoothing and two replay windows all verify. This establishes
initial consistency, not a benchmark win or a full-training preservation claim.

The receipt checks all16 unchanged sources and62 preserved arguments; only
the teacher checkpoint changes. GPU peak125.578814GiB and host77.356819GiB
pass limits.8.863GiB free preserves the8GiB reserve after budgeting initial
and temporary checkpoint writes. No real data is loaded in preflight.

The registered matched-teacher200-update training now starts at
PID1558880/tool6791, with first-update watcher30038. Initialization and teacher
are the same independently reproduced synthetic full-context update200 file;
all other training/prior/architecture/inference settings remain fixed. No outer
evaluation is authorized, and the full goal remains active.

The first matched-teacher update passes its receipt. Initial inner metrics
exactly reproduce the eligible checkpoint; all16 source pairs and preflight/
training arguments match. The teacher and initialization are the same verified
synthetic checkpoint. First loss7.677726, gradient norm33.653976 and pre-update
anchor MSE6.7123e-15 are finite. Fixed smoothing, exact visible targets and two
replays verify. Selected step0 serialization remains SHA
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
Watcher10593 observes validation50 while PID1558880/tool6791 continues.

### Matched-teacher validation50: ranking improves, rating guards still fail

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7483889273 | .7453467606 |
| RMSE | .8815765602 | .8461743473 |
| AUC | .8437793345 | .8669502977 |
| MAP | .0702261331 | .0541438582 |
| NDCG@10 | .0852756617 | .0745977304 |
| Precision@10 | .0526894866 | .0373681865 |
| Recall@10 | .0959947478 | .0967616358 |

Relative to the old-teacher soft-choice update50, Netflix NDCG improves
.0637525→.0745977 and Recall.0833769→.0967616, but MAE/RMSE worsen to
.7453468/.8461743. MovieLens ranking also improves while rating errors rise.
Only MovieLens MAE, RMSE and NDCG pass:2/4 MF guards and1/4 BPR targets.
Key`[1,1.0109571318386883,1.1124819656498566]` is rejected; step0 remains selected.

The observation receipt verifies all16 source pairs, frozen inference/baselines,
same synthetic initialization/teacher, canonical0/50 keys, selected bytes and
all six logged synthetic updates with exact visible targets and two replays.
Saved metrics reproduce without inference/calibration; macro user-MSE rises
.00841227/.01859480 versus initialization. The200-update schedule continues
at PID1558880/tool6791; watcher47121 now targets validation100. No outer run.

### Conditional-personalized top-k union prepared as an isolated target control

An inventory of current campaign training configs finds no trial combining
native user/item-specific conditional relevance with top-k union. Conditional
trials used all candidates; union trials used globally shared visible-frequency
targets. The earlier shared synthetic reference shows counts outperform the
incumbent, but explicitly does not establish compatibility with rating accuracy.

Register an optional`full_context_conditional_union_anchor50_200` against the
completed hard-visible-union anchor50 control. Change only the relevance target
to native synthetic conditional held-out-positive probability. Keep the hard
objective and that comparison's original synthetic rating teacher, deliberately
isolating one behavioral target change rather than mixing in the live
matched-teacher change. The newly available half-count default is inactive for
the hard objective. No production source or live run is changed.

All78 focused CPU tests pass, including six paired-prior cases proving every
episode tensor/metadata field except relevance targets remains exact, with
identical protected rating/teacher losses and gradients but different ranking
gradients. All108 frozen legacy cases reproduce bitwise again. No DWIP weights
or real training data are loaded for these tests. Latent-target uncertainty
remains and the combination may fail; no benchmark improvement is inferred.

This optional control has not started. Finish/audit the live matched-teacher
run first; prioritize independent verification if its selected candidate passes
all eight comparisons. Otherwise require new full-size preflights/source and
argument receipts. Conditional targets are not visible frequencies, so a future
receipt must not claim visible-mask teacher equality. The goal remains unmet.

### Matched-teacher validation100 rejected; continue the full schedule

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7500633152 | .7487701534 |
| RMSE | .8824926333 | .8490408100 |
| AUC | .8426084686 | .8717365113 |
| MAP | .0722448969 | .0557520097 |
| NDCG@10 | .0876023081 | .0769398271 |
| Precision@10 | .0525672372 | .0379771276 |
| Recall@10 | .0968104714 | .0990265603 |

Ranking improves modestly from50, but rating errors worsen on both datasets.
Only MovieLens MAE, RMSE and NDCG pass, still2/4 MF guards and1/4 BPR targets.
Key`[1,1.0156004785077635,1.0879137371726766]` is rejected and selected step0
remains unchanged. No trained checkpoint is promoted.

The observation receipt verifies all16 current/archived sources, frozen
inference/baselines, matching synthetic initialization/teacher, canonical
0/50/100 keys and all11 logged synthetic updates with visible-target equality
and two replays. Saved metrics reproduce without inference/calibration. Macro
user-MSE rises.00805831/.02287569 versus initialization despite mean prediction
shifts -.00357298/-.01110920; do not fit or apply a real-data correction.

The registered validations150/200 and full200-update schedule continue at
PID1558880/tool6791. Watcher79926 now awaits terminal status for the completed
audit; its two-hour observation deadline does not stop training. The optional
conditional-union control remains unlaunched. No outer evaluation runs.

The conditional-union preflight verifier is prepared separately from all16
live training sources. It distinguishes the pending matched-teacher prerequisite
from the older completed hard-union comparison, and correctly refuses while
the prerequisite has no terminal completion receipt. It will check original
teacher preservation, the single behavioral target change, inactive half-count
default, native conditional metadata, legacy equivalence and both full-size
replays. It explicitly rejects a spurious visible-mask equality claim. No new
GPU preflight or training is started by this CPU-only refusal check.

### Matched-teacher200 completes without promotion; conditional-target preflight starts

All200 registered updates finish. Validation150 and200 both pass only MovieLens
MAE, RMSE and NDCG, as at50/100. Final MovieLens MAE/RMSE are
.7510446777/.8831323424 and NDCG/Recall.0871547661/.0966486664; Netflix
.7491426344/.8493600986 and .0746576798/.0972695281. No trained checkpoint
passes the joint goal or replaces the eligible initialization.

The campaign audit and completion receipt verify all16 live/archived sources,
frozen architecture/inference/baselines, all five selection keys, all21 logged
synthetic updates, exact visible targets and two replays. Selected step0's
254 tensors exactly equal initialization. Saved150/200 diagnostics reproduce
without inference/calibration. Only after terminal audit, its byte-identical
step0 blob is deduplicated while all paths and other artifact hashes remain
unchanged; free space becomes8.886GiB. Zero new outer evaluations.

The conditional-union prerequisite is now met. Its fresh preflight starts at
PID1571152/tool52832 after archiving16 sources. Compare to the completed hard
visible-union anchor50 control, preserving that comparison's original frozen
synthetic rating teacher and changing only the ranking target to native
user/item-specific conditional relevance. The half-count default is inactive
for the hard objective. No production source, architecture or inference changes
are made. Require active finite ranking gradients, native conditional metadata,
two replays and exact source/argument receipts before training; visible-mask
teacher equality is inapplicable. The full goal remains active and unmet.

### Conditional-target full-size checks pass; registered training starts

Both full-size modes pass. Losses5.860647/5.041301 and gradient norms
56.302784/39.746510 are finite, with active hard-union ranking pairs, native
conditional synthetic metadata, frozen original rating teacher and two verified
replays. The receipt checks all16 sources and61 preserved original arguments;
only the relevance target changes behaviorally versus the completed hard
visible-union anchor50 comparison. The half-count argument is inactive.

GPU peak125.592375GiB and host77.354923GiB pass limits.8.885GiB free preserves
the8GiB reserve after initial/temporary checkpoint budgeting. No real data is
loaded in preflight, and no visible-frequency equality is claimed.

The registered200-update conditional-target run starts at PID1571341/tool72196;
first-update watcher23948 checks it against the successful preflight. All other
prior/training settings and public architecture/inference remain unchanged.
This is readiness evidence only, not a metric win or goal completion.

The first conditional-target update now passes its receipt: initial inner
metrics exactly reproduce the eligible checkpoint, all16 live/archive/preflight
sources match, and training arguments differ from preflight only by action/output.
Synthetic initialization and the original frozen rating teacher verify. First
loss6.058582 and gradient norm71.507225 are finite; native conditional targets,
active hard-union ranking and two replays verify, without a visible-frequency
equality claim. Step0 serialization remains SHA
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
Watcher71126 observes validation50 while PID1571341/tool72196 continues.

### Conditional-target validation50: less rating damage, still rejected

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7465932485 | .7390645703 |
| RMSE | .8792687838 | .8395555720 |
| AUC | .8322165218 | .8427967221 |
| MAP | .0671948150 | .0462902428 |
| NDCG@10 | .0814424978 | .0652449931 |
| Precision@10 | .0507334963 | .0331055993 |
| Recall@10 | .0909503882 | .0849483615 |

Compared with the registered hard-visible-union anchor50 control at50, Netflix
MAE/RMSE improve.742938/.843363→.739065/.839556 and NDCG/Recall improve
slightly. Nevertheless Netflix loses both MF guards and both BPR targets, while
MovieLens Recall remains below BPR. Only2/4 MF guards and1/4 BPR targets pass.
Key`[1,1.0030426302493933,1.2239254510693425]` is rejected; no promotion.

The observation receipt reads the actual selected payload rather than assuming
step0, verifies its state signature/finiteness and confirms that all254 tensors
still equal initialization. It checks all16 sources, frozen inference/baselines,
canonical0/50 keys and all six logged synthetic updates with active conditional
ranking and two replays. No visible-mask equality is claimed. Saved metrics
reproduce without inference/calibration; macro user-MSE rises.00315295/.00550375
versus initialization. The200-update schedule continues at PID1571341/tool72196;
watcher73645 targets validation100. Zero new outer evaluations; goal unmet.

### Optional personalized-exposure prior prepared without changing live training

The isolated sampler and preregistration for
`full_context_conditional_union_affinity1to3_anchor50_200` strengthen only the
synthetic exposure affinity coefficient from U(0,1) to U(1,3), preserving density
recalibration and the current conditional-union optimization/teacher recipe.
All16 live sources remain unchanged. No trainer/CLI integration is deployed,
and no new GPU preflight or training has started.

Seventeen CPU staging tests pass, including six default tensor/metadata cases
that reproduce production exactly. An immutable six-case synthetic episode
reference was captured before deployment for a nontrivial legacy check later.
The CPU-only latent-oracle probe uses no neural weights, gradients, real data
or CUDA. Across12 smaller paired cases, rating-oracle expected-DCG efficiency
improves.21471→.33401 and positive-mass recall.08205→.18005, with11/12 DCG
improvements. These are not benchmark NDCG/Recall scores. Different generated
masks/crops/users, one regression, lower local support density and weaker
visible-count utility limit the evidence; neural transfer remains unproven.

The current200-update run must finish and be audited first. If its selected
candidate passes all8 comparisons, independent inner/Cornac reproduction takes
priority. Otherwise complete isolated integration/tests, deploy only after the
live process terminates, reproduce frozen default/legacy references, and require
fresh full-size two-mode synthetic preflights with exact source/argument receipts.
Preserve the original frozen synthetic rating teacher, architecture/inference,
single-GPU lock and8GiB reserve. Zero new outer evaluations; goal remains unmet.

### Exposure-affinity trainer/CLI integration passes isolated checks

The staged trainer now forwards the optional affinity range and the staged BPR
CLI exposes `--exposure-affinity-low/high`, defaulting to the original0/1 range.
Invalid nonfinite, negative or reversed ranges fail before model/data loading.
No production source is edited while PID1571341 continues the registered run.

The combined44-test CPU suite passes. It includes the original17 sampler checks,
all six immutable default episode references, old-namespace compatibility, both
context modes through the joint-window training path, CLI forwarding, and finite
active conditional-ranking gradients on synthetic dummy logits. Tests forbid
real-data loading/neural inference and verify that CUDA never initializes.
AST checks constrain staged edits to sampler, training-episode wiring and CLI/
argument validation; every shared inference helper and all16 live sources match.
This is integration evidence, not a trained-model or benchmark improvement.

`exposure_affinity_prior_integration_receipt.json` records source/test hashes.
The standalone default-reference verifier also reproduces all six cases, and
the prospective preflight verifier correctly refuses to proceed without the
current run's completed terminal audit. Its future admission gate preserves
all optimization settings, verifies exactly three allowed source changes,
requires independent default/legacy reference receipts, full-size two-mode
synthetic preflights, original frozen teacher, active conditional ranking and
two replays, exact snapshot/arguments, campaign lock and checkpoint headroom.
No new GPU job or outer evaluation starts. Goal remains active and unmet.

### Conditional-target validation100: still fails the joint gate

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7478434842 | .7402226307 |
| RMSE | .8800300272 | .8403235895 |
| AUC | .8298057037 | .8452244971 |
| MAP | .0679398797 | .0472111340 |
| NDCG@10 | .0820083331 | .0661745792 |
| Precision@10 | .0501222494 | .0330016337 |
| Recall@10 | .0906689363 | .0856528343 |

Versus50, NDCG improves slightly on both datasets and Netflix Recall increases,
but both datasets' rating errors rise and MovieLens Recall declines slightly.
Only MovieLens MAE/RMSE/NDCG beat the frozen references:2/4 MF guards and1/4 BPR
targets. Key`[1,1.0040069767083963,1.2155249017490535]` is rejected; no promotion.

The reusable observation audit first reproduces the existing50 observation,
then verifies100: all16 live/archive sources, frozen inference/baselines,
synthetic checkpoint/teacher ancestry, all canonical validation keys and11
logged synthetic conditional-ranking updates with two replays. It reads the
actual selected checkpoint, checks signature/finiteness and confirms its254
tensors remain equal to initialization at step0. No visible-frequency equality
is asserted. Saved-error diagnostics reproduce without inference/calibration;
macro user-MSE increases.00346183/.00625117 versus initialization despite
negative mean prediction shifts on both datasets.

Training PID1571341/tool72196 continues the unchanged200-update schedule;
watcher58587 observes validation150. The stronger exposure-affinity prior is
staged only and cannot replace this live trial. Zero new outer evaluations;
the incumbent retains all four MF rating wins, but the full goal remains unmet.

### Stronger-exposure prior generates valid full-size CPU episodes

With the current GPU trial still live, a CPU-only staged probe tests the exact
training prior geometry:2000×2000 context,12288×4096 parent,4096 rating queries
and64 ranking users with2000 negatives. Three seeds and both context modes yield
12 episodes/six baseline-versus-stronger pairs. The range remains the registered
U(1,3); no optimization or production setting changes.

Latent-rating-oracle expected-DCG efficiency increases.18506→.23344 and positive-
mass recall.03021→.05772. Five of six paired DCG values improve, but one falls
by.05512. Visible-count efficiency drops.71867→.27689 and average local support
density.07821→.05534. These are synthetic oracle diagnostics, not DWIP scores,
benchmark NDCG/Recall, neural learnability or proof of transfer. Masks/crops/users
can differ across paired priors; the counterevidence remains explicit.

All episodes retain detached finite native conditional targets and64 ranking
users, with density/rank/noise/holdout draws preserved. Peak CPU RSS3.1642GiB;
no neural weights, neural inference/gradients, real data or CUDA initialization.
All16 live sources and frozen inference verify unchanged. The receipt is
`exposure_affinity_full_size_cpu_receipt.json`. Full-size GPU preflight remains
mandatory after the running conditional trial terminates and is audited.
The previous goal turn made progress; this turn adds full-size prior evidence
while training continues. Zero new outer evaluations; goal remains unmet.

### Conditional-target validation150 rejected; registered completion remains pending

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7481106263 | .7408102750 |
| RMSE | .8799255566 | .8407926218 |
| AUC | .8308177201 | .8476853050 |
| MAP | .0683870288 | .0477772587 |
| NDCG@10 | .0822830517 | .0672224663 |
| Precision@10 | .0502444988 | .0336402792 |
| Recall@10 | .0905900994 | .0876124302 |

NDCG on both datasets and Netflix Recall improve versus100, but Netflix rating
errors rise and MovieLens Recall falls slightly. Only2/4 MF guards and1/4 BPR
targets pass. Key`[1,1.0048040328068126,1.2018884767734217]` is rejected. The
actual selected checkpoint remains step0 with all254 initialization tensors
unchanged; no promotion. The observation audit verifies all16 live/archive
sources, frozen inference/baselines, canonical keys and16 logged synthetic
conditional-ranking updates with two replays. Saved-error diagnostics reproduce
without new inference, training or calibration.

The exposure-affinity deployment patch is additionally dry-run against temporary
copies: all three outputs exactly match the44-test staging hashes. Production
remains untouched. Training PID1571341/tool72196 continues; watcher62550 now
targets terminal status, not a premature stop. Only after completion/audit and
the joint-gate decision may the staged prior proceed to deployment and full-size
GPU preflight. Zero new outer evaluations; goal remains active and unmet.

### Conditional200 completes without promotion; stronger prior deployed

The registered run exits0 after200 updates. Final MovieLens MAE/RMSE are
.7479932825/.8797668733 and NDCG/Recall.0826514477/.0914197234; Netflix values
are.7402932168/.8403074111 and.0647963087/.0847950044. All trained validations
pass only2 MF guards and1 BPR target. The actual selected step0 remains exactly
equal to the254-tensor initialization; no promotion. Campaign and completion
audits verify all21 synthetic conditional-ranking logs, both window replays,
all16 sources, original teacher, frozen inference/baselines and canonical keys.
Final saved-error diagnostics reproduce without new inference/calibration.

Only after process termination and successful audit, the duplicate completed
step0 blob is safely hard-linked to the byte-identical completed control. Both
paths and every other artifact stay unchanged; receipt5 restores8.875GiB free.
No live checkpoint is deduplicated.

The preregistered stronger-exposure prior is then deployed via the dry-run-tested
patch, changing exactly three training sources. The new permanent prior tests
and the complete recsys/BPR CPU suite pass:540 tests. Six immutable predeployment
default episodes match every tensor/metadata value exactly;108 frozen legacy
loss/diagnostic/component/gradient cases also reproduce exactly. The remaining
13 training sources, shared inference helpers, architecture and baselines stay
unchanged. No real data enters neural training.

Fresh full-size two-mode synthetic preflight now runs at PID1582186/tool87154
for`full_context_conditional_union_affinity1to3_anchor50_200`. The only behavioral
change is the registered exposure-affinity prior U(0,1)→U(1,3). Initialization,
original synthetic rating teacher anchor50 and all optimization settings match
the completed conditional-union control. Exact source/argument and memory gates
must pass before training starts. Zero new outer evaluations; goal remains unmet.

### Stronger-exposure full-size preflight passes; synthetic training starts

Both context modes pass: losses5.892203/4.959862, gradient norms69.081245/38.543087
and exposure-affinity coefficient2.584229 are finite and in the registered range.
Native conditional ranking remains active; the original synthetic rating teacher
is frozen and both window-gradient replays verify. The admission receipt checks
all16 sources and every original optimization argument, with exactly the two
registered prior endpoints added. No visible-frequency equality is claimed.

Peak GPU125.68732GiB/host77.35630GiB pass limits. Free8.87541GiB covers the8GiB
reserve plus initial/temporary checkpoint writes. No real data is loaded during
preflight. The registered200-update trial starts at PID1582427/tool85289; watcher
65035 targets the first synthetic update and initial-validation reproduction.
Architecture, inference, initialization and original teacher remain fixed.
This is training readiness, not a benchmark win. Zero new outer evaluations;
the full eight-comparison goal remains active and unmet.

The first stronger-affinity update now passes its audit. Initial inner metrics
reproduce the eligible incumbent exactly, and the saved step0 has all254
initialization tensors with the unchanged state signature. All16 live/archive/
preflight sources match; train/preflight arguments differ only by action/output.
Synthetic initialization and the original frozen rating teacher verify.
First loss5.834102, gradient norm66.232468 and active ranking3.772094 are finite;
the generated affinity coefficient1.014577 lies within[1,3], native conditional
targets verify and both window replays pass. No visible-frequency equality is
claimed. The checkpoint serialization SHA remains
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
PID1582427/tool85289 continues; watcher64955 now targets validation50. These
are integrity/readiness checks, not new benchmark wins. Goal remains active.

### Optional exposure–item-quality coupling prepared during the live affinity run

The prior currently has independent popularity noise plus a comparatively weak
item-bias term in its synthetic exposure logits. An isolated optional control
strengthens only that item-bias coefficient from U(0,1.5) to U(1.5,4.5), preserving
the active affinity U(1,3), density calibration, RNG draw order and every other
generation/optimization setting. It is preregistered against the current trial
as`full_context_conditional_union_affinity1to3_quality1p5to4p5_anchor50_200`.
All16 live sources remain unchanged; nothing is deployed or trained for this
optional preparation.

A full-size CPU-only synthetic probe uses three seeds and both context modes:
12 episodes/six pairs at2000×2000 context and12288×4096 parent. Latent-rating-
oracle expected-DCG efficiency increases.23344→.43967 and positive-mass recall
.05772→.13269; visible-count efficiency increases.27689→.37453. Both scorer
efficiencies improve in all six pairs. These are not neural scores or benchmark
NDCG/Recall: different masks/crops/users can enter each variant, average local
support density declines.05534→.04991, and learnability/transfer remain unproven.
No real data, neural weights, neural inference/gradients or CUDA are used in the
probe; peak CPU RSS3.1611GiB. All six baseline episode hashes and utilities exactly
reproduce the independently saved earlier full-size affinity-prior cases.

Twenty-six sampler tests pass before integration. Twelve immutable default
episode references span both affinity ranges0/1 and1/3, two context modes and
three seeds. The staged trainer/CLI adds opt-in quality endpoints defaulting to
0/1.5, with early rejection of invalid ranges. The combined59-test CPU suite
passes, including frozen references, old namespaces, joint-window argument
forwarding, both modes' finite synthetic dummy-logit ranking gradients, and
explicit refusal of real-data/model access or CUDA initialization. AST checks
limit edits to sampler, training-episode forwarding and CLI/argument validation.
A temporary-copy deployment dry-run exactly reproduces all three staged hashes.

The current affinity200-update trial must finish and be audited first. If its
selected candidate passes all8 comparisons, independent inner/Cornac reproduction
takes priority. Otherwise any deployment must pass frozen default/legacy checks,
fresh full-size two-mode GPU preflights and exact source/argument/memory receipts,
with the original synthetic teacher, single-GPU lock and8GiB reserve preserved.
The previous goal turn made progress; this turn adds a synthetic prior hypothesis
with measured full-size evidence and tested integration. Zero new outer
evaluations; the full goal remains active and unmet.

The standalone quality-prior default verifier also reproduces all12 immutable
cases. Its prospective preflight verifier correctly refuses to proceed while
the affinity trial lacks a completed terminal audit. The future gate checks
both affinity and quality coefficient ranges, exact preservation of existing
arguments, the three staged source changes, frozen default/legacy references,
original teacher, conditional targets, two replays and memory/headroom. No
production source, live training recipe or evaluation protocol is changed.

### Stronger-affinity validation50: less rating damage, no BPR target wins

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7462900270 | .7385520609 |
| RMSE | .8786471896 | .8389538068 |
| AUC | .8328176758 | .8441396471 |
| MAP | .0655231863 | .0451646102 |
| NDCG@10 | .0796142108 | .0633273270 |
| Precision@10 | .0502444988 | .0322738749 |
| Recall@10 | .0904758730 | .0832810421 |

Compared with the registered conditional-union control at50, all four rating
metrics improve slightly but all four NDCG/Recall targets decline. Both Netflix
MF guards still fail, and MovieLens NDCG no longer beats BPR. Only2/4 MF guards
and0/4 BPR targets pass. Key`[1,1.0023236830362352,1.248289188543926]` is rejected;
the actual selected checkpoint remains the254-tensor initialization at step0.
No promotion is made.

The observation receipt verifies all16 live/archive sources, frozen inference
and baselines, canonical0/50 keys, synthetic lineage, active conditional ranking
and both replays in all six logged updates. It checks the nondefault affinity
range against each logged coefficient. Saved-error diagnostics reproduce
without new inference/calibration; macro user-MSE rises.00173923/.00399376 versus
initialization. The source/preflight/audit evidence continues to enforce
synthetic-only neural training; no real values are fitted into predictions.

Training PID1582427/tool85289 continues the registered200-update schedule;
watcher2771 now targets validation100. The optional quality-coupling prior stays
staged, with its terminal-audit admission gate intact. Zero new outer evaluations;
the incumbent retains all four MF wins and the complete goal remains unmet.

### Proposed quality prior passes density-boundary readiness checks

A separate CPU-only stress probe checks that extending exposure coefficients
does not silently saturate the existing[-20,20] density-calibration bracket.
Sixteen full-size episodes use maximal proposed affinity3/quality4.5/popularity
spread2.8, both density endpoints.004/.08, Gaussian/clustered families and four
fixed seeds. Every case has finite detached exposure/relevance probabilities,
retains bracket margin and attains the requested mean exposure probability
within6.52671e-8 absolute error (required tolerance1e-6).

The probe observes sampler-local calibration values using a temporary Python
return hook; neither staged sampler nor live production is edited. No neural
weights, neural inference/gradients, real data or CUDA are used, and all16 live
sources verify unchanged. This finite stress sample is not an exhaustive bound,
neural learnability result or benchmark improvement. The optional quality prior
remains undeployed; the current affinity run and its registered schedule continue.
The previous goal turn made progress; this adds prior-readiness evidence without
weakening the eight-comparison goal or its synthetic-only constraint.

### Stronger-affinity validation100: Netflix ranking regresses; no promotion

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7471392959 | .7392151772 |
| RMSE | .8790878849 | .8390998874 |
| AUC | .8322009638 | .8437160593 |
| MAP | .0661021037 | .0418740515 |
| NDCG@10 | .0808502366 | .0572811627 |
| Precision@10 | .0503667482 | .0287836031 |
| Recall@10 | .0920279065 | .0756602434 |

All four rating errors increase versus50. MovieLens ranking improves, but NDCG
remains strictly below BPR's.0809004866; rounding does not create a win. Netflix
NDCG/Recall regress, with Recall below the initialization's.0772930029. Only2/4
MF guards and0/4 BPR targets pass. Key`[1,1.002640508910905,1.3123477575323688]`
is rejected. The actual selected step0 still equals all254 initialization tensors.

The observation audit verifies all16 live/archive sources, frozen inference and
baselines, canonical0/50/100 keys, synthetic lineage, affinity ranges, active
conditional ranking and both replays in all11 logged updates. Saved-error
diagnostics reproduce without inference/calibration; macro user-MSE rises
.00130248/.00360534 versus initialization despite negative mean prediction
shifts. No candidate is promoted and no new outer evaluation runs.

PID1582427/tool85289 continues the registered200-update schedule; watcher11864
targets validation150. The optional quality prior is staged only. Its completed
CPU integration/default checks and finite density-boundary stress sample do not
replace the required post-termination audit, full-size GPU preflight or benchmark
gate. The complete goal remains active and unmet.

### Stronger-affinity validation150: still rejected; awaiting registered completion

| Metric | MovieLens100K | Netflix small |
|---|---:|---:|
| MAE | .7477474301 | .7395052519 |
| RMSE | .8794929434 | .8395274295 |
| AUC | .8330380162 | .8461728802 |
| MAP | .0658470063 | .0431735512 |
| NDCG@10 | .0801476226 | .0593929223 |
| Precision@10 | .0502444988 | .0297192930 |
| Recall@10 | .0915966270 | .0783863909 |

All four rating errors rise versus100. Netflix ranking partially recovers while
MovieLens ranking declines, but all four BPR targets and both Netflix MF guards
still fail. Only2/4 MF guards and0/4 BPR targets pass. The canonical key
`[1,1.00303395403856,1.288478509386367]` is rejected; actual selected step0 still
equals all254 initialization tensors. No candidate is promoted.

The observation audit checks all16 live/archive sources, frozen inference and
baselines, canonical0/50/100/150 keys, synthetic lineage, affinity ranges, active
conditional ranking and two replays in all16 logged updates. Saved errors
reproduce without new inference/calibration; macro user-MSE increases
.00229570/.00569048 versus initialization despite negative mean prediction shifts.

PID1582427/tool85289 continues the registered200-update schedule. Watcher52287
now observes terminal status; an observation timeout would not authorize a
restart or stop. The quality prior remains staged until completion/audit and
the eight-comparison decision. Zero new outer evaluations; goal remains unmet.

### Affinity200 completes without promotion; quality-coupling prior deployed

The registered affinity1to3 schedule completes200 updates with exit0 and a
passing terminal campaign audit. Validation200 has MAE/RMSE
.7478275953/.8795875895 on MovieLens and .7392221126/.8393291282 on Netflix;
NDCG@10/Recall@10 are .0808882889/.0934643260 and .0580621846/.0764647357.
Only2/4 MF guards and0/4 BPR targets pass, including the strict MovieLens NDCG
comparison despite similar rounded values. All trained validations fail the
joint gate. Selected step0 retains all254 initialization tensors; no promotion.

The completion receipt verifies all21 logged synthetic updates, active native
conditional ranking, original teacher anchor50, affinity[1,3], two window
replays, all16 sources and frozen inference/baselines. Deduplication receipt6
preserves all paths/artifacts while hard-linking only the completed byte-identical
step0 checkpoint. Free disk after deduplication is8.857GiB.

With the prerequisite complete and no all8 candidate awaiting reproduction,
the registered quality deployment patch now changes only the sampler,
training-episode forwarding and CLI validation. All three production hashes
match the tested staging receipt. The synthetic exposure item-quality coefficient
changes from U(0,1.5) to U(1.5,4.5); affinity[1,3], architecture, public inference
and every optimization setting remain unchanged. Permanent production tests,
12-case frozen default verification and108-case legacy loss/gradient verification
must pass before a fresh two-mode full-size GPU preflight and training.
No new GPU trial has started at this entry. Zero new outer evaluations; the
incumbent still retains all four MF wins and the BPR goal remains unmet.

### Quality prior passes production compatibility gates

The permanent quality-prior regression suite passes41 tests; the full BPR/recsys
suite passes581 tests. The immutable12-case default reference reproduces every
tensor/metadata value exactly across both context modes and affinity ranges.
All108 frozen legacy loss/diagnostic/component/gradient cases remain exact.
Deployment receipt and the16-source preflight snapshot record all hashes.

A fresh full-size two-mode synthetic GPU preflight starts as PID1598199/tool96823.
No training starts until its quality-aware admission verifier passes. Separate
first-update and scheduled-validation helpers now check both coefficient ranges;
historical helpers and their receipt hashes remain unchanged. No new real-data
inference or outer evaluation is performed in these CPU checks or GPU preflight.

### Registered quality-coupling training starts after GPU admission

Both full-size preflight modes pass with finite active conditional ranking and
two verified gradient-window replays. Losses are3.237393/3.319711 and gradient
norms36.31834/34.93383. Both synthetic episodes have affinity2.58423 and quality
2.95454 within the registered ranges. Peak allocated GPU memory125.663GiB and
host RSS77.360GiB pass the existing limits; disk headroom preserves8GiB reserve
including initial/temporary checkpoint writes. The admission receipt verifies
all16 sources, unchanged inference/baselines, original frozen synthetic rating
teacher and exact preservation of every optimization argument.

Training PID1598460/tool14965 now runs the registered200-update schedule with
validation every50 and save-best only. No inference changes, real-data gradient
updates, calibration or outer evaluation are authorized. Initial inner
reproduction and the quality-aware first-update audit are pending. This is a
new hypothesis under test, not a benchmark win or completed goal.

### Quality-coupling first update and initial inner reproduction pass

The quality-aware first-update receipt passes. Initial inner scores exactly
reproduce the incumbent on both datasets, and selected step0 retains all254
initialization tensors. It verifies all16 live/training/preflight source hashes,
exact preflight-to-training arguments, frozen public inference/baselines,
synthetic ancestry and original frozen teacher. The first logged update has
finite active conditional top-k ranking, affinity[1,3], quality[1.5,4.5], teacher
anchor50 and two verified replays. No real data enters gradient updates.

PID1598460/tool14965 continues200 updates. Watcher86807 observes validation50,
then runs saved-error diagnostics and the quality-aware validation audit without
new inference. Watcher timeouts never authorize stopping or restarting training.
The goal remains active and unmet; initial scores retain4/4 MF wins and0/4 BPR
wins. No candidate has been promoted and zero new outer evaluations have run.

### Synthetic factorial probe exposes an exposure-prior tradeoff

While the registered quality-coupling run remains live and untouched, a CPU-only
probe halves the independent exposure-popularity scale from[1.4,2.8] to[.7,1.4].
Six current-prior episodes exactly reproduce the independently archived quality
probe, including tensor hashes and utilities. A second probe reuses these12
episodes and generates12 additional episodes with affinity[0,1], completing a
2x2 factorial at the full2000x2000 context/12288x4096 parent geometry. Quality
coupling remains[1.5,4.5]; shared density/rating/noise/holdout draws are preserved.

| Affinity / independent popularity | Rating-oracle DCG efficiency | Count DCG efficiency | Oracle positive-mass recall | Local support density |
|---|---:|---:|---:|---:|
| Current[1,3] / [1.4,2.8] | .439673 | .374532 | .132690 | .049909 |
| [1,3] / [.7,1.4] | .675495 | .291601 | .162019 | .068238 |
| [0,1] / [1.4,2.8] | .330799 | .801458 | .090432 | .040439 |
| [0,1] / [.7,1.4] | .624932 | .785079 | .117153 | .057028 |

Reducing independent popularity alone improves oracle efficiency in6/6 pairs
but weakens count utility in6/6. Reducing affinity alone reverses that tradeoff.
The combined variant improves mean oracle and count efficiency; count improves
6/6, oracle only4/6, and oracle positive-mass recall declines. It is not a clear
win on every synthetic diagnostic. Lower affinity also removes personalized
heterogeneity; masks, crops, users and local support density differ.

These are latent-oracle/count-reference results, not DWIP inference or real
recommendation metrics. Peak CPU RSS stays below3.16GiB, CUDA is uninitialized,
no real data/neural models/gradients are used, and all16 active sources plus
frozen inference remain unchanged. The probe identifies a possible future
prior tradeoff, not a selected recipe: finish the current registered trial and
prioritize any all8 winner. No further GPU preflight/training or outer evaluation
is started by this diagnostic. Goal remains active and unmet.

### Quality-coupling validation50 reduces rating damage but fails the joint gate

The scheduled trained-checkpoint inner metrics are:

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7458175319 | .7373272547 |
| RMSE | .8773354990 | .8373466967 |
| AUC | .8275935657 | .8309456198 |
| MAP | .0635396151 | .0378663799 |
| NDCG@10 | .0781731258 | .0525130495 |
| Precision@10 | .0491442543 | .0271201545 |
| Recall@10 | .0887522350 | .0695080315 |

Only2/4 MF guards and0/4 BPR targets pass. Netflix misses MF by.00005884 MAE
and.00033783 RMSE. The strict key[1,1.0004036195919273,1.4011870244702218]
is rejected; selected step0 remains identical to initialization. Both Netflix
ranking targets regress below initialization, while MovieLens ranking improves
but remains below BPR. No promotion is made.

The quality-aware observation receipt verifies all16 live/archive sources,
frozen inference/baselines, canonical selection keys, synthetic lineage,
affinity/quality coefficient ranges, active conditional ranking and both replays
in all six logged updates. Saved-error diagnostics reproduce all metrics without
new inference or calibration. MovieLens macro user-MSE declines.00109673;
Netflix rises.00084912. Their mean prediction shifts are-.00531977/+.00324131,
so this is not a shared global offset that can simply be removed.

PID1598460/tool14965 continues the registered200 updates. Watcher24878 now
targets validation100. No outer evaluations run and the joint goal remains unmet.

### Exposure-balance fallback registered and tested only in isolation

The factorial evidence motivates a conditional next prior: restore exposure
affinity[0,1],halve independent exposure-popularity sigma to[.7,1.4],and preserve
quality coupling[1.5,4.5]. Compare against the current quality trial and preserve
every optimization setting, initialization, original frozen synthetic teacher,
architecture and public inference. This changes only synthetic exposure generation;
it is not a real-data fit or a claim that lowering heterogeneity will transfer.

The existing sampler and training-episode forwarder already support popularity
ranges. Isolated trainer staging adds only two CLI flags and finite/nonnegative/
ordered argument validation.22 CPU tests pass: exact effective legacy arguments,
same default episodes in both modes, invalid ranges, zero/fixed ranges, CLI and
joint-window forwarding, finite active conditional-ranking gradients and AST
scope limited to `episode_arguments`/`main`. Tests explicitly load the production
reference by path to avoid accidentally comparing the staged module to itself.
No neural model, real-data loader or CUDA initialization is permitted in tests.

The patch dry-run uses only a temporary copy and exactly reproduces the tested
staged hash. All16 live sources and frozen inference remain unchanged. The new
admission verifier correctly refuses to proceed before the current quality200
completion/audit receipt exists. Restored default affinity metadata is absent by
design; future audits must verify registered arguments/source and that absence,
while checking logged popularity sigma[.7,1.4] and quality[1.5,4.5].

No production deployment or new GPU preflight/training has occurred. Finish and
audit the live trial first; independently reproduce any all8 winner before
considering fallback. Otherwise require production compatibility, immutable
default/legacy checks, fresh full-size two-mode preflights, exact source/argument
receipts and8GiB reserve. Current registered training is never changed or stopped
to make this preparation possible. Goal stays active and unmet.

### Quality-coupling validation100 rejected; Netflix ranking regresses further

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7465602089 | .7394699846 |
| RMSE | .8769425575 | .8384380321 |
| AUC | .8271638061 | .8290142264 |
| MAP | .0652213429 | .0349085439 |
| NDCG@10 | .0786951135 | .0464348555 |
| Precision@10 | .0491442543 | .0234962127 |
| Recall@10 | .0881448447 | .0610272332 |

Only2/4 MF guards and0/4 BPR targets pass. The key
[1,1.0029861190652924,1.5162184849380227] is rejected. Selected step0 remains
all254 initialization tensors; no checkpoint is promoted. Netflix rating and
ranking metrics worsen versus50. MovieLens RMSE/NDCG improve slightly, but its
MAE rises and Recall declines.

The observation receipt verifies all16 live/archive sources, frozen baselines
and inference, canonical0/50/100 keys, synthetic lineage, both exposure ranges,
active conditional ranking and two replays in all11 logged updates. Saved
metrics reproduce without new inference/calibration. Mean user-MSE declines
.00403424/.00052816 versus initialization while within-user centered error rises
.00130968/.00084019. These MSE decompositions do not override the actual
macro-MAE/RMSE guards; Netflix still fails both.

The saved Netflix profile shows mean top10 item support falling from1004.70
at initialization to825.09 at50 and706.20 at100. First1000-item share falls
.74903 to.59883 to.51897, while top10 ceiling share changes only.06651 to.05978.
This describes the ranking shift, not proof of causality or permission for
filtering/calibration. The staged exposure-balance settings were registered
from synthetic factorial evidence before this100-update result.

PID1598460/tool14965 continues the registered200-update schedule. Watcher56160
targets validation150. A quality-aware terminal verifier is prepared and its
live-process refusal checked; it cannot issue completion before terminal status,
tool exit, audit and all registered evidence. No production sources change,
no outer evaluation runs, and the goal remains active and unmet.

### Quality-coupling validation150 still fails; await terminal audit

MovieLens MAE/RMSE are.7467601215/.8771775926 and NDCG/Recall@10 are
.0804746061/.0886320059. Netflix MAE/RMSE are.7395893638/.8386252598 and
NDCG/Recall@10 are.0474886945/.0621912986. Ranking partly recovers versus100,
but all four rating errors rise. Only2/4 MF guards and0/4 BPR targets pass;
key[1,1.0031480399207926,1.489273132893287] is rejected. All254 selected tensors
remain initialization at step0; no candidate is promoted.

The scheduled observation audit passes for all16 live/archive sources, frozen
inference/baselines, canonical0/50/100/150 keys, synthetic lineage and all16
logged updates with conditional ranking, both prior ranges and two replays.
Saved-error diagnostics reproduce without model inference or fitted correction.

PID1598460/tool14965 continues to200. Terminal watcher75211 waits for process
exit and complete status before running the final saved-error diagnostic and
campaign audit. The completion receipt additionally requires the observed main
tool exit code and the quality-aware terminal verifier. The exposure-balance
fallback remains staged and undeployed until these gates pass and no all8 winner
requires independent reproduction. Zero new outer evaluations; goal unmet.

### Quality200 completes without promotion; exposure-balance CLI deployed

The full registered schedule exits0. Final inner metrics are:

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7469188079 | .7404504483 |
| RMSE | .8770557942 | .8392433530 |
| AUC | .8284737351 | .8309203087 |
| MAP | .0660659811 | .0351685126 |
| NDCG@10 | .0800445834 | .0468653078 |
| Precision@10 | .0507334963 | .0237338482 |
| Recall@10 | .0902404417 | .0608679093 |

Every trained validation50/100/150/200 passes only2/4 MF guards and0/4 BPR
targets. Final key[1,1.0043159789361595,1.5021240317166176] is rejected. Selected
step0 exactly retains all254 initialization tensors. Campaign audit passes;
its238 changed tensors refer to the older campaign root, not this initialization.
The terminal receipt verifies the complete schedule, main exit0, absent PID,
all21 synthetic conditional updates, affinity/quality ranges, original teacher,
two replays, all16 sources, exact registered arguments and frozen inference/
baselines. No all8 candidate exists; no promotion or outer evaluation occurs.

Completed-only deduplication receipt7 atomically hard-links the byte-identical
step0 checkpoint, retaining both paths and85 other artifacts unchanged. Free
disk rises from8.554 to8.844GiB. No live checkpoint is deduplicated.

With terminal gates satisfied, the exposure-balance deployment patch changes
only trainer CLI options and argument validation; its hash exactly matches the
22-test staged source. The sampler/forwarder already support the registered
prior ranges. Permanent popularity-range tests pass15 cases. Broader production
tests and immutable default/legacy verification run before a fresh full-size GPU
preflight. No new neural training starts at this entry. The original incumbent
still has all four MF wins and no BPR target wins; the goal remains unmet.

### Exposure-balance compatibility fix verified before GPU admission

The broader suite catches four failures in minimal legacy configurations that
omit popularity keys. The new range validator indexed these keys directly.
Replace only that lookup with non-mutating defaults.get fallbacks, matching the
other optional-prior validators. This does not change any registered experiment
argument or the generated prior. Add a permanent empty-config regression test.

Preserve all v1 staging/receipt artifacts and the failed production log. New v2
staging uses the immutable comparison source snapshot as its legacy reference;
23 staged tests and16 permanent focused tests pass. The full production BPR/
recsys suite passes597 tests. All12 frozen default episodes and108 legacy
loss/diagnostic/component/gradient cases reproduce exactly. The complete v2
patch dry-run against the original comparison source matches tested production.
Only CLI and argument-validation functions change; no architecture, sampler,
objective or shared inference helper changes. No GPU work preceded this fix.

### Exposure-balance full-size preflight passes; synthetic training starts

The two-mode preflight passes with losses1.653570/1.608682, gradient norms
11.31167/10.18029, finite active conditional ranking and two verified replays.
Popularity sigma.812578 and quality coefficient2.954537 fall inside the
registered ranges. Affinity[0,1] is the unchanged sampler default and correctly
omits optional affinity metadata; arguments/source and metadata absence verify.
Peak allocated GPU memory125.673GiB and host RSS77.306GiB pass. Free disk8.842GiB
preserves the8GiB reserve including initial and temporary checkpoint writes.

The v2 admission receipt verifies all16 sources, unchanged inference/baselines,
original frozen synthetic rating teacher, default/legacy compatibility and exact
preservation of all optimization arguments. PID1607996/tool34431 now runs the
registered200-update control with validation every50 and save-best only.
Only the synthetic exposure prior differs from the completed quality trial:
affinity[0,1],independent popularity spread[.7,1.4],quality coupling[1.5,4.5].

The first-update watcher85552 awaits initial inner reproduction and the
balance-aware provenance/range audit. No neural update uses real data, no
inference transformation is added and no outer evaluation starts. Synthetic
oracle/count alignment motivated this trial; neural transfer remains unproven.
The goal remains active and unmet.

### Exposure-balance first update and initial reproduction pass

The balance-aware first-update receipt passes. Initial inner metrics exactly
reproduce the eligible incumbent on both datasets; selected step0 retains all254
initialization tensors. All16 live/training/preflight sources and exact arguments,
frozen inference/baselines, synthetic ancestry and original frozen teacher verify.
The first update has finite active conditional ranking, default affinity metadata
absence, registered popularity/quality ranges and two verified window replays.

PID1607996/tool34431 continues the registered200-update synthetic schedule.
Watcher89124 targets validation50, then read-only saved-error diagnostics and
the balance-aware observation audit. Observation timeouts never authorize a
restart or stop. No trained-checkpoint benchmark win is yet established, no
candidate is promoted and zero new outer evaluations have run. Goal remains
active and unmet; the eligible incumbent still retains all four MF rating wins.

### Synthetic collaborative reference adds little over visible counts

A CPU-only diagnostic examines whether a fitted collaborative reference extracts
substantially more ranking information from the current balanced exposure prior.
It exactly reproduces six full-size synthetic episodes from the earlier factorial
probe, then fits isolated Cornac BPR references using only each visible mask.
The reference uses10 factors,200 iterations,LR.001,regularization.01,seed123 and
one CPU thread. No latent ratings, relevance targets, hidden queries or parent
counts enter fitting. All local IDs remain declared, including unsupported ones;
the unchanged BPR initialization/negative sampling handles them without filtering.

| Synthetic scorer | Expected DCG efficiency | Positive-mass recall |
|---|---:|---:|
| Visible item count | .785079 | .146658 |
| BPR reference | .785140 | .146845 |
| BPR item bias only | .785017 | .146865 |
| Latent expected-rating oracle | .624932 | .117153 |

BPR exceeds visible-count efficiency in only3/6 cases, and its own bias-only
ordering in4/6. This gives no compelling reason to add a collaborative fitting
stage. These conventional fits are isolated synthetic evaluation references,
never neural teachers/targets. The active neural training recipe is unchanged.

Nine CPU tests verify support-only fitting, rating-value invariance, unsupported
IDs, rejection of non-synthetic/invalid contexts and stable rank encoding. Raw
BPR score ranks are mapped into[1,4.5] only for this diagnostic utility helper,
avoiding that helper's clipping; no public DWIP prediction changes. BPR utility
uses float64 versus archived float32 count utilities, so tiny differences must
not be interpreted as robust gains. This six-case expected-utility diagnostic
is not a Cornac benchmark or evidence of neural transfer.

Fits/scoring take1.08–10.76 seconds per case and peak CPU RSS is3.237GiB.
No real data or neural model is loaded, CUDA remains uninitialized, and all16
live training sources plus frozen inference stay unchanged. PID1607996 continues
the registered run; its first trained validation remains pending. Goal unmet.

### Balanced conditional validation50 retains all MF guards, not BPR wins

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7456249395 | .7351856834 |
| RMSE | .8772072660 | .8355482287 |
| AUC | .8202875652 | .8125328657 |
| MAP | .0628702408 | .0333168503 |
| NDCG@10 | .0778661490 | .0468650963 |
| Precision@10 | .0476772616 | .0240308926 |
| Recall@10 | .0893344502 | .0605713057 |

All4 MF rating guards pass, but0/4 BPR targets pass. MovieLens ranking improves
versus initialization, while Netflix ranking regresses. The feasible but worse
key[0,2.031554726041974,1.5145952890156928] is rejected; selected step0 remains
all254 initialization tensors. No checkpoint is promoted merely for retaining
the rating guards.

The observation receipt verifies all16 live/archive sources, frozen inference/
baselines, canonical0/50 keys, synthetic lineage and all six logged updates:
conditional ranking, default-affinity metadata absence, popularity/quality
ranges, original teacher anchor50 and two replays. Saved metrics reproduce
without new inference or fitted correction. Mean user-MSE declines
.00105067/.00095529 versus initialization, while within-user centered error
changes+.00035906/+.00001308. Netflix top10 mean item support falls to725.64
and first1000-item share to.51718; these are descriptive, not permission for
calibration or candidate filtering.

PID1607996/tool34431 continues the registered200-update schedule. Watcher29111
targets validation100. A balance-aware terminal verifier is prepared and its
live-process refusal tested. Goal remains active and unmet; zero new outer
evaluations have run.

### Observable-target control registered without modifying live training

Conditionally register `full_context_balanced_visible_union_anchor50_200`.
Change only `ranking_relevance_target` from native conditional relevance to
`visible_item_frequency`, keeping the current balanced exposure prior and every
other optimization argument. Targets come from the exact generated visible
mask; no fitted BPR/MF reference becomes a neural teacher. The original frozen
synthetic rating teacher and anchor50 remain unchanged. Architecture and public
inference stay frozen; there is no count blending or score transformation.

The rationale is synthetic observability: current-prior visible counts match
the isolated BPR reference, whereas native latent targets still contain
unobserved variation. The current balanced trial also retains all rating guards
at50. Neither fact proves the new target will preserve guards while improving
BPR ranking; earlier visible-target controls on different priors damaged ratings.

Eleven CPU tests pass. Across three seeds and both modes, every nonrelevance
episode field is bitwise identical, the protected rating/anchor loss and logit
gradients are exact, and total ranking gradients are finite and changed. Tests
verify exact detached visible-mask targets, CLI forwarding, rejection of
tampered labels/real provenance, no neural/data loading and no CUDA initialization.
All108 frozen legacy cases remain bitwise exact. All16 active sources and frozen
inference are unchanged; the implementation already exists and no production
edit is required. The preflight verifier correctly refuses before current
200-update completion/audit.

No new GPU preflight or training starts. Finish/audit the live trial first and
prioritize independent reproduction of any all8 winner. Otherwise require the
registered target-only comparison, fresh full-size two-mode GPU checks, exact
visible-mask equality, default-affinity metadata absence, popularity/quality
ranges, original teacher, two replays and storage headroom. Goal stays unmet.

### Balanced conditional validation100 keeps MF wins, worsens Netflix ranking

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7456827089 | .7354345238 |
| RMSE | .8760269516 | .8350441168 |
| AUC | .8167551797 | .7979808768 |
| MAP | .0637504189 | .0264304688 |
| NDCG@10 | .0784460951 | .0350049578 |
| Precision@10 | .0482885086 | .0179414823 |
| Recall@10 | .0888715183 | .0441062195 |

All4 MF rating guards still pass and both RMSEs improve versus50, although both
MAEs rise slightly. No BPR target passes; Netflix NDCG/Recall regress further.
The key[0,2.7899449020234486,1.858149232438401] is worse than both50 and the
incumbent. Selected step0 remains all254 initialization tensors; no promotion.

The100-update observation audit passes: all16 live/archive sources, frozen
inference/baselines, canonical0/50/100 keys, synthetic lineage and all11 logged
updates with default affinity,popularity/quality ranges,active conditional
ranking,original teacher anchor50 and two replays. Saved errors reproduce
without new model inference or fitted correction. The observable-target
follow-up remains prepared only; no live source or optimization setting changes.

PID1607996/tool34431 continues the registered200 updates. Watcher68991 targets
validation150. Terminal watcher39416 waits for process exit and complete status
before final saved-error diagnostics/campaign audit. Only after observing the
main tool exit code may the prepared balance-aware completion verifier issue
its receipt. Any all8 winner takes precedence over the optional target control.
No new outer evaluations; goal remains active and unmet.

### Balanced conditional validation150 still fails the joint target

MovieLens MAE/RMSE are.7462400420/.8766545513 and NDCG/Recall@10 are
.0802569401/.0878854047. Netflix MAE/RMSE are.7358762317/.8354661115 and
NDCG/Recall@10 are.0348994820/.0449341217. All four rating errors rise versus100,
though all4 MF guards still pass. No BPR target passes. The key
[0,2.738540729409826,1.8446107689600235] remains worse than initialization;
selected step0 retains all254 tensors and no checkpoint is promoted.

The observation audit verifies all16 live/archive sources, frozen inference/
baselines, canonical0/50/100/150 keys, synthetic lineage and all16 logged updates
with the registered prior, native conditional ranking, original teacher anchor50
and two replays. Saved metrics reproduce without new inference or calibration.

PID1607996/tool34431 continues to200; terminal watcher39416 remains active.
First-update, scheduled-validation and completion helpers for the registered
visible-target follow-up are prepared and syntax-checked without touching live
sources. They require actual visible-mask equality, not the current conditional
target's inapplicable equality flag; the terminal helper verifies only the
registered target changes and all other optimization arguments remain fixed.
No new GPU experiment or outer evaluation starts before terminal admission.
The goal remains active and unmet.

### Balanced conditional200 completed; visible-frequency control ready

The registered balanced conditional run completed all200 updates with exit0.
Its terminal audit and completion receipt pass: all21 logged updates are
synthetic-only, all16 sources and frozen inference/baselines match, and the
original rating teacher, anchor50 and two window replays remain intact.
All trained validations retain4/4 MF guards but achieve0/4 BPR targets.
Final MovieLens MAE/RMSE are.7464115233/.8767449804 and NDCG/Recall@10 are
.0804583623/.0880462735. Netflix values are.7359017624/.8354833705 and
.0331175746/.0419495490. Final key[0,2.9333789061,1.9250659108] is rejected;
selected step0 equals all254 initialization tensors. No promotion or independent
winner reproduction is due. The generic audit's238 changed tensors compare
against the older campaign root, not this trial's immediate initialization.

Completed-only deduplication8 retains both checkpoint paths and all85 other
artifacts, recovering headroom from8.535751 to8.826408GiB. No live checkpoint
was deduplicated. A fresh combined CPU suite passes608 tests (597 production
plus11 preparation cases); all108 frozen legacy cases remain exact.

Proceed to the preregistered balanced visible-frequency control: change only
the synthetic ranking target, preserve the balanced prior and all other
optimization arguments, and require a new two-mode full-size GPU preflight
with the exact16-source snapshot before training. No production edits,
real-data gradients, inference corrections or new outer evaluations.

### Balanced visible-frequency GPU admission passes; synthetic run starts

Fresh full-size2000-row,two-window GPU checks pass in both context modes.
Peak allocated GPU/host memory is125.660/77.362GiB. Losses1.363289/1.370744
and gradient norms7.634030/8.790152 are finite; exact visible-mask labels,
original frozen synthetic rating teacher and two gradient replays verify.
The admission receipt matches all16 sources and every preserved argument;
8.821167GiB free satisfies the8GiB reserve including initial/temporary saves.

Registered trial `full_context_balanced_visible_union_anchor50_200` starts
under PID1623092/tool66723 for200 updates with validation every50. Only the
synthetic ranking target changes versus the balanced conditional comparison.
Initial inner reproduction and first-update provenance remain pending. No
production source,architecture or public-inference changes; no real-data
gradients or BPR/MF neural teachers. All8 comparisons and independent inner
reproduction still gate another outer evaluation. Goal active and unmet.

### Visible-frequency first update verifies; synthetic tradeoff probe completes

Initial inner metrics reproduce the incumbent exactly, as do all254 model
tensors. The first-update receipt verifies all16 sources, exact registered
arguments, synthetic ancestry, original frozen teacher, actual visible-mask
targets and two replay backwards. PID1623092/tool66723 continues the registered
200 updates. Watcher81412 audits50/100/150; terminal watcher81379 waits for
completion before saved-error diagnostics and campaign audit. The final receipt
still requires the main training tool's observed exit code.

A separate CPU-only diagnostic measures the minimum additional squared rating
error needed to make latent expected ratings globally monotone in visible item
counts. Per-user risk weights are recovered from synthetic conditional exposure
and ordinal probabilities; equal-count items remain freely ordered. This is an
oracle projection on generated data only, never a neural teacher or inference
correction. Twelve focused tests pass; the fresh combined suite passes620.

| Synthetic prior | Extra rating-MSE floor | Rating-oracle DCG efficiency | Count DCG efficiency |
|---|---:|---:|---:|
| Current balanced | .101497 | .624932 | .785079 |
| Popularity sigma[0,.7] | .074108 | .849538 | .806428 |
| Quality coefficient[4.5,7.5] | .067986 | .611520 | .888993 |

Each variant covers three seeds and both full-size context modes. All6 balanced
episodes reproduce archived hashes exactly. Lower nuisance reduces the risk
floor and improves the rating oracle in6/6 cases, and counts in5/6. Stronger
quality reduces the floor in6/6 but improves the rating oracle in only2/6;
stronger exposure coupling is not automatically better aligned for top10.

This is exploratory synthetic evidence, not benchmark improvement or proof of
top-k feasibility. Global monotonicity is stricter than the training top-k loss;
zero margins and latent-oracle knowledge make its own constrained risk estimate
optimistic. Crops/masks/users change with the prior, so these are not per-cell
causal comparisons. Peak CPU RSS3.148GiB; no neural model, real data or CUDA was
loaded. All16 live sources and public inference remain unchanged. No follow-up
GPU run is registered or started on this evidence; finish the active trial first.

### Conditional lower-nuisance follow-up registered before validation50

Register `full_context_low_nuisance_visible_union_anchor50_200` as a fallback
only. Change independent exposure-popularity sigma from[.7,1.4] to[0,.7]; keep
visible-frequency targets,affinity[0,1],quality[1.5,4.5],the original frozen
synthetic rating teacher and every optimizer/loss/window setting unchanged.
This choice follows the CPU evidence before the active trial's trained inner
results; stronger quality was not chosen because its rating-oracle ranking
declined on average. No production source change is required.

Six focused CPU preparation tests pass over three seeds and both context modes.
The zero lower bound reaches the generator, independent latent metadata draws
are preserved, labels exactly equal detached generated visible-mask counts,
and synthetic provenance/geometry/known-query checks pass. Model/data loaders
are forbidden during tests; CUDA stays uninitialized and all16 sources match.
Preparation receipt records the exact registration,evidence,test and config
hashes. These6 tests are separate from the earlier620-test combined run.

PID1623092's registered comparison continues unchanged. No fallback GPU
preflight or training starts. A dedicated preflight verifier is not yet
prepared: require audited200-update completion,priority reproduction of any
all8 winner,exact preserved arguments/sources,fresh two-mode full-size GPU
checks and8GiB reserve before admission. Goal active and unmet;zero new outer
evaluations and no real-data gradients.

The lower-nuisance admission verifier is now prepared and syntax-checked. Its
prerequisite gate correctly refuses while the balanced visible-frequency run
is incomplete,without starting GPU work or issuing an admission receipt.
It requires all16 comparison sources,exact registered prior-only arguments,
the six-case preparation receipt,unchanged108-case legacy objective evidence,
fresh two-mode full-size finite gradients,visible-mask equality and storage
headroom. A passing full-size GPU admission has not yet occurred.

### Balanced visible-frequency validation50 retains MF wins but misses BPR

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7460512172 | .7359750349 |
| RMSE | .8780294383 | .8365500540 |
| AUC | .8215312258 | .8167132854 |
| MAP | .0635326830 | .0349750817 |
| NDCG@10 | .0784432321 | .0498151705 |
| Precision@10 | .0484107579 | .0254418536 |
| Recall@10 | .0889620701 | .0644851037 |

All4 MF guards pass, but0/4 BPR targets pass. Compared with initialization,
MovieLens ranking improves,Netflix ranking regresses,and all four rating
errors rise. The feasible key[0,1.9082534593,1.4560987366] is worse than the
incumbent and rejected. Selected step0 remains all254 initialization tensors;
there is no promotion merely for retaining rating wins.

The observation receipt verifies all16 live/archive/preflight sources,frozen
inference/baselines,canonical0/50 scores,synthetic lineage,exact visible-mask
labels,registered popularity/quality ranges,original teacher anchor50 and two
replays in all six logged updates. Saved errors reproduce without new inference
or any fitted correction. The lower-nuisance fallback remains CPU-prepared only.

PID1623092/tool66723 continues the registered200 updates. Watcher81412 now
awaits100/150;terminal watcher81379 awaits complete status and process exit.
Only the main tool's observed exit0 can admit the final completion receipt.
Goal remains active and unmet;zero new outer evaluations,no real-data gradients,
and no architecture/public-inference changes.

### Lower-nuisance monitoring helpers prepared while comparison continues

First-update,scheduled-validation and completion helpers are prepared and
syntax-checked for the registered prior-only fallback. They require sigma[0,.7]
and exact visible-mask targets while preserving all optimization arguments.
The terminal receipt explicitly checks only the registered exposure-popularity
range change; it does not mislabel this as a ranking-target change.

The CLI correctly refuses the live campaign lock. A separate direct verifier
call also refuses the confirmed live PID before reading future trial artifacts.
The initial negative test expected the PID error at the CLI and instead hit the
earlier lock safeguard; both refusal paths are now explicitly verified. No
production change was needed. All16 active sources remain unchanged and the
future trial directory does not exist. Continue PID1623092's full200 schedule;
no new GPU preflight or neural training starts during this preparation.

### Balanced visible-frequency validation100 passes one BPR target, not the goal

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7465605320 | .7369701834 |
| RMSE | .8773855040 | .8368337090 |
| AUC | .8183761922 | .8021247804 |
| MAP | .0673781383 | .0274606706 |
| NDCG@10 | .0815062887 | .0368121903 |
| Precision@10 | .0479217604 | .0187137977 |
| Recall@10 | .0897939813 | .0458775238 |

MovieLens NDCG@10 now strictly exceeds BPR's.0809004866; its Recall@10 still
misses. Netflix ranking regresses further. All4 MF rating guards pass,although
Netflix margins shrink to.00029823 MAE and.00017515 RMSE. This is1/4 BPR wins,
not joint success. The feasible key[0,2.6822267670,1.7885864377] is worse than
both50 and initialization and is rejected. Selected step0 stays exactly the
initialization;no candidate is promoted.

The100-update observation verifies all16 live/archive sources,frozen inference
and baselines,canonical0/50/100 keys,synthetic ancestry,and all11 logged updates
with exact visible-mask targets,the registered prior,original teacher anchor50
and two gradient replays. Saved errors reproduce without inference or fitted
correction. A fresh combined CPU suite passes626 tests;the registered fallback
and its four monitoring/admission helpers remain preparation only.

PID1623092/tool66723 continues to200. Watcher81412 now awaits150;terminal
watcher81379 waits for complete status and process exit. No fallback preflight
or training starts before terminal admission;no new outer evaluation is run.
The goal remains active and unmet,with unchanged architecture/public inference
and no real-data gradients.

### Lower-nuisance benefit persists on preregistered disjoint synthetic seeds

Before generation,register seeds911/912/913,both full-size context modes,and
exactly two fixed priors:current balanced versus the already registered lower
nuisance[0,.7]. Require mean risk-floor reduction and improvements in both
rating-oracle and visible-count DCG efficiency;report all12 cases without
selective exclusions. These seeds are disjoint from the exploratory901–903
panel. No new parameter search or model/benchmark evaluation is performed.

| Prior | Extra rating-MSE floor | Rating-oracle DCG efficiency | Count DCG efficiency |
|---|---:|---:|---:|
| Balanced | .082483 | .644472 | .844101 |
| Lower nuisance | .047977 | .881540 | .853576 |

All three preregistered aggregate criteria pass. Risk falls and rating-oracle
ranking improves in6/6 paired cases;counts improve in4/6,not universally.
The new evidence supports the prior hypothesis but still proves neither neural
learnability nor transfer to either real benchmark. As before,crops/masks/users
change and this global count-order oracle risk is not the training top-k loss.
No projected diagnostic score becomes a neural teacher or public prediction.

The fallback preflight now verifies this report's source/registration hashes,
all12 case identities,the recomputed means and the three directions,along with
its existing terminal/source/synthetic/GPU/storage gates. Its preparation receipt
records the robustness hashes. No fallback source snapshot or GPU run exists;
PID1623092's current200-update comparison remains unchanged.

A read-only look at saved validation100 errors also finds Netflix top10 mean
item support falling1004.70->554.19 and first1000-item share.74903->.38350.
Within-user centered rating-error contribution rises.00163408 while the
user-bias-squared contribution falls.00136805. These descriptive diagnostics
are not a fitted correction or evidence that a common offset would solve the
joint task;inference stays frozen. Goal remains active and unmet.

### Balanced visible-frequency validation150 loses the Netflix rating guards

MovieLens MAE/RMSE are0.7471294505/0.8781214437 and NDCG/Recall@10
are0.0836981892/0.0917722772. Netflix values are
0.7375437977/0.8373654978 and0.0380286102/0.0485037799.
Both Netflix MF guards now fail. MovieLens retains its two rating wins and its
NDCG win over BPR,but no other BPR target passes:2/4 MF guards and1/4 BPR wins.
The infeasible key[1, 1.0004260818581776, 1.7209790597980277] is rejected. Selected step0 remains all254
initialization tensors;the eligible incumbent is not replaced by this failure.

The observation audit verifies all16 sources,frozen inference/baselines,
canonical0/50/100/150 selection keys,synthetic ancestry and all16 logged updates
with exact visible-frequency targets,the registered prior,original frozen
teacher anchor50 and two replays. Saved metrics reproduce without new inference
or calibration. The registered200-update training schedule remains unchanged.

Watcher81412 finishes its50/100/150 observations successfully. PID1623092/
tool66723 continues;terminal watcher81379 waits for complete status and process
exit. The lower-nuisance fallback remains prepared only. No new outer evaluation
or real-data gradient updates;goal active and unmet.

### Balanced visible-frequency200 completes without promotion; lower-noise preflight starts

The full200-update schedule completes in8912.369 seconds with main tool exit0.
Terminal saved-error diagnostics,campaign audit and completion receipt pass.
Final MovieLens MAE/RMSE are.7474054282/.8782799323 and NDCG/Recall@10 are
.0832749980/.0915207442. Netflix values are.7374017690/.8372246624 and
.0346372116/.0442117113. Final2/4 MF guards and1/4 BPR wins fail the joint goal;
key[1,1.0002578215,1.8395320502] is rejected. Selected step0 remains all254
initialization tensors. The generic campaign audit's238 changed tensors compare
with an older campaign root,not this trial's immediate initialization.

All21 logged updates verify synthetic visible-frequency targets,registered
prior ranges,original frozen rating teacher anchor50 and two replays. All16
live/archive/preflight sources and frozen inference/baselines match. No all8
candidate requires independent reproduction;no promotion or new outer evaluation.

Completed-only checkpoint deduplication9 retains both paths and all85 other
artifacts,restoring free space from8.523323 to8.813992GiB. No live checkpoint
is deduplicated. The lower-nuisance16-source snapshot exactly matches this
completed comparison and pins registration,preparation and completion hashes.

Start the preregistered full-size two-mode synthetic GPU preflight under tool
20870. Only exposure-popularity sigma changes from[.7,1.4] to[0,.7];all other
prior/optimization/teacher/inference settings remain fixed. Require the prepared
admission verifier,including the preregistered disjoint-seed evidence and8GiB
headroom,before any new training. The goal stays active and unmet.

### Lower-nuisance full-size GPU admission passes; prior-only training starts

Fresh2000-row,two1000-item-window synthetic checks pass in both context modes.
Peak allocated GPU/host memory is125.628/77.362GiB. Losses1.290815/1.432730 and
gradient norms7.980828/10.010550 are finite. Generated popularity sigma.112578
is within[0,.7];quality coefficient2.954537 is unchanged. Exact visible-mask
labels,original frozen synthetic rating teacher and two gradient replays verify.

The admission receipt verifies the completed comparison,preregistered disjoint
synthetic panel,all16 identical sources and the exact two changed prior arguments.
Every optimization argument and frozen inference/baseline remains unchanged.
Free8.813141GiB satisfies the8GiB reserve with initial and temporary saves.
No real data or BPR/MF reference is used as a neural teacher or target.

Start `full_context_low_nuisance_visible_union_anchor50_200` under PID1641513/
tool96187 for200 synthetic updates with validations every50. First-update
watcher5465 awaits initial inner reproduction and provenance verification;
watcher88982 covers50/100/150;terminal watcher62587 waits for process exit and
complete status before final diagnostics/audit. A completion receipt additionally
requires observing the main training tool's exit code. No trained result is yet
claimed;all8 comparisons and independent reproduction still gate outer evaluation.
Goal active and unmet;architecture/public inference remain unchanged.

### Lower-nuisance initial reproduction and first-update audit pass

Initial inner metrics reproduce the eligible incumbent exactly,as do all254
initialization tensors and the architecture signature. The first-update receipt
verifies all16 live/archive/preflight sources,exact registered training arguments,
frozen inference/baselines,synthetic initialization/teacher lineage,the original
rating teacher,actual visible-frequency labels,sigma[0,.7],quality[1.5,4.5] and
two gradient replays. Loss and gradient norm are finite;no checkpoint promotion.

PID1641513/tool96187 continues the registered200 synthetic updates. Validation
watcher88982 awaits50/100/150;terminal watcher62587 awaits completion. No trained
inner result is yet available from this new prior. The goal remains active and
unmet;no new outer evaluation,real-data gradients or architecture/inference changes.

### Broader training catalog registered without changing live training or inference

Frozen Netflix small has4895 items,spanning five public1000-item windows;
current synthetic joint ranking spans only two. Contiguous training crops can
omit head or tail blocks entirely. A two-window input-size match is therefore
not a full-catalog ranking match. This motivates a controlled broader-slate
fallback,not a claim that catalog width alone explains the observed regressions.

Register `full_context_four_window_visible_union_anchor50_200`:change only
`training_item_windows=4` and `negatives_per_user=4000`. Keep the4096-item parent,
2000x1000 encoded windows,current lower-nuisance prior,visible targets,original
synthetic teacher and all remaining optimizer settings. Architecture/public
inference stay fixed. Four windows are broader,not an exact five-window match;
expected extra compute and full-size GPU resources are not yet measured.

Twelve full-size CPU cases compare2/4 windows across three seed components and
both context modes through the actual training episode wrapper. Global latent
metadata draws match,while crop columns and rating queries may differ. Every
four-window case has queries in all four windows and exact detached visible-mask
targets. Coverage increases from2000/4096 to4000/4096 parent columns. One
contiguous example's mean item supports are43.903,4.590,1.126,.175:the added
windows expose substantially colder candidates in the same loss. The report's
support-span statistic floors its denominator at1;it is not an unfloored ratio.
Peak CPU RSS is3.541GiB. No DWIP checkpoint,real data or CUDA is loaded.

### Four-window numerical checks distinguish gradients from Adam equivalence

The first toy suite passes logits,loss,gradient and RNG comparisons but fails
four overstrong pure-ranking Adam parameter comparisons. The maximum gradient
difference is5.96e-8;near-zero offset gradients become a.000748754 parameter
difference at toy LR.001. Reversing toy replay entries matches the reference
summation exactly,which identifies order sensitivity,not an intrinsically
better optimizer:the exact pure score-shift derivative is zero and the
reference's nonzero residual is also rounding. No production replay order is
changed and no benchmark benefit is inferred.

Preserve the failed log and both numerical diagnostics. The revised15-case
suite checks gradient/RNG equality within FP32 tolerance,SGD updates for the
pure-ranking fixture,and Adam updates for a nondegenerate protected-loss
fixture. It also verifies four-window forwarding and joint gradients that
correct offsets invisible to separately ranked windows. All15 pass;the fresh
combined suite passes641. This does not establish general bitwise Adam parity.

A separate actual-DWIP four-window CUDA gradient verifier is prepared for two
reduced-size context modes. It requires exact logits/losses,relative gradient
L2 error<.02,cosine>.999,frozen original teacher and unchanged inference/sources;
it performs no optimizer step. The incomplete-comparison gate correctly refuses
while PID1641513 is live. This CUDA check and a fresh full-size two-mode GPU
preflight are both still required. The full-size admission verifier is not yet
prepared;no four-window source snapshot,GPU run or training directory exists.
Finish/audit the current200-update comparison and prioritize any all8 winner.
Goal remains active and unmet;no real-data gradients or new outer evaluations.

### Lower-nuisance validation50 retains MF guards but regresses ranking

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7460135007 | .7332661873 |
| RMSE | .8781305101 | .8344343757 |
| AUC | .8077051059 | .8000842703 |
| MAP | .0550869444 | .0297368580 |
| NDCG@10 | .0680486089 | .0416604701 |
| Precision@10 | .0419315403 | .0209713352 |
| Recall@10 | .0803702416 | .0549430027 |

All4 MF guards pass. Netflix MAE/RMSE improve versus initialization,but all four
ranking metrics targeted against BPR decline versus initialization;0/4 BPR wins.
The feasible key[0,2.2396650407,1.6935120092] is worse than the incumbent and is
rejected. Selected step0 remains all254 initialization tensors;no promotion.
The synthetic oracle alignment improvement has not established neural transfer.

The observation receipt verifies all16 live/archive sources,frozen inference/
baselines,canonical0/50 keys,synthetic lineage and all six logged updates with
sigma[0,.7],exact visible-frequency labels,original teacher anchor50 and two
replays. Saved errors reproduce without new inference or calibration.
PID1641513/tool96187 continues200 updates;watcher88982 now awaits100/150 and
terminal watcher62587 remains active. No new outer evaluations;goal unmet.

The four-window full-size admission verifier is now prepared and its live-
comparison refusal verified. It requires the reduced actual-DWIP CUDA checks
plus a fresh full-size four-window,two-mode preflight;neither has run. No new
GPU trial starts. Future training audits must respect the unchanged replay
implementation's legitimate skipping of query-empty windows;both admission
fixtures must nevertheless verify four replays. Four-window first-update,
scheduled-observation and terminal helpers are not yet prepared.

### Four-window monitoring prepared; CPU and terminal safety checks pass

Prepare the four-window first-update, scheduled-observation and completion
helpers without changing any of the16 live training sources. They enforce the
registered4000-candidate/four1000-item-window geometry, retain the exact prior,
synthetic lineage and original teacher checks, and record per-update replay
coverage. The shared metadata checker rejects inconsistent integer metadata
and replay counts that cannot contain the logged queries. It deliberately
permits legitimately query-empty windows: the arithmetic is a necessary
capacity bound, not independent proof of exact per-window coverage. The frozen
replay implementation supplies that verification, and both GPU admission
fixtures still explicitly require four replays.

The19 focused metadata tests pass. A fresh expanded combined suite covering
all BPR/recsys tests and the campaign's six preparation/probe test files passes
669 tests in6.47s. This includes the previously641-case suite, the19 metadata
cases and9 additional synthetic-reference cases; it is not merely summed from
separate executions. Compilation passes for all three new monitoring helpers.
The terminal CLI refuses the held campaign lock, and direct verification
refuses live PID1641513 before reading completion artifacts or initializing
CUDA. Preserve both refusals in `four_window_terminal_refusal_checks.log`.

Preparation and status receipts now bind helper/test hashes. Full-size GPU
admission additionally requires these hashes and terminal refusal evidence.
No four-window source snapshot, GPU preflight or training has started. The
existing lower-nuisance200-update run continues unchanged; its last audited
validation remains50 with all4 MF guards and0 BPR targets. Goal remains active
and unmet, with zero new outer evaluations and no real-data gradient updates.

### Lower-nuisance validation100 preserves MF wins but further loses ranking

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7460981927 | .7331415315 |
| RMSE | .8776980499 | .8339936335 |
| AUC | .8086515994 | .7906697209 |
| MAP | .0569395254 | .0266991831 |
| NDCG@10 | .0696158021 | .0359028646 |
| Precision@10 | .0431540342 | .0178969256 |
| Recall@10 | .0810949872 | .0448459653 |

The scheduled observation passes its integrity audit and saved-error metrics
reproduce without new inference or calibration. All4 strict MF guards pass,
but0/4 BPR targets pass. The worse feasible key[0,2.7439240396,1.8921795570]
is rejected; all254 selected step0 tensors remain exactly initialization.

Relative to initialization, Netflix rating errors improve while its top10
first1000-item fraction falls from.7490 to.5060 and mean item support falls
from1004.70 to626.50. The score-ceiling fraction rises from.0665 to.1072.
These are descriptive inner-validation diagnostics, not causal evidence, and
they do not enter synthetic labels or training. They are consistent with the
previously registered broader-catalog hypothesis, not proof it will work.

PID1641513/tool96187 continues the unchanged200-update schedule. Watcher88982
now awaits150;terminal watcher62587 remains assigned to completion. Finish and
audit the current trial before any four-window CUDA check/preflight/training.
All16 production sources, architecture and public inference remain unchanged.
Goal stays active and unmet; zero new outer evaluations.

### Lower-nuisance validation150 remains ineligible for promotion

Scheduled validation150 passes its observation and saved-error reproduction
audits. MovieLens MAE/RMSE are.7459616801/.8778700731 and NDCG/Recall@10 are
.0721738635/.0844319433. Netflix values are.7328933661/.8338039185 and
.0371299368/.0465963115. All4 MF guards pass;0/4 BPR targets pass. The key
[0,2.6408511394,1.8237748520] remains worse than initialization and is rejected.
Selected step0 remains exactly initialization. All16 sources,synthetic lineage,
original frozen rating teacher and public inference pass the ongoing audit.

Watcher88982 completes successfully after observing100/150. PID1641513/tool96187
remains live;terminal watcher62587 is still assigned to the200-update completion.
Do not infer training exit from a watcher timeout or start another GPU job yet.
The broader four-window control remains preregistered and CPU-prepared only.

### Lower-nuisance200 finishes without promotion; four-window GPU checks start

Observe the main training tool96187 exit0 and terminal watcher62587 exit0.
The terminal audit verifies the full200-update schedule, all five canonical
validations,21 logged synthetic updates, original teacher, unchanged16sources
and frozen inference/baselines. Every validation retains all4 MF guards and
passes0/4 BPR targets. Final MovieLens MAE/RMSE are.7461612581/.8779018767 and
NDCG/Recall@10 are.0727205479/.0837349318. Netflix values are.7330529536/
.8338984764 and.0360042855/.0453959349. The final key
[0,2.7106815311,1.8596702774] is rejected;selected step0 remains all254 tensors
exactly initialization. No candidate promotion or new outer evaluation occurs.

Completion receipt confirms the fallback start condition. Completed-only
checkpoint deduplication10 retains both byte-identical paths and preserves114
other artifact hashes, restoring8.8063GiB free. No live checkpoint is touched.

The actual-DWIP four-window CUDA replay check now runs and passes both reduced
128x256 fixtures with four64-column windows. Logits and losses match exactly,
RNG state is preserved and all238 gradient tensors are compared. Relative-L2
gradient errors are.00174367/.00658807, with cosine.99999850/.99997861. Peak
GPU allocation is2.5081GiB. This is synthetic-only and performs no optimizer
step; it does not establish full-size memory admission or bitwise Adam parity.

Archive the unchanged16 production sources before the fresh full-size two-mode
preflight. PID1656625/tool87390 starts that preflight with only the registered
training geometry and candidate-budget changes. Four-window training has not
started. The full-size admission verifier, including storage reserve, must pass
before launching the registered200-update trial. Goal remains active and unmet.

### Four-window full-size admission fails memory cap; staged runtime fix tested

The full-size two-mode preflight exits1. Both losses and gradients are finite,
both modes verify four replays and exact visible-frequency labels, and the
original teacher,architecture and synthetic-only checks pass. However GPU peak
allocation is129.144718GiB, exceeding the unchanged128GiB admission cap. Host
peak is77.3554GiB. Preserve the failed report,log and source snapshot. Do not
admit training,raise the cap,or reduce the four-window catalog to force a pass.

The detached logit-collection pass currently constructs and discards autograd
graphs before replay. Prepare an isolated copy of the window-training helper
that wraps only that pass in `torch.no_grad()`. Gradient-enabled verified replay,
train mode,RNG handling,joint loss,query coverage,precision and geometry remain
unchanged. All16 production files stay unchanged; this is not deployed yet.

The staged combined suite passes671 tests,including two new checks that the
collection pass disables gradients and the replay restores gradient-enabled
forwarding for all four windows. Keep staging-only tests in the staging folder
so unchanged production tests do not accidentally require an undeployed fix.
Reduced actual-DWIP staged CUDA tests pass both modes with exact within-run
logits/losses and preserved RNG. Relative gradient L2 errors are.00173147/
.00161940 and cosine.99999851/.99999870. Cross-run fixture losses need not be
bitwise identical;no general Adam equivalence is claimed.

Preregister the runtime memory proposal in
`four_window_memory_staging_registration.json`, retaining the128GiB cap and
all full-size arguments. Tool11219 runs a fresh full-size two-mode staged
preflight. Its snapshot explicitly declares the staged module as the sole
runtime override rather than mislabeling unchanged production hashes as its
implementation. Even a successful staged result requires a source-bound
deployment/admission amendment before production training. No real-data
gradients,new outer evaluation or four-window training has occurred.

### Phase profiling identifies replay peak; cached-logit offload passes full size

The isolated no-grad collection proposal also exits1 at exactly129.144718GiB.
It does not solve the admission failure and is not deployed. Preserve its
failed report,log,registration and snapshot rather than replacing the evidence.

A fresh synthetic-only phase profile of the original production implementation
localizes peak allocation on a2000x4000 first-observation episode:

| Phase | Peak GPU GiB |
|---|---:|
| Synthetic episode generation | 3.1369 |
| Frozen rating teacher | 44.6723 |
| Detached student collection | 93.1403 |
| Joint objective forward | 19.5728 |
| Joint objective backward | 28.8702 |
| Verified window-gradient replay | 129.1400 |

This is a single synthetic diagnostic,not benchmark evaluation or training;
no optimizer step occurs. The peak is during replay,not the collection graph.
The5000-bin per-query reference logits remain cached across all four windows.

Prepare a second isolated optimization-only patch: under the existing offload
option,store detached reference logits on CPU,assemble the same joint GPU
logits,and copy only each current reference window for exact replay verification.
Keep the original gradient-enabled collection;do not combine the unsuccessful
no-grad change. There are no prior,loss,teacher,optimizer,schedule,architecture,
inference,precision,geometry or candidate-budget changes.

The fresh v2 combined CPU suite passes671 tests in6.96s. Two additional cache
integrity tests verify detached CPU references and rejection of corrupted
reference logits. Reduced actual-DWIP CUDA tests explicitly verify all four
reference caches are on CPU and pass exact within-run logits/losses,RNG and
gradient checks. Relative-L2 errors are.00173287/.00161779 and gradient cosines
are.99999850/.99999870. No general bitwise Adam claim is made.

Preregister v2 before its full-size probe. Tool93888 exits0:both full-size
context modes have finite gradients,exact visible-frequency targets and four
verified replays. GPU peak falls to126.804054GiB,below the unchanged128GiB cap;
host peak is79.630245GiB,below160GiB. The full catalog and every effective
argument except output path are preserved. The runner/snapshot explicitly
bind the sole staged runtime override;all16 production files remain unchanged.

`four_window_memory_staged_v2_preflight_receipt.json` is staged readiness,not
production training admission. No GPU job or four-window trial is live now.
Next register and deploy the exact v2 memory-only source amendment,rebind the
source-aware admission/monitoring helpers,and run fresh production admission
before the200-update comparison. Preserve both failed attempts and do not
start via the original failed preflight. Goal remains active and unmet,with
zero new outer evaluations and no real-data gradients.

### Exact cache-offload amendment deployed; fresh production admission running

Reverify no GPU job or four-window training directory exists. Preregister the
memory amendment before deployment in
`four_window_visible_union_memory_registration.json`. Deploy exactly the staged
v2 window helper,with SHA256
05d3febffd701a36442d5220192f0c245bc21cc155bcbaa78fac2a767979cf02.
The other15 production sources remain identical to the completed comparison;
architecture,public inference,baselines,prior and objective are unchanged.
Do not deploy the unsuccessful no-grad collection proposal.

The first two fresh combined CPU runs correctly fail historical preparation
hash guards that still require the old window helper. Preserve both logs and
the original historical tests. Add memory-amended copies of the three source-
bound preparation suites:their assertions resolve exactly the single registered
source change while retaining before/after checks on all16 sources. No archived
configuration or test expectation is silently rewritten. Move the two cache-
integrity cases into the normal window-replay regression tests as well.

The resulting production combined suite passes671 tests in7.17s;focused four-
window preparation passes15. All amended monitoring helpers compile,terminal
CLI lock refusal and direct live-PID refusal pass without CUDA initialization,
and a shared source resolver binds the base registration,amendment,comparison,
staged source and successful staged memory receipt. The deployment receipt
verifies exact staged bytes,frozen inference/baselines and15unchanged sources.

Fresh production actual-DWIP CUDA replay checks pass both reduced fixtures,
including four CPU reference-logit caches,exact within-run logits/losses,RNG
preservation and all238 gradient tensors. Relative-L2 errors are.00173010/
.00161819 and gradient cosines.99999850/.99999870. Preserve the existing caveat
that this does not imply bitwise Adam equivalence.

Prepare memory-amended source-bound first-update,observation,completion and
full-size admission helpers. Archive all16 current production sources before
the fresh full-size two-mode production preflight,tool16002. Its128GiB GPU cap,
four-window geometry,64ranking users and all optimization/prior arguments remain
unchanged. No four-window training starts until this fresh admission passes.

### Four-window production admission passes; registered synthetic training starts

The fresh production preflight tool16002 exits0. The amended admission verifier
also exits0 and binds the base registration,exact single-source memory amendment,
staged evidence,current16-source snapshot and archived comparison sources. GPU
peak is126.804054GiB and host peak79.679703GiB. Both full-size context modes
have finite gradients,exact visible-frequency targets,original frozen synthetic
teacher anchor50 and four verified replays. Frozen architecture,inference and
baseline registrations match. Storage has8.79795GiB free,with8GiB reserve
including the initial and temporary checkpoint requirements. No cap is relaxed.

Start `full_context_four_window_visible_union_anchor50_200` under PID1662022/
tool17511. The registered200-update schedule uses seed71,LR2e-6,full scope,
EMA.95,64ranking users,four independently encoded2000x1000windows and4000
negatives/user. The prior,loss,teacher and public inference otherwise match the
completed lower-nuisance comparison. The memory amendment changes only where
detached replay-reference logits are stored. Initial checkpoint and frozen
teacher both retain synthetic-only ancestry. All neural examples and targets
come from the synthetic generator;benchmark data remains evaluation-only.

First-update watcher92312 is live. Scheduled watcher46486 awaits50/100/150 and
terminal watcher94562 waits for process exit before saved-error diagnostics and
campaign audit. Completion still requires observing main tool17511's actual
exit code;do not infer completion from a watcher or restart on a timeout.
No new candidate metric win is claimed at launch;the goal remains active and
unmet with zero new outer evaluations.

### Four-window initial reproduction and first synthetic update pass

First-update watcher92312 exits0. Initial inner metrics exactly reproduce the
eligible incumbent,and all254 initial tensors/state signatures match. The
step0 checkpoint serialization SHA remains
f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802.
This is initialization reproduction,not a new BPR win;initial comparisons remain
4/4 MF rating guards and0/4 BPR ranking targets.

The first synthetic update has249389 combined queries. The query-capacity
bound requires all four windows and replay verifies four. Finite active ranking
gradients,exact detached visible-mask frequency targets,default affinity,
popularity[0,.7],quality[1.5,4.5] and original synthetic teacher anchor50 pass
the first-update checks. Train/preflight arguments match except action/output,
and all16 current live/archive/preflight sources match the registered memory
amendment. Frozen public inference,baselines and synthetic ancestry are intact.

PID1662022/tool17511 remains live. Scheduled watcher46486 awaits50/100/150;
terminal watcher94562 remains assigned to final diagnostics/audit. Do not
modify or restart the registered200-update run. No benchmark ranking gain is
claimed yet and no new outer evaluation has occurred;goal remains active.

### Optional synthetic risk-budget anchor prepared without changing live training

While PID1662022/tool17511 trains,prepare an isolated alternative to strict
teacher-prediction imitation. The current squared prediction anchor penalizes
departures even if both synthetic rating risks improve. The proposed soft
penalty instead sums separate positive-excess expected macro-RMSE and expected
macro-MAE hinges against the same detached synthetic teacher. Improvement in
one metric does not cancel the other metric's positive excess in the loss.
The existing synthetic rating and ranking terms are otherwise unchanged.

The prototype uses the existing parent-count expected-macro-RMSE surrogate
with32 antithetic draws and shared episode_seed+7919 for student/teacher;
macro-MAE is analytic. It does not change public inference or add calibration.
This is a batch macro-risk penalty,not a guarantee for each user,each optimizer
step,or real-data metrics. Monte Carlo and sampled-parent replication remain
approximations. Its units differ from the existing prediction-MSE anchor,so
anchor50 is not automatically a justified coefficient for a future trial.

The isolated suite passes51 CPU tests. Six cases reproduce the default legacy
loss,diagnostics,protected/ranking components and logit gradients exactly; six
more verify that only the protected teacher term changes. Other cases cover
separate metric excesses,zero penalty/gradient for equal or jointly improved
risks,user-macro weighting,global RNG preservation,query alignment,finite
generated-episode gradients,detached teachers/probabilities,synthetic-only
provenance and double-precision gradient checks. CPU fixtures use synthetic
predictions/context-bias references,no DWIP checkpoint or real-data teacher.

Preserve the initial test failures:clamp_min's nonzero equality subgradient was
corrected to ReLU,and generated full-catalog fixtures were fixed to request the
required synthetic relevance targets. All16 live production source hashes are
unchanged. No CLI integration,new GPU work or model training uses this proposal.

Record `risk_budget_anchor_proposal.json` as unintegrated preparation,not a
registered training trial. Any follow-up requires synthetic-only coefficient/
gradient calibration,explicit opt-in integration,source-bound preregistration
and fresh full-size admission after the current200-update run completes. An
all8 winner takes priority. The real BPR/MF goal and outer gate remain unchanged.

### Four-window validation50 retains MF guards but regresses ranking

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7459089407 | .7331592070 |
| RMSE | .8779531511 | .8343293764 |
| AUC | .8059473639 | .7943664851 |
| MAP | .0556058238 | .0285885314 |
| NDCG@10 | .0683946893 | .0397170796 |
| Precision@10 | .0419315403 | .0199613842 |
| Recall@10 | .0798768033 | .0511316711 |

All4 strict MF rating guards pass;0/4 BPR targets pass. The feasible key
[0,2.4066086549,1.7608219582] is worse than initialization and is rejected.
All254 selected step0 tensors remain exactly initialization;no promotion.
Compared with the two-window lower-nuisance control at50,the four-window run
slightly improves all four rating errors,but Netflix NDCG/Recall are lower.
Broader catalog coverage has not yet established neural ranking transfer.

The source-amended scheduled observation passes. Saved-error diagnostics
reproduce all rating/ranking metrics without new inference or calibration and
preserve input hashes. Relative to initialization,Netflix top10 first1000-item
fraction falls from.7490 to.5970,mean item support from1004.70 to726.55,and
score-ceiling fraction rises from.0665 to.0996. These are descriptive inner
diagnostics,not causal proof and not labels for synthetic training.

PID1662022/tool17511 remains live under the unchanged200-update schedule.
Watcher46486 now awaits100/150;terminal watcher94562 remains active. All16
current production sources,architecture,inference,teacher and synthetic-only
lineage remain unchanged. The optional risk-budget objective stays isolated
and unintegrated. Goal remains active and unmet;zero new outer evaluations.

### Synthetic risk-budget gradient calibration preregistered;live training unchanged

The four-window200-update comparison remains live as PID1662022/tool17511,
with latest logged update60 at this preparation. Validation50 remains the
latest audited result:4/4 MF rating guards,0/4 BPR ranking targets,no checkpoint
promotion. Watcher46486 awaits100/150 and terminal watcher94562 remains active.
All16 production source hashes still match the live registered configuration.

Prepare `calibrate_risk_budget_gradients.py` without deploying the risk-budget
objective. It compares legacy prediction-MSE and proposed macro-risk anchor
parameter gradients through the same synthetic-only incumbent and original
frozen synthetic teacher. Both components replay every query-bearing window,
including exactly-zero risk gradients;replay preserves CPU/CUDA RNG and checks
exact logits. Student and teacher state tensors must remain exactly unchanged.
No optimizer step,benchmark data loading or benchmark-based coefficient fit is
permitted. Full2000x4000 geometry uses the unchanged128GiB GPU/160GiB host caps.

Register seed components911/912/913,episode20,both first-observation contexts,
and model RNG seed20260923 before any calibration. For active cases,match the
legacy50-weighted anchor parameter-gradient norm with the nearest log-grid
weight to the median log norm ratio. Grid [.25,.5,1,2,4,8,16,32,64];ties choose
the larger protection weight. At least2 active cases per context are required;
insufficient coverage or an out-of-grid median fails without a coefficient.
This matches norm scale only,not directions,Adam updates or benchmark transfer.

`risk_budget_gradient_calibration_registration.json` binds the16 production
sources,staged risk implementation/tests,calibrator,synthetic checkpoints and
the imported replay-metadata checker. Its SHA256 is
`9183c16c3a08be0300c31df20303a7d5320ed686e2799471488251468278c37a`.
Snapshotting includes the registration and auxiliary checker. Actual calibration
requires comparison exit0,absent PID,completed200-update schedule and audited
terminal receipt,and no all8 winner. The gate is rechecked after acquiring the
campaign lock. Any all8 winner takes priority over the fallback.

The expanded standalone CPU suite passes96 tests:the original51 objective
tests,coefficient-scale cases,two-component/zero-gradient parameter replay,
source binding,and completion/winner/PID/hash gates. Intermediate suites passed
78,82 and84 tests;this is not a combined production+staging regression count.
The actual CLI exits1 with the intended unfinished-comparison error before
model loading/CUDA. Preserve the first missing-registration failure separately;
it is not evidence for that gate. See
`risk_budget_gradient_calibration_preparation_receipt.json` and
`risk_budget_gradient_calibration_gate_tests.log`.

No actual-DWIP calibration has run,no coefficient is selected,and no new GPU
job,production deployment or trainer/CLI integration occurred. After current
training completes without an all8 winner,run the preregistered calibration.
A pass still requires opt-in integration/default compatibility,registered
training arguments and fresh full-size admission before candidate training.
The batch macro-risk anchor remains a soft synthetic penalty,not a guarantee
for individual users,optimizer steps or real metrics. Goal remains active and
unmet;zero new outer evaluations.

### Risk-budget trainer opt-in staged while the registered comparison trains

Prior goal turn classified as progress:it preregistered synthetic-only gradient
calibration and verified96 CPU tests plus the pre-CUDA unfinished-run refusal.
Direct PID/tool observation confirms the same four-window job remains live;
latest logged update70 is finite and validation50 is still the latest result.
There is no terminal condition and no reason to restart or shorten its schedule.

Prepare an isolated `risk_budget_training_staging` tree,without editing any of
the16 live production sources or any source bound by calibration. The future
patch is restricted to the synthetic trainer,ranking-prior objective,and one
new synthetic-risk helper. Architecture/public inference remain untouched.
An explicit `--teacher-anchor-mode macro_risk_budget` opts in;default
`prediction_mse` retains legacy behavior. Invalid risk weights/risk settings
are rejected before model loading. The trainer forwards the mode through its
existing objective and includes the new helper in future source snapshots.
That future source map has17 entries;do not reuse historical16-source guards.

The helper is byte-identical to the already-registered calibration helper.
The staged ranking objective differs from the earlier prototype only in its
package-qualified helper import and rejecting nonfinite risk-anchor weights.
The CPU pytest bootstrap substitutes staged modules only inside that test
process;it is not a standalone GPU training launcher. CUDA is hidden for tests.

Focused integration passes39 cases. Twelve cases establish exact legacy
trainer loss/diagnostics/components/logit-gradient compatibility,and twelve
establish exact opt-in equivalence to the registered risk prototype,covering
both context orders and ordinary/component-return modes. CLI parsing/dispatch,
invalid-setting rejection,and helper inclusion in source snapshots are covered.
The broader staged regression passes638 tests:599 existing BPR/recsys tests
plus those39 integration tests. The separate unchanged calibration suite still
passes96. These counts are distinct,not a claim to rerun all historical campaign
tests. Preserve the initial8 invalid-fixture failures:component-return fixtures
incorrectly enabled both projection and replay,which the existing trainer
correctly rejected. Fixtures now disable replay only for projection cases;
the actual proposed training retains replay and disables projection.

See `risk_budget_training_staging_proposal.json` for the three-path write scope,
base/staged source hashes,commands/logs and deployment conditions. No coefficient
has been selected;the50 used by equivalence fixtures is not a proposed risk
training weight. Calibration must run using the current unchanged production
sources after the200-update comparison finishes without an all8 winner. Only
then may a passing coefficient be bound to training registration and the staged
patch deployed,followed by production regressions and fresh full-size admission.
No new GPU job or model update uses the risk-budget proposal yet. Goal remains
active and unmet,with zero new outer evaluations.

### Four-window validation100 is audited and rejected;MF wins remain intact

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7462049291 | .7328706787 |
| RMSE | .8782342130 | .8340262906 |
| AUC | .8072957531 | .7886846904 |
| MAP | .0570138968 | .0267285012 |
| NDCG@10 | .0699875710 | .0361667047 |
| Precision@10 | .0429095355 | .0180751522 |
| Recall@10 | .0824313528 | .0452068275 |

All4 strict MF guards pass,but0/4 BPR targets pass. The feasible selection key
[0,2.7220207452,1.8756184289] is worse than both initialization and validation50.
The checkpoint is rejected;all254 selected step0 tensors remain exactly the
synthetic initialization. There is no candidate promotion or outer evaluation.

MovieLens NDCG/Recall improve versus50,but Netflix ranking declines further,
despite improved Netflix MAE/RMSE. Against the paired two-window control at100,
the four-window run has slightly higher NDCG/Recall on both datasets,while rating
differences are mixed;neither control establishes ranking transfer over the
incumbent. Broader synthetic catalog coverage alone has not solved the goal.

The scheduled source-amended observation passes,including all11 logged synthetic
updates and four verified replays each. Saved-error diagnostics reproduce all
rating/ranking metrics without new inference,fit or model modification and
preserve artifact hashes. Netflix top10 first1000 fraction falls from.749027 to
.507634,mean item support from1004.70 to626.38,and ceiling fraction rises from
.066508 to.111882. These remain descriptive inner diagnostics,not causal proof
or real-data training targets.

PID1662022/tool17511 remains live after100 under the original200-update schedule.
Watcher46486 now awaits150;terminal watcher94562 remains assigned to completion.
All16 production sources,synthetic ancestry and frozen inference remain intact.
The calibrated-risk fallback remains staged only;do not deploy before the
registered comparison completes and the already-preregistered calibration runs.
Goal remains active and unmet;zero new outer evaluations.

### Four-window validation150 passes integrity but remains below BPR

The previous goal turn made progress by staging opt-in integration and auditing
validation100. This turn verifies the same live process through updates110,
120,130 and140,then observes validation150;no restart or schedule change occurs.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7461569260 | .7327690147 |
| RMSE | .8782616772 | .8338749628 |
| AUC | .8099855901 | .7880381642 |
| MAP | .0594693819 | .0270891724 |
| NDCG@10 | .0721815871 | .0371503211 |
| Precision@10 | .0444987775 | .0188177633 |
| Recall@10 | .0823471941 | .0465349867 |

All4 strict MF rating guards pass;0/4 BPR targets pass. The feasible key
[0,2.6443313116,1.8321043902] improves versus100 but remains worse than
initialization and50. The checkpoint is rejected and selected step0 remains
exactly the synthetic initialization. No candidate is promoted.

The scheduled observation validates all16 production sources,the unchanged
inference/baselines,synthetic-only logged updates through150 and replay coverage.
Saved-error diagnostics reproduce all metrics without new inference or fitting
and preserve artifact hashes. Netflix top10 first1000 fraction is.487272,mean
item support630.63,and ceiling fraction.114496;these are descriptive diagnostics,
not real-data training targets or a causal account of the ranking shortfall.

Scheduled watcher46486 exits0 after its50/100/150 observations. Training
PID1662022/tool17511 is still live at elapsed3:40;terminal watcher94562 awaits
the registered200-update completion. Risk-budget calibration remains unexecuted
and all staged trainer/prior changes remain undeployed. Preserve the original
schedule,then audit actual exit/completion before conditional calibration. Goal
remains active and unmet;zero new outer evaluations.

### Four-window200-update trial completes without promotion;calibration starts

Main training tool17511 exits0 after all200 registered updates. Terminal watcher
94562 also exits0 after final saved-error diagnostics and campaign audit.
The completion verifier with `--pid 1662022 --training-exit-code 0` then
succeeds;the PID is absent and the completion receipt
passes all source,synthetic-lineage,teacher,inference,schedule and selection checks.

| Final200 Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7462645915 | .7328432871 |
| RMSE | .8780824371 | .8338169518 |
| AUC | .8096739349 | .7850813971 |
| MAP | .0597868161 | .0259934646 |
| NDCG@10 | .0724171392 | .0354880903 |
| Precision@10 | .0441320293 | .0179266300 |
| Recall@10 | .0824446630 | .0446699316 |

All five validations0/50/100/150/200 retain all4 MF rating guards and have0/4
BPR wins. Every trained checkpoint is rejected. Selected step0 remains exactly
the initialization,with all254 state tensors intact and file SHA256
`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`.
The generic audit's238 changed tensors compare against the older campaign root,
not this trial's immediate initialization. No new incumbent or outer evaluation.

The registered four-window control therefore does not establish the desired
ranking transfer. Final Netflix top10 first1000 fraction is.465305,mean item
support603.68,and ceiling fraction.117199. Final saved metrics reproduce without
another neural evaluation. Preserve the full completed schedule and artifacts.

After observing actual exit0 and the completion receipt,no GPU process remains.
Start the already-preregistered `calibrate_risk_budget_gradients.py` under the
campaign lock as PID1674921/tool77875,logging to
`risk_budget_gradient_calibration.log`. This uses the same synthetic incumbent,
original frozen synthetic rating teacher,three fixed seed components and two
context orders. It computes/replays parameter gradients only;no optimizer step
or real-data loading. All16 production sources remain unchanged,so calibration
matches its source-bound registration. Risk-budget trainer/prior integration
remains staged and undeployed;no coefficient or training admission is claimed
before calibration finishes and its gates pass. Goal remains active and unmet.

### Synthetic gradient calibration selects32;source-bound risk trial preregistered

Calibration tool77875 exits0 after all six registered actual-DWIP cases. Both
seed911 contexts have zero risk penalty/gradient because both synthetic rating
risks improve over the frozen teacher. Seeds912/913 provide two active cases
per context,meeting coverage. Both components replay four windows in every
case,including zeros;RNG is preserved and student/teacher states remain exactly
unchanged. No optimizer update or real dataset is used.

Active norm-scale targets are107.4634,103.9330,17.0102 and13.3216. Their log-median
corresponds to42.0466;the preregistered nearest-log-grid rule selects32. Selection
is independently recomputed from saved parameter-gradient norms before training
registration. Nonzero gradient-direction cosines range.3091-.7772,so norm matching
must not be described as equivalent directions or Adam updates. Peak allocation
is124.533700GiB GPU and90.736511GiB host,below unchanged128/160GiB caps.

Register `full_context_four_window_risk_budget32_200` before deployment in
`risk_budget_training_registration.json`,SHA256
`0b973121d2146a20913485efec97d57b8cce0e4272225c7e9d062f2efa9009b3`.
Relative to the completed four-window control,only trial name,anchor mode and
anchor weight differ. Same synthetic incumbent/original teacher,seed71,200 steps,
validation50,LR2e-6,EMA.95,full scope,geometry,ranking/rating/boundary losses and
all priors remain fixed. Weight32 is selected from synthetic gradients only.

Under the campaign lock,deploy exactly the staged synthetic trainer,ranking-prior
objective,and new `recsys_teacher_risk_budget.py`. The helper is byte-identical
to the calibrated one. All14 other production sources and frozen inference
remain unchanged;the new manifest has17 sources. The old16-source calibration
and completed-trial guards are historical evidence,not applicable live-source
checks after this authorized amendment. Their original reports/snapshots remain
preserved;do not rewrite them or treat expected old-hash failures as regressions.

Fresh production regression passes638 CPU tests. Its compatibility bootstrap
loads the pre-deployment trainer/objective from the calibration snapshot,so the
legacy comparison is independent of the newly deployed objective. A separate
28-case verifier suite covers registered arguments,separate risk hinges,zero
penalties,provenance and replay-corruption rejection. Production source hashes
are bound in a fresh full-size preflight snapshot and preparation receipt.

Start full-size two-context preflight as PID1675398/tool44366,with weight32 and
the actual production CLI. It must exit0 and pass the new17-source admission
verifier before any risk-budget training starts. Calibration success alone is
not training admission or benchmark evidence. The goal remains active and unmet;
no new outer evaluation or risk-budget optimizer update has occurred yet.

### Risk-budget32 passes full-size admission and starts synthetic training

Production preflight tool44366 exits0. Both context variants use the registered
2000x4000/four-window geometry,original frozen synthetic teacher and risk weight32.
Source-bound admission verifies all17 live/preflight sources,all arguments,
finite active ranking gradients,separate risk hinges,four exact replays per
context and unchanged inference. Peak allocation is126.804054GiB GPU and
79.690353GiB host,within unchanged128/160GiB caps. The verifier's28 focused CPU
cases pass again after adding actual baseline/split hash checks.

After completion/audit,atomically hard-link only the byte-identical completed
four-window step0 checkpoint to its retained common source. Both paths and
checkpoint bytes are preserved;201 metadata/reference artifact hashes remain
unchanged. Deduplication11 restores8.779091GiB free before the new initial save,
preserving the8GiB temporary-save reserve. No live checkpoint is deduplicated.

Start `full_context_four_window_risk_budget32_200` as PID1675615/tool27269.
The200-update schedule,validation50,seed71,LR2e-6,EMA.95,full trainable scope,
four-window replay and all priors remain registered and unchanged. Only the
synthetic teacher anchor changes from prediction imitation50 to separate
macro-risk excess hinges32. No real data,real-trained initialization or real
teacher enters gradient updates;public architecture and inference stay fixed.

Initial inner metrics reproduce exactly. The first-update watcher54249 exits0:
all254 selected initial tensors equal the incumbent,all17 live/training snapshot
sources match,and249389 synthetic queries replay four verified windows. First
loss1.33145666 and gradient norm5.02352762 are finite. Both risk excesses and
the anchor are correctly zero on this episode;the unpenalized state already
improves both synthetic risks versus the teacher. This is not a real-metric
safety guarantee or evidence of a BPR win.

Prepare source-amended scheduled/terminal verifiers using the tested common
risk-diagnostic checks. The expanded29-case suite also verifies refusal to
complete a live process before artifact loading. Watcher68330 now awaits
50/100/150 and audits saved errors,selection,all17 sources and synthetic risk
hinges. Terminal watcher43488 awaits200,then runs saved-error/campaign audits.
After actual main tool27269 exit0 is observed,run
`verify_risk_budget_training_completion.py --pid 1675615 --training-exit-code 0`.
The terminal watcher does not substitute its exit for the training exit.

No trained benchmark validation for this new objective exists yet. Incumbent
MF wins remain preserved,and the full BPR-plus-MF goal remains active and unmet.
Do not modify or restart the live registered run;prioritize any all8 candidate's
independent inner/Cornac reproduction before the existing gated outer protocol.
Zero new outer evaluations have occurred throughout this goal campaign.

### Risk-budget validation50 improves matched-control ranking,but not the incumbent

The previous turn made progress by completing the control,calibrating/deploying
the new objective,and starting the audited synthetic trial. This turn directly
verifies the same live PID through updates10/20/30/40,then observes validation50.
The risk anchor is active at10/30 and correctly zero at20/40;no source or schedule
change is made while training runs.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7467366308 | .7334714624 |
| RMSE | .8817216912 | .8359350012 |
| AUC | .8197417396 | .8241380612 |
| MAP | .0593207281 | .0377868971 |
| NDCG@10 | .0743100343 | .0548420580 |
| Precision@10 | .0454767726 | .0275657211 |
| Recall@10 | .0869321895 | .0730810878 |

All4 strict MF rating guards pass;0/4 BPR targets pass. The feasible key
[0,1.6837998167,1.3818717247] is worse than initialization and is rejected.
Selected step0 remains exactly initialization;no checkpoint promotion.

Against matched imitation-anchor50 at update50,MovieLens NDCG/Recall increase
by.005915/.007055 and Netflix by.015125/.021949. All four rating errors are
higher than that matched control,but remain below MF. Against the incumbent,
MovieLens NDCG/Recall are slightly higher while Netflix remains lower. Thus the
new optimization mitigates the prior control's ranking degradation but has not
improved overall incumbent selection or achieved any BPR target. Netflix RMSE
has only.001074 margin under MF,so the strict guard remains important.

The scheduled observation passes all17 live/archive source checks,synthetic
updates through50,separate risk-hinge arithmetic and four-window replay coverage.
Saved-error diagnostics reproduce all metrics without new inference,fitting or
model modification,and preserve input hashes. Netflix top10 mean item support
is968.79,first1000 fraction.710085,and ceiling fraction.085816. These are
descriptive inner diagnostics,not real-data targets or a causal explanation.

PID1675615/tool27269 remains live under the unchanged200-update schedule.
Watcher68330 now awaits100/150;terminal watcher43488 remains active. Preserve
synthetic-only gradients,original teacher,architecture and public inference.
The full eight-comparison goal remains active and unmet;zero new outer results.

### Risk-budget validation100 retains MF guards;Netflix ranking slips further

Previous goal turn classified as progress:it audited the first trained inner
validation without promoting a worse checkpoint. This turn verifies live
updates60/70/80/90 and then the scheduled100-update observation. The registered
process,training sources,synthetic prior and schedule remain unchanged.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7467536712 | .7333860140 |
| RMSE | .8803086192 | .8351532219 |
| AUC | .8202307193 | .8136998371 |
| MAP | .0633300544 | .0351370012 |
| NDCG@10 | .0785852530 | .0507277335 |
| Precision@10 | .0482885086 | .0251596614 |
| Recall@10 | .0886186776 | .0650971414 |

All4 strict MF guards pass;0/4 BPR targets pass. MovieLens NDCG/Recall improve
versus50,while both Netflix ranking targets decline. The feasible key
[0,1.8903122259,1.4445708720] is worse than50 and initialization,and is rejected.
Selected step0 remains exactly the synthetic initialization. MovieLens NDCG
is still below BPR's.0809004866;being close is not a win. Netflix RMSE margin
below MF improves to.001856,but the ranking deficit remains substantial.

Versus the matched imitation-anchor control at100,MovieLens NDCG/Recall are
higher by.008598/.006187 and Netflix by.014561/.019890. All four rating errors
remain higher than that control,but below MF. This supports a relative benefit
of the new optimization,not an incumbent improvement or joint goal success.

Observation verifies all17 live/archive sources,all11 logged synthetic updates,
risk-hinge arithmetic and replay coverage. Saved errors reproduce every metric
without new inference,fitting or model modification,and preserve input hashes.
Netflix top10 mean support is822.51,first1000 fraction.575078,and ceiling
fraction.096970. These are descriptive inner diagnostics,not training targets.

PID1675615/tool27269 remains live at elapsed2:30 after validation100. Watcher
68330 now awaits150;terminal watcher43488 awaits registered completion. Continue
all200 updates unchanged,then observe actual main exit and perform terminal
verification. Architecture/inference and synthetic-only ancestry stay fixed;
the full eight-comparison goal is active and unmet,with zero new outer results.

### Top-k-specific synthetic rating-cost probe registered;live training unchanged

The older synthetic count-order diagnostic constrains the entire catalog. That
does not measure the cost of changing only top10 membership/order. Register a
CPU-only oracle probe on six fresh synthetic cases (seeds921/922/923 and both
context modes) using the current four-window prior. Compare fixed visible-count
and conditional-oracle top10 orders at margins0/.01/.05/.1. Recover rating-risk
weights from synthetic conditional exposure; no real data or neural model is
loaded. Project latent expected ratings with weighted isotonic regression, with
unconstrained tail order and scores bounded to[1,5]. All58 focused tests pass,
including40 comparisons against an independent constrained optimizer.

This is a read-only feasibility diagnostic, not a training target, neural
feasibility proof or a benchmark result. Fixed count tie-breaking can add cost;
zero-margin ties do not guarantee the intended ranking. Population root-MSE is
not the finite-count expected macro-RMSE objective. Expected synthetic utility
ratios are not Cornac metrics. The current risk-budget200 trial and all17
production sources remain untouched. No follow-up training is registered.

### Top-k oracle costs favor small gaps;finite-gap optimizer prototype remains isolated

The six-case CPU probe completes with exit0 and3.570618GiB peak RSS. All17
production source hashes remain unchanged. The first attempt stopped after
the two921 cases because independently accumulated FP32 means differed by
more than1e-6. Its log is preserved. A disclosed arithmetic-only amendment
normalizes the ordinal probabilities in float64 and uses their exact mean,
checking the stored FP32 mean within5e-6. All six cases then rerun under the
same seeds,prior,targets and margins; no partial-result training decision.

| Top10 target | Adjacent gap | Extra synthetic rating-MSE | Population root-MSE delta | Expected MAE delta |
|---|---:|---:|---:|---:|
| Visible counts | 0 | .002630 | .002026 | .004183 |
| Visible counts | .01 | .003482 | .002700 | .005736 |
| Visible counts | .05 | .039565 | .032867 | .083566 |
| Visible counts | .1 | .313126 | .200161 | .277034 |
| Conditional oracle | 0 | .000657 | .000434 | .001166 |
| Conditional oracle | .01 | .001186 | .000856 | .002408 |

The separate global-count-order zero-margin reference costs.051288. Its
count ties are free,unlike the fixed top10 tie order,so these are different
constraint families rather than a universal lower-bound theorem. In all six
cases the top10 zero-margin cost is smaller. Count-order expected-DCG efficiency
is.828054 versus.672381 for latent expected-rating order; conditional-oracle
efficiency is1 by definition. These are synthetic utility ratios,not Cornac
metrics. Oracle knowledge and unconstrained tail scores do not establish what
DWIP can learn or what rating cost occurs on either real benchmark.

Prepare only `finite_gap_topk_prototype.py`,outside production. It retains the
two-sided union weights but replaces unbounded pairwise logistic pressure with
a Huberized hinge that stops rewarding signed gaps above.01. At a tied score
its scalar derivative exactly equals the current logistic derivative; almost
equal relevance targets still receive hard corrective pressure when misordered.
The.01 candidate is an exploratory choice made after this synthetic probe,
not a preregistered decision. It is the smallest positive margin tested; larger
gaps sharply increase the synthetic rating cost. Full-model gradients and
Adam updates are not thereby matched,and no transfer benefit is claimed.

All95 focused CPU tests pass:58 projection/provenance tests and37 prototype
tests,including exact fractional-cutoff enumeration,sparse user-macro handling,
finite differences,zero residual gradient after adequate ordering,and undiluted
20-to2000-item losses. No neural checkpoint or real data is loaded. The proposal
requires completed/audited risk-budget200 training,priority reproduction of any
all8 winner,separate exact integration registration,legacy/replay tests and fresh
full-size GPU admission. No production deployment or next training registration
has occurred. The current trial remains unchanged;goal active and unmet.

### Finite-gap integration passes isolated regression;conditional follow-up registered

Previous goal turn classified as progress:it completed the synthetic top10-cost
probe and prepared the finite-gap prototype. This turn directly confirms the
registered PID1675615 is live,then stages its trainer integration without changing
the running code. The production amendment would touch exactly four paths:
the BPR trainer,the shared synthetic-episode forwarder,the ranking objective,
and a new finite-gap helper. The shared forwarder changes only its synthetic
relevance-target objective allowlist;all normalization/inference functions stay
byte/AST-equivalent. Future training would archive18 sources rather than17.

The staged new mode is `topk_finite_gap` with margin.01,visible-frequency targets
and ordinary two-sided top-k union. It preserves the existing rating objective,
synthetic risk-budget anchor32,clipping/boundary auxiliary,prior,teacher and
four-window replay. It reports the fraction of ranking weight whose score gaps
remain below the margin;zero ranking pressure after sufficient ordering is
intentional,not a numerical failure. No production source has changed.

All131 focused tests and730 combined tests pass (599 existing regressions).
This includes48 exact archived legacy loss/diagnostic/component/logit-gradient
comparisons across four previous ranking objectives and both anchor modes,
12 exact full-objective comparisons with the registered prototype,and8
four-window gradient/RNG/AdamW replay cases including saturated ranking losses.
Initial test-only errors used catalog queries for ordinary pairwise loss and
recomputed shared predictions before asserting bitwise gradients. Corrected
fixtures use genuine user-matched generation and the archived full objective
with only its ranking helper substituted;no tolerance relaxation or production
patch was needed. Real/model loaders are forbidden in integration tests.

`finite_gap_training_registration.json` conditionally registers
`full_context_four_window_finite_gap001_risk_budget32_200`. Relative to the
active risk-budget control,only its name,ranking objective and new.01 margin
change. The registration binds the exact four-path amendment,synthetic evidence,
archived control,original teacher,initialization and frozen protocol. It does
not authorize changing/stopping the live trial or starting overlapping GPU work.
Finish and audit all200 control updates first;prioritize independent inner
reproduction of any all8 candidate. Otherwise require exact deployment,fresh
production regressions,two-mode full-size GPU admission and existing memory/
storage caps. A dedicated new preflight verifier is still pending. No new real
or outer evaluation occurred,and the eight-comparison goal remains unmet.

### Risk-budget validation150 passes MovieLens NDCG;all MF guards remain intact

The unchanged run progresses through updates130/140/150. The scheduled watcher
68330 completes with exit0 after auditing150. Main training tool27269 and
terminal watcher43488 are each directly confirmed live;the former remains
PID1675615 at elapsed3:40. The watcher exit is not the main training exit.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7470705747 | .7340925142 |
| RMSE | .8807604717 | .8358256247 |
| AUC | .8250867206 | .8159078280 |
| MAP | .0687544117 | .0366179599 |
| NDCG@10 | .0839963771 | .0530411586 |
| Precision@10 | .0500000000 | .0264072479 |
| Recall@10 | .0911931445 | .0672141842 |

All4 MF rating guards pass. MovieLens NDCG exceeds BPR's.0809004866,but its
Recall and both Netflix ranking targets still fail:1/4 BPR wins,not the goal.
Both datasets' NDCG/Recall improve versus100,but the feasible selection key
[0,1.8307731283,1.3864564063] remains worse than initialization and50. Therefore
the candidate is rejected;selected step0 retains all254 initialization tensors.
Netflix RMSE headroom is.00118324 under MF. No trained150 checkpoint is saved
under the registered save-best-only policy.

Relative to the matched imitation-anchor control at150,MovieLens NDCG/Recall
increase by.011815/.008846,and Netflix by.015891/.020679. All four rating errors
are higher than that control but below MF. This is evidence of a better
rating/ranking compromise for this optimization,not overall incumbent promotion.

The observation verifies all17 live/archive sources,synthetic visible targets,
risk-hinge arithmetic and four-window replay coverage through150. Saved-error
diagnostics reproduce all metrics without new inference,fitting or artifact
mutation. Netflix top10 mean item support is818.84,first1000 fraction.542047
and ceiling fraction.095559;these remain descriptive inner diagnostics only.

Continue the registered200 updates unchanged. The finite-gap alternative is
staged and conditionally registered only;its production bootstrap correctly
refuses before deployment and before GPU/model loading. Finish/audit this run
and prioritize any all8 candidate before using that fallback. Synthetic-only
ancestry,architecture,inference and the outer gate are unchanged. Zero new
outer evaluations;the full goal remains active and unachieved.

### Finite-gap admission and monitoring prepared while risk-budget control continues

Previous goal turn classified as progress:it completed isolated integration,
registered the conditional follow-up and audited validation150. This turn
directly confirms PID1675615 remains live and reaches update160;the active
200-update schedule,17 production sources and selected checkpoint are unchanged.

Prepare separate finite-gap admission/first-update,scheduled-observation and
completion helpers. All80 CPU verifier/monitoring tests pass. The prerequisite
requires the control's actual exit0/completion receipt,all five scheduled
validations,archived17 sources,frozen baselines/inference and no all8 candidate
awaiting reproduction. Tests explicitly ensure a winner has priority even when
a receipt falsely says there is none. The current incomplete-control gate
correctly refuses before CUDA initialization and creates no admission receipt.

Future admission requires the exact18-source deployment,fresh730-case production
regressions,the two registered full-size synthetic contexts,active finite-gap
ranking,original teacher,separate risk hinges,four-window replay and unchanged
128/160GiB memory and8GiB storage caps. Storage checks include an initial plus
temporary checkpoint. Scheduled/completion audits allow genuinely saturated
finite-gap losses to be zero,while verifying consistent zero unsatisfied weight;
they do not misclassify the intended stopping behavior as a training failure.

First-update checks retain exact initialization reproduction and all254 state
tensors/dtypes/shapes. The new helpers remain preparation,not evidence of GPU
admission or benchmark improvement. No production source is deployed,no new GPU
job starts,and no real data enters neural gradients. Finish/audit the current
run before any conditional fallback;goal active with zero new outer results.

### Risk-budget200 completes without promotion;finite-gap production preflight starts

Main tool27269 exits0 after the registered200 updates. Terminal watcher43488
also exits0 after final saved-error diagnostics and the campaign audit. The
completion verifier runs only after those separate observed exits and passes:
all21 logged updates,17 source snapshots,synthetic risk hinges/replays,original
teacher,frozen inference and all five scheduled validations reproduce.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7474429960 | .7348981606 |
| RMSE | .8807230470 | .8362858043 |
| AUC | .8265584342 | .8155014068 |
| MAP | .0705266912 | .0359782337 |
| NDCG@10 | .0855912507 | .0516942092 |
| Precision@10 | .0507334963 | .0256497846 |
| Recall@10 | .0922276557 | .0648690604 |

At200,all4 MF guards pass and MovieLens NDCG again exceeds BPR. Its Recall and
both Netflix ranking targets still fail. The key[0,1.8969586055,1.4058029213]
is worse than initialization;all trained checkpoints were rejected. Selected
step0 preserves all254 initialization tensors. Netflix RMSE has only.00072306
margin under MF. The generic audit's238 changed tensors compare against the
older campaign root,not this run's immediate initialization. No promotion or
new outer evaluation occurs.

The conditional finite-gap prerequisite now passes with no all8 candidate to
reproduce. Completed-only deduplication12 atomically links the byte-identical
selected0 checkpoint to the completed imitation control,retaining both paths,
all provenance and223 protected artifact hashes. Free space rises8.460381→
8.751049GiB. No live checkpoint is deduplicated.

Deploy exactly the four registered production paths via apply_patch. All18
resulting source hashes match their tested staging versions and are archived
for preflight;model/inference sources remain unchanged. Fresh production
regression tests pass730 cases using the immutable control as legacy reference.
The source-bound preparation receipt and storage check pass. Historical17-source
live guards are now intentionally obsolete;preserve their snapshots/receipts
rather than rewriting their expected hashes or rerunning them against new code.

Fresh two-mode full-size synthetic preflight starts as PID1689692/tool85736,
with log `full_context_four_window_finite_gap001_risk_budget32_preflight.log`.
Only the ranking surrogate and.01 margin differ from the completed risk-budget
control;the original teacher,anchor32,prior,optimizer,geometry and inference
remain fixed. No new neural training is admitted until actual preflight exit0
and the18-source admission verifier pass. The full goal remains active/unmet.

### Finite-gap full-size admission passes;registered synthetic training starts

Preflight tool85736 exits0. Both registered synthetic contexts have finite,
active ranking gradients and four verified window replays. Loss/gradient norm
are.835999/1.181068 for first-observation and.809256/1.264596 for catalog context.
Finite-gap ranking losses are.006740/.007285,and unsatisfied ranking-weight
fractions are.153755/.141716. Both synthetic teacher-risk hinges are zero.
These fractions are objective diagnostics,not recommendation accuracy or BPR
metrics;the local tied-score slope match does not imply global gradient matching.

Peak allocated GPU memory126.804054GiB and host RSS79.744862GiB remain below
128/160GiB. The18-source admission verifies the original teacher,exact arguments,
archived control completion,fresh730-case production regression evidence and
storage headroom for initial/temporary checkpoints above the8GiB reserve.
It records actual main preflight exit0;no real-data gradient target or teacher
is introduced.

The registered trial starts as PID1690319/tool23937 with200 updates and inner
validation every50. Only the finite-gap ranking surrogate and.01 margin change;
all prior,teacher,anchor32,optimizer,architecture and public-inference settings
are preserved. First-update watcher56505 verifies initialization reproduction;
scheduled watcher34026 handles50/100/150;terminal watcher52999 prepares final
diagnostics/audit after process exit. Terminal-watcher success never substitutes
for observing actual main training exit before completion verification.

Initial inner reproduction and first-update evidence remain pending at this
entry. There is no trained benchmark result for finite-gap ranking yet,and no
new outer evaluation. Goal active and unmet;do not alter or restart the live run.

### Finite-gap first update verified;read-only control comparison recorded

First-update watcher56505 exits0. Its receipt verifies all18 current/archive
sources,the registered recipe,synthetic teacher/risk hinges and four replayed
query-bearing windows. Initial inner metrics exactly reproduce the incumbent;
all254 state tensors remain equal at selected step0. This is initialization
reproduction,not evidence about a trained finite-gap checkpoint. Main tool23937
and PID1690319 remain live;the200-update schedule is unchanged.

`finite_gap_first_update_control_comparison.json` compares the two already
logged first synthetic updates without another model execution. Only trial
name,ranking objective and the new.01 margin differ between configurations;
all45 compared logged episode/replay metadata fields agree. Synthetic expected
macro-RMSE differs by-.000000656 and total rating loss by-.000017762. Both runs
have zero synthetic teacher-risk hinges. However,forward diagnostics are not
bitwise equal,including those of the frozen teacher;the cause is not established
by these logs. Equal seeds and metadata do not certify unarchived tensor equality.

The pre-clipping total gradient norm is5.023528 for the logistic control versus
.639865 for finite-gap. With the unchanged.5 global clipping threshold,the
approximate multipliers are.099532 and.781414. This does not establish gradient
component directions or AdamW update magnitudes. The effective candidate mean
also differs11.28125 versus11.3125,so this is not an exact fixed-prediction
objective comparison. Ranking scalars.436755 versus.003748 are not comparable
accuracy measures. The finite-gap unsatisfied-weight fraction is.114234;only
scheduled benchmark validation can establish transfer or retained rating wins.

Correct stale active status pointers to this trial's preflight,optimization,
initial validation and first-update receipt. Preserve completed risk-budget200
metrics separately rather than presenting them as trained finite-gap results.
No production sources,checkpoint,inference or training settings change. No new
real or outer evaluation occurs. Continue scheduled50/100/150 observations and
terminal auditing;the eight-comparison goal remains active and unmet.

The live run subsequently reaches logged update10 without changes. Loss.962580
and pre-clipping gradient norm15.247684 are finite;four-window replay and exact
loss-component/risk-hinge arithmetic verify. Expected synthetic macro-RMSE is
.748740 versus teacher.748415,so the registered risk penalty activates on excess
.000324905. Student MAE.604966 remains below teacher.606492;its excess is zero.
Ranking loss.078370 and unsatisfied-weight fraction.415952 are training
diagnostics,not benchmark measures. Only validation0 exists;the first trained
inner result remains scheduled at50. Both scheduled and terminal watchers are
directly confirmed live. No additional model run or real evaluation is started.

### Finite-gap validation50 retains MF wins but does not improve ranking

Previous goal turn classified as progress:first-update integrity and the
read-only matched comparison were completed. This turn directly verifies the
same live PID1690319/main23937 and waits on scheduled watcher34026 rather than
changing or restarting training. Logged20/30/40/50 updates remain finite;the
registered risk hinge alternates between active and inactive across episodes.
Its logged total gradient norms do not isolate component gradients or prove
which component controls AdamW updates.

Watcher34026 audits validation50 and remains live for100/150. All18 production
and snapshot sources,synthetic targets/teacher,risk-hinge arithmetic,finite-gap
diagnostics and four-window replay evidence verify. Saved-error diagnostics
reproduce every saved metric with no new inference or fitting.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7460723760 | .7323761840 |
| RMSE | .8804098969 | .8348163545 |
| AUC | .8179469387 | .8199278021 |
| MAP | .0582601852 | .0359204054 |
| NDCG@10 | .0715527454 | .0526055990 |
| Precision@10 | .0442542787 | .0265854745 |
| Recall@10 | .0819460840 | .0715065601 |

All4 MF rating guards pass,but all4 BPR ranking comparisons fail. The feasible
selection key[0,1.7208759877,1.4363995106] is worse than initialization's
[0,1.5920447872,1.3476380407]. All four ranking targets decline from initialization.
The candidate is rejected;selected step0 still preserves all254 initialization
tensors. Save-last remains disabled;there is no saved trained50 checkpoint to
promote or use as a teacher. Netflix RMSE margin under MF is.0021925090.

`finite_gap_validation50_control_comparison.json` records all seven metrics and
input hashes. Compared with matched logistic risk-budget50,finite-gap reduces
MovieLens MAE/RMSE by.000664/.001312 and Netflix by.001095/.001119,but reduces
their NDCG/Recall by.002757/.004986 and.002236/.001575 respectively. This is a
rating/ranking tradeoff,not a gain toward the full goal. Netflix top10 mean
support falls1004.70→938.17,first1000 share.749027→.704277 and ceiling share
rises.066508→.087316 versus initialization;these are descriptive inner results.

Possible later investigation is synthetic-only fixed-prediction full-model
ranking-gradient scale calibration:the finite-gap tied-score slope match did
not calibrate aggregate parameter gradients. No cause,coefficient,new recipe
or fallback is established or registered now. Finish/audit the unchanged200
updates and prioritize any all8 candidate's independent inner/Cornac gate.
Architecture,inference and ancestry stay unchanged;zero new outer evaluations.
Goal remains active and unmet.

### Conditional ranking-gradient calibration prepared;live finite-gap trial unchanged

Previous goal turn classified as progress:validation50 and matched-control
comparison established that all MF guards still pass but ranking declines.
This turn directly confirms the same PID1690319 and reaches logged update60,
with finite loss.848310 and gradient norm.775252. The registered200 updates,
all18 production sources,architecture,inference and checkpoints remain intact.

Prepare `calibrate_finite_gap_gradients.py` and conditionally preregister six
synthetic-only cases:seeds931/932/933 crossed with first-observation/catalog
contexts,episode20,model seed20260923 and the existing four2000x1000 windows.
The unchanged synthetic initialization produces one fixed set of logits per
case. Logistic and finite-gap.01 losses use those identical predictions,
visible-frequency targets,clipping and top-k-union candidate weights. Each
loss's logit gradient is replayed separately through identical window/RNG
states to measure full-model parameter-gradient norms and direction cosine.
No teacher model forward,optimizer step or real-data gradient is involved.

The preregistered rule uses the median log ratio of pure logistic/finite-gap
gradient norms and nearest coefficient from[.25,.5,1,2,4,8,16,32,64],larger
numerical tie. Require at least2 active cases per context and an in-grid median;
zero gradients are not replaced by an arbitrary scale. A future separately
registered trial would compensate `ranking_boundary_weight=1/ranking_weight`
to keep the absolute boundary coefficient1 and the rating/risk objectives
unchanged. Selected weight1 must not trigger a duplicate training run.

All68 final CPU preparation tests pass. Six generated-episode cases exactly
match both production ranking losses and logit gradients;four coefficient
cases preserve the protected loss/gradient and absolute boundary term. Four
paired replay/RNG cases cover genuine four-window queries and a legitimate
query-empty window. Tests also cover grid/coverage failures,source binding,
incomplete/live/winning-trial refusal,and precedence for any all8 candidate.

Preserve the initial preparation failures and registration revisions. The
first CPU replay fixture inadvertently used the CUDA autocast context and
initialized CUDA,without loading DWIP or running calibration. The corrected
fixture passes an explicit null CPU context and forbids CUDA initialization,
real/model loaders and checkpoint loading. A second fixture incorrectly
expected four replays for a seed with only three query-bearing windows;its
corrected coverage assertion does not weaken production replay requirements.
Subsequent process inspection shows only the existing training GPU process;
no live weights,settings or schedule were modified.

The actual registered CLI refuses before CUDA,model loading or lock acquisition
because finite-gap200 is incomplete. Preparation is not calibration,coefficient
selection or training admission. Finish/audit this run with actual main exit0
and prioritize any all8 candidate's independent inner/Cornac gate. Only a
completed nonwinner permits this serial calibration;fresh registration and
full-size preflight remain necessary before any future training. Norm matching
does not match gradient directions,total clipping or AdamW steps,and establishes
no recommendation gain. Zero new outer evaluations;goal active and unmet.

### Finite-gap validation100 preserves rating guards while Netflix ranking degrades

Previous goal turn classified as progress:the conditional gradient calibration
and68 CPU checks were completed without changing live production training.
This turn directly polls PID1690319 and watcher34026 while the same registered
run advances through70/80/90/100. Observation timeouts are live waits,not process
termination;no restart,early stopping or schedule modification occurs.

Watcher34026 audits100 and remains live for150. All18 live/archive sources,
the11 logged synthetic updates,visible teachers,finite-gap/risk diagnostics and
query-bearing window replays verify. Saved metrics reproduce without another
model execution or fitting. Main23937 and terminal watcher52999 remain assigned
to the unchanged200-update trial.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7459543769 | .7324577212 |
| RMSE | .8777791957 | .8335736172 |
| AUC | .8175451093 | .8073940613 |
| MAP | .0604648355 | .0310555268 |
| NDCG@10 | .0756284705 | .0440885855 |
| Precision@10 | .0474327628 | .0223674439 |
| Recall@10 | .0865909602 | .0572979027 |

All4 MF rating guards pass and all4 BPR ranking targets fail. MovieLens ranking
recovers above initialization,but Netflix falls further below both50 and0.
The selection key[0,2.1476165191,1.5892186645] is worse than initialization and50;
selected0 still exactly preserves all254 initialization tensors. The rejected
trained100 weights are not saved. Netflix RMSE headroom under MF is.0034352463.

`finite_gap_validation100_control_comparison.json` binds all seven metrics and
input hashes. Relative to logistic control100,MovieLens MAE/RMSE improve by
.000799/.002529 and Netflix by.000928/.001580,but their NDCG/Recall decline by
.002957/.002028 and.006639/.007799. Netflix top10 mean support is718.41,
first1000 share.533299 and ceiling share.097564;these are descriptive inner
diagnostics,not new inference corrections or fitted ranking targets.

A read-only source check rejects one possible explanation:public candidate
inference normalizes the full observed matrix before blocking,and the synthetic
episode also fits parent normalization before cropping. There is no newly
identified per-window-normalization discrepancy to patch. This does not rule
out other prior/population/catalog differences or establish their effects.

The six-case gradient-calibration registration remains unchanged from its
post50 preparation;no coefficient is chosen from these100 metrics. Preserve
the live200-update trial and wait for150/200. Independent inner/Cornac
reproduction of any all8 candidate takes priority over conditional calibration.
Architecture,inference and synthetic-only ancestry are unchanged. No new outer
evaluation;the full goal remains active and unachieved.

### Finite-gap validation150 rejected;scheduled watcher completes normally

Previous goal turn classified as progress:validation100 and the matched
comparison were audited. This turn directly confirms the same live process
through110/120/130/140/150,without modifying or restarting training. The
registered schedule,synthetic-only ancestry and all18 production sources remain
unchanged. Watcher34026 exits0 after auditing50/100/150;that is not a training
exit. Main PID1690319/tool23937 continues toward200,and terminal watcher52999
remains separate.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7463688112 | .7337070098 |
| RMSE | .8777028136 | .8343143345 |
| AUC | .8202852956 | .8059283551 |
| MAP | .0636586504 | .0299626428 |
| NDCG@10 | .0774106110 | .0422004872 |
| Precision@10 | .0475550122 | .0215208674 |
| Recall@10 | .0865734067 | .0535491601 |

All4 MF rating guards pass,but no BPR ranking comparison passes. Netflix ranking
declines again from100. The key[0,2.2979617603,1.6423763893] is worse than all
previous validations. Selected0 retains all254 initialization tensors;rejected
trained150 weights are not saved. Netflix RMSE margin under MF is.0026945290.

`finite_gap_validation150_control_comparison.json` records all seven metrics,
input hashes and the unchanged selection. Relative to logistic control150,
MovieLens MAE/RMSE are lower by.000702/.003058 and Netflix by.000386/.001511,
but NDCG/Recall are lower by.006586/.004620 and.010841/.013665 respectively.
Unlike that control,this run does not pass MovieLens NDCG at150. Netflix top10
mean item support is660.25,first1000 share.469880 and ceiling share.098114;
these remain descriptive inner diagnostics only.

The observation verifies all16 logged synthetic updates,risk hinges,finite-gap
termination diagnostics,query-bearing window replays,frozen baselines/inference
and18 live/archive sources. Saved-error diagnostics reproduce the saved metrics
without new model execution,fitting or input mutation. The prepared gradient
calibration remains unrun and unmodified;no coefficient is selected.

Finish200 unchanged,observe actual main23937 exit separately,and wait for
terminal diagnostics/audit before running the completion verifier. Any all8
candidate takes priority for independent inner/Cornac reproduction. Only an
audited nonwinner permits the conditional serial gradient calibration. No new
outer evaluation;goal remains active and unmet.

### Finite-gap200 completes without promotion;synthetic gradient calibration starts

Previous goal turn classified as progress:validation150 and its matched
comparison were audited. This turn observes main tool23937 exit0 after all200
registered updates,then separately observes terminal watcher52999 exit0 after
saved-error diagnostics and the campaign audit. Completion verifier22297 exits0
and verifies all21 logged synthetic updates,18 live/archive sources,original
teacher,risk/replay diagnostics,all five validations and checkpoint selection.

| Metric | MovieLens | Netflix |
|---|---:|---:|
| MAE | .7467395673 | .7345237463 |
| RMSE | .8778007932 | .8349096945 |
| AUC | .8213519137 | .8057757871 |
| MAP | .0654953121 | .0290063393 |
| NDCG@10 | .0803102963 | .0404614806 |
| Precision@10 | .0487775061 | .0205851775 |
| Recall@10 | .0901134520 | .0516260424 |

Every validation retains4 MF rating wins and0 BPR ranking wins. MovieLens NDCG
at200 remains strictly below BPR's.0809004866;do not round it into a win. The
final key[0,2.3835629555,1.6642767976] is worse than initialization and all earlier
validations. Every trained candidate is rejected;selected0 preserves all254
initialization tensors. No rejected trained weights are saved or promoted.
The generic audit's238 changed tensors compare against the older campaign root,
not this trial's immediate initialization.

`finite_gap_validation200_control_comparison.json` confirms the full pattern:
at50/100/150/200,all four rating errors are lower than the matched logistic
risk-budget control,but all four ranking targets are lower too. Finite-gap1
therefore improves that rating/ranking tradeoff in the wrong direction for the
full objective. No all8 candidate exists for independent reproduction,and no
new outer evaluation is authorized.

The conditionally registered gradient-calibration gate now passes on the
audited completed nonwinner. After confirming the GPU has no compute process,
start the unchanged six-case `calibrate_finite_gap_gradients.py` as
PID1712560/tool21017,logging to `finite_gap_gradient_calibration.log`. This
compares pure ranking parameter gradients on identical synthetic predictions;
it performs no optimizer step and uses no real-data targets or teacher.
The original synthetic teacher file/ancestry and initialization remain fixed.

Model/source preservation,memory caps,all six case results and the coefficient
are still pending at this entry. Calibration success will not itself admit
training or establish benchmark transfer. Weight1 must not launch a duplicate
run;any distinct accepted weight still needs separate registration and fresh
full-size preflight with the absolute boundary coefficient held at1. The full
goal remains active and unmet,with zero new outer evaluations.

### Synthetic gradient calibration selects weight2;fresh training admission pending

Calibration tool21017 exits0. The six preregistered cases all have finite,active
ranking gradients over238 trainable tensors and four verified window replays
for each loss. The source-bound calibrator preserves all model-state tensors,
the synthetic initialization and original teacher file;it performs no optimizer
step and loads no teacher model or real-data targets. Completion audit10394
exits0,verifying the case/log/report agreement,live/archive sources,ancestry,
RNG/replay evidence,preparation hashes and registered coefficient rule.

| Seed | Context | Logistic/finite-gap gradient norm | Direction cosine |
|---|---|---:|---:|
| 931 | first-observation | 1.562118 | .992607 |
| 931 | catalog | 1.577295 | .991726 |
| 932 | first-observation | 1.962999 | .604373 |
| 932 | catalog | 25.683326 | .230034 |
| 933 | first-observation | 1.880191 | .999044 |
| 933 | catalog | 1.879901 | .999134 |

The log-median target is1.8800457008,so the registered grid selects ranking
weight2. Set a future boundary multiplier to.5 to preserve its absolute
coefficient1;do not double that auxiliary unintentionally. The large932 catalog
ratio and low direction cosine remain disclosed:one scalar does not match all
cases,directions,total clipped gradients or AdamW updates. These results do
not establish any recommendation gain or resolve the eight-comparison goal.

Peak allocated GPU124.239539GiB and host RSS91.062069GiB pass the unchanged
128/160GiB caps. The completion receipt is
`finite_gap_gradient_calibration_completion_receipt.json`. The calibration
registration remains unchanged from preparation,and no coefficient was chosen
from the later100/150/200 real inner results.

After both training and calibration processes are absent,completed-only
deduplication13 atomically links the byte-identical finite-gap selected0 blob
to the completed risk-budget control. Both checkpoint paths,all provenance and
205 protected artifact hashes remain unchanged. Free space rises8.440243→
8.730911GiB;no live checkpoint is touched.

Next candidate is `full_context_four_window_finite_gap001_weight2_risk_budget32_200`:
same synthetic initialization,teacher,prior,seed71,LR2e-6,EMA.95,200 updates and
validation every50;only trial name and the coupled2/.5 coefficients change.
No production-source amendment is needed. This candidate is not yet registered
or admitted:prepare its source-bound checks and run fresh two-mode full-size
preflight before training. Existing argument/finite-gap diagnostic helpers are
parameterized by registration,but the previous hard-coded control/preparation
gate must not be reused unchanged. Preserve the8GiB initial/temporary write
reserve and all original comparison/outer gates. Goal active and unmet.

### Calibrated weight2 registration and fresh preflight — 2026-09-23

The proposed weight2 control is now registered in
`finite_gap_weight2_registration.json` (SHA256
`475680bdd4970e2d73aad0f2cdbadf504d65a1f23b129183141cfd522b953d65`).
Relative to completed finite-gap weight1,only name,ranking weight2 and boundary
multiplier.5 change. The absolute boundary coefficient remains1;the original
synthetic rating teacher,risk32,prior,initialization and complete200/50 schedule
are preserved. No production,inference or architecture source changes.

The new source-bound `finite_gap_weight2_trial.py` registers,prepares,admits and
supervises this control without modifying the historical frozen verifiers.
It reuses only their parameterized source/argument/finite-gap checks. The
comparison must be complete with no all8 candidate;the six-case calibration
receipt and its selection must reproduce. New CPU tests cover exact command
parsing,2/.5 coupling,argument drift,synthetic diagnostics,two-mode replay,
resource failures,saturation and nonzero preflight exit refusal. All183 tests
pass,including68 frozen calibration and80 historical verifier tests. The
previous730 production regression tests are reused with exactly identical18
source hashes;they were not rerun or misreported as fresh.

`finite_gap_weight2_preparation_receipt.json` binds tests,registration and
preflight source snapshots. Free space8.726326GiB retains more than8GiB after
initial-plus-temporary checkpoint allowance. A fresh full-size two-mode
synthetic preflight starts in tool66218,PID1716831. Training is not yet admitted:
observe actual exit0 and validate this new report before launching the
registered supervisor. That supervisor watches first update,50/100/150 inner
validations,and terminal200 diagnostics/audit;verification failures stop its
child rather than silently continuing. An all8 candidate requires independent
inner/Cornac reproduction before the existing outer gate. No new outer
evaluation;goal remains active and unmet.

The fresh preflight now completes with observed tool66218 exit0 and absent
PID1716831. Source-bound admission passes in
`finite_gap_weight2_preflight_receipt.json`:GPU126.804054GiB and host79.748108GiB
are below128/160GiB caps. Both contexts have active finite ranking,finite
positive gradient norms and four verified replays. The teacher stays frozen;
no real data is loaded. Initial-plus-temporary checkpoint reserve still passes.
The two preflight losses are.84279513 and.81683207,with ranking components
.00676742 and.00743418. Matched seed does not mean bit-identical GPU predictions
across historical processes:episode metadata agrees exactly,but small numerical
prediction/target differences occur. The claim is unchanged protected objective
and coefficients,not identical cross-process scalar losses.

Registered training launches under supervisor tool19139,childPID1717414.
First-update and50/100/150/terminal200 checks run within that same supervisor;
their receipts distinguish observations from completed-trial evidence. The
previous matched trial took about4.9hours. This launch is not benchmark success
and does not alter the incumbent checkpoint. Await first-update verification.

The first-update observation now passes in
`full_context_four_window_finite_gap001_weight2_risk_budget32_200/inner_validation_0_observation.json`.
Its step0 title refers to the initial inner validation;it also verifies the
actual first synthetic update. Loss.90216607,preclip gradient norm.73120046,
ranking.00373485,zero active teacher-risk hinge and four replays are finite and
registered. All18 live/archived sources,2/.5 weights and original frozen
synthetic teacher verify. Selected0 contains exactly the254 initialization
tensors (serialization SHA`f806b21ac9e3f9ce5d6acd3cc8337c7d0182c0a73a8880fc514d4f0076ec2802`).
Initial metrics reproduce exactly with4 MF wins and0 BPR wins;this is not a
trained-candidate gain. Supervisor19139 and childPID1717414 remain live,with
validation50 next. Do not start another GPU job or deduplicate its live
checkpoint. No new outer evaluation;goal active and unmet.

Continuation classifies the preceding registration/preflight/launch turn as
progress and this turn as a verified wait. Supervisor19139 is polled live twice
for300seconds;OS checks independently confirm trainingPID1717414 and
supervisorPID1717343. A fresh CPU-only verification(tool91731 exit0)checks every
logged update so far,1 and10,against the registered synthetic diagnostics,
arguments,18 source hashes,original teacher and checkpoint ancestry. Update10
loss1.04464948,preclip gradient norm15.8101883,ranking.07788095 and teacher-risk
hinge.00046700 are finite;four replays remain verified. The global clipping
threshold is still.5,not the logged preclip norm. There is no trained-candidate
validation yet. Leave the live registered run unchanged and await validation50;
initial MF wins are not evidence of trained-weight guardrail retention.

### Calibrated weight2 validation50 retains MF wins but is rejected

The next continuation waits on the actual live supervisor19139 through the
first scheduled validation. OS checks confirm both supervisorPID1717343 and
trainingPID1717414 remain live;the preceding turn was a verified wait and this
turn yields new inner-evaluation evidence. The registered schedule is unchanged.

| Inner metric at weight2 update50 | MovieLens 100K | Netflix small |
|---|---:|---:|
| MAE | .7467178722 | .7336695454 |
| RMSE | .8817764609 | .8363201514 |
| AUC | .8210868063 | .8264137830 |
| MAP | .0598321638 | .0385071447 |
| NDCG@10 | .0729550969 | .0559649906 |
| Precision@10 | .0455990220 | .0283528888 |
| Recall@10 | .0831380551 | .0744135685 |

All four MF rating guards pass,but no BPR target passes. Netflix RMSE's strict
margin over MF is only.0006887122. The unchanged selection key
`[0,1.653648988424322,1.3852420949391542]` rejects this candidate;selected0
retains the exact initialization tensors and its original serialization hash.
All four ranking targets remain below the incumbent,so this is not progress
toward the full gate merely because the rating guards pass.

Compared with finite-gap weight1 at50,weight2 raises MovieLens NDCG/Recall by
.00140235/.00119197 and Netflix by.00335939/.00290701,while worsening all four
rating errors. The worst BPR ratio improves1.720876→1.653649 but remains worse
than the incumbent1.592045. Relative to matched logistic risk-budget32,weight2
improves both Netflix ranking targets but worsens both MovieLens targets;
rating changes are mixed. Norm calibration did not reproduce logistic behavior
or solve the transfer problem.

The scheduled receipt `inner_validation_50_observation.json` verifies all
logged updates1/10/20/30/40/50,synthetic-only targets,original frozen teacher,
18 live/archived sources,four-window replays and finite-gap/risk diagnostics.
Saved-error diagnostics reproduce every archived metric without new inference,
training or fitted correction. Macro prediction shifts are+.04049069 and
+.04209818;these are descriptive,not applied calibrations.

`compare_finite_gap_weight2_validation.py --step 50` independently recomputes
canonical selection keys and descriptive matched-step metric differences from
the saved inner reports,after checking the registration and scheduled audit.
Tool29057 exits0 and writes
`finite_gap_weight2_validation50_control_comparison.json`,binding its inputs and
analysis source. It does not load a neural model,modify training or perform
additional inference. This is one training seed on previously used inner data,
not blind evaluation or a statistical significance claim.

Training and supervisor remain live after the audit;validation100 is next.
Keep200 updates and all unchanged gates. No new fallback trial,promotion or
outer evaluation;the full objective remains active and unmet.

### Stopped at the user's explicit request — 2026-09-23

The user stops this joint ranking/rating goal. The goal service reports
`paused`. SIGTERM stops trainingPID1717414;the supervisor records actual child
exit-15 in `process_exit.json` and tool19139 exits1 because the registered
schedule did not finish. Both training and supervisorPID1717343 are verified
absent,and the GPU compute-process list is empty.

Last logged update is60;last completed inner validation is50,with4 MF wins and
0 BPR wins. Selected0,all logs and evaluation artifacts remain preserved. The
interrupted trial is marked `terminated_by_user`,not completed or promoted.
The separately prepared CPU proximal-ranking prototype remains isolated and
unregistered for neural training;it never changes the live production sources
or model. No further training,evaluation,calibration,prototype development or
automatic follow-up is authorized without a new explicit user request.
The joint objective is not achieved;stopping is not a success claim.
