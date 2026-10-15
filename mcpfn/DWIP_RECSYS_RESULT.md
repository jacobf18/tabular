# DWIP v2-2 recommender-system result — corrected

**Goal achieved with synthetic-only training and unchanged DWIP v2-2. The
earlier real-trained completion claim remains retracted.**
Only synthetic data may train the neural model. The previous final checkpoint
used real inner-training ratings, violating that requirement even though it
excluded validation/test labels. Its numerical scores are not eligible evidence
of success. Both real-adapted runs and every descendant are ineligible as
initializations, teachers or final candidates.

## Confirmed synthetic-only candidate

- Run: `runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/`
- Checkpoint: `best_validation_checkpoint.pth`, EMA step **100**.
- SHA256: `1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770`.
- Required inference: **2000x1000** public overlapping-tail blocks, selected
  through `runs/recsys_goal_20260920/user_context_inner_screen.json`.
- Hash-bound lineage receipts verify continuation from the audited clean
  synthetic step-125 root, never from either real-trained run.
- Unchanged DWIP v2-2: 254 state tensors, 77,998,688 state elements; no MF
  factors, extra head or inference prediction ensemble.

### Frozen outer scores

| Metric | MovieLens MF | MovieLens synthetic DWIP | Netflix MF | Netflix synthetic DWIP |
|---|---:|---:|---:|---:|
| MAE, lower | 0.743013 | **0.727878** | 0.735777 | **0.732989** |
| RMSE, lower | 0.899831 | **0.887358** | 0.862117 | **0.861721** |
| AUC, higher | 0.744503 | **0.803480** | 0.733257 | **0.769587** |
| MAP, higher | 0.054848 | **0.074635** | 0.019811 | **0.023371** |
| NDCG@10, higher | 0.076114 | **0.099393** | 0.032350 | **0.034902** |
| Precision@10, higher | 0.067538 | **0.086057** | 0.023722 | **0.024645** |
| Recall@10, higher | 0.046308 | **0.067176** | 0.024162 | **0.029007** |

**All 14 comparisons win.** Full precision results are preserved in this run's
`context_2000x1000_test/results.csv` and `comparison.json`. This is the seventh
eligible candidate recipe evaluated on the
previously observed benchmark, not a fresh blind-test result. All 14 inner
comparisons win. Netflix outer MAE's margin is **0.002787811**, and RMSE wins
by **0.000395432**. The RMSE margin is small; no statistical-significance claim
is made. No neural updates occur in the final inference-geometry experiment.

The previous clean candidate (`full_context_expected_mae_250`, step 125)
won 12/14, with Netflix MAE/RMSE 0.739807547 / 0.865899839. The current model
improves both rating metrics over that original clean checkpoint; all results
remain preserved. Relative to these same selected weights at 1000x1000,
Netflix MAE improves by 0.001011567 and RMSE by 0.001868777. MovieLens scores
are unchanged; all five Netflix ranking scores decrease but still beat MF.

## Synthetic training improvement

The selected trial optimizes expected finite-query user-RMSE under synthetic
ordinal probabilities: exact expected absolute error for singleton users and
32 antithetic noise draws for multi-query users. It retains NLL, ranking and
expected-MAE auxiliaries. Increasing the synthetic query cap from 1024 to 4096
reduces the pooled singleton-user fraction from 90% to 23%, versus 24% in
Netflix inner evaluation. No real rating enters these objectives.

The current run uses 12288x4096 synthetic parents and 1000x1000 neural crops,
with mixed Gaussian/clustered factors and uniform-rating contamination drawn
per episode from 0-0.15. Its user-RMSE surrogate replicates sampled query profiles
to synthetic parent query counts; this approximates, rather than exactly
computes, full-parent prediction risk. It also draws synthetic user-specific
Gaussian noise scales with episode log-standard-deviation 0.15-0.6. It uses
seed 56, LR 5e-6, EMA .95, 400 updates and validation every 100. Step zero is
eligible; the lowest worst-of-four inner rating ratio selects step 100.
The architecture is unchanged. This trial adds the registered context-only
user-scale normalization to both synthetic training and evaluation inference.

The final improvement comes from **more user context at inference**, not a
different architecture or real-data update. An inner-only screen rejects
1000x2000 item windows but selects 2000x1000 user windows, lowering the worst
inner rating ratio from 0.996470996 to 0.994059818 with bitwise-identical weights.
The full inner gate precedes the single registered outer confirmation. This
uses more memory: query-only peak allocation grows from **22.549 to 44.340 GiB**,
and full Netflix public inference peaks at **93.481 GiB**. A synthetic-input preflight verifies query/public
parity within 6.89e-8 and preservation of observed ratings.

## Training and evaluation boundary

Neural gradient updates use synthetic episodes exclusively. Evaluation uses
the frozen outer 80/20 split (seed 123), fixed 10% inner holdout (seed 20260920),
unknown exclusion, relevance threshold >=4 and user-macro metrics. Checkpoint
selection uses only inner validation. Baseline MF fitting remains evaluation
infrastructure, never neural pretraining or a teacher.

Inference uses inner-selected **2000x1000** overlapping-tail public blocks, predictive mean,
[1,5] clipping and observed-rating preservation. Context-only two-way ridge
centering estimates user/item offsets from observed real evaluation context
(regularization 3, 30 iterations), then restores them after prediction. This is
disclosed inference preprocessing, not neural weight training; it does not use
held-out labels. Use the candidate evaluator rather than default column scaling.
After centering, each row is scaled using its observed residual variance,
shrunk with prior-count 20 and bounded to [0.5,2]. The inverse transform restores
raw rating units before clipping. No held-out value enters scale estimation.

## Retractions and safeguards

`mixed_observed_adaptation_200` and `mixed_macro_rating_rank_200` are ineligible.
Their original artifacts and completion audit are preserved, with adjacent
`eligibility_retraction.json` notices superseding any completion interpretation.
Synthetic fine-tuning of these checkpoints does not clean their ancestry.

The active trainer and preflight no longer expose real-data adaptation. The
observed-rating episode module is removed. Training, preflight and candidate
evaluation require either a content-hashed audited synthetic root or a verified
synthetic lineage receipt. Missing, changed or non-synthetic provenance fails
closed before model loading. Every neural training episode must explicitly
identify synthetic gradient data. These are workflow safeguards, not proof
against deliberately forged provenance or arbitrary external training code.

## Rejected follow-ups

The no-MAE loss ablation did not improve inner selection: its selected
step-zero weights exactly matched the preceding user-noise checkpoint, so it
received no duplicate outer evaluation. The user-scaled trial then improved
both Netflix rating metrics, but did not close the remaining RMSE gap.

The follow-up `synthetic_user_scaled_mean_anchor_400` adds weight 0.25 on
user-macro squared error to analytic synthetic conditional mean ratings while
retaining the successful prior, normalization and MAE auxiliary. It finishes
400 updates without improving the fixed selector. Its selected step-zero
weights and inference configuration exactly match the confirmed checkpoint;
no duplicate outer evaluation is performed. This loss change is not promoted.

Both training runs are finished; no GPU job is active. The wider-item inference
screen also fails inner selection and is not promoted. The separately registered
wider-user screen succeeds and supplies the final 2000x1000 geometry. No model
ensemble, MF factors, extra head or real-trained teacher is introduced.

## Correction validation

The initial correction passes 69 focused tests across `test_dwip_recsys_adaptation.py`,
`test_dwip_recsys_experiment.py` and `test_dwip_recsys.py`. Coverage includes
prohibited flags, unknown weights, real-trained ancestors, altered receipts,
synthetic-only sampling, storage protection and unchanged objective math.
That historical artifact audit accepts the then-selected clean checkpoint, rejects both actual
real-trained checkpoints, verifies the selected step and original state
keys/shapes/dtypes, and recomputes 12/14 wins from the stored outer scores.
No new neural training or outer inference is used for these checks.

The final recipe is audited in
`synthetic_user_scaled_parent_risk_400/context_2000x1000_completion_audit.json`:
fixed query arrays, checkpoint/config/source/split/baseline hashes, original
architecture and all twelve checkpoint ancestors verify. Independently computed
rating scores and independently reconstructed outer ranking scores agree with
Cornac, including every user's five ranking metrics and tie semantics. It
confirms **14/14 inner and 14/14 outer wins**. The earlier `benchmark_audit.json`
remains historical evidence of 13/14 at the old 1000x1000 geometry.
Current focused tests: **183 passed**, including synthetic-mixture and
user-specific-noise math, parent-query-risk gradients, provenance guards,
context-variability diagnostics and deterministic preflight coverage. The
context-only user-scale normalizer passes unit/inference-parity tests and is
used by the confirmed checkpoint. The optional conditional-mean auxiliary
passes analytic gradient, eligibility and default-equivalence tests; it is
not used by the confirmed checkpoint.

Reproduce the eligible candidate with a new output directory:

```bash
PYTHONPATH=mcpfn/src:mcpfn/scripts /usr/bin/python mcpfn/scripts/evaluate_dwip_recsys_candidate.py \
  --checkpoint mcpfn/runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth \
  --context-screen mcpfn/runs/recsys_goal_20260920/user_context_inner_screen.json \
  --partition test \
  --output mcpfn/runs/recsys_goal_20260920/synthetic_reproduction_new
```
