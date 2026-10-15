# Jellyfish counterfactual evaluation: data-access prerequisite

## Status

Reviewed on 2026-10-12. The paper and existing MatrixFormer+ causal inference
code have been inspected. No Jellyfish evaluation has been run, and no model
has been trained. The required firm-month panel is unavailable in this workspace.

The linked paper is **Artificial Intelligence in the Firm: Bottlenecks in
Software Production**, by Fiona Chen and James Stratton, current version August
4, 2026. Source: <https://fion.ac/jellyfish.pdf>.

Downloaded PDF SHA-256:
`27f8d1fe0a2c145e700ae8835c997533fb4483e21a11f9c2385186bdb9f7c096`.

The abstract and Section 2.1 describe proprietary Jellyfish data covering 718
firms and 725,938 workers, January 2021 through March 2026. The PDF and the
author's research page contain no identified public replication-data download.
Published aggregate figures cannot reconstruct the underlying panel.

## What the paper estimates

Section 3.1 uses firm-by-month event studies with firm and month fixed effects,
time-varying controls for baseline firm size, and firm-clustered standard errors.
Event time -1 is the omitted category; periods outside the main event window
are binned. Appendix Figure 1 compares alternative staggered-DiD estimators.

Two distinct treatment contrasts require separate analyses:

- **Assistants:** initial AI adoption, measured using the earliest Copilot or
  Cursor seat activation. The paper describes a 16-month diffusion window to
  avoid overlap with agent adoption.
- **Agents:** the additional effect of agent adoption relative to assistant
  adoption. The treatment dates use the earliest Claude Code API or GitHub
  agent signal. This is an incremental effect, rather than the effect of all AI
  relative to no AI.

Illustrative published pooled estimates, for comparison rather than prediction
ground truth:

| Outcome | Agent effect (SE) | Assistant effect (SE) | Source |
|---|---:|---:|---|
| Lines added per worker-month | 1494.39 (582.83) | 420.10 (370.59) | Figure 6 |
| Commits per worker-month | 4.57 (1.12) | 1.40 (0.64) | Figure 6 |
| Pull requests per worker-month | 1.22 (0.29) | 0.19 (0.17) | Figure 6 |
| Issues resolved per worker-month | 0.12 (0.17) | 0.09 (0.10) | Figure 7 |
| Review duration, days | 3.45 (1.60) | -0.38 (0.97) | Figure 11 |

## Required data

An authorized, anonymized firm-month export is sufficient; individual workers'
identities, source code, and Jira text are unnecessary. CSV or Parquet should
include the following, either as separate tables or repeated firm-level fields:

- Stable anonymous firm ID and calendar month.
- Outcomes in the paper's units, including their valid-observation indicators.
  Original activity counts and the applicable worker denominators can be used
  if the analysis outcomes have yet to be constructed.
- Assistant and agent adoption dates for every firm. Distinguish confirmed
  non-adoption from an unknown or unavailable adoption date.
- Baseline firm-size controls, their reference dates, and any original sampling
  weights or analysis-sample flags.
- Original sample restrictions, event-window limits, tail bins, and aggregation
  rules for each published result. Replication code would resolve these choices.

Preserve missing observations separately from genuine zero activity. Exact
paper replication also requires matching denominator construction and control
interactions; these details should come from the authors' analysis files.

## Proposed frozen-model protocol

1. Form a firm-by-calendar-month matrix for each outcome and treatment contrast.
   Use the original outcome units and declared sample restrictions. Retain the
   general and specialist experts as diagnostics alongside their fixed blend.
2. Remove all exposed outcomes from inference inputs for the chosen contrast.
   Every treated firm's outcomes from its adoption month onward are hidden,
   including outcomes of other treated firms. Never-treated and not-yet-treated
   firm-months provide untreated support.
3. For the agent contrast, establish the assistant-only counterfactual and donor
   eligibility using both adoption dates and the original sample rules. Simply
   treating firms without assistants as comparable controls would change the
   estimand. Censor or otherwise handle second-treatment exposure explicitly in
   the assistant analysis.
4. Use the existing frozen MatrixFormer+ checkpoints, column normalization, and
   observed-error convex blending. Normalization statistics and blend weights
   use only the permitted observed cells. Neither neural checkpoint receives
   gradient updates. Original firm-by-time orientation is the primary analysis;
   any orientation sensitivity is declared separately.
5. Predict counterfactual untreated outcomes, then compute observed-minus-imputed
   outcome gaps and event-time summaries. Match published sample weights and
   pooling rules before comparing effect estimates. A matrix-completion effect
   estimator is an alternative specification, not an exact TWFE replication.
6. For predictive validation, predeclare block holdouts in known untreated
   periods, using pseudo-adoption dates and suffix masks. Hide each placebo
   unit's subsequent outcomes from its inference context; continue to exclude
   actual treated outcomes everywhere. Fit scaling and blending again on the
   reduced observed panel. Report RMSE, MAE, mean placebo effect, and event-time
   placebo gaps on common supported cells.
7. Report per-month untreated donor counts and per-firm pre-treatment support.
   Agent adoption exceeds 95% near the sample end, according to Section 2.4.
   Restrict unsupported horizons rather than assigning credible causal estimates
   to all-treated months or firms without a pre-treatment history.
8. Compare the paper's event-study specification and suitable heterogeneous-DiD
   or untreated-only imputation baselines on the same sample. Report matched
   samples and support separately whenever an estimator drops observations.
   Fit classical causal baselines as required, while keeping both neural
   checkpoints frozen.
9. For uncertainty, resample at the firm level and repeat the full estimation
   pipeline, including normalization, blending, and imputation. Label any
   cheaper conditional-on-predictions intervals separately. Report placebo and
   pretrend diagnostics; causal interpretation still requires identification
   assumptions such as no anticipation, suitable untreated-outcome structure,
   and absence of relevant spillovers.

Actual treated counterfactual outcomes are unobserved. Therefore, treated-block
RMSE and error relative to the true treatment effect are unavailable on this
observational dataset. Disagreement with the paper's point estimates is a
robustness result, not evidence that either estimator is more accurate.

## Existing integration points

- `mcpfn/scripts/evaluate_causal_matrixformer_plus.py`: frozen expert factory,
  causal column-standardization recipe, and original-orientation blend.
- `mcpfn/src/tabimpute/routing.py`: `TwoExpertRouter(policy="observed_blend")`.
- General checkpoint:
  `mcpfn/runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth`.
- Specialist checkpoint:
  `mcpfn/runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth`.

The existing causal runner evaluates other saved benchmark panels. It does not
yet load Jellyfish data. A paper-specific loader and evaluator should be added
once the panel schema and original treatment/sample rules are available.

No synthetic proxy or previously evaluated causal panel should be labeled as a
result on the Jellyfish study.
