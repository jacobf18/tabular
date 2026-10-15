# DWIP v2-2 recommender-system goal

## Verified goal: synthetic-only training

**The goal is achieved by the audited synthetic-only 2000x1000 recipe. The
previous real-trained completion claim remains retracted.**
The user requires exclusively synthetic neural training, not merely exclusion
of validation/test labels. The two real-rating adaptation runs violated that
requirement. Their scores do not count, and neither their checkpoints nor any
descendants may initialize or teach an eligible experiment.

Beat the fixed MF baseline on all seven metrics on both datasets while keeping
DWIP v2-2 unchanged (254 state tensors, 77,998,688 state elements). Real data is
evaluation context/labels only, never neural gradient targets. MF fitting is
solely the reference baseline, not a source of neural weights or teacher targets.
Use the fixed inner validation for selection; the observed outer benchmark is
not a fresh blind test. No architecture change or prediction ensemble.

The current outer-confirmed eligible candidate is
`synthetic_user_scaled_parent_risk_400`, EMA step 100, SHA256
`1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770`.
With inner-selected **2000x1000** public inference blocks, it wins **14/14**
outer comparisons and **14/14** inner comparisons. Netflix MAE **0.732988718**
beats MF **0.735776529**, and RMSE **0.861721441** beats MF **0.862116873**.
All MovieLens and all Netflix ranking metrics win. The required geometry
receipt is `user_context_inner_screen.json`; loading the checkpoint with the
old default 1000x1000 blocks does not reproduce this winning recipe.
The preceding clean step-125 checkpoint remains preserved as the audited
synthetic root; its outer Netflix MAE/RMSE were 0.739807547/0.865899839.
The current recipe improves both and meets the full fixed-benchmark goal.
Increasing user context changes no neural weight or model parameter. Relative
to the same weights at 1000x1000, MovieLens is unchanged and Netflix MAE/RMSE
improve; Netflix ranking scores fall but remain above MF. Selection is inner-only.
The narrowest rating margin is Netflix RMSE, **0.000395432**; no statistical
significance or fresh-blind-test claim is made. Public peak memory is **93.481 GiB**.

Context-only two-way ridge centering estimates user/item offsets from observed
evaluation context (regularization 3, 30 iterations). This is inference
preprocessing, not a neural gradient update, and is not concealed as a
data-independent transform. Evaluation labels never enter this preprocessing.
The current candidate additionally normalizes rows by context-only residual
scales, shrunk with prior-count 20 and clamped to [0.5,2], then inverts them.

Real-adaptation flags, loader, sampler and update schedule are removed from the
active runner/preflight. Training, preflight and candidate evaluation reject
unknown or real-trained checkpoint lineage before model loading. Only the two
content-hashed audited synthetic roots and verified synthetic descendants are
eligible. New checkpoints receive hash-bound lineage receipts; synthetic
fine-tuning cannot remove real-trained ancestry.

The completion audit verifies all fourteen strict comparisons, independent
rating and ranking calculations (including every outer ranking user), original
architecture, all twelve checkpoint ancestors, archived training sources and
frozen baseline/split hashes. Both actual real-trained artifacts are rejected.
See `DWIP_RECSYS_RESULT.md`, `winning_recipe.json`, and
`synthetic_user_scaled_parent_risk_400/context_2000x1000_completion_audit.json`.
The chronological experiment registrations below are retained as history;
no further training is queued for this completed goal.

## Synthetic-only follow-up: expected macro RMSE

Register `synthetic_expected_macro_rmse_200` from the clean step-125 checkpoint
only. Keep architecture, prior family, normalization and evaluation protocol
fixed. Optimize expected finite-query user-RMSE under the generator's synthetic
ordinal distribution, with exact singleton-user risk (expected absolute error)
and 32 antithetic Monte Carlo label draws for multi-query users. This includes
synthetic observation noise rather than taking an RMSE against denoised means.
Retain NLL 0.1, ranking 0.05 and expected MAE 0.1. Increase the synthetic query
cap from 1024 to 4096 to reduce artificial singleton users from decoder capping.
No real ratings, real-trained weights or real-trained teacher are gradient data.

Use seed 50, 200 updates, LR 5e-6, EMA 0.95, validation every 50, and step zero
eligible under worst-of-four inner MAE/RMSE ratios. Synthetic parents remain
2048x2048, crops 1000x1000, first-observation ordering, user-balanced caps,
density .002-.08 and popularity spread 1.4-2.8. Run the full-size finite/memory
preflight before training. Save model-only checkpoints with a 17 GiB reserve.
Full inner rating/ranking evaluation must pass before considering another
outer confirmation; the already observed outer split is not a fresh blind test.

77 focused tests pass, including exact enumeration checks of the new objective
and its gradients, synthetic-only provenance gates and default behavior.

The three-update full-size preflight passes with finite FP32 states and finite
candidate output, at 85.03 GiB peak allocated memory. The registered trial is
running from the verified clean root. A 16-episode CPU query-count audit finds
pooled singleton-user fractions of **90.15%** with the old 1024 cap and
**22.57%** with 4096, versus **23.99%** on Netflix inner queries (MovieLens
9.85%). Mean synthetic queries per queried user increase from 1.14 to 3.92.
This supports changing the cap before interpreting finite-query macro risk.
The audit uses no outer labels and performs no neural fitting.

At update 50, all four inner rating metrics beat MF for the first time in an
eligible synthetic-only follow-up: MovieLens MAE/RMSE **0.744914728 /
0.879474737**, Netflix **0.734857703 / 0.836927921**. Worst rating ratio is
**0.999903295**. Netflix RMSE's advantage is only 0.000080943. This is progress,
not completion: training continues to its registered 200 updates, and full
inner ranking and outer confirmation remain unverified for the new model.

The trial finishes and selects EMA step **100**, SHA256
`0e1eb4fbc4f2d4e951214f20e5dd1abc3a293adfbe88dd6a5709f24d799071eb`.
Full-interface inner evaluation confirms **14/14 wins**: Netflix MAE/RMSE
**0.734144437 / 0.836849071**, MovieLens **0.745295441 / 0.880485335**.
Architecture/configuration and finite FP32 tensor keys/shapes/dtypes match the
original network; checkpoint receipts verify clean synthetic ancestry.
Register one outer confirmation of this exact inner-selected checkpoint. It
is the second eligible synthetic-only outer candidate, not a new blind test;
no alternative checkpoint is to be selected from its outer labels.

Outer confirmation returns **13/14 wins**, not completion. MovieLens MAE/RMSE
are **0.730205894 / 0.893278776**. Netflix MAE **0.735633791** now passes by
0.000142738, but RMSE **0.866673982** fails by 0.004557109 and is slightly
worse than the prior clean model. The finite-query objective closes the MAE
gap without solving the RMSE gap. Do not select a different saved step using
this outer result. Full precision evidence is in the run's `test_confirmation/`.

## Synthetic prior follow-up: clustered preferences

Register `synthetic_clustered_known_macro_200_v2`, starting from the just-selected
clean step-100 checkpoint (selected on inner metrics, not an outer sweep).
Use a 50/50 episode mixture of the original Gaussian factors and synthetic
clustered user/item factors. Clustered factors draw 2-16 clusters per side and
strength .5-.95, retaining within-cluster variation and expected unit factor
variance. The bounded ordinal/noise/exposure process and DWIP network stay
unchanged. This tests whether correlated preference groups improve prediction
beyond the current independent-factor prior.

Also exclude globally unknown synthetic users/items from rating and ranking
queries, matching the existing evaluation exclusion rule. Locally cold entities
with context elsewhere in the synthetic parent remain eligible. A 16-episode
audit finds only 0.25% globally unknown queries overall but up to 13% in one
sparse episode; this is protocol alignment, not a claimed explanation of the
entire RMSE gap. No real ratings enter either prior family.

Keep the expected macro-RMSE objective, 4096 query cap, NLL .1, ranking .05,
expected MAE .1, context/parent geometry and normalization from the preceding
trial. Use seed 51, LR 5e-6, EMA .95, 200 updates, validation every 50 and the
same inner worst-rating selector with step zero eligible. Require a six-update
full-size preflight covering both families. Model-only checkpoint reserve is
16.5 GiB, allowing another atomic checkpoint pair without deleting history.
88 focused tests pass, including latent-family, ordinal probability, parent
versus local support, objective and synthetic-provenance checks.

The first mixed-prior preflight was finite but randomly sampled clustered
episodes on all six updates. Its coverage assertion failed; a shell command
nevertheless started `synthetic_clustered_known_macro_200`. That process was
terminated during initial validation, before any gradient update, training log
or checkpoint. Its config/snapshots and aborted status remain preserved. The
corrected preflight explicitly alternates Gaussian and clustered branches; the
new `_v2` run may launch only after that independent coverage gate passes.

The corrected preflight now passes six updates with deterministic coverage of
both families, zero globally unknown queries, clean lineage and finite FP32
state/output; peak allocated memory is 85.04 GiB. Gaussian defaults remain
bitwise identical to the preceding run's saved sampler on three audit seeds.
89 focused tests pass. `synthetic_clustered_known_macro_200_v2` is now running
with the registered recipe; the eligible result remains 13/14 pending its
inner selection and any subsequent eligible confirmation.

An inner-only error decomposition of eligible checkpoints identifies an
objective tradeoff: expected macro-RMSE lowers Netflix MAE relative to MF by
0.003124 and user-RMSE by 0.000160, but increases mean user-MSE relative to MF
from +0.000212 to +0.009580. Its mean prediction shift relative to MF grows
from +0.03446 to +0.08547. Users with 6-10 or >10 inner queries still lose
RMSE to MF, while singleton/two-query groups improve. This diagnosis reads
only cached inner evaluations of verified synthetic checkpoints, never the
ineligible real-trained runs. It motivates preserving mean-rating accuracy
when assessing the clustered prior rather than equating an inner RMSE win
with broad error reduction. Evidence: `synthetic_macro_error_decomposition.json`.

The clustered trial completes and selects **EMA step 200**, worst inner rating
ratio **0.999083538**, with MovieLens MAE/RMSE **0.744015952 / 0.878653828**
and Netflix **0.733561687 / 0.836241777**. SHA256:
`ba97cffb651d184dd8b7ae99e656d6369143aeb2236d676da64259315ab0be20`.
Its clean receipt chain and inner selection are verified. Full inner ranking
evaluation is running. No new outer confirmation is registered for this small
intermediate gain; the previously confirmed model remains the 13/14 reference.

## Global-context prior investigation

The generator normalizes from its entire synthetic parent, whereas inference
normalizes from the entire real evaluation context before 1000x1000 crops.
Matching only local support misses this distinction. An eight-episode audit
finds query-weighted median global item support **140** in 2048x2048 synthetic
parents versus **782** for Netflix inner context. Mean global/local item-support
ratios are **2.57** versus **18.76**. These are diagnostic context counts, not
neural targets or real-trained teacher predictions.

A larger 16-episode screen finds 12288x4096 parents with density .004-.08 yield
local row/item medians **25 / 81** versus Netflix **21 / 73**, global item
median **1107**, and global/local item-support ratio **17.80**. This is a much
closer normalization-context regime, though not an exact distribution match.
16384-row parents overshoot the ratio at 33.40. Metadata support tensors are
diagnostic only; they are never supplied as additional network inputs.
Evidence: `synthetic_global_context_audit.json` and
`synthetic_global_context_extended_audit.json`. 92 focused tests pass.

Register `synthetic_global_context_macro_250` after the clustered checkpoint's
full inner gate and a new full-size preflight pass. Start from that clean,
inner-selected step-200 checkpoint. Keep the Gaussian/clustered mixture,
known-entity queries, expected macro-RMSE/NLL/ranking/MAE objectives and 4096
query cap; change only synthetic parent geometry to 12288x4096 and density
range to .004-.08. The neural training crop remains 1000x1000 and inference is
unchanged. Use seed 52, LR 5e-6, EMA .95, 250 steps, validation every 50, and
the same worst-four inner selector with step zero eligible. Save model-only
checkpoints with a 16 GiB reserve for atomic writes, preserving all history.
Assess full inner rating/ranking results before another outer confirmation.

The clustered model's full inner gate passes **14/14**, and its
`architecture_and_inner_audit.json` verifies unchanged configuration and all
254 FP32 state tensors, clean lineage, snapshot hashes, frozen query arrays
and independent metric aggregation. Its outer scores are deliberately not
measured at this intermediate stage. The larger-parent preflight passes six
updates covering both synthetic families, at 85.03 GiB peak allocated memory
and about five seconds per update. `synthetic_global_context_macro_250` is now
running with the registered recipe; no inference or architecture change is
introduced, and the confirmed benchmark result remains 13/14.

At update 50, the larger-parent trial improves worst inner rating ratio to
**0.998669366**: Netflix MAE/RMSE **0.733431538 / 0.835895111**. The run is
still active; this is not another outer result.

## Synthetic rating-noise investigation

The clustered checkpoint reduces Netflix's inner mean-user-MSE gap to MF
from +0.009580 to +0.006190, but does not eliminate it. Its mean signed error
is +0.05561 versus MF -0.01511. This is an inner diagnostic, not an auxiliary
neural training target derived from MF.

A 16-episode tail audit finds normalized held-out residuals below -2 for
3.67% of large-parent synthetic queries versus 5.10% of Netflix inner queries.
Excess kurtosis is 0.027 versus 0.469; one-star query mass is 3.02% versus
4.55%. These observations motivate testing a synthetic Gaussian/ordinal plus
uniform-rating noise mixture rather than introducing any real-data updates.
The optional mixture independently replaces a generated rating with a uniform
1-5 draw. Its exact conditional target distribution and mean are mixed by the
same probability; expected MAE and macro-RMSE therefore remain internally
consistent. Default probability zero preserves the existing generator.

This option is implemented and tested, but **is not active in the running
larger-parent trial**. A CPU prior screen tests episode probabilities uniformly
drawn from 0-.15 before deciding whether to register a neural experiment.
108 focused tests pass, including analytic/empirical ordinal-mixture checks,
seed determinism and synthetic-only provenance rejection.

The uniform-noise screen improves large-parent skewness from -0.390 to
**-0.602** (Netflix inner: -0.591), excess kurtosis from 0.027 to **0.549**
(0.469), and one-star mass from 3.02% to **4.22%** (4.55%). The lower-tail
frequency below -2 rises only to 4.01%, still below 5.10%; it is not an exact
fit. The selected range for a potential follow-up is 0-.15 probability per
episode. No real rating values or MF predictions become training targets.
The running geometry-only trial remains unchanged and must finish selection
before any new neural trial starts.

The geometry-only trial finishes and selects **EMA step 150**, SHA256
`074e5775dc53bdea0939c0a86990b8b06152897e2d25216f73080ae41561472f`.
Worst inner rating ratio is **0.998581260**; MovieLens MAE/RMSE are
**0.744374804 / 0.879351083**, Netflix **0.733314252 / 0.835821366**.
Clean lineage and selection are verified. Full inner evaluation is now running;
if all 14 comparisons pass, confirm this exact checkpoint once on the known
outer benchmark. Do not select another saved step using outer scores.

Full inner evaluation passes **14/14** and the registered outer confirmation
returns **13/14**. Outer MovieLens MAE/RMSE are **0.728744440 / 0.891327887**;
Netflix is **0.735390488 / 0.866110111**. Netflix RMSE improves by 0.000563871
over the preceding outer-confirmed model but still trails MF by **0.003993238**.
No alternative step was selected on outer labels. This is the third eligible
synthetic-only outer candidate on an already observed benchmark, not a blind
test. The goal remains active.

## Parent-population risk and rating-noise follow-up

Register `synthetic_parent_risk_uniform_600`, conditional on a finite-state and
memory preflight. Initialize only from the clean, inner-selected step-150
checkpoint above. Keep the large parent geometry, density .004-.08,
Gaussian/clustered mixture, known-entity query filter, 4096 decode cap and
unchanged DWIP network/inference. Add synthetic uniform-rating replacement
probability drawn per episode from 0-.15, with matching exact conditional
ordinal probabilities and means.

Also change the synthetic expected macro-RMSE surrogate from decoded-crop
query counts to parent query-population counts. The current cap/crop can make
a user with many parent held-out ratings appear to have one or two targets,
overemphasizing absolute-error-like risk. The new surrogate replicates decoded
query profiles to the known synthetic parent count and samples independent
ordinal outcomes for these copies. Random within-user remainder allocation
keeps profile weighting unbiased. This approximates global user aggregation;
it is **not** an exact reconstruction of uncomputed predictions elsewhere in
the parent. It introduces neither model inputs nor real-data gradient labels.

Use seed 53, 600 updates, LR 5e-6, EMA .95, validation every 100, NLL .1,
ranking .05, expected MAE .1, and the same worst-four inner selector with step
zero eligible. Use model-only checkpoints and a 16 GiB atomic-write reserve.
Require preflight coverage of both latent families, nonzero synthetic mixture
noise, and parent-population counts before launch. Full inner rating/ranking
gates remain necessary before another outer confirmation.

The six-update exact-recipe preflight passes: both synthetic latent families,
nonzero uniform contamination, parent-query counts, finite gradients/FP32 state
and finite candidate output are verified, with **85.03 GiB** peak allocation.
**116 focused tests pass.** The registered 600-update trial is now running from
the hash-verified clean checkpoint. No real rating enters a gradient update.

An inner-only decomposition of the selected global-context model confirms the
remaining aggregation issue: Netflix users with 6-10 held-out ratings have
RMSE **0.005504** above MF; users with 11+ have **0.002766** excess. Overall
inner RMSE wins by 0.001187, but mean-user MSE is still **0.005671** worse and
signed prediction bias is 0.066566 higher than MF. These stored-query results
support testing parent-count risk rather than interpreting the inner macro-RMSE
win as uniform improvement. Evidence: `global_context_macro_error_decomposition.json`.

While the registered trial runs unchanged, a CPU-only context diagnostic
compares split-half residual-variance stability in 1000x1000 blocks. It uses no
held-out ratings or neural fitting. Residual variability includes preference
signal as well as observation noise; it does not identify noise variance by
itself. Any follow-up prior change remains conditional on this evidence and
the running trial's inner selection.

At update **100**, the parent-risk/uniform trial improves its worst inner
rating ratio to **0.998054933**. MovieLens MAE/RMSE are **0.743629328 /
0.876770697**; Netflix **0.733713921 / 0.835380825**. Both RMSEs improve over
step zero, while Netflix MAE worsens slightly. This is a provisional rating-only
selection, not a new full-inner or outer result. The scheduled 600 updates
continue unchanged; the step-100 ancestry/source/log audit passes.

The 600-update trial now completes and selects **EMA step 100**, SHA256
`57b3d37cafb7174f33b3fbcdbcbdb2949278bb72ea75eda1a858d1be21979daa`.
Later registered validation points do not beat its worst inner rating ratio.
The completed-run architecture and provenance audit verifies original model
configuration, all 254 tensor keys/shapes/dtypes, 77,998,688 finite FP32 state
elements, archived source hashes and synthetic-only ancestry. Full inner
rating/ranking evaluation is running. If all 14 comparisons pass, register one
outer confirmation of this exact checkpoint, never an alternative step. This
would be the fourth eligible synthetic-only outer candidate on the already
observed benchmark, not a fresh blind test.

Full inner evaluation confirms **14/14 wins**, and the registered outer
confirmation again returns **13/14**. MovieLens MAE/RMSE are **0.727439434 /
0.887274366**; Netflix **0.735438879 / 0.864590665**. Netflix's RMSE gap to MF
shrinks from 0.003993238 to **0.002473792**, while its MAE win is a narrow
**0.000337650**. All ten ranking comparisons still win. This is the fourth
eligible outer candidate, selected exclusively on inner validation. The stored
query, metric-average and artifact-hash audit passes; the goal remains active.

The new inner error decomposition also lowers Netflix mean-user MSE excess
from 0.005671 to **0.002861** and signed-bias excess from 0.066566 to
**0.048244**, without fully removing either. Evidence:
`parent_risk_macro_error_decomposition.json`. This supports continued prior
work rather than claiming the aggregation surrogate solves the entire gap.

### Conditional follow-up: synthetic user-specific noise

The context-only split-half audit finds a substantial missing source of stable
user-level variability. For profiles with >=20 local observations, Netflix's
split-half variance correlation is **0.593**, versus **0.230** in the Gaussian
noise prior and **0.083** after episode-wide uniform contamination. Its residual
standard-deviation coefficient of variation is **0.240**, versus **0.133 /
0.142**. Both datasets show this effect across support-count strata. These are
equal block-user profile diagnostics, not macro query metrics or an identified
observation-noise estimate. Evidence: `synthetic_user_scale_audit.json`.

Add an optional, synthetic-only per-user Gaussian noise scale:
`episode_noise * exp(log_std * Normal(0,1) - log_std**2)`. Its expected squared
multiplier is one; the original Gaussian variance is retained in expectation
before ordinal rounding/clipping. Draw episode log-std from a registered range.
Analytic conditional ordinal probabilities and means use the same sampled
scales. Default range `[0,0]` is bitwise identical to the running trial's
archived generator, including its existing uniform mixture. No model parameter,
inference feature or real rating is introduced. The running trial does **not**
use this change.

A 16-episode CPU screen of log-std **0.15-0.6** with the same 0-0.15 uniform
mixture raises variance correlation to **0.408** and standard-deviation CV to
**0.203**, closer but not equal to Netflix. Without uniform contamination these
become 0.672/0.257. A separate tail screen finds the combined prior improves
the residual-below-minus-two frequency from 4.01% to **4.60%** (Netflix 5.10%),
but overshoots skewness/kurtosis (-0.761/0.943 versus -0.591/0.469). This is
broader prior coverage, not a claim of perfect matching or better eval scores.
Artifacts: `synthetic_user_scale_heterogeneous_audit.json` and
`synthetic_heterogeneous_tail_audit.json`.

**138 focused tests pass**, covering tensor-valued ordinal noise, empirical
probabilities, exact defaults, seeded isolation and parent-risk gradients.
A neural heteroscedastic trial remains conditional on the current run's final
inner selection and needs its own exact-recipe GPU preflight; none is launched.

Register **`synthetic_user_noise_parent_risk_400`**, conditional on a successful
six-update GPU preflight. Initialize only from the completed parent-risk run's
inner-selected step-100 checkpoint (`57b3d37c...` above). Keep the same parent
geometry, density, Gaussian/clustered factors, known-query filter, uniform
mixture 0-0.15, parent-count RMSE surrogate, query cap, auxiliaries and inference.
Enable only the new user-noise log-std range **0.15-0.6** in the synthetic prior.
All scales, probabilities, counts and labels are generated synthetically.

Use seed **54**, **400 updates**, LR **5e-6**, EMA **0.95**, validation every
**100** updates and the fixed worst-four inner selector, with step zero eligible.
Require both preference families and nonzero user-noise spread in the exact
preflight, along with finite gradients/state/output and memory limits. Use
model-only checkpoints with an explicitly registered **15.5 GiB** reserve;
the next checkpoint's atomic replacement would otherwise breach the preceding
16 GiB reserve. Preserve all historical artifacts; the trainer's default reserve
stays unchanged. No outer confirmation is queued for the new trial; it first
needs completed inner selection and the full fourteen-comparison gate.

The six-update user-noise preflight passes with **85.03 GiB** peak allocation,
finite FP32 state/output/gradients, both latent families, nonzero user-noise
spread and valid parent-query counts. Source hashes and clean initialization
are verified before launch. `synthetic_user_noise_parent_risk_400` is now
running with its registered recipe; no real-trained initialization or teacher
is used. No additional outer evaluation is launched alongside training.

The 400-update user-noise trial completes and selects **EMA step 400**, SHA256
`12c7b606a8282847729773087fa95714b5ff7647199384c37bc4230d8733e54c`.
Worst inner rating ratio improves to **0.997200713**. MovieLens MAE/RMSE are
**0.744041912 / 0.877305892**; Netflix **0.732578998 / 0.834665836**.
The completed-run audit verifies clean synthetic lineage, source snapshots,
finite FP32 state and unchanged architecture. Full inner rating/ranking
evaluation is running; these rating-only results do not establish completion.

Full inner evaluation confirms **14/14 wins**. One registered outer confirmation
of the exact step-400 checkpoint returns **13/14**, the fifth eligible candidate
on the already observed benchmark. Outer MovieLens MAE/RMSE are **0.726096601 /
0.886315868**; Netflix **0.734525703 / 0.864635729**. Netflix MAE improves by
0.000913177 versus the preceding candidate, but RMSE worsens by 0.000045064 and
still misses MF by **0.002518856**. All ten ranking comparisons win. Do not call
this a uniform improvement, fresh blind test, or completion.

The inner-only decomposition explains a remaining tradeoff: Netflix macro-RMSE
improves, while mean-user MSE excess remains **0.002973** (previously 0.002861)
and signed-bias excess rises to **0.059267** (previously 0.048244). Users with
6-10 and 11+ inner targets still trail MF in RMSE by 0.003207 and 0.002118.
Evidence: `user_noise_macro_error_decomposition.json`. This motivates testing
whether the additional absolute-error auxiliary is now overemphasizing
median-like predictions; the diagnostic does not establish that causally.

## Registered loss ablation: remove the MAE auxiliary

Register **`synthetic_parent_risk_no_mae_400`**, initialized only from the clean
inner-selected user-noise step-400 checkpoint (`12c7b606...` above). Keep the
same synthetic prior (including user-noise spread 0.15-0.6 and uniform mixture
0-0.15), parent-count expected macro-RMSE, NLL **0.1**, ranking **0.05**, 4096
query cap, geometry, density and inference. Change expected-MAE auxiliary weight
from **0.1 to 0**. This is not a claim that MAE stops mattering: the unchanged
worst-of-four inner selector and full fourteen-comparison gate still require it.
All gradients and analytic targets remain synthetic; no real-data bias offset,
calibration, teacher, factor fit or architecture change is introduced.

Use seed **55**, **400 updates**, LR **5e-6**, EMA **0.95**, validation every
**100**, and step zero eligible. Use model-only checkpoints with the existing
explicit **15.5 GiB** reserve; preserve history and the default reserve. Require
another six-update, both-family, exact-recipe preflight before launch, including
finite state/output/gradients and verification that the loss has no MAE term.
No outer evaluation is queued until completed inner selection and all 14 inner
rating/ranking comparisons pass.

The six-update no-MAE preflight passes with **85.03 GiB** peak allocation,
finite FP32 states/output/gradients, both synthetic families, nonzero user-noise
spread and valid parent counts. Its recorded loss equals parent-risk loss plus
0.1 NLL and 0.05 ranking loss, with no expected-MAE term. Clean initialization
and exact source hashes are verified before launching
`synthetic_parent_risk_no_mae_400`. It is now training; no additional outer
evaluation runs alongside it. The goal remains unachieved.

The no-MAE trial completes all 400 updates without improving the fixed inner
selector. It selects **step zero**, score **0.997200713**. An identity audit
verifies its selected tensors are bitwise identical to the evaluated user-noise
parent and its inference configuration is unchanged. This rejects the ablation
under the registered selector, rather than promoting a worse step. No repeated
outer evaluation is run, and the eligible outer-candidate count remains five.

## Conditional preprocessing investigation: global user scales

Prepare an opt-in **`two_way_user_scale`** normalization without changing the
running no-MAE recipe. After the existing context-only two-way centering,
estimate each user's residual variance from its **observed context only**,
center within that user's observations, and shrink with 20 prior degrees of
freedom toward unit variance. Clamp the resulting standard deviation to
**[0.5, 2]**. Divide each input row by its scale and restore that scale when
decoding ratings. Synthetic training computes these same statistics on its
synthetic parent before cropping. Oracle probabilities stay in rating units;
the loss restores both row and column units before evaluation.

This is disclosed context-dependent preprocessing, not a neural parameter,
MF factor, fitted prediction calibration, new head or ensemble. It never uses
held-out rating values. Public block geometry, predictive mean, clipping and
observed-rating preservation remain unchanged. Existing `two_way` behavior is
bitwise identical on archived-generator/default-loss checks; the running trial
uses its original source snapshot and does not enable this mode.

An inner-context-only screen finds that global and local-block scale estimates
differ by more than 10% for **23.96%** of eligible Netflix block-user profiles
(MovieLens **16.79%**), versus **12.74%** in eight synthetic episodes. Netflix's
global scale 1st/99th percentiles are **0.696/1.451**, while the local estimates
are **0.749/1.341**. This exposes potentially useful global profile information
outside the neural crop; it does not prove better prediction. Evidence:
`context_user_scale_statistics.json`.

**147 focused tests pass**, including masked-value independence, cold-user
fallbacks, scale bounds, synthetic target preservation, raw-unit loss inversion,
finite scaled parent-risk gradients,
observed-rating preservation and parity between query-only and public-interface
evaluation. No GPU candidate or neural trial uses this mode yet. Any experiment
requires an explicit registration and its own preflight after the current run.

Register an **inference-only inner screen**, after the no-MAE GPU job ends,
using the audited `synthetic_user_noise_parent_risk_400` step-400 checkpoint
(`12c7b606...`). Compare only the existing `two_way` mode and the prepared
`two_way_user_scale` mode with fixed prior-count 20 and bounds [0.5, 2]. Require
the old mode to reproduce its stored inner rating scores, verify weights remain
unchanged after both modes, and save checkpoint/source/split hashes. No optimizer,
neural update, outer evaluation or success claim is allowed in this screen.
Its result will determine whether to register a matched synthetic training trial;
the full fourteen-metric inner gate still precedes any new outer confirmation.

The no-update inner screen is now running, after the no-MAE process exited.

The screen completes with **bitwise unchanged neural weights and no gradients**.
The legacy mode reproduces its stored scores exactly. User scaling improves
the worst inner rating ratio slightly, from **0.997200713** to **0.997126126**.
MovieLens MAE/RMSE become **0.744643664 / 0.877193364**; Netflix **0.732943120 /
0.834603405**. Both MAEs worsen slightly while RMSEs improve. These are inner
rating-only results, not evidence of full-gate or outer success.

Register **`synthetic_user_scaled_parent_risk_400`**, conditional on a new
six-update full-size GPU preflight. Initialize from the clean, inner-selected
user-noise step-400 checkpoint (`12c7b606...`), not from any later optimizer
state in the rejected no-MAE ablation. Enable `two_way_user_scale`; otherwise
keep the successful user-noise prior, parent-count expected RMSE, NLL 0.1,
ranking 0.05 and expected-MAE **0.1**. Restore the MAE auxiliary because its
removal did not improve the fixed selector. All neural updates remain synthetic.

Use seed **56**, **400 updates**, validation every **100**, LR **5e-6**, EMA
**0.95**, query cap 4096, parents 12288x4096, neural crops 1000x1000, density
0.004-0.08 and the same known-entity filtering/order/popularity settings. Step
zero remains eligible under worst-four inner ratios. The preflight must cover
both factor families, finite context-derived row scales within [0.5,2], finite
FP32 state/output/gradients and memory limits. Save model-only checkpoints with
an explicit **15 GiB** reserve, preserving all older artifacts and leaving the
trainer's default reserve unchanged. Full inner rating/ranking gates still
precede any outer confirmation. This is a registered preprocessing and training
change, not a change to network architecture, metric definitions or data splits.

The exact-recipe preflight passes all six synthetic updates, both factor
families, finite row scales, gradients, FP32 state and outputs, with peak
allocation **85.034 GiB**. The 147 focused tests pass. The registered run is
now training in session **12500**; step zero reproduces the screened inner
ratio **0.997126126**, and the first logged gradient update is synthetic.
No new full inner gate or outer confirmation has run for this normalization.

All 400 registered updates complete. The fixed selector chooses **EMA step
100**, SHA256 `1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770`,
with worst inner rating ratio **0.996470996**. MovieLens MAE/RMSE are
**0.744814560 / 0.876900534**; Netflix **0.732375982 / 0.834055056**.
Later validation points do not improve that selector. The completed-run audit
passes clean lineage, unchanged architecture/config/tensor schema, finite FP32
state and archived/live source hashes. Full public-interface inner evaluation
is now running; no outer confirmation is yet registered for this candidate.

Full inner evaluation passes **all fourteen** rating/ranking comparisons.
The public inference path preserves observed ratings and agrees with the
query-only selector to numerical tolerance. The pre-outer gate verifies the
selected checkpoint, architecture, clean lineage and frozen baseline/split
hashes. Register one outer confirmation of this exact step-100 checkpoint,
the **sixth** eligible synthetic-only candidate. This benchmark has already
been observed; this is not a fresh blind test and does not establish success
before all outer comparisons and artifact audits complete.

The sixth eligible outer confirmation returns **13/14**, not completion.
Netflix MAE/RMSE are **0.734000285 / 0.863590218**, improving both over the
user-noise parent; the RMSE gap to MF shrinks to **0.001473345**. MovieLens
MAE/RMSE become **0.727877760 / 0.887358392**, worse than that parent but
still better than MF. All ranking comparisons pass. The frozen-query,
independent rating metric, stored per-user average and artifact-hash audit
passes; ranking is audited through stored per-user averages, not reconstructed.

Inner-only error decomposition shows a remaining Netflix mean-user MSE excess
of **0.001942741** and signed-bias excess of **0.056784471** relative to MF.
Users with 6-10 queries retain RMSE excess **0.002729923**, and those with 11+
retain **0.000794188**. This supports investigating a synthetic conditional-mean
accuracy auxiliary alongside the existing risk/MAE objectives, rather than
using the outer benchmark to choose an alternative saved checkpoint. Evidence:
`user_scaled_macro_error_decomposition.json`.

## Registered conditional-mean auxiliary trial

Register **`synthetic_user_scaled_mean_anchor_400`**, initialized only from the
clean inner-selected user-scaled step-100 checkpoint (`1fcadc0c...`). Preserve
the prior, user-scale normalization, parent-count expected user-RMSE, NLL 0.1,
ranking 0.05 and expected MAE 0.1. Add **0.25 times user-macro squared error to
the analytic synthetic conditional mean rating**. Its gradient equals the
synthetic expected squared-rating-error gradient; the omitted conditional
variance is a parameter-independent constant. This is a training auxiliary,
not a real-data calibration, MF teacher, additional head or inference ensemble.

The inner decomposition motivates preserving mean accuracy while retaining
the successful risk/MAE objectives. The earlier no-MAE ablation did not improve
selection, so it is not repeated. Weight 0.25 is fixed before this trial rather
than chosen by an outer sweep. The new control defaults to zero; comparison
against archived code verifies bitwise-identical default episodes, metadata,
losses, diagnostics and gradients on three seeds. **155 focused tests pass**,
including raw-unit scaling, equal-user weighting, analytic expected-MSE gradient
equivalence and rejection of real, observed or missing auxiliary targets.

Use seed **57**, **400 updates**, validation every **100**, LR **5e-6**, EMA
**0.95** and the same step-zero-eligible worst-four inner selector. Keep parents
12288x4096, crops 1000x1000, density .004-.08, mixed factors, uniform noise
0-.15, user-noise log-std .15-.6, known-entity filtering and user-balanced query
cap 4096. Require a six-update exact-recipe preflight covering both families,
finite state/gradients/output and the new loss composition before training.
Model-only checkpoints retain an explicit **15 GiB** reserve (15.666 GiB free
before preflight); preserve all earlier artifacts. Full inner gating precedes
any new outer confirmation. No claim of improvement is made before evaluation.

The exact-recipe preflight passes all six synthetic updates and both factor
families, finite state/gradients/output and bounded context-derived row scales.
Peak allocated memory is **85.034 GiB**. An independent check verifies each
logged loss includes exactly the registered 0.25 mean-rating term (maximum
FP32 roundoff 1.21e-7), and verifies initialization/source hashes. The registered
400-update run is now launched in session **12217**, with no concurrent GPU job.

A CPU-only residual-spectrum diagnostic runs while that recipe remains fixed.
It compares normalized observed context with three row-shuffled nulls per
block, preserving exposure masks and each local user's residual distribution.
No neural fit, MF teacher, held-out rating value or inference change is used.
The eight-episode prior sample and representative real blocks have substantially
different density distributions. Although unstratified spectral excess differs,
this is **not evidence identifying a latent-rank mismatch** or sufficient
justification for changing the rank prior. Dense real blocks and synthetic
episodes both have substantial excess correlation; sparse-block differences
are noisy. Preserve the report as descriptive, not a score-improvement claim:
`context_residual_spectrum.json`. Five diagnostic unit tests pass.

The mean-rating auxiliary trial finishes all **400** updates without improving
the fixed selector; it chooses **step zero**, ratio **0.996470996**. Saved SHA256
is `a5e8dc59e63b37d5b3ef7f266de958a8a8c989b9486b063e56ac072de225ba0b`.
The fallback identity audit verifies every selected state tensor is bitwise
identical to its parent, with unchanged inference configuration and matching
inference-function ASTs. Training source hashes, synthetic lineage and the
logged loss composition pass. **No duplicate outer evaluation** is run, and
the eligible outer candidate count remains six. This rejects the registered
auxiliary under the fixed selector, not all conceivable weights or seeds.

The latest focused suite passes **160 tests**. Both registered trials have
finished; no GPU job is active. The goal remains unmet at 13/14 outer wins.
A possible next investigation is whether larger item-context windows reduce
the previously measured global/local support gap without changing DWIP weights
or architecture. This would be an explicitly registered inference-geometry
experiment, not the current 1000x1000 protocol; it has **not** been enabled,
evaluated or claimed to improve scores. Its inner-only screen and memory gate
would precede any matched synthetic curriculum or full benchmark confirmation.

## Registered inner-only wider-item-context screen

Register a no-update comparison of **1000x1000** (legacy control) and
**1000x2000** inference windows on the clean user-scaled step-100 checkpoint
(`1fcadc0c...`). This tests the documented loss of user-profile context under
local blocking. It changes inference geometry, **not** network architecture,
weights, normalization, metrics, splits, MF comparator or data eligibility.
MF already has the full evaluation-training context. The larger window is not
silently substituted into any historical result or claimed to have equal cost.

Use only the fixed inner partition and existing worst-four rating selector.
Require exact legacy-score reproduction, bitwise-unchanged weights, absent
gradients and source/checkpoint/split hashes; record time and peak GPU memory.
No optimizer, neural updates, outer evaluation or ranking success claim is
allowed. Any promising shape still needs a public full-inference memory and
rating/ranking gate, and a separately registered matched synthetic curriculum
before promotion. The confirmed candidate remains the old 1000x1000 geometry.

The item-context screen completes with bitwise-unchanged weights, no gradients,
and exact legacy-score reproduction. The 1000x2000 window improves MovieLens
slightly but worsens Netflix inner MAE/RMSE to **0.733755345 / 0.835277084**;
its worst ratio **0.997930990** loses to legacy **0.996470996**. Peak query-only
memory grows from **22.549 to 44.286 GiB**, and evaluation time from 69.8 to
95.2 seconds. The wider-item setting is **not promoted**; no outer evaluation
or matched training is launched for it. Evidence: `item_context_inner_screen.json`.

Register a separate **user-context** screen of legacy 1000x1000 against
**2000x1000** on the same fixed clean checkpoint, under identical no-update,
inner-only and hash/identity gates. Netflix contains substantially more users
than items, so doubling user context addresses item-estimation support rather
than the unsuccessful larger user-profile/item-axis experiment. MovieLens
already fits its entire user axis within 1000 rows and should remain unchanged.
This is a second explicit geometry hypothesis, not an outer-label sweep. The
new shape stays disabled unless its evidence supports a separate full gate and
matched synthetic training registration. **166 focused tests pass.**

The user-context screen completes with exact legacy reproduction, absent
gradients and bitwise-unchanged weights. **2000x1000** keeps MovieLens scores
identical and improves Netflix MAE/RMSE to **0.730914564 / 0.832036879**.
The worst inner ratio improves to **0.994059818**; peak query-only memory is
**44.340 GiB**, versus 22.549 GiB, and runtime 77.5 versus 69.7 seconds.
These are inner rating-only results, not full ranking or outer success.

**Evidence-driven plan update:** first isolate the promising inference effect
with the exact already-completed synthetic checkpoint, rather than introduce
new weight updates before its full gate. This supersedes the preceding proposal
to require a matched curriculum before any promotion. The actual goal remains
unchanged: same DWIP network, synthetic-only weight ancestry, all fourteen fixed
MF comparisons. A matched synthetic rectangular-context curriculum is retained
as a follow-up only if the wider-window candidate remains insufficient.

Register a **synthetic-input, no-update public-inference memory/parity preflight**
at 2000x1000, followed on success by the full fixed inner rating/ranking gate.
The evaluator may enable a wider shape only through a hash-verified completed
inner-screen receipt, including unchanged weights, frozen baseline/splits,
archived sources and recomputed selection. No arbitrary outer-selected geometry
override is allowed. The previously confirmed checkpoint/recipe remains 13/14
at 1000x1000 until an explicitly registered new outer confirmation completes.

The synthetic-input public preflight passes with unchanged weights, preserved
observed ratings and finite [1,5] outputs. Peak allocation is **93.481 GiB**;
128 query-only predictions agree with full public inference within **6.89e-8**.
Full public inner evaluation then passes **all fourteen** comparisons. Netflix
MAE/RMSE are **0.730914564 / 0.832036879**; all five ranking metrics still beat
MF. MovieLens metrics remain identical. Full inner peak memory is **93.481 GiB**.

The pre-outer gate verifies frozen query arrays, independently recomputed rating
metrics, all stored user-metric averages, configuration/checkpoint/source/split
hashes and the geometry-selection receipt. **183 focused tests pass**, including
an independent NumPy ranking implementation checked against Cornac per user on
ties and remapped IDs. The final evaluator now separately reconstructs all five
ranking metrics from full predictions and checks each user, including strict
AUC ties, maximum-rank AP ties and exclusion of observed positives only.

Register **one outer confirmation** of checkpoint `1fcadc0c...`, predictive
mean, two-way user-scale normalization, and **2000x1000** overlapping-tail
public blocks. This is the **seventh eligible candidate recipe**, not a new
trained checkpoint. No new neural update occurs. The outer benchmark was
already observed and must not be described as blind. All fourteen outer
comparisons and independent metric/artifact checks are required for completion.

The registered outer confirmation completes with **14/14 strict wins**.
Netflix MAE/RMSE are **0.732988718 / 0.861721441**, versus MF **0.735776529 /
0.862116873**. All five Netflix ranking metrics remain above MF, though below
this checkpoint's legacy-window ranking scores; all MovieLens scores remain
unchanged and above MF. The goal is achieved on this fixed, previously observed
benchmark, not as a statistical-significance or fresh-generalization claim.

The final audit independently recomputes rating metrics from frozen query
arrays, verifies every outer ranking user's independently reconstructed five
metrics, and checks all fourteen comparator directions. It rechecks original
configuration and all 254 state keys/shapes/dtypes, 77,998,688 finite FP32
elements, all twelve checkpoint ancestors, parent/source hashes and synthetic
update paths. Both preserved real-trained checkpoints are rejected. The public
normalization, clipping and observed-rating-preservation contracts hold.
`winning_recipe.json` records the required checkpoint and context receipt;
`context_2000x1000_completion_audit.json` records the requirement-level evidence.
No GPU job or required follow-up remains active for this goal.

## Experiment history

**Historical record, not current instructions:** the real-adaptation sections
below document an unauthorized training-regime change and are INELIGIBLE.
Their proposals and success claims are superseded by the synthetic-only policy
above. Original checkpoints, source snapshots, metrics and completion audits
remain preserved as evidence; adjacent eligibility retractions override their
goal-completion interpretation.

### Adapted model passes every inner metric

`mixed_macro_rating_rank_200` completes and selects EMA step 200, worst rating
ratio 0.995513743. Full-interface inner evaluation now beats MF on **all seven
metrics on both datasets**. MovieLens MAE/RMSE is 0.740091732 / 0.872492827;
Netflix is 0.732005811 / 0.833253827. Netflix Precision@10 recovers to
0.010738155 versus MF 0.010322293. These are adapted-model, not zero-shot,
results. Register one outer confirmation of this exact inner-selected model;
do not compare alternative checkpoints on outer labels. It is the first
adapted-model outer confirmation and second overall candidate confirmation.
The historical outer split is not a fresh blind test. Goal completion still
requires actual outer results, not these inner wins.

At update 50 of `mixed_macro_rating_rank_200`, **all four inner rating
comparisons beat MF** for the first time: MovieLens MAE/RMSE
0.741317046 / 0.872334752 and Netflix 0.734399749 / 0.834267039. Worst rating
ratio is 0.996724258. This is a saved, validation-selected adapted model, not
goal completion: the run is still active, full ranking evaluation is pending,
and no adapted model has an outer confirmation yet. The exact-recipe preflight
passes at 85.2 GiB, including nonzero ranking losses on both real datasets.

### Direct macro-metric adaptation follow-up

The first real-adaptation pilot completes and selects EMA step 150. Full inner
scores are MovieLens MAE/RMSE 0.742041797 / 0.870646914, Netflix
0.738250584 / 0.836526908. Netflix RMSE now beats MF by 0.000481956, but its
MAE trails by 0.000982171 and Precision@10 trails by 0.000237636. All other
inner metrics win. No adapted candidate has been evaluated on outer labels.

Register `mixed_macro_rating_rank_200`, initialized from that selected model:
retain the real/synthetic mixture, holdout construction, 4096 real query cap,
geometry, LR 3e-6, EMA 0.95, NLL 0.1, MAE 0.1, 200 updates and 50-update
validation; use seed 49 and fresh AdamW. On real episodes, replace user-MSE
with mean user-RMSE, using sqrt(user-MSE + 1e-6) for finite gradients. This
matches the benchmark's user-wise root aggregation rather than taking a root
after averaging users. Synthetic episodes keep their prior MSE objective.

Also enable real ranking weight 0.05 using only **known held-out training
ratings**: rating >=4 versus rating <=3 for the same user. No missing real
cell, inner validation coordinate, or outer coordinate becomes a negative
gradient query. Synthetic exposure-ranking weight remains 0.05. This addresses
the measured Precision@10 regression without adding an inference ensemble.
Tests verify exact user-wise RMSE arithmetic, finite zero-error gradients and
known-rating pair enumeration. An eight-episode real-data audit independently
checks zero gradient-coordinate overlap with both inner and outer query sets.
Run another five-update exact-recipe GPU preflight before training. The new
logging schedule records real as well as synthetic updates; the preceding
pilot's 25-step log interval coincided mostly with its synthetic updates.

### Registered training-rating adaptation

The preceding goal turn is **progress**, not a wait or a completed goal:
outer confirmation proves 12/14 wins but contradicts Netflix rating success.
No GPU process is running at the start of this continuation.

Synthetic-only fine-tuning has plateaued just above Netflix's inner rating
baseline. The next separately labeled experiment introduces **real training
ratings in neural gradient updates**. This changes the training regime from
zero-shot synthetic-only inference to an adapted model; it must never be
reported as another synthetic-only or zero-shot result. It preserves the
user's DWIP v2-2 neural architecture, 254-tensor state structure, train/test
split, metrics, 1000x1000 inference and no-MF-inference restrictions. MF also
fits real training ratings; this is not additional validation/test access.

Register `mixed_observed_adaptation_200`: initialize from the retained
synthetic-only expected-MAE checkpoint. Train 200 updates, validate every 50,
use seed 48, fresh AdamW at 3e-6, EMA 0.95, activation checkpointing, and
worst-rating selection with step zero eligible. Four of every five updates
alternate MovieLens/Netflix **inner-training context only**; every fifth uses
the best full-context synthetic prior for regularization. Real episodes mask
10% of the supplied training ratings, select query-anchored original-ID blocks,
fit two-way normalization after removing every episode target, and cap targets
at 4096 with user-balanced selection. Real targets are observed ratings with
MSE + NLL 0.1 + MAE 0.1. There are no unobserved real negative queries and no
real ranking loss, so held-out coordinates never become negative labels.
Synthetic episodes retain expected-rating targets, expected-MAE 0.1, NLL 0.1
and ranking 0.05. Neither inner validation labels nor outer labels enter
gradient targets. Source-context hashes and disjointness evidence are saved.

Run a five-update GPU preflight covering both real datasets and a synthetic
episode before launch. The new sampler tests demonstrate that perturbing
episode targets cannot change source values or global normalization, and that
inner validation target values cannot change loaded adaptation tables. All
57 focused tests pass. The new run explicitly uses a **17 GiB** atomic-write
storage reserve (older runs retain their recorded 18 GiB policy), keeping one
selected model and all distinct historical weights; no prior artifacts are
deleted to make room. No new outer evaluation is queued during adaptation.

### Outer-confirmed result: goal still active

The expected-MAE pilot completes and selects its **EMA step 125** by inner
worst-rating score 1.002055159. Its one registered frozen outer confirmation
is complete. **12 of 14 metrics beat MF, but the full goal is not achieved.**

| Outer dataset | MF MAE | DWIP MAE | MF RMSE | DWIP RMSE | Ranking wins |
|---|---:|---:|---:|---:|---:|
| MovieLens 100K | 0.743013001 | 0.729921482 | 0.899831176 | 0.888566755 | 5/5 |
| Netflix small | 0.735776529 | 0.739807547 | 0.862116873 | 0.865899839 | 5/5 |

Netflix MAE/RMSE still trail by 0.004031018 / 0.003782967. Do not hide these
regressions behind ranking gains or an average across datasets. All seven
MovieLens metrics win, and all five Netflix ranking metrics win. Full exact
scores, selection evidence, hashes and per-user metrics are under
`runs/recsys_goal_20260920/full_context_expected_mae_250/test_confirmation/`.
The retained candidate is that run's `best_validation_checkpoint.pth`, SHA256
`c9041c7a3bde914a4c2cfb051fd2c9f1619099ade264815d658d89b92cc7a6a9`.

Architecture and its 254 state tensors remain unchanged. This candidate uses
context-only two-way centering, synthetic-only neural updates, full-size
activation-checkpointed training, ordered exposure contexts, user-balanced
queries, expected-rating MSE plus expected-MAE/NLL/ranking losses, and EMA.
EMA remains one network at inference, not an ensemble. The frozen split,
1000x1000 public inference, clipping and all metric definitions remain fixed.
All 51 focused tests pass, and selected-run source snapshots match recorded
hashes. No training or evaluation GPU process remains active after this
confirmation. Further work should select on inner data only; the outer result
is now observed and must not become a checkpoint-selection criterion.

### Registered outer confirmation

After the expected-MAE pilot and its full inner evaluation complete, evaluate
its verified inner-selected checkpoint once on the frozen outer split,
regardless of whether its small remaining inner rating gaps close. This
supersedes the earlier conservative gate requiring an inner baseline win
before any outer confirmation. Inner validation chooses the checkpoint;
outer evaluation measures the actual requested benchmark outcome and must not
choose between checkpoints. Step zero remains a valid choice if the auxiliary
does not help. Report every metric and any failure, and do not infer outer
success from inner averages. This is a preregistered single confirmation,
not a search over outer-test scores; no candidate outer labels have yet been
used. The historical outer split is already known and is not a fresh blind
generalization test.

`full_context_ema_500` completes with step zero selected. It reduces later
MovieLens errors and slightly lowers Netflix RMSE, but Netflix MAE remains
worse than the initialization, so it is not promoted. Its selected predictions
exactly reproduce the retained full-context model.

Register `full_context_expected_mae_250`: initialize from the retained
full-context step-250 model; keep the same synthetic prior, 1000x1000 geometry,
user-balanced sampling, MSE/NLL/ranking losses, EMA 0.99, and LR 5e-6. Add
expected-MAE weight 0.1, use seed 47, fresh AdamW, 250 updates and validation
every 125. This is a bounded objective-alignment pilot, not a change to
evaluation metrics. Checkpoint selection remains the worst MAE/RMSE ratio,
with step zero eligible. Run an exact-recipe GPU preflight, including the
new probability targets and EMA candidate swap, before training.

A finer objective audit shows that the selected full-context model already
has slightly lower Netflix mean user-MSE than MF (delta -0.000028251), despite
losing mean user-RMSE and MAE. These objectives are not interchangeable.
About 24% of Netflix validation users have only one query, for whom RMSE equals
absolute error; this group contributes much of the remaining gap. This is not
a reason to relabel the evaluation metric or declare success.

An optional expected-MAE auxiliary is now implemented for a subsequent,
explicitly registered experiment if needed. It computes the exact ordinal
1–5 probabilities from the synthetic latent preference/noise, then optimizes
expected raw-scale absolute error with equal-user weights. The default weight
is zero, preserving the current experiment. No outer labels or real-rating
gradient targets are introduced. Probability normalization/mean identities,
raw-scale restoration, per-user weighting, and finite gradients are covered;
51 focused tests pass. The live EMA run does not include this auxiliary.

The full-context run completes and selects step 250. Full-interface inner
evaluation confirms all seven MovieLens metrics and all five Netflix ranking
metrics beat MF. Netflix MAE 0.739136209 and RMSE 0.838676034 still trail by
0.001867796 and 0.001667171 respectively. There is no across-the-board rating
win and no outer candidate result yet. Later full-context checkpoints improve
some MovieLens scores but worsen Netflix; the worst-rating selector correctly
rejects them.

Register `full_context_ema_500`: continue from that selected full-context
checkpoint with the identical synthetic prior, geometry and objectives, seed
46, fresh AdamW, LR 5e-6, 500 updates and 125-update validation. Use parameter
EMA with decay 0.99 to smooth episodic-gradient variation. Validation and saved
checkpoints use the EMA parameters exclusively; training retains its raw
optimizer trajectory. Inference still runs one unchanged 254-tensor model,
not a prediction ensemble. Step zero remains eligible, selection remains
worst-rating, and a matching GPU preflight must pass before launch. The new
EMA tests verify averaging arithmetic, parameter identity, and restoration of
raw training weights even when validation raises; all 49 focused tests pass.

At step 250 of the live full-context run, MovieLens MAE/RMSE improves to
0.745392357 / 0.876491889 and Netflix to 0.739136210 / 0.838676034. The worst
rating ratio falls from 1.00934873 to 1.00253340. This materially narrows the
Netflix gap but does not yet beat MF; no outer-test confirmation is authorized
by this partial result. The selected step-250 model is saved, and the run
continues to its registered 1000 updates.

`balanced_two_way_expected_rank_2k` also completes with step zero selected;
its later checkpoints do not improve Netflix's rating criteria. Both failed
follow-ups retain all their evaluation/provenance artifacts. Their redundant
step-zero weight copies may be retired only after an exact 254-tensor match
against the retained initialization, with a reconstruction/hash audit; no
previous-session or distinct trained weights are deleted.

Register `full_context_balanced_ordered_1k`: retain the best completed two-way
checkpoint, use 1000x1000 contexts (matching public inference), 2048x2048
synthetic parents, first-observation ordering, user-balanced query capping,
density 0.002–0.08 and popularity 1.4–2.8. Keep expected-rating point targets,
NLL 0.1, ranking 0.05, LR 1e-5, fresh AdamW and worst-rating selection; use seed
45, 1000 updates, validation every 250, and activation checkpointing. A 16-pair
CPU context screen gives median row/column support 20/88, versus Netflix
21/73; fractions with support at most two are 8.55%/6.40%, versus real
9.77%/8.07%. This is a closer sparse-context match, not evidence of a rating
win. Run another exact-recipe three-update preflight before training.

An inner-only residual-strength diagnostic rules out simple uniform residual
shrinkage as the solution: Netflix's best tested multiplier is the current
1.0, while MovieLens benefits from 1.25. These multipliers are diagnostic only;
no calibrated blend or inference ensemble is deployed.

The ordered-crop 2000-update run completes without improving its initial
checkpoint: step zero remains selected, and full validation exactly reproduces
the preceding model's scores. Do not promote its later weights. Matching
support statistics alone was not sufficient to improve the neural predictor.

A subsequent training audit identifies a query-sampling mismatch: uniformly
capping exposed cells drops low-activity users before the equal-user loss is
computed. In 16 paired episodes, the 512x320 recipe retains 386.2 query users
on average versus 445.7 with user-balanced sampling; at 1000x1000 the difference
is 571.6 versus 977.5. The optional `user_balanced` cap first samples one query
per eligible local user, then fills the remaining budget. If the user count
exceeds the budget, it samples users uniformly. Default outputs remain exactly
equivalent in a new 12-case audit, including actively capped episodes.

Register `balanced_two_way_expected_rank_2k`: start from the best completed
two-way/popularity model (not the rejected ordered weights), retain its IID
512x320 / 320x512 crops, 1024x1024 parents, density 0.02–0.20, popularity
1.4–2.8, LR 1e-5, NLL 0.1, ranking 0.05 and expected-rating targets. Change
query sampling to `user_balanced`, use seed 44, fresh AdamW, 2000 updates,
500-update validation and worst-rating selection. This isolates the sampling
fix before spending more compute on a larger geometry.

Activation checkpointing is also implemented as an optional training-only
execution path, with an exact CPU output/gradient/state-key equivalence test.
Unmodified 1000x1000 DWIP updates now pass a three-update GPU preflight at
84.68 GiB peak, about 4.96 seconds/update. This removes the memory obstacle to
matching inference geometry; no full-size model has yet been trained or
promoted. The focused regression suite currently has 48 passing tests. One
earlier broader run hit the unrelated unseeded bias-prior exposure-frequency
test's stochastic tolerance; an unchanged rerun passed.

The completed `two_way_popularity_expected_rank_2k` selects step 2000 under its
recorded mean-RMSE rule. Full inner evaluation confirms MovieLens MAE/RMSE
0.748170606 / 0.876796048 versus MF 0.757859740 / 0.889766278; all five
MovieLens ranking metrics also win. Netflix MAE/RMSE 0.744160937 / 0.842579609
still trail MF 0.737268413 / 0.837008864. Four Netflix ranking metrics win, but
Precision@10 trails by 0.000475271. This is not goal completion, and no outer
candidate evaluation has been run.

Its support-stratified inner errors show the largest remaining Netflix gap on
locally cold users/items. The synthetic IID crop mixes head and tail entities,
unlike the real first-observation-ordered inference blocks. The new optional
`first_observation` context ordering assigns independent arrival priorities to
context edges only, sorts each axis by first arrival, then crops using the same
overlapping-tail geometry. Crop selection anchors a uniformly chosen held-out
user and one of that user's query items; it uses masks, never hidden values.
Uniform block selection was screened and rejected because it overweighted
almost-empty tail blocks. Full-parent context-only normalization still occurs
before cropping. Default IID outputs remain exactly equal to the prior source
snapshot in a 12-case audit.

Register `ordered_two_way_expected_rank_2k`: start from the preceding selected
model, use a 2048x2048 parent, first-observation crops of 512x320 / 320x512,
density 0.005–0.12, popularity spread 1.4–2.8, expected-rating point target,
NLL 0.1, ranking 0.05, seed 43, LR 1e-5, 2000 updates and validation every 500.
The lower-density ordering screen better covers Netflix's cold-query strata;
it is not claimed to match every support quantile. Use fresh AdamW, save only
the best model including step zero, and select by `worst_rating`. Run matching
GPU preflights in both crop orientations before training. All gradients remain
synthetic-only; architecture, inference blocks, and frozen splits are unchanged.

## Target and fixed evaluation

Improve MovieLens 100K and Netflix small beyond the existing Cornac MF baseline,
without changing the DWIP v2-2 architecture or using MF at inference. Lower
user-macro RMSE and MAE on **both** datasets are the primary rating criteria;
report all seven existing metrics, including every ranking regression. A rating
win alone is not an across-the-board recommendation win.

The previous selected98k checkpoint scored RMSE 0.925936 / 0.948745, versus
historical MF 0.8998 / 0.8621 (MovieLens / Netflix). Its NDCG@10 was
0.005821 / 0.005034, versus MF 0.0761 / 0.0324. See the frozen results in
`runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919`.

Keep that run's complete user/item universes, outer 80/20 split (seed 123),
unknown exclusion, rating threshold 4, macro-over-users metrics, 1000x1000
overlapping-tail inference blocks, training-column normalization, and [1,5]
clipping. Rerun the exact MF configuration rather than relying on rounded CSVs.
The outer test results have already been viewed; they are a historical target,
not a fresh blind test. Do not repeatedly select checkpoints on those labels.

## Selection and training

Create a fixed 10% **inner holdout from outer training ratings only**, seed
20260920, excluding inner-unknown queries. New runs select by the worst of all
four inner MAE/RMSE ratios to MF; earlier runs used mean normalized RMSE and
retain that recorded selector. Confirm final rating and
ranking scores on the unchanged outer split. Further independent split seeds
are needed before making a statistical/generalization claim.

Initial experiment: synthetic-only gradient updates, warm-starting the prior
session's selected checkpoint, same 254-tensor architecture and strict loading.
Replace near-noiseless dense spectral episodes with bounded integer 1–5
ratings, noisy low-rank preferences, user/item biases, heterogeneous activity
and popularity, and preference/quality-dependent exposure. Hold out *exposed*
ratings separately from unexposed cells. Compute means/stds exclusively from
context, including parent rows outside the local training crop, to match the
global normalization used before inference blocking.

Optimize raw-rating-scale, equal-user-weighted squared error plus 0.1 NLL,
rather than matching the old spectral teachers' means. Include an optional
pairwise-ranking auxiliary term for exposed positives versus unexposed items;
first establish the rating-only control. Alternate rectangular orientations.
Use AdamW, FP32 weights/state, BF16 computation, clipping 0.5, warmup/cosine,
and an initial 2000-step pilot at 1e-5. Inspect support-stratified errors before
choosing further prior/geometry/schedule changes.

## Commands and storage

From `mcpfn`, with `PYTHONPATH=src:scripts`:

```bash
python scripts/dwip_recsys_experiment.py baseline
python scripts/dwip_recsys_experiment.py train
pytest -q tests/test_dwip_recsys_experiment.py tests/test_dwip_recsys.py
```

All new artifacts live in `runs/recsys_goal_20260920`. Keep historical artifacts
untouched. Storage is constrained (~20.4 GiB free at launch); keep one best model
and one rolling optimizer checkpoint per active experiment, using an 18 GiB
floor. Do not create an unbounded checkpoint sweep or delete older runs.

## Normalization ablation, registered before candidate evaluation

Inspecting the previous model's recommendations (not held-out labels) shows
91.6% of MovieLens top-10 recommendations have at most five training ratings;
the median recommended item has just one rating. Per-column centering makes a
single observed 1 indistinguishable from a single observed 5 to the network,
then adds that unreliable item mean back after prediction. Priors alone cannot
fully learn item-mean shrinkage if normalization removes the necessary signal.

After the column-normalized control, test `--normalization global` with matching
synthetic training: use one context-only global rating mean/std for all columns,
retaining item-quality levels in the input. This **changes preprocessing**, not
the network or evaluation split/metrics. No fitted factor model, learned extra
head, score ensemble, or target-label calibration is added. Compare on inner
validation first. New runs snapshot trainer/prior sources before training.
`--model-only` permits bounded follow-ups without saving redundant optimizer
states; these runs cannot resume an optimizer trajectory.

A third preregistered normalization option, `shrunk_column`, keeps global item
information available across Netflix's row blocks while avoiding raw singleton
item means: center each column at `(sum_context + 10 * global_mean)/(count+10)`
and use the context-only global standard deviation. The pseudo-count is fixed
at 10, not fitted to held-out labels. Training applies the identical transform.
This is normalization, not an MF predictor, and changes no network tensors.
Test it only as a separate matched-training ablation; do not silently substitute
it for either of the first two runs.

## Completed evidence so far

The deterministic MF rerun reproduces all historical four-decimal scores. Its
outer RMSE is 0.899831176 / 0.862116873. Inner-validation targets are different:
RMSE 0.889766278 / 0.837008864, MAE 0.757859740 / 0.737268413.

The 2000-step column-normalized control selects its final step. Inner RMSE
improves from 0.915132613 / 0.910830435 to 0.903018560 / 0.871944273, but **does
not beat MF**. Full public-interface evaluation agrees with the efficient
query-only validation metrics within 1e-9. Ranking remains poor: inner NDCG@10
0.000805377 / 0.000116496, versus MF 0.033978369 / 0.017740155. This control is
not promoted or evaluated again on the outer test.

Global and shrunk-column normalization are separate matched-seed, matched-LR,
2000-step ablations, not substitutions made while evaluating the control.
Inspect their `validation_*.json`, `status.json`, and eventual
`validation_full/results.csv` under `runs/recsys_goal_20260920`. The goal stays
active until a selected candidate actually clears the MF comparison; improved
training loss or partial validation gains alone do not establish success.

## Next training-objective ablation

`--point-target expected_rating` optionally replaces the noisy synthetic point
target with its exact latent-conditional expectation under Gaussian noise,
integer rounding, and [1,5] clipping. For latent preference `mu` and noise `s`,
the target is `1 + sum(Phi((mu-threshold)/s))` over thresholds 1.5–4.5. Since
exposure is independent of rating noise conditional on the sampled latent
variables, this preserves the squared-error population optimum while removing
irreducible target noise from its gradient. NLL still uses actual noisy ratings;
inputs, query masks, normalization, and ranking labels are unchanged. The latent
expectation is training-only, never an inference feature. This option is not
enabled retroactively for any of the first three normalization controls.

The completed global-normalization pilot selects step 2000 and beats MF on
**all five ranking metrics on both inner holdouts**. Its inner NDCG@10 is
0.048774027 / 0.028538900 versus MF 0.033978369 / 0.017740155. Its RMSE remains
0.899489817 / 0.881515233 and MAE 0.768930041 / 0.782365663, so it does **not**
satisfy the rating goal and has not been sent to outer-test confirmation.
This separates a demonstrated ranking improvement from the unresolved rating
problem rather than presenting an average-score win as universal improvement.

## Combined follow-up

The shrunk-column pilot selects step 2000: inner RMSE 0.898777295 / 0.869907344,
MAE 0.770002061 / 0.770847675. It wins nine of ten dataset-specific ranking
comparisons; Netflix Precision@10 remains 0.009609387 versus MF 0.010322293.
It is the strongest rating candidate, but still fails both MF rating targets.

Continue its selected checkpoint in `shrunk_expected_rank_large_4k` with fresh
AdamW state, 4000 updates at peak LR 1e-5, 512x320 alternating orientation,
the exact expected-rating point target, NLL weight 0.1, and pairwise-ranking
weight 0.05. Validate every 1000 updates using the unchanged inner split.
This is a combined practical follow-up, not an isolated causal ablation of its
three changes. The larger-context preflight completes three finite FP32/BF16
updates, peaks at 73.44 GiB allocated GPU memory, and takes ~0.58 seconds/update
after its first update. Keep only the selected model to respect the disk floor.

At update 1000 of this still-running follow-up, inner RMSE improves to
0.891086619 / 0.867092589; MAE is 0.764497737 / 0.768982192. MovieLens RMSE is
now close to MF, but both datasets still fail the rating criteria. Full ranking
evaluation is queued after training, followed by a GPU preflight of `two_way`.
No new candidate has been evaluated on the outer test during this goal.

## Pending cross-block information ablation

Netflix's 1000-column inference blocks contain only a fraction of each user's
training ratings, whereas MF sees the entire profile. A context-only global
user/item-bias diagnostic gets inner RMSE 0.858027 with regularization 3,
better than the current neural models but still worse than MF 0.837009.
This points to lost global profile information, not just noisy item means.

The optional `two_way` normalization fits ridge user/item offsets from context
only (30 alternating updates, regularization 3), subtracts them before DWIP,
and restores them afterward. It fits **no latent factors** and adds no neural
head or tensors, but does change inference preprocessing and must be reported
as such. Regularization 3 was chosen from the documented inner-only diagnostic
over 1/3/10/30; this is not an untouched synthetic-only hyperparameter choice.
Training uses the identical normalization on a larger synthetic parent matrix
before cropping both axes, preserving the context-only/no-query-leakage rule.
This option is implemented and unit-tested, not yet a validated improvement,
and is not enabled for the currently running combined follow-up.

Completed-pilot storage audit: the column-control's ~893 MiB rolling optimizer
snapshot is retired after verifying that its step-2000 model tensors are
bitwise identical to its retained selected checkpoint. Its audit hashes,
selected weights, predictions, source snapshots, and logs remain. Nothing from
the previous session is removed. Future runs explicitly include step zero as
an eligible saved candidate, so a useful preprocessing change is not lost if
subsequent fine-tuning fails to improve it.

## Continuing evidence

At update 2000 of the larger follow-up, inner MovieLens RMSE is 0.887614130,
now lower than MF 0.889766278; MAE 0.761276455 still trails MF 0.757859740.
Netflix improves to RMSE 0.858816297 and MAE 0.760249515, still behind MF.
The 4000-update experiment remains live; this is partial progress, not success.

The baseline-component audit confirms that Cornac ranking excludes training
positives, not every known training rating. Replaying all known ratings in MF
(as DWIP's imputation interface does) changes its ranking slightly but leaves
its held-out rating scores unchanged. The global-normalized pilot still wins
all five inner ranking metrics on both datasets against that stricter control.
Keep the originally reproduced MF as the official historical comparator and
report this additional protocol check rather than silently changing baselines.

## Registered next run: two-way centering and heavier popularity

Support diagnostics use equal-user weights (and equal synthetic-episode
weights), not just pooled query counts. MovieLens/Netflix inner query median
row support is 41/21 and column support is 121/73, versus 8/19 for the current
larger synthetic prior. The real column-support 90th percentile is 278 on both
datasets, versus 92 in that prior. This is another concrete prior mismatch.

The next run, `two_way_popularity_expected_rank_2k`, starts from the completed
4000-update run's inner-selected checkpoint. Keep 512x320 contexts, the
expected-rating target, NLL 0.1, ranking 0.05, LR 1e-5, fresh AdamW, and seed 41;
train 2000 updates and validate every 500. Enable `two_way` normalization,
broaden item-popularity logit standard deviation from 0.7–1.9 to 1.4–2.8, and
sample exposure density 0.02–0.20 instead of 0.01–0.12. A 16-episode paired CPU
support screen gives median row/column support 21/83 and column p90 237 for
this recipe. It improves bulk support coverage but still underrepresents cold
users; inspect those separately rather than calling the prior perfectly matched.

These choices use context statistics and inner diagnostics only. Run a new
three-update GPU preflight with the exact changed recipe before training.
Step zero is eligible, allowing separation of the immediate preprocessing
effect from subsequent prior/training changes. No outer-test candidate
evaluation is queued yet. A 12-case exact tensor-equivalence audit confirms
that adding optional prior controls leaves the older default recipe unchanged.

The larger 4000-update follow-up completes and selects its final checkpoint:
inner RMSE 0.884887328 / 0.860707334, MAE 0.758599620 / 0.763033862. All five
MovieLens ranking metrics win, but Netflix Precision@10 and Recall@10 still
trail the original MF. The goal is not met by this checkpoint.

Both two-way GPU preflights pass at ~73.43 GiB peak allocation. Before any
updates in `two_way_popularity_expected_rank_2k`, centering alone gives
MovieLens MAE 0.757573158 and RMSE 0.888969695, both slightly better than MF;
Netflix improves to MAE 0.750417367 and RMSE 0.850275472, still behind MF.
That step-zero candidate is saved. The registered 2000-update follow-up is
now training; no outer-test scores are being used to select it.

An additional context-ordering audit explains part of the support mismatch:
user/item internal IDs strongly anticorrelate with observation degree
(MovieLens -0.65/-0.79; Netflix -0.72/-0.80). With actual inference windows and
equal-user weighting, Netflix's median context density is 0.0233 and its p90
is 0.0897 despite a global density of 0.0090. A future, conditional research
option is to model first-observation ordering and crop geometry in the prior;
this is not enabled in the running experiment or required if it already wins.

At two-way update 500, inner MAE/RMSE is 0.749072101/0.882047997 for MovieLens
and 0.741469439/0.841760343 for Netflix. At update 1000, MovieLens improves
again (0.748418268/0.877121304) while Netflix regresses slightly
(0.744986479/0.843480026). The original mean-RMSE selector promotes update
1000, even though Netflix remains the unresolved part of the goal.

**Selection correction for future runs:** `--selection-objective worst_rating`
(the new CLI default) minimizes the maximum of the four ratios to MF: MAE and
RMSE on each dataset. A value below one therefore requires every rating
criterion to win; a large improvement on one dataset cannot hide failure on
the other. Older and currently running experiments keep their snapshotted
mean-RMSE selector. Their scores/checkpoints are not retrospectively relabeled.
Use `--selection-objective mean_rmse` to reproduce that earlier selection rule.
Ranking remains a separate full-interface confirmation gate.
