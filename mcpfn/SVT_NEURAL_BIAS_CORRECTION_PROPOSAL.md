# SVT-Guided Neural Bias Correction for Matrix Completion

## Motivation

The goal is to build a matrix completion model that keeps the stability and
theoretical bias of classical low-rank solvers, while gaining the adaptability of
a neural in-context learner. Pure neural row/column architectures have shown
that they can learn structure, but pointwise accuracy has been difficult to push
below strong matrix-completion baselines. Pure SVT/SoftImpute is stable and often
excellent, but it has fixed assumptions about the spectrum, missingness, and
noise. The middle ground is to use SVT as a differentiable algorithmic backbone
and train a neural model to diagnose when that backbone is right, when it is
wrong, and how to correct its bias from the observed matrix itself.

## Core Idea

The model should be viewed as an internal routing and bias-correction system on
top of an SVT trajectory, not as a manually routed collection of prior-specific
experts. SVT produces a sequence of candidate completions, for example at
unroll depths 0, 16, 32, and 64. These candidates expose different inductive
biases: shallow steps preserve more of the neural initialization and can work
well for spectral-tail cases, while deeper steps impose stronger low-rank
completion and are better for clean latent-factor regimes.

The neural part then performs in-context analysis of the observed matrix. It can
read row/column embeddings, missingness structure, spectral statistics, and
held-out observed-entry CV signals. From these, it learns a policy over the SVT
trajectory and, ultimately, a correction to the resulting predictions. This lets
the model behave like "SVT plus learned judgment": the algorithm supplies a
stable completion path, and the neural network learns when and how to deviate.

## Proposed Architecture

The architecture has three conceptual layers:

1. **Algorithmic backbone.** A differentiable SVT/SoftImpute-style unroll
   generates intermediate completions. This gives the model a strong default
   low-rank prior and a stable information path from observed entries to missing
   entries.

2. **Internal diagnosis.** A controller consumes matrix-level evidence:
   missingness features, spectrum features, observed-entry CV errors across SVT
   depths, and learned row/column summaries. Its job is not to identify a named
   prior, but to infer which parts of the SVT trajectory are trustworthy for the
   current matrix.

3. **Neural bias correction.** A decoder or residual head corrects systematic
   SVT errors. This is where the model can learn patterns that classical SVT
   cannot represent, such as MNAR bias, distorted spectral tails, heteroskedastic
   residual structure, or cases where the best completion is not the raw
   low-rank projection.

## Training Strategy

Training should combine pointwise accuracy with policy quality. The model should
still optimize missing-entry prediction loss, because RMSE is the target metric.
At the same time, auxiliary losses can make the internal policy learn faster:
depth distillation from per-depth missing-entry error, observed-entry
cross-validation regret, and calibration losses for the distributional decoder.
The important distinction is that these losses supervise internal judgment, not
external prior labels.

The current experiments support this direction. The controller can already learn
meaningful regime-dependent behavior: routing rank-like matrices to deeper SVT
and spectral-tail matrices to shallower SVT. The remaining failure mode is
misrouting or under-correcting regimes such as broad-rank latent factors and
MNAR matrices. Those failures are useful because they expose exactly where the
neural diagnostic and correction layers need more signal.

## Why This Is Better Than Explicit Routing

Explicit prior routing is brittle: it works only when the deployment problem
matches one of the priors we anticipated. The better objective is to make the
model perform its own internal analysis of the matrix and learn a continuous
correction policy. Then new problems are handled by similarity in observed
structure, not by matching a hard-coded prior category.

In short, the proposed model is an in-context matrix-completion system with SVT
as its scaffold: classical completion provides the reliable algorithmic path,
and the neural network learns the bias corrections needed when that path is
wrong.
