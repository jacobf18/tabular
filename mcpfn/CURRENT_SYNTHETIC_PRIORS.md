# MatrixFormer: current synthetic training priors

**Recipe snapshot: 2026-11-08.** This describes the active **million-matrix continuation**, not every generator available in the repository and not the historical pretraining mixture. It is intended as a standalone reference for making a synthetic-priors figure.

- Repository: `/root/tabular`.
- Active run: `/mnt/matrixformer_scratch/scientific_prior_training_20261108/million_v1`.
- Sampling specification: `mcpfn/scripts/million_matrix_prior.py`.
- Training entry point: `mcpfn/scripts/train_million_matrix_continuation.py`.
- Run documentation: `mcpfn/MILLION_MATRIX_CONTINUATION.md`.

All gradient-training matrices in this continuation are generated synthetically. The real-data MissBench/UCI reruns are separate evaluations; their new repeated-mask implementations are **not** the training generators described here. These are statistical, domain-inspired priors, not calibrated physical simulators.

## 1. Top-level mixture

Each optimizer update accumulates gradients from **four independently generated matrices: one from each domain**. The plan is 250,000 updates, giving 1,000,000 usable matrices, or 250,000 per domain. These are planned totals, not a statement that training has finished.

| Domain | Fraction of matrix draws | Weight in the training loss | Main structure |
|---|---:|---:|---|
| Scientific panels | 25% | 0.35 | Smooth dynamics, transport, nonlinear measurements, mixtures, processes, profiles |
| General imputation | 25% | 0.35 | Spectral matrices and latent-factor matrices, with diverse masks and sparse support |
| Recommendation | 25% | 0.20 | Sparse ordinal user–item ratings with exposure bias |
| Score panels | 25% | 0.10 | Bounded-score-like matrices with partially observed new rows |

**For a figure depicting how many matrices are generated, use 25/25/25/25, not 35/35/20/10.** The latter are loss weights. Within general imputation, native and structured-mask episodes alternate equally.

## 2. Scientific panels: ten active families

Each family occurs once per ten science draws, so each contributes approximately **2.5% of all training matrices**. Family and mask selections form a deterministic cycle covering all ten families crossed with all six science masks every 60 optimizer updates. Random parameters and matrix dimensions still vary by seed.

### 2.1 Family catalog

| Display name | Implementation key | Construction | Useful visual signature |
|---|---|---|---|
| Smooth sensor networks | `smooth_sensors` | Shared smooth latent factors mixed across channels, plus independent smooth channel curves with weight 0.25 | Correlated traces with local channel-specific deviations |
| Dynamic factor panels | `dynamic_factors` | AR(1) latent states with persistence 0.6–0.995, mixed with smooth factors; weak nonlinear response | Correlated but irregular temporal fluctuations |
| Transport and delayed pulses | `transport` | Two to six Gaussian pulses with channel-dependent delays, widths and amplitudes; weak low-rank background and a `log1p` transform | Similar peaks shifted across channels |
| Nonlinear calibration | `nonlinear_calibration` | Linear latent projections; about 35% of channels stay linear, others use tanh, quadratic and sinusoidal responses; channel-specific drift | Shared latent signal expressed through different response curves |
| Grouped regimes | `regime_panel` | Channels assigned to latent groups, a shared background, and one to three smooth step changes with group-specific offsets | Groups of channels changing together |
| Chemical mixtures | `chemical_mixture` | Softmax mixture proportions times positive endmember signatures, with a small pairwise interaction | Changing compositions and correlated measurement channels |
| Treatment processes | `treatment_process` | Positive flows and loads, load/flow concentrations, residence time, reaction attenuation, saturation and interactions | Nonlinear dependencies between process measurements |
| Assays and covariates | `assay_covariates` | Six latent factors, random projections, per-column identity/tanh/ReLU/square/sine/softplus transforms, and a latent-threshold regime shift | Different marginal shapes linked by common covariates |
| Locally smooth measurements | `smooth_local` | Four shared cubic-spline states plus independent channel splines | Smooth shared trajectories with local departures |
| Repeated spatial profiles | `spatial_profiles` | Two to five casts, each with reset depth, displaced transitions, exponential decay and interactions | Repeated depth profiles with cast-dependent shifts |

### 2.2 Original six families: shared ingredients

The first six families are implemented in `mcpfn/src/tabimpute/prior/scientific_panels.py`.

- Time increments are lognormal, with log-standard deviation chosen from 0, 0.3 or 0.8. Cumulative times are rescaled to `[0, 1]`.
- The latent dimension is sampled uniformly from 2 through `min(7, number_of_columns)`.
- Each initial smooth factor combines six sine waves, random phases and a linear trend. Frequencies are log-uniform from 0.15 to 8; amplitudes are damped by `max(frequency, 1)^1.5`.
- Column loadings are Gaussian, scaled by the inverse square root of latent dimension. Nonlinear transformations can make the final matrix rank exceed the latent dimension.
- `nonlinear_calibration` and `chemical_mixture` replace the smooth latent factors with independent Gaussian row factors in half their draws; they are not always time-series-like.
- Structural columns are standardized, then receive heteroscedastic Gaussian noise. The base noise scale is log-uniform from 0.002 to 0.2, with lognormal channel multipliers.
- With probability 0.2, the draw also receives sparse outliers: each cell has probability 0.005 of an added `Normal(0, 0.5)` perturbation.
- Columns are then given random scales `LogNormal(0, 1)` and offsets `Normal(0, 3)`.
- With probability 0.5, three fully observed covariates are appended: time, `sin(2π time)` and `cos(2π time)`.

For example, before noise and rescaling, a smooth sensor panel has the form

\[
S_{ij}=\sum_{k=1}^{r}F_k(t_i)W_{kj}+0.25\,G_j(t_i).
\]

**The returned science target includes generated measurement noise and outliers. It is not the noiseless structural signal.** The model is trained to impute missing measurements.

### 2.3 Four additional process families

These are implemented in `mcpfn/src/tabimpute/prior/scientific_processes.py`.

`treatment_process` and `assay_covariates` use six latent states, with a 50/50 choice between independent Gaussian rows and smooth cubic-spline states with 12 knots. Representative treatment-process relationships are

\[
\text{concentration}=\frac{\text{load}}{\text{flow}},\qquad
\text{residence}=\frac{\exp(0.25z_4)}{\text{flow}},\qquad
\text{outlet}=\text{concentration}\,
\exp[-\kappa\,\text{residence}\,(\text{oxygen}+0.4)].
\]

Features also include saturating concentration responses and cross-concentration interactions. Channels are shuffled/repeated selections of these features with weak random mixing.

`smooth_local` uses 4–14 spline knots for shared states and six knots for independent channel curves, whose mixing weight is sampled from 0.05–0.4. `spatial_profiles` constructs two to five casts; its first column is a depth covariate and is always observed after masking.

All four standardize structural columns and add heteroscedastic Gaussian noise. Its base scale is log-uniform from 0.001–0.04 for `smooth_local`, and 0.01–0.15 for the other three. Final scales are `LogNormal(0, 0.8)` and offsets `Normal(0, 2)`. The depth column also receives the generated noise and affine transform. These four do **not** append the optional three time features.

### 2.4 Scientific missingness

All six mechanisms start with independent missing probability sampled uniformly from 0.02–0.30, then optionally add structure. The final missing fraction can therefore be substantially higher.

| Mask key | Additional structure |
|---|---|
| `isolated` | No additional structure beyond independent missing cells |
| `bursts` | One to three contiguous outages per channel, up to roughly one third of the row axis each |
| `network_outage` | One to three shared outage intervals affecting a random subset of channels |
| `scheduled_assays` | Roughly one third of channels observed only at one phase of a periodic schedule, subject to the initial missingness |
| `self_mnar` | Additional sigmoid missingness depending on the standardized value of the cell itself; column-specific slope in `[-3, 3]` |
| `covariate_mar` | Additional sigmoid missingness depending on the first column; that covariate is kept fully observed |

The generator repairs support to leave at least four observations per measurement column. Science episodes then use means and standard deviations computed from observed entries only, applying those statistics to both source and target. Missing input cells are represented by `NaN`.

## 3. General-imputation priors

There are two equally frequent branches. Each accounts for **12.5% of all training matrices**.

### 3.1 Matrix-value generators

Both branches use the following four underlying generator families:

| Family | Requested rank | Spectrum or factor construction |
|---|---|---|
| Low rank | 2–16 | Mixed exponential-head/power-law-tail spectrum; tail exponent 0.2–2.0 |
| Slowly decaying spectrum | 2–160 | Same spectral construction; tail exponent 0.0–0.5 |
| Steeply decaying spectrum | 2–160 | Same spectral construction; tail exponent 1.0–2.5 |
| Latent-factor anchor | 1–11 | Product of row and column factors drawn from heterogeneous distributions |

Ranks are clipped to feasible matrix dimensions. The three spectral families construct

\[
S=U\operatorname{diag}(s_0,\ldots,s_{r-1})V^\top,\qquad
X^{\mathrm{measurement}}=S+\epsilon.
\]

The **target is the complete clean signal `S`**, while the observed source contains Gaussian noise with standard deviation sampled uniformly from 0 to 0.001. The active `spectrum_target_rank=10000` retains every generated component; it is not a smaller-rank truncation target.

The exact spectrum, before optional area scaling, is

\[
\begin{aligned}
b_k&=(1-e^{-h})e^{-k/\tau_b},\\
\widetilde{s}_k&=\left[b_k e^{-(k/\tau_e)^p}
 +(1-b_k)a(1+k)^{-\alpha}\right]
 e^{-\lambda(k/\max(r-1,1))^q},\\
s_k&=A\,\widetilde{s}_k/\widetilde{s}_0.
\end{aligned}
\]

The continuous parameters are sampled uniformly: `h ∈ [0,3]`, `τ_b ∈ [1,20]`, `τ_e ∈ [1,30]`, `p ∈ [1,3]`, `a ∈ [0.2,1]`, `λ ∈ [0,1]`, `q ∈ [1,4]`, and `A ∈ [1,5]`. The exponent `α` depends on the family in the table. Thus the family names describe a distribution of spectral shapes, not a single fixed decay curve.

Each singular-vector basis independently chooses uniformly among:

- **Haar:** Gaussian features orthogonalized by QR.
- **Coherent:** lognormally row-scaled Gaussian features, then QR; leverage can concentrate on a subset of rows or columns.
- **Smooth latent geometry:** random Fourier cosine features of random latent positions, with small perturbations, then QR. This does **not** imply smoothness in the displayed row order.
- **Offset:** Gaussian features with one nearly constant feature, then QR.

Half the episodes request the original amplitude regime and half request area-scaled amplitudes. For a spectral matrix with actual pre-transpose dimensions `R × C`, the latter multiplies its clean singular values by

\[
g(R,C)=\max\left(1,\sqrt{\frac{RC}{240\cdot80}}\right)
\]

before noise and normalization. Small matrices therefore have identical amplitudes in both regimes. Latent anchors are not area-scaled.

The latent anchor has `S = UVᵀ`; each factor vector independently chooses Gaussian, Laplace, Student-t, spike-and-slab, or centered Dirichlet coordinates. It retains intrinsic dimensions **10–50 rows by 5–50 columns**. Feature warping and quantization are disabled in the active recipe. This family does not add the spectral generator's Gaussian observation noise.

### 3.2 Native branch: eight components

The native branch cycles equally over:

| Index | Component |
|---:|---|
| 0 | `low_rank_mcar` |
| 1 | `low_rank_profile_mixed` |
| 2 | `slow_tail_mcar` |
| 3 | `slow_tail_profile_mixed` |
| 4 | `steep_tail_mcar` |
| 5 | `steep_tail_profile_mixed` |
| 6 | `anchor_latent_mcar` |
| 7 | `anchor_latent_profile_mixed` |

Each component contributes approximately 1/8 of native draws, or 1.5625% of all matrices. MCAR components begin with missing probability 0.4. The profile-mixed components choose among `mcar`, `mar`, `mnar`, `mnar_censoring`, `mnar_two_phase_subset`, `mnar_skip_logic`, and `mnar_cluster_level` with weights 0.30, 0.20, 0.20, 0.10, 0.10, 0.05, and 0.05. Their nominal missing probability is 0.4 with probability 0.25; otherwise it is `0.05 + 0.70 × Beta(2,2)`.

Native matrices transpose with probability 0.5. After the initial mask, `broad_support` augmentation:

- Leaves the original mask unchanged with probability 0.25.
- Otherwise chooses a row-oriented or column-oriented query slice, selecting 10–40% of positions along that axis.
- Restricts retained support in that slice using a fraction sampled log-uniformly from 0.01–0.60, intersected with the original observed mask. Allowed locations can be shared or independently sampled across query positions.
- Trains on missing cells in the selected query slice, rather than necessarily every missing cell in the matrix.
- Recomputes observed-only normalization for augmented episodes, using pooled within-column variance shrinkage with `variance_df=1`.

The active driver supplies forced shapes, which disables the inherited sampler's separate 20% small spectral replay switch. Do not depict that replay switch as an additional active branch.

### 3.3 Structured branch: four generators × twelve masks

The structured branch crosses the four matrix generators with the following implementation keys, cycling through all 48 combinations every 96 optimizer updates:

| Mask key | Naming guidance |
|---|---|
| `mcar` | MCAR |
| `mar` | Column-dependent MAR (`Col-MAR` in the paper) |
| `mnar` | Self-MNAR |
| `mnar_censoring` | Censor-MNAR |
| `mnar_two_phase_subset` | Display as Phase-MAR; implementation retains its historical key |
| `mnar_skip_logic` | Skip-logic mechanism, an extra family beyond the eleven-pattern MissBench suite |
| `mnar_cluster_level` | Display as Clust-MCAR; implementation retains its historical key |
| `mar_neural` | Neural mechanism; do not infer the paper's category solely from this historical key |
| `mar_block_neural` | Block-neural mechanism; likewise retains its historical key |
| `mnar_latent_factor` | Display as Latent-MCAR; implementation retains its historical key |
| `mnar_polarization` | Pol-MNAR |
| `mnar_soft_polarization` | Soft-Pol-MNAR |

The requested missing probability is either 0.2 or 0.6, equally likely. **These are nominal settings, not a guarantee of 20% or 60% realized missingness:** some inherited mechanisms do not use `p_missing` directly. For plots, compute missingness from the returned mask.

This branch uses observed-only raw-source column normalization with DF1 variance shrinkage, no extra native support augmentation, and no native random transpose. The neural masks retain their original configurations, including internal seed-42 settings; they are not the newly implemented tiled, repeated-seed benchmark masks. Full-source normalization used to construct value-dependent synthetic masks is distinct from observed-only normalization of the model input.

## 4. Recommendation prior

Implementation: `mcpfn/src/tabimpute/prior/recsys_rating_episode.py`, called through `MixedSampler.sample` in `mcpfn/scripts/single_model_distillation.py`.

A **12,288-user × 4,096-item parent population** is generated before extracting a smaller training context. A typical latent preference is

\[
P_{ij}=\mu+b_i+c_j+\gamma A_{ij}.
\]

- Rank is sampled from 2–16; affinity `A` is a 50/50 mixture of Gaussian low-rank and clustered-affinity constructions.
- The global mean is 3.1–3.9, with Gaussian user and item biases; signal strength is 0.2–0.85.
- Heteroscedastic Gaussian rating noise has base scale 0.55–1.0 and user-specific lognormal variation. Ratings are rounded and clipped to integers 1–5.
- A sampled fraction of 0–15% of cells is replaced by uniformly random integer ratings.
- Exposure depends on user activity, heavy-tailed item popularity, affinity and item quality. Parent exposure density is sampled log-uniformly from 0.004–0.08; this is not uniform independent masking.
- Of the exposed ratings, 15–30% are held out. Queries involving users or items without observed parent support are excluded.
- Context cropping uses first-observation ordering. Queries are sampled in a user-balanced manner, capped at 4,096 rating targets per episode.
- Observed-context normalization removes fitted user/item biases and applies residual and user-specific scales (`two_way_user_scale`).

**Target distinction:** the sampler also computes an auxiliary expected-rating field because it is called with `point_target="expected_rating"`. However, this continuation passes `ratings.target` and `ratings.rating_query` to its loss. It supervises the **normalized sampled ratings**, not that auxiliary expectation. Unexposed negative-query cells are not training targets in this continuation.

For a plot, show a cropped ordinal-rating matrix alongside its exposure/holdout mask. The realized support density after cropping need not equal the parent density.

## 5. Score-panel prior

Implementation: `sample_score_panel` in `mcpfn/src/tabimpute/prior/scientific_processes.py`.

Rows represent model-like entities and columns represent task-like scores, but all values are synthetic:

\[
X_{ij}=\operatorname{sigmoid}(a_i^\top w_j+b_j+c_i)+\epsilon_{ij}.
\]

- Latent rank is 2–11 for the active dimensions.
- Abilities and loadings are Gaussian, with inverse-square-root rank scaling of the loadings; row and column biases have standard deviations 0.6 and 0.7.
- Gaussian score noise has standard deviation sampled uniformly from 0.005–0.08.
- There is **no final clipping after adding noise**, so a few generated values can lie outside `[0,1]`.
- The first 60–85% of rows are context rows, with independent 10% missingness.
- Remaining rows are missing except for a shared randomly chosen subset of 1–15 observed columns.
- Model inputs and targets use observed-only column normalization.

For a figure, this is a natural block-mask example: densely observed historical rows above sparse new rows. It is not a copy of a real LLM evaluation matrix.

## 6. Requested versus realized dimensions

The initial shape proposal is:

| Probability | Proposed shape, rows × columns |
|---:|---|
| 80% | Uniform over `32×8`, `64×16`, `96×24`, `128×32`, `192×48`, `256×64` |
| 18% | Uniform over `384×96`, `512×128`, `768×128` |
| 1.8% | `1024×256` |
| 0.1% | `1536×512` |
| 0.1% | `2000×1000` |

**This is not the final distribution of matrix sizes.** Domain-specific rules are applied afterward:

| Domain/branch | Actual shape adjustments |
|---|---|
| Original six science families | May append three fully observed time-feature columns |
| Four process science families | Use requested dimensions without those appended features |
| Native spectral imputation | Use proposed dimensions, then may transpose |
| Latent anchors | Keep intrinsic 10–50 rows × 5–50 columns; native branch may transpose |
| Structured neural masks | Spectral shapes `24×12` with probability 0.85 or `48×24` with probability 0.15 |
| Other structured masks | Spectral shapes `240×80` with probability 0.85 or `1024×256` with probability 0.15 |
| Recommendation context | At least 128 rows and 48 columns, extracted from the larger parent population |
| Score panel | Clamp to 96–512 rows and 16–128 columns |

The latent-anchor override still applies inside both structured-mask shape categories. Use logged `rows`, `columns`, `observed_cells`, and `query_cells` for empirical size summaries. Degenerate structured draws may retry up to 16 seeds, with rejected attempts recorded; the million-matrix counter counts usable episodes, not rejected attempts.

## 7. What is not in this active mixture

- `damped_oscillators` and `spatial_field` exist as transfer-only scientific families but are excluded from training. `spatial_profiles`, in contrast, is active.
- Other historical generators in the repository are not automatically part of this run. In particular, the active general-imputation mixture is the four value-generator families above, not every latent/nonlinear/robust-PCA generator defined in the library.
- Real scientific observations, real user ratings, real LLM scores, MissBench matrices and UCI matrices are not used as gradient-training examples in this continuation. Real benchmarks are monitored separately.
- The fixed synthetic-validation episodes are separate from training draws.

The run is a continuation from `retained_v2/snapshots/step_3000/checkpoint.pth`, not training from scratch. A frozen original combined MatrixFormer supplies an additional retention loss on the non-science domains. Its predictions do not constitute an additional data-generation family. The primary losses are target NLL, truth Huber loss and a squared CDF loss; the science domain has no teacher-retention term.

## 8. Guidance for the separate priors figure

1. **Separate matrix values from missingness.** Use a target/value heatmap and a binary mask rather than asking one heatmap to explain both. Neutral gray can indicate unobserved input cells, not zero values.
2. **Use spectra for spectral priors.** Show normalized singular-value curves for low-rank, slow-tail and steep-tail examples, alongside heatmaps if space permits. Compute the signal spectrum from the clean generator output; the noisy source has an additional noise floor.
3. **Use traces for temporal priors.** A few channels versus time reveal transport delays, regime changes and smooth local variation more clearly than a large undifferentiated heatmap.
4. **Keep the science target label honest.** Label it “complete generated measurements,” not “noise-free truth.” For the spectral generator, “clean signal” is appropriate.
5. **Show normalization explicitly.** Raw units illustrate heterogeneous offsets and scales; model-space values illustrate what the network receives. Do not mix the two under a single color-scale label.
6. **Do not invent visible clusters.** Low rank alone need not yield rectangular blocks; Fourier-basis smoothness is in latent coordinates, not necessarily displayed row order. Label any sorting used for visualization.
7. **Report actual masks and sizes.** Distinguish requested missingness from realized missing fraction, and missing cells from loss-query cells in sparse-support episodes.
8. **Use reproducible, independent illustration seeds.** Predeclare examples without selecting them for favorable model performance. A small illustration is not a claim about the active size distribution.

An economical layout would use four domain panels, with a separate science-family gallery or supplemental figure. One panel per domain cannot represent all ten scientific families or all twelve structured masks.

### CPU-only sampling entry points

The following creates one illustrative scientific episode in memory. It does not load a model, launch training, or modify the active run:

```bash
cd /root/tabular
PYTHONPATH=mcpfn/src:mcpfn/scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import numpy as np
from tabimpute.prior.scientific_panels import sample_scientific_episode

episode = sample_scientific_episode(
    73100001,
    family="smooth_sensors",
    pattern="bursts",
    rows=128,
    columns=24,
)
complete_measurements = episode.target * episode.scale + episode.center
observed_measurements = np.where(episode.query, np.nan, complete_measurements)
print(episode.metadata)
print("Shape:", complete_measurements.shape)
print("Missing fraction:", float(episode.query.mean()))
PY
```

For the four process families, use `sample_process_episode` from `tabimpute.prior.scientific_processes` with the same `seed`, `family`, `pattern`, `rows`, and `columns` arguments. For score panels, use `sample_score_panel(seed, rows=..., columns=...)`. All return source, target, query, center, scale and metadata; reconstruct raw values as in the example.

For a structured spectral example, the following Python calls return CPU tensors and both raw and normalized arrays:

```python
from single_model_complete_missingness_prior import CompleteMissingnessSampler

sampler = CompleteMissingnessSampler()
episode, arrays = sampler.draw(
    family="mcar",
    component="slow_tail_mcar",
    shape=(240, 80),
    regime="area_scaled",
    requested_missingness=0.6,
    seed=73100002,
)
```

`arrays` contains `raw_source`, `raw_target`, `query`, `model_source`, `model_target`, `observed_mean`, and `observed_scale`. The raw source is the complete noisy matrix before masking; apply `query` for its observed-only display. This example samples the structured branch, not the native sparse-support branch.

For exact active specifications, `training_specs(step)` in `million_matrix_prior.py` returns all four draw specifications for a given update. The active training seed base is `1_810_000_000`, distinct from validation (`2_310_000_000`) and preflight (`2_410_000_000`). Inspecting specifications is lightweight; `MillionSampler.draw` itself uses CUDA and should not be launched casually alongside the active jobs. The recommendation generator accepts `device="cpu"`, but its full parent population is substantially larger than the displayed context; reducing that population would be an illustrative variant rather than the exact recipe.

## 9. Implementation map

All paths below are relative to `/root/tabular`.

| Purpose | Source |
|---|---|
| Domain mixture, schedules and shape overrides | `mcpfn/scripts/million_matrix_prior.py` |
| Active training objective and retention | `mcpfn/scripts/train_million_matrix_continuation.py` |
| Six original science families and science masks | `mcpfn/src/tabimpute/prior/scientific_panels.py` |
| Four added science families and score panels | `mcpfn/src/tabimpute/prior/scientific_processes.py` |
| Spectrum generation and noisy/clean matrix pairs | `mcpfn/src/tabimpute/prior/spectrum_training_set_generation.py` |
| Singular-vector geometry | `mcpfn/scripts/general_spectral_geometry.py` |
| Native amplitude mixture and transpose path | `mcpfn/scripts/single_model_area_prior.py` |
| Structured imputation episodes and raw plotting arrays | `mcpfn/scripts/single_model_complete_missingness_prior.py` |
| Structured mask dispatch and configurations | `mcpfn/scripts/diagnose_single_model_missingness_coverage.py` |
| Latent factors and inherited imputation mask classes | `mcpfn/src/tabimpute/prior/training_set_generation.py` |
| Sparse row/column support augmentation | `mcpfn/scripts/rowcol_fixed_validation.py` |
| Observed-support variance shrinkage | `mcpfn/scripts/spectral_support_normalization.py` |
| Active recommendation wrapper | `mcpfn/scripts/single_model_distillation.py` |
| Rating and exposure generation | `mcpfn/src/tabimpute/prior/recsys_rating_episode.py` |

This document only records the active recipe. Creating it does not change training, evaluation, checkpoint selection, or paper figures.
