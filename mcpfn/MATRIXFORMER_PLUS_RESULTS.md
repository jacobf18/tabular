# MatrixFormer+ benchmark results

**MatrixFormer+** is the observed-error-fitted convex blend of the general
MatrixFormer model and the recommender specialist:

`MatrixFormer+ = (1 - w) × MatrixFormer + w × RecSys specialist`, with `0 ≤ w ≤ 1`.

The weight minimizes squared error on raw predictions of observed cells, using
observed-column standardization for tabular/LLM matrices and global scaling for
ratings. There is one weight per input matrix (per inference batch for LLM
evaluation). This names the existing `observed_blend` result, not the hard router
or the masked-observation blend. No new checkpoint is trained and no evaluations
are rerun for this naming change.

Bold denotes the best value among these three methods for each benchmark/metric;
exact ties are bolded together. Values are rounded to six decimal places.

## Overall tabular NRMSE ↓

Average case-level NRMSE within each dataset, then average datasets equally.
MissBench includes all 11 patterns at 40% missingness (462 cases, 42 datasets).
UCI includes all three patterns, five missingness rates and ten repeats
(1,800 cases, 12 datasets). These balanced suites also give equal pattern
weighting. This is a macro-average, not RMSE pooled over all missing cells.

| Benchmark | MatrixFormer | RecSys specialist | MatrixFormer+ |
|---|---:|---:|---:|
| MissBench overall | 1.674085 | 1.844990 | **1.579644** |
| UCI overall | 1.015805 | 1.156767 | **0.997922** |

MatrixFormer+ reduces overall NRMSE by 5.64% on MissBench and 1.76% on UCI
relative to general MatrixFormer.

## Tabular imputation: NRMSE ↓

MissBench: 42 datasets, all 11 patterns, 40% missingness, full-matrix context.
UCI: 12 datasets, three patterns, missingness rates 10–50%, ten repeats,
2000×1000 context windows. Cases are averaged within datasets before weighting
datasets equally. There is no row train/test split.

| Benchmark | Missingness pattern | MatrixFormer | RecSys specialist | MatrixFormer+ |
|---|---|---:|---:|---:|
| MissBench | MCAR | 0.893521 | 0.956852 | **0.865539** |
| MissBench | MAR | 1.126568 | 1.214635 | **1.085904** |
| MissBench | MNAR | **1.917464** | 2.494735 | 1.978498 |
| MissBench | MAR Neural | 0.924937 | 0.980997 | **0.878489** |
| MissBench | MAR Block Neural | 1.091410 | 1.133611 | **1.069641** |
| MissBench | MNAR Polarization | 3.822571 | 3.716195 | **3.065141** |
| MissBench | MNAR Soft Polarization | 2.374008 | 2.315790 | **2.270023** |
| MissBench | MNAR Latent Factor | 0.942377 | 0.975954 | **0.906569** |
| MissBench | MNAR Cluster Level | 0.908557 | 0.954987 | **0.886365** |
| MissBench | MNAR Two-Phase Subset | 0.925732 | 0.995789 | **0.902942** |
| MissBench | MNAR Censoring | 3.487793 | 4.555342 | **3.466970** |
| UCI | MCAR | 0.697792 | 0.779228 | **0.684916** |
| UCI | MAR | 0.835767 | 0.915139 | **0.820466** |
| UCI | MNAR | 1.513857 | 1.775935 | **1.488384** |

MatrixFormer+ improves 13 of 14 pattern summaries versus general MatrixFormer.
The exception is MissBench MNAR; the regression is retained in the table.

## Recommender systems: RMSE / MAE ↓

Metrics are macro-averaged over test users in original rating units, using the
frozen outer splits. Each expert retains its established recipe: global
normalization and 9976×1000 streamed context for the smaller matrices, or
10000×1000 for MovieLens 1M/10M; two-way centering, user scaling and 2000×1000
context for the specialist. Both clip to the observed training rating range:
[0.5,5] for MovieLens 10M and [1,5] otherwise.

| Dataset | Metric | MatrixFormer | RecSys specialist | MatrixFormer+ |
|---|---|---:|---:|---:|
| MovieLens 100K | RMSE | 0.920160 | **0.887358** | **0.887358** |
| MovieLens 100K | MAE | 0.752393 | **0.727878** | **0.727878** |
| Netflix small | RMSE | 0.886902 | **0.861721** | **0.861721** |
| Netflix small | MAE | 0.760755 | **0.732989** | **0.732989** |
| MovieLens 1M | RMSE | 0.896905 | **0.876720** | **0.876720** |
| MovieLens 1M | MAE | 0.734689 | **0.714240** | **0.714240** |
| MovieLens 10M | RMSE | 0.866433 | **0.833401** | **0.833401** |
| MovieLens 10M | MAE | 0.706832 | **0.678188** | **0.678188** |

The observed-error fit assigns specialist weight 1.0 on all four datasets, exactly
recovering its predictions. These comparisons combine deployed expert recipes,
not checkpoint weights alone. MovieLens 1M/10M use the separately audited
`matrixformer_plus_large_recsys_20261004` extension. Its general predictions
match the frozen original results exactly; both metrics improve over general
MatrixFormer, but MF/SVD/PMF remain stronger on these two larger datasets.

## LLM benchmark completion: R² ↑

R² at five mutual-information-selected tasks, averaged across ten folds.
The full campaign also evaluates entropy/random selection and budgets 1–15;
these are representative MI@5 summaries, not averages across all those settings.

| Dataset | MatrixFormer | RecSys specialist | MatrixFormer+ |
|---|---:|---:|---:|
| MTEB | 0.763884 | 0.693816 | **0.765135** |
| MMLU | **0.899200** | 0.762836 | **0.899200** |
| Merged | 0.407349 | 0.328567 | **0.427926** |
| BenchPress | 0.182682 | 0.105368 | **0.204020** |

Across all 180 dataset/selection/budget settings, MatrixFormer+ improves R² in
158, worsens it in 18, and ties in four versus general MatrixFormer. These are
dependent settings, not independent significance tests.

## Provenance and interpretation

- General checkpoint: `runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth`.
- Specialist checkpoint: `runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth`.
- Audited campaign: `runs/benchmark_evaluations/two_expert_router_v2_20261004/`.
- Audited MovieLens 1M/10M extension: `runs/benchmark_evaluations/matrixformer_plus_large_recsys_20261004/summary.csv` and `final_audit.json`.
- Audited transpose/average comparison (2026-10-05): `runs/benchmark_evaluations/matrixformer_orientation_v2_20261004/RESULTS.md`; protocol and interpretation in `ORIENTATION_EVALUATION.md`. These variants do not replace the original results above.
- Numeric sources: `tabular/summary.csv`, `recsys/summary.csv`, and `llm/summary.csv` inside that campaign; MatrixFormer+ maps exactly to `observed_blend`.
- The original `RESULTS.md`, manifests, predictions and audits are unchanged. That report also retains hard-routing and masked-observation controls.
- Raw observed-cell predictions see their own input values. Fitting uses no benchmark-hidden targets, but reconstruction error is in-sample evidence, not held-out validation.
- MatrixFormer+ requires both expert inference passes. These previously inspected benchmarks support an exploratory comparison, not a new independent confirmation.
