# Combined MatrixFormer on causal panels

The November 2 evaluation adds the fixed combined MatrixFormer checkpoint to
the existing CPS, Kansas and California-smoking placebo benchmarks. There are
3,000 frozen draws in 12 designs. No benchmark inputs, baseline predictions or
historical scores are overwritten.

## Run and export

From the `tabular` repository root:

```bash
PYTHONPATH=mcpfn/scripts:mcpfn/src python -m pytest -q mcpfn/tests/test_single_matrixformer_causal.py
python mcpfn/scripts/evaluate_single_matrixformer_causal.py
python mcpfn/scripts/export_single_matrixformer_causal_results.py --paper-root /root/MatrixFormerPaper
cd /root/MatrixFormerPaper && python scripts/plot_causal_ensemble_results.py
```

The runner accepts `--checkpoint` (the file must match the frozen SHA256) and
`--output`; the exporter accepts `--run` and `--paper-root`. Resume verifies
checkpoint, config, inputs, source code, environment and saved output hashes.
The first draw of each design is checked against the historical teacher in all
three orientations: all 36 reconstructed matrices match bit-for-bit.

Checkpoint:
`/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth`

SHA256: `0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49`.

## Protocol

- Complete the entire panel, with states as rows; no chunking.
- Standardize each column using observed entries only, run predictive-mean
  inference with FP32 weights and BF16 autocast, undo scaling and restore
  observed entries exactly. Keep the original causal SDPA settings.
- Evaluate the standard panel and its transpose with their own observed-column
  statistics, plus the average of the two reconstructed panels, not an average
  of metrics. The paper restores standard orientation, requiring one forward pass,
  as designated primary in the original evaluation and used in the other domains.
  An intermediate manuscript used averaging after reviewing all orientations;
  that post-evaluation choice is not retained. The evaluator's frozen manifest
  and numeric results remain unchanged; the exporter selects `matrixformer`
  by changing only the paper's `shown` flags. All orientations remain exploratory
  ablations, including improvements and regressions.
- Report hidden-block RMSE and absolute implied treatment-effect error. True
  treatment effects are zero: the latter is the absolute difference of the
  means of true and imputed hidden entries.
- Retain the per-draw min/max scale set by MatrixFormer*, SDiD, SC, DiD,
  SoftImpute, K-NN and HyperImpute. The new model does not change these
  endpoints; its scores can fall outside [0, 1]. This is not the
  standard-deviation-normalized RMSE used for MissBench/UCI.
- Average draws within each design, then weight designs equally within each
  benchmark. The 95% normal interval has half-width
  `1.96 * sqrt(sum(design_standard_error**2)) / number_of_designs`.
  Designs are fixed and draws are treated as independent; their shared source
  panels mean these intervals may underestimate broader uncertainty.

## Results and artifacts

Paper-reported normalized scores (mean ± 95% interval half-width): MatrixFormer
uses standard orientation; MatrixFormer* retains its archived standard orientation.

| Benchmark | MatrixFormer RMSE | MatrixFormer* RMSE | MatrixFormer effect error | MatrixFormer* effect error |
|---|---:|---:|---:|---:|
| California smoking | 0.107 ± 0.011 | 0.159 ± 0.013 | 0.247 ± 0.018 | 0.323 ± 0.020 |
| CPS | 0.185 ± 0.010 | 0.177 ± 0.009 | 0.317 ± 0.014 | 0.305 ± 0.012 |
| Kansas | 0.079 ± 0.018 | 0.093 ± 0.020 | 0.111 ± 0.021 | 0.125 ± 0.022 |

The combined model in standard orientation has the lowest point estimates for
both metrics on California smoking and Kansas; the archived ensemble has the
lowest point estimates on CPS. Averaging improves RMSE relative to standard on
all three benchmarks but improves effect error only on CPS. All orientations
remain reported as ablations; intervals do not adjust for selection among them.

The full run, predictions and provenance are saved under
`mcpfn/runs/benchmark_evaluations/single_matrixformer_causal_20261102/`.
Its `RESULTS.md` includes every baseline and orientation. Portable metrics,
input/code hashes and score CSVs are copied to
`/root/MatrixFormerPaper/results/single_model/causal/`.
The exporter reconstructs every score and verifies that the 42 historical
reference means and intervals agree with their six-decimal archive.

The root manuscript's main causal figure shows eight methods. The appendix
adds an orientation-control bar plot, in the same style and with confidence
intervals. Both MatrixFormer and MatrixFormer* are blue. The separate
historical PNAS adaptation is not updated.
