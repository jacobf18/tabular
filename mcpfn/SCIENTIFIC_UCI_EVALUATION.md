# Full UCI evaluation of the scientific continuation

The November 8, 2026 evaluation uses the **actual step-3,000 scientific
checkpoint**, selected before this full evaluation. It does not use
`best_retention_checkpoint.pth`: that file still contains the unchanged original
fallback because no trained checkpoint passed every original retention check.

Checkpoint:
`/mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/snapshots/step_3000/checkpoint.pth`

SHA256: `e7f5aca853bc0925ad5d5fa4d724914144c882614696895c24e10ec97333b12b`

Results:
`/mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/full_uci_step3000_v1/`

## Completed results

All 1,800 cases finished successfully in 535 seconds. Every one of the 36
overlapping screening cases exactly reproduces its earlier NRMSE. Original and
saved candidate predictions were independently rescored for every case.

| Pattern | Original NRMSE ± 95% CI | Scientific checkpoint NRMSE ± 95% CI | Change |
|---|---:|---:|---:|
| Overall | 1.0170 ± 0.3051 | 0.9677 ± 0.2876 | -4.84% |
| MCAR | 0.6885 ± 0.1623 | 0.6485 ± 0.1533 | -5.81% |
| Col-MAR | 0.8264 ± 0.1630 | 0.7852 ± 0.1583 | -4.99% |
| Self-MNAR | 1.5361 ± 0.6784 | 1.4695 ± 0.6527 | -4.33% |

Eleven of twelve dataset averages improve. Blood transfusion is the exception:
0.6250 → 0.6437 (+2.99%). Every missingness-rate aggregate improves, by 3.74–5.81%.
The paired overall difference (candidate minus original) is
**-0.0493 ± 0.0309**, using the twelve dataset-level paired differences.
This confirms the direction of the screening-subset improvement without
resolving the separate BenchPress retention regression.

## Scope and comparability

- All **1,800 frozen cases**: 12 datasets, three missingness patterns, five
  missingness settings (10%, 20%, 30%, 40%, 50%), and ten repeats.
- Patterns: MCAR, Col-MAR, and Self-MNAR. The saved historical identifiers remain
  `MCAR`, `MAR`, and `MNAR` for compatibility.
- Native orientation, predictive mean, BF16 autocast, and observed-only
  column-standardization before inference.
- Maximum context of 2,000 rows by 1,000 columns, with the existing overlapping
  tail-window averaging. All rows are retained; there is no row train/test split.
- The original combined MatrixFormer is the reference. Its saved predictions are
  independently rescored on every case rather than trusting summary tables.
- The **unchanged historical NRMSE scorer** averages column RMSE divided by the
  standard deviation of the held-out true values, with the existing
  constant-column rules. This scoring denominator is different from the
  observed-only standard deviation used for model input preprocessing.
- Average cases within each dataset, then weight datasets equally. The 95% t
  intervals use **12 dataset means**, not 1,800 independent cases. Paired
  difference intervals also use dataset-level differences.

These are previously inspected development benchmarks, not untouched tests.
This evaluation does not train, modify model weights, select among checkpoints,
change the completed run's acceptance rules, or promote a replacement model.
The user's updated preference is to prioritize UCI improvements while tolerating
MissBench tradeoffs in future decisions; the historical gates remain unchanged.

## Run or resume

```bash
cd /root/tabular
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=mcpfn/src:mcpfn/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u mcpfn/scripts/evaluate_scientific_uci.py \
  --checkpoint /mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/snapshots/step_3000/checkpoint.pth \
  --output /mnt/matrixformer_scratch/scientific_prior_training_20261108/retained_v2/full_uci_step3000_v1
```

The driver takes an exclusive output lock. Rerunning the same command verifies
and reuses cached predictions; changed checkpoints, sources, or protocols are
rejected. Use scratch storage: predictions are not written to the nearly full
root filesystem.

## Outputs

- `summary.md`: overall, per-pattern, and per-dataset comparison tables.
- `overall.csv`, `per_pattern.csv`, `per_dataset.csv`,
  `per_missingness.csv`, `per_pattern_rate.csv`: machine-readable summaries.
- `per_case.csv`, `cases/`, `predictions/`: all individual results and predictions.
- `manifest.json`, `sources/`: checkpoint/protocol/source hashes and source copies.
- `audit.json`: successful rescoring and hash verification after all cases finish.
- `status.json`, `evaluation.log`, `launch.json`: completion state and worker log.
- `error.json`: diagnostic traceback if execution fails.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mcpfn/src:mcpfn/scripts \
  python -m pytest -q -p no:cacheprovider \
  mcpfn/tests/test_scientific_uci_evaluation.py \
  mcpfn/tests/test_chunked_uci.py \
  mcpfn/tests/test_dwip_missbench_uci.py
```
