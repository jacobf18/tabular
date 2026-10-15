# Submodular Benchmark Selection — Code & Data Release

This directory contains the complete code and assembled score matrices used in
the experiments for *Submodular Benchmark Selection* (Smola, 2026).

```
release/
├── code/        # Python modules and experiment scripts (see code/README.md)
├── data/        # Score matrices and per-leaderboard sources (see data/README.md)
├── figures/     # Output: PDF figures (created on first run)
└── logs/        # Output: JSON/NPZ logs (created on first run)
```

## Quick start

The release is self-contained. From this directory:

```bash
# 1. (optional) regenerate per-dataset covariance estimates from the matrices
python code/estimate_covariance.py

# 2. main greedy CV experiments — Figures 2, 3, 4 (~30 min, dominated by Merged EM)
python code/eval_greedy_all.py

# 3. entropy vs. mutual information — Figures 5, 6, 7
python code/eval_entropy_vs_mi.py

# 4. eigenvalue spectrum — Figure 1
python code/plot_eigenspectrum.py

# 5. selection-order plots — Figures 9, 10, 11 (depends on 2)
python code/plot_selection_order.py
python code/plot_mi_selection_all.py

# 6. normality diagnostics — Figure 12
python code/run_normality.py

# 7. appendix experiments
python code/eval_benchpress.py     # Figures 13, 14
python code/eval_tabimpute.py      # Figure 15  (requires GPU + tabimpute pkg)
python code/eval_logit.py          # Figures 16-18
```

All scripts are invoked from the release root and resolve paths through
`code/path_config.py`, so they also work when invoked from another directory.
Set `BENCHSELECT_DATA_DIR` or `BENCHSELECT_EXPERIMENT_ROOT` to redirect inputs
or outputs for scratch runs.

## Dependencies

- Python ≥ 3.9
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `tabimpute` (only for `eval_tabimpute.py`; requires CUDA)
- Tests: `pytest`

## Data

Three primary score matrices and a fourth from BenchPress live in `data/`:

| Matrix       | Shape       | % observed | Source                                |
|--------------|-------------|------------|---------------------------------------|
| MMLU         | 5452 × 57   | 100.0      | per-subject MMLU leaderboard          |
| MTEB         | 263 × 56    | 77.3       | MTEB embedding leaderboard            |
| Merged       | 118 × 114   | 31.1       | 9 leaderboards, canonicalized models  |
| BenchPress   | 83 × 49     | 33.8       | BenchPress release matrix             |

See `data/README.md` for the file format and per-leaderboard sources.

## Reproducing all figures from a clean state

```bash
# regenerate every figure shown in the paper (~1 hour total on CPU)
python code/plot_eigenspectrum.py
python code/eval_greedy_all.py
python code/eval_entropy_vs_mi.py
python code/plot_selection_order.py
python code/plot_mi_selection_all.py
python code/run_normality.py
python code/eval_benchpress.py
python code/eval_logit.py
# TabImpute takes about 2.5--3h on a 12 GB GPU
python code/eval_tabimpute.py
```

Outputs land in `figures/` (and `figures_logit/` for the logit experiments).

## License

Released under Apache 2.0.  Underlying benchmark data inherits the licenses of
the source leaderboards — see `data/README.md` for citations.
