# MissBench: the TabImpute benchmark

This folder reproduces the experiments in the TabImpute paper. There are two command-line tools:

- `run_benchmark.py` downloads the datasets, samples missingness masks, runs the imputation methods, and scores them. Results go to `results/<experiment>.csv`.
- `make_figures.py` turns those results into every figure and table in the paper. Output goes to `figures/`.

Run both from this folder (`mcpfn/benchmark`).

## Setup

```bash
pip install -e "..[benchmark]"
```

That covers TabImpute and the CPU baselines in the `quick` method group. For every baseline in the paper, install the extra packages and the git submodules that hold ReMasker, CACTI and DiffPuter:

```bash
pip install -e "..[benchmark-all]"
```

```bash
git submodule update --init remasker CACTI DiffPuter
```

TabImpute downloads its default checkpoint from HuggingFace (`Tabimpute/TabImpute`) on first use. A GPU is used automatically when one is available (`--device` overrides it). TabImpute on CPU is slow: a few minutes per table.

## Quick start

```bash
python run_benchmark.py list
```

```bash
python run_benchmark.py run --methods quick --datasets SMSA slump FacultySalaries
```

```bash
python make_figures.py all
```

`run` performs all four steps (`download`, `missingness`, `impute`, `evaluate`). Each step can also be run on its own with the same options. Steps skip work that already exists, so an interrupted run can simply be restarted. Pass `--overwrite` to redo imputations.

## Reproducing the paper

### 1. Main benchmark (Tables 1, 3, 4 and Figs. 1, 4, 5)

41 OpenML datasets × 11 missingness patterns at p = 0.4, with all 15 methods of the paper:

```bash
python run_benchmark.py run --experiment openml --methods paper
```

For the CPU timing bar in Fig. 4b, also run TabImpute on CPU for the MCAR tables:

```bash
python run_benchmark.py impute --methods tabimpute_cpu --patterns mcar --device cpu
```

For Fig. 6, run not-MIWAE with a Self-MNAR missingness model:

```bash
python run_benchmark.py impute --methods notmiwae_self_mnar
```

### 2. Pattern-specific TabImpute models (Tables 2, 5, 7 and the purple bars in Figs. 1, 5, 6)

These checkpoints are not on HuggingFace. Train them with the scripts in `mcpfn/scripts`, then pass one `--checkpoint PATTERN=PATH` per model:

```bash
python run_benchmark.py impute --checkpoint self_mnar=ckpts/self_mnar.ckpt censor_mnar=ckpts/censor_mnar.ckpt
```

Pattern keys are listed by `python run_benchmark.py list`. Table 7 compares against a model trained on a nonlinear factor model: `--checkpoint tabimpute_nonlinear=PATH`.

After adding methods, re-run `evaluate` to refresh the results CSV:

```bash
python run_benchmark.py evaluate --experiment openml
```

### 3. Missingness levels (Fig. 7)

```bash
python run_benchmark.py run --experiment openml_mcar_sweep --methods tabimpute hyperimpute missforest gain
```

### 4. Figures and tables

```bash
python make_figures.py list
```

```bash
python make_figures.py paper
```

| Item | Paper | Needs |
|---|---|---|
| `fig1` | Fig. 1: RMSE overall, MCAR, Censor-MNAR, Self-MNAR | `openml` |
| `fig3` | Fig. 3 (right): example missingness masks | nothing |
| `fig4a` | Fig. 4a: TabImpute vs. MissForest scatter | `openml` |
| `fig4b` | Fig. 4b: runtime per entry | `openml` |
| `fig5` | Fig. 5: RMSE for all patterns | `openml` |
| `fig6` | Fig. 6: Self-MNAR with not-MIWAE (Self-MNAR) | `openml` |
| `fig7` | Fig. 7: RMSE vs. fraction missing | `openml_mcar_sweep` |
| `table1`, `table3` | RMSE ± 95% CI (top 5 / all methods) | `openml` |
| `table2`, `table5` | Specialized TabImpute models | `openml` + checkpoints |
| `table4` | Column-wise R² ± 95% CI | `openml` |
| `table7` | Linear vs. nonlinear factor model | `openml` + checkpoint |
| `table8` | Missingness pattern parameters | nothing |
| `table9` | OpenML datasets | nothing |

Tables are written as LaTeX and use the pattern macros in `figures/macros.tex` together with the `booktabs`, `multirow` and `arydshln` packages.

Extra items that are not in the paper: `uci`, `uci_table`, `runtime`, `r2_bars`, `wasserstein_bars`, `mae_bars`, `win_rate`, `win_rate_table`, `critical_difference`. `--methods` restricts which baselines are shown, and `--compare A B` changes the pair in `fig4a`.

## Other experiments

UCI datasets from the HyperImpute paper, MCAR at p = 0.1 to 0.5 with 10 masks each:

```bash
python run_benchmark.py run --experiment uci --methods quick
```

Runtime scaling of TabImpute and TabImpute V2 on random 10-column matrices with 10 to 1000 rows:

```bash
python run_benchmark.py runtime
```

## How it works

- `missbench/datasets.py`: the hard-coded OpenML dataset IDs (paper Table 9) and the UCI datasets.
- `missbench/patterns.py`: the 11 missingness patterns and their parameters (paper Table 8). Masks are seeded per dataset, pattern, level and repeat, so re-running gives the same masks. `--seed` changes them. After masking, each column is standardized with the mean and standard deviation of its observed entries, as in the paper.
- `missbench/methods.py`: every imputation method with the settings used in the paper. To add a method, add a `Method` entry to `METHODS`.
- `missbench/experiment.py`: missingness, imputation and scoring.
- `missbench/metrics.py`: the reported RMSE is the column-wise RMSE normalized by the standard deviation of the true missing values in each column, averaged over columns.
- `missbench/plots/`: figures and tables.

Data and imputations are stored under `datasets/` (change it with `--data-dir`):

```
datasets/openml/<dataset>/dataset.pkl                    complete data
datasets/openml/<dataset>/<PATTERN>_<p>/missing.npy      standardized data with missing values
datasets/openml/<dataset>/<PATTERN>_<p>/true.npy         standardized ground truth
datasets/openml/<dataset>/<PATTERN>_<p>/<method>.npy     imputation
datasets/openml/<dataset>/<PATTERN>_<p>/<method>.json    runtime and device
```

This is the same layout the paper results were produced with. `evaluate` also recognizes the file names those earlier runs used (for example `masters_mcar.npy` for TabImpute), so existing result folders can be scored and plotted without recomputing.

The `categorical/` folder holds the older, unmaintained scripts for the categorical OpenML datasets. They are not part of the paper or of these tools.
