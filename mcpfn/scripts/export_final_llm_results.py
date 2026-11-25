"""Verify a complete final-model LLM evaluation and prepare paper figures and prose."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from recsys_evaluation_common import sha256, write_json


DATASETS = [('mteb', 'MTEB'), ('mmlu', 'MMLU'), ('merged', 'Merged'), ('benchpress', 'BenchPress')]
KEYS = ['dataset', 'selection', 'budget', 'fold', 'method']


def validate_table(table, baseline):
    expected = {(dataset, selection, budget, fold, method) for dataset, _ in DATASETS
                for selection in ('mi', 'entropy', 'random') for budget in range(1, 16)
                for fold in range(10) for method in ('matrixformer', 'kernel_ridge')}
    if table.duplicated(KEYS).any() or set(table[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError('Missing or duplicate final LLM settings')
    if not np.isfinite(table[['rmse', 'relative_rmse', 'n_eval']]).all().all() or (table.n_eval <= 0).any():
        raise ValueError('Nonfinite or empty LLM scores')
    counts = table.groupby(KEYS[:-1]).n_eval.nunique()
    if not counts.eq(1).all():
        raise ValueError('Unequal query support across methods')
    current = table[table.method == 'kernel_ridge']
    original = baseline[baseline.method == 'kernel_ridge']
    joined = current.merge(original, on=KEYS, validate='one_to_one')
    if len(joined) != len(current) or len(joined) != len(original):
        raise ValueError('Kernel-ridge baseline coverage changed')
    for field in ('rmse', 'relative_rmse'):
        np.testing.assert_allclose(joined[field+'_x'], joined[field+'_y'], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(joined.n_eval_x, joined.n_eval_y)


def summary(table):
    bootstrap_indices = np.random.default_rng(0).integers(0, 10, (10000, 10))
    rows = []
    for (dataset, selection, budget, method), group in table.groupby(KEYS[:3] + ['method']):
        ordered = group.sort_values('fold')
        if ordered.fold.tolist() != list(range(10)):
            raise ValueError('Uncertainty requires all ten folds')
        values = ordered.rmse.to_numpy()
        lower, upper = np.percentile(values[bootstrap_indices].mean(axis=1), [2.5, 97.5])
        rows.append(dict(dataset=dataset, selection=selection, budget=budget, method=method,
                         rmse=values.mean(), ci_lower=lower, ci_upper=upper, folds=10))
    return pd.DataFrame(rows)


def matched_mi_subset(subset, table):
    subset = subset.assign(selection='mi')
    expected = {(dataset, 'mi', budget, fold, method) for dataset in ('merged', 'benchpress')
                for budget in (5, 10, 15) for fold in range(10)
                for method in ('matrixformer', 'kernel_ridge', 'logit_ridge')}
    if subset.duplicated(KEYS).any() or set(subset[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError('Unexpected greedy-MI subset coverage')
    checked = subset.query("method == 'kernel_ridge'").merge(table, on=KEYS, validate='one_to_one')
    if len(checked) != 60:
        raise ValueError('Unexpected greedy-MI baseline subset')
    for field in ('rmse', 'relative_rmse'):
        np.testing.assert_allclose(checked[field+'_x'], checked[field+'_y'], atol=1e-9, rtol=0)
    selected_model = subset[subset.method == 'matrixformer'][KEYS].merge(table, on=KEYS, validate='one_to_one')
    if len(selected_model) != 60:
        raise ValueError('Incomplete final-model subset')
    return pd.concat([subset[subset.method != 'matrixformer'], selected_model[subset.columns]], ignore_index=True)


def results_text(panels, logit):
    wide = panels.pivot(index=['dataset', 'selection', 'budget'], columns='method', values='rmse')
    scores = wide.xs(('mi', 15), level=('selection', 'budget'))
    comparisons = [f"${scores.loc[key, 'matrixformer']:.3f}$ versus ${scores.loc[key, 'kernel_ridge']:.3f}$ on {label}"
                   for key, label in DATASETS]
    wins = (wide.matrixformer < wide.kernel_ridge).groupby('dataset').sum()
    text = '\\paragraph{Results.}\n'
    text += ('\\Cref{fig:llm-rmse-mi} shows standardized RMSE under greedy MI selection. '
             'At $k=15$, MatrixFormer and kernel ridge achieve ' + ', '.join(comparisons[:-1])
             + ', and ' + comparisons[-1] + ', respectively. '
             '\\Cref{fig:llm-rmse-entropy-random} reports greedy entropy and random selection. '
             f'Across the 180 combinations of dataset, selection rule and budget, MatrixFormer has lower '
             f'fold-mean RMSE in {int(wins.sum())}: ' + ', '.join(f'{int(wins[key])} of 45 on {label}' for key, label in DATASETS)
             + '. These counts summarize correlated settings, not independent tests.\n\n')
    selected = logit.query("method == 'logit_ridge' and budget == 15").groupby('dataset').rmse.mean()
    text += ('The selection release also provides a logit-space variant of kernel ridge. '
             'Under the same greedy-MI tasks at $k=15$, its RMSE is '
             f"${selected['merged']:.3f}$ on Merged and ${selected['benchpress']:.3f}$ on BenchPress, "
             f"compared with ${scores.loc['merged', 'matrixformer']:.3f}$ and "
             f"${scores.loc['benchpress', 'matrixformer']:.3f}$ for MatrixFormer.\n\n")
    return text


def export(args):
    audit = json.loads((args.evaluation / 'audit.json').read_text())
    if not audit['complete'] or audit['settings'] != 1800 or not audit['all_predictions_rescored']:
        raise ValueError('Require a complete, rescored evaluation')
    if sha256(args.evaluation / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Evaluation manifest changed')
    if sha256(args.evaluation / 'per_fold.csv') != audit['per_fold_sha256']:
        raise ValueError('Audited fold scores changed')
    manifest = json.loads((args.evaluation / 'manifest.json').read_text())
    table = pd.read_csv(args.evaluation / 'per_fold.csv')
    baseline_path = args.paper / 'results/llm_rmse_all_folds.csv'
    validate_table(table, pd.read_csv(baseline_path))
    logit_path = args.paper / 'results/llm_rmse_mi_folds.csv'
    logit = matched_mi_subset(pd.read_csv(logit_path), table)
    panels = summary(table)
    data = args.paper / 'results/final_model'
    data.mkdir(parents=True, exist_ok=True)
    table.to_csv(data / 'llm_per_fold.csv', index=False)
    panels.to_csv(data / 'llm_summary.csv', index=False)
    logit.to_csv(data / 'llm_mi_folds.csv', index=False)
    (args.evaluation / 'paper_results.tex').write_text(results_text(panels, logit))
    write_json(data / 'llm_validation.json', dict(checkpoint_sha256=manifest['checkpoint_sha256'],
                config_sha256=manifest['config_sha256'], evaluation_manifest_sha256=audit['manifest_sha256'],
                frozen_index_sha256=manifest['frozen_index_sha256'], settings=1800, panels=180,
                all_predictions_rescored=True, unchanged_kernel_ridge_verified=True,
                baseline_csv_sha256=sha256(baseline_path), logit_csv_sha256=sha256(logit_path)))
    subprocess.run([sys.executable, str(args.paper / 'scripts/plot_llm_rmse_curves.py')], check=True)
    print(panels.query("selection == 'mi' and budget == 15").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    export(parser.parse_args())


if __name__ == '__main__':
    main()
