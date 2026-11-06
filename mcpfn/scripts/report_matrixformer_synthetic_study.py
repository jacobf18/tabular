"""Audit diagnostics and plot frozen synthetic experiments in the paper style."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import seaborn as sns

from recsys_evaluation_common import sha256, write_json
import synthetic_matrixformer_study as study


CONDITION = ['family', 'rank', 'noise', 'observed_fraction', 'rows', 'columns']


def summarize(frame, groups, metrics):
    rows = []
    for keys, selected in frame.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,)))
        row['matrices'] = len(selected)
        for metric in metrics:
            values = selected[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(values))
            half = float(student_t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
            row.update({metric: mean, f'{metric}_ci95_half': half})
        rows.append(row)
    return pd.DataFrame(rows)


def verify_diagnostics(output):
    specification = json.loads((output / 'protocol.json').read_text())
    records = 0
    for case in specification['cases']:
        if not study.selected_attention_case(case):
            continue
        path = output / 'cases' / case['case_id']
        data = study.generate(case)
        original = pd.read_csv(path / 'attention.csv')
        recalculated = []
        with np.load(path / 'attention.npz') as attention:
            targets = attention['targets']
            for layer in range(8):
                for axis in ('column', 'row'):
                    recalculated += [dict(layer=layer, axis=axis, **row) for row in
                                     study.attention_statistics(attention[f'{axis}_{layer}'], targets, data, axis)]
        keys = ['layer', 'axis', 'query', 'head']
        expected = pd.DataFrame(recalculated).sort_values(keys).reset_index(drop=True)
        expected = expected.astype({name: float for name in expected.columns if name not in keys})
        actual = original.sort_values(keys).reset_index(drop=True)[expected.columns]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        removal = pd.read_csv(path / 'interventions.csv')
        with np.load(path / 'predictions.npz') as predictions:
            baseline = predictions['matrixformer']
        for row in removal.itertuples(index=False):
            donors = json.loads(row.donor_indices)
            indices = (donors, row.column) if row.axis == 'row' else (row.row, donors)
            assert data['observed'][indices].all() and not data['observed'][row.row, row.column]
            assert len(donors) == len(set(donors)) == row.removed
            np.testing.assert_allclose(row.original_prediction, baseline[row.row, row.column], rtol=1e-12)
            np.testing.assert_allclose(row.truth, data['truth'][row.row, row.column], rtol=1e-12)
            recalculated_error = (row.prediction - row.truth) ** 2 - (row.original_prediction - row.truth) ** 2
            np.testing.assert_allclose(row.squared_error_change, recalculated_error, rtol=1e-10, atol=1e-12)
        records += len(removal)
    return records


def score_plots(summary, figures, style):
    low_rank = summary[summary.family == 'low_rank']
    model = low_rank[low_rank.method == 'matrixformer'].pivot(index='rank', columns='observed_fraction', values='rmse')
    figure, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.05))
    sns.heatmap(model, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1.2, cbar=False,
                ax=axes[0], annot_kws={'fontsize': style.LABEL_SIZE})
    axes[0].set_title('MatrixFormer RMSE')
    for axis, method, title in zip(axes[1:], ('two_way_knn', 'uncentered_svd'), ('vs. two-way K-NN', 'vs. uncentered SVD')):
        baseline = low_rank[low_rank.method == method].pivot(index='rank', columns='observed_fraction', values='rmse')
        relative = np.log10(model / baseline)
        sns.heatmap(relative, annot=True, fmt='.1f', cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                    cbar=False, ax=axis, annot_kws={'fontsize': style.LABEL_SIZE})
        axis.set_title(title + '\nlog10(RMSE ratio)')
    for axis in axes:
        axis.set_xlabel('Observed fraction')
        axis.set_ylabel('Generating rank' if axis is axes[0] else '')
        axis.tick_params(length=0)
    style.layout(figure)
    style.save(figure, figures / 'synthetic_phase')
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.55), sharey=True)
    selected = summary[(summary.family == 'clusters') & (summary.observed_fraction == .4)]
    order = selected[selected.noise == .25].sort_values('rmse').method.tolist()
    colors = [style.BLUE if method == 'matrixformer' else '#cf9d46' if method.startswith('oracle') else style.NEUTRAL for method in order]
    for axis, noise in zip(axes, (0., .25, 1.)):
        values = selected[selected.noise == noise].set_index('method').loc[order]
        positions = np.arange(len(order))
        axis.barh(positions, values.rmse, color=colors, height=.7)
        axis.errorbar(values.rmse, positions, xerr=values.rmse_ci95_half, fmt='none', color=style.ERROR, elinewidth=.7, capsize=1.5)
        axis.set_yticks(positions, [study.METHOD_NAMES[method] for method in order])
        axis.invert_yaxis()
        axis.set_title(f'Noise SD = {noise:g}')
        axis.set_xlabel('RMSE')
        style.style_axes(axis)
        for label, method in zip(axis.get_yticklabels(), order):
            if method == 'matrixformer':
                label.set_color(style.BLUE)
                label.set_fontweight('bold')
    style.layout(figure)
    style.save(figure, figures / 'synthetic_neighbors')
    plt.close(figure)

    noise_summary = summary[summary.family == 'noise']
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.4), sharey=True)
    order = noise_summary[noise_summary.observed_fraction == .4].sort_values('rmse').method.tolist()
    for axis, rate in zip(axes, (.2, .8)):
        values = noise_summary[noise_summary.observed_fraction == rate].set_index('method').loc[order]
        positions = np.arange(len(order))
        axis.barh(positions, values.rmse, color=[style.BLUE if method == 'matrixformer' else style.NEUTRAL for method in order], height=.7)
        axis.errorbar(values.rmse, positions, xerr=values.rmse_ci95_half, fmt='none', color=style.ERROR, elinewidth=.7, capsize=1.5)
        axis.set_yticks(positions, [study.METHOD_NAMES[method] for method in order])
        axis.invert_yaxis()
        axis.axvline(1, color='#777777', linestyle='--', linewidth=.7)
        axis.set_title(f'{rate:.0%} observed; independent noise')
        axis.set_xlabel('RMSE')
        style.style_axes(axis)
    style.layout(figure)
    style.save(figure, figures / 'synthetic_null')
    plt.close(figure)


def attention_plots(output, figures, attention_summary, intervention_summary, style):
    specification = json.loads((output / 'protocol.json').read_text())
    example = next(case for case in specification['cases'] if case['family'] == 'clusters'
                   and case['noise'] == .25 and case['observed_fraction'] == .4 and case['seed'] == 0)
    path = output / 'cases' / example['case_id']
    data = study.generate(example)
    figure, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 3.8))
    with np.load(path / 'attention.npz') as captured:
        for row, direction in enumerate(('row', 'column')):
            labels = data[f'{direction}_labels']
            order = np.argsort(labels, kind='stable')
            neighborhood = (labels[:, None] == labels[None, :])[np.ix_(order, order)]
            sns.heatmap(neighborhood, ax=axes[row, 0], cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
            matrix = np.mean([captured[f'map_{direction}_{layer}'] for layer in range(8)], axis=0)
            special = len(matrix) - len(labels)
            matrix = matrix[special:, special:][np.ix_(order, order)]
            sns.heatmap(matrix, ax=axes[row, 1], cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
            axes[row, 0].set_title(f'True {direction} groups')
            axes[row, 1].set_title(f'{direction.capitalize()} attention\n(all heads and layers)')
            axes[row, 0].set_ylabel('Query')
            axes[row, 0].set_xlabel('Key (sorted for display only)')
            axes[row, 1].set_xlabel('Key (special tokens omitted)')
    selected = attention_summary[(attention_summary.family == 'clusters') & (attention_summary.noise == .25)]
    axis = axes[0, 2]
    for direction, color, marker in (('row', style.BLUE, 'o'), ('column', '#777777', 's')):
        subset = selected[selected.axis == direction].sort_values('layer')
        axis.errorbar(subset.layer + 1, subset.group_excess, yerr=subset.group_excess_ci95_half,
                      color=color, marker=marker, markersize=2.4, linewidth=.9, capsize=1.5, label=direction.capitalize())
    axis.axhline(0, color='#999999', linestyle='--', linewidth=.6)
    axis.set_xlabel('Layer')
    axis.set_ylabel('Same-group mass minus chance')
    axis.set_title('Neighborhood alignment')
    axis.legend(frameon=False)
    style.style_axes(axis, grid='y')
    axis = axes[1, 2]
    selected = intervention_summary[(intervention_summary.family == 'clusters') & (intervention_summary.noise == .25)]
    order = ('attention', 'random', 'oracle')
    selected = selected.set_index('arm').loc[list(order)]
    axis.bar(np.arange(3), selected.squared_error_change, color=[style.BLUE, style.NEUTRAL, '#cf9d46'], width=.65)
    axis.errorbar(np.arange(3), selected.squared_error_change, yerr=selected.squared_error_change_ci95_half,
                  fmt='none', color=style.ERROR, elinewidth=.7, capsize=1.5)
    axis.set_xticks(np.arange(3), ['Attention', 'Random', 'Oracle'], rotation=30, ha='right')
    axis.axhline(0, color='#999999', linewidth=.6)
    axis.set_ylabel('Change in query squared error')
    axis.set_title('Remove three observed donors')
    style.style_axes(axis, grid='y')
    style.layout(figure)
    style.save(figure, figures / 'synthetic_attention')
    plt.close(figure)
    shutil.copyfile(path / 'attention.npz', output / 'report/example_attention.npz')
    atomic_example = {name: data[name] for name in ('context', 'row_labels', 'column_labels')}
    from recsys_evaluation_common import atomic_npz
    atomic_npz(output / 'report/example_matrix.npz', **atomic_example)


def report(output, paper):
    sys.path.insert(0, str(paper / 'scripts'))
    import figure_style as style
    style.apply()
    audited = json.loads((output / 'audit.json').read_text())
    specification = json.loads((output / 'protocol.json').read_text())
    if specification['profile'] != 'main' or audited['cases'] != len(specification['cases']):
        raise ValueError('Paper report requires the complete main profile')
    removal_count = verify_diagnostics(output)
    destination = output / 'report'
    destination.mkdir(exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    scores = pd.read_csv(output / 'scores.csv')
    from synthetic_uncentered_svd_control import audit as audit_svd
    controls = pd.DataFrame(audit_svd(output))
    study.METHOD_NAMES['uncentered_svd'] = 'SVD (uncentered)'
    scores = pd.concat([scores, controls[scores.columns]], ignore_index=True)
    if scores.duplicated(['case_id', 'method']).any():
        raise ValueError('Duplicate scores')
    summary = summarize(scores, CONDITION + ['method'], ['rmse', 'mae', 'signal_rmse'])
    if not (summary.matrices == 10).all():
        raise ValueError('Incomplete seed inventory')
    summary.to_csv(destination / 'scores_summary.csv', index=False)
    matrixformer = scores[scores.method == 'matrixformer'][['case_id', 'rmse']].rename(columns={'rmse': 'matrixformer_rmse'})
    paired = scores.merge(matrixformer, on='case_id', validate='many_to_one')
    paired['rmse_difference'] = paired.rmse - paired.matrixformer_rmse
    summarize(paired, CONDITION + ['method'], ['rmse_difference']).to_csv(destination / 'paired_differences.csv', index=False)
    scores.to_csv(destination / 'scores_by_matrix.csv', index=False)
    attention = pd.read_csv(output / 'attention.csv')
    metrics = ['donor_mass', 'special_mass', 'self_mass', 'missing_key_mass', 'group_mass', 'group_chance', 'group_excess', 'top5_group_fraction']
    matrix_attention = attention.groupby(CONDITION + ['case_id', 'seed', 'layer', 'axis'], as_index=False)[metrics].mean()
    matrix_attention.to_csv(destination / 'attention_by_matrix.csv', index=False)
    attention_summary = summarize(matrix_attention[matrix_attention.family == 'clusters'], CONDITION + ['layer', 'axis'], metrics)
    attention_summary.to_csv(destination / 'attention_summary.csv', index=False)
    removal = pd.read_csv(output / 'interventions.csv')
    query_keys = CONDITION + ['case_id', 'seed', 'query', 'axis', 'arm']
    query_removal = removal.groupby(query_keys, as_index=False)[['squared_error_change', 'prediction_shift']].mean()
    matrix_removal = query_removal.groupby(CONDITION + ['case_id', 'seed', 'arm'], as_index=False)[['squared_error_change', 'prediction_shift']].mean()
    matrix_removal.to_csv(destination / 'interventions_by_matrix.csv', index=False)
    intervention_summary = summarize(matrix_removal, CONDITION + ['arm'], ['squared_error_change', 'prediction_shift'])
    intervention_summary.to_csv(destination / 'interventions_summary.csv', index=False)
    paired_values = query_removal.pivot(index=CONDITION + ['case_id', 'seed', 'query', 'axis'], columns='arm',
                                       values=['squared_error_change', 'prediction_shift'])
    paired_removal = pd.DataFrame(index=paired_values.index)
    paired_removal['attention_minus_random'] = paired_values['squared_error_change']['attention'] - paired_values['squared_error_change']['random']
    paired_removal['shift_attention_minus_random'] = paired_values['prediction_shift']['attention'] - paired_values['prediction_shift']['random']
    difference_metrics = ['attention_minus_random', 'shift_attention_minus_random']
    paired_removal = paired_removal.reset_index().groupby(CONDITION + ['case_id', 'seed'], as_index=False)[difference_metrics].mean()
    summarize(paired_removal, CONDITION, difference_metrics).to_csv(destination / 'paired_interventions.csv', index=False)
    score_plots(summary, figures, style)
    attention_plots(output, figures, attention_summary, intervention_summary, style)
    for name in ('protocol.json', 'manifest.json', 'audit.json'):
        shutil.copyfile(output / name, destination / name)
    shutil.copyfile(output / 'uncentered_svd/manifest.json', destination / 'uncentered_svd_manifest.json')
    write_json(destination / 'report_audit.json', dict(intervention_rows_checked=removal_count,
               attention_recomputed_from_weights=True, scoring_audit=audited,
               uncentered_svd_cases_rescored=len(controls),
               interval_unit='Ten independent matrices per condition; average heads, queries and donor resamples within each matrix first',
               intervals='Two-sided 95% Student-t intervals, no multiple-testing adjustment',
               paired_difference_sign='Positive favors MatrixFormer for errors; positive indicates greater harm from attention-selected removal for interventions',
               reporting_policy='All registered conditions retained, no checkpoint or favorable-head selection',
               source_sha256=sha256(Path(__file__)), style_sha256=sha256(paper / 'scripts/figure_style.py'),
               inputs={name: sha256(output / name) for name in ('scores.csv', 'attention.csv', 'interventions.csv')},
               figures={path.name: sha256(path) for path in figures.iterdir() if path.is_file()}))
    print(summary[(summary.observed_fraction == .4) & ((summary.family != 'low_rank') | summary['rank'].isin([2, 16, 32]))]
          [['family', 'rank', 'noise', 'method', 'rmse', 'rmse_ci95_half']].to_string(index=False))
    print('\nPaired removal effects:')
    print(pd.read_csv(destination / 'paired_interventions.csv').to_string(index=False))
    print(f'Figures and audited portable report: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('/root/MatrixFormerPaper'))
    arguments = parser.parse_args()
    report(arguments.output, arguments.paper)
