"""One takeaway per synthetic paper figure; retain full diagnostic grids separately."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

from recsys_evaluation_common import sha256, write_json
from report_matrixformer_synthetic_study import summarize
from report_synthetic_cluster_representations import COLORS, attention_image, load_example


def finish(figure, output, style, legend=False):
    if legend:
        handles, labels = figure.axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False, fontsize=7)
    figure.tight_layout(pad=.5, rect=(0, .1 if legend else 0, 1, 1))
    style.save(figure, output)
    plt.close(figure)


SOLVERS = ('matrixformer', 'softimpute', 'uncentered_svd', 'two_way_knn')
DISPLAY_LAYERS = (1, 4, 8)
NEIGHBOR_NAMES = {'matrixformer': 'MatrixFormer', 'two_way_knn': 'Two-way K-NN',
                  'oracle_neighbors': 'Oracle neighbors', 'oracle_block': 'Oracle block mean'}
NEIGHBOR_RATES = (.2, .4)


def solver_comparisons(scores):
    selected = scores[scores.family.isin(['low_rank', 'clusters']) & scores.method.isin(SOLVERS)].copy()
    keys = ['family', 'rank', 'noise', 'observed_fraction', 'method']
    expected = {('low_rank', rank, 0., rate, method) for rank in (2, 4, 8, 16, 32)
                for rate in (.2, .4, .6, .8) for method in SOLVERS}
    expected |= {('clusters', 4, noise, rate, method) for noise in (0., .25, 1.)
                 for rate in (.2, .4, .6, .8) for method in SOLVERS}
    if selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('The comparison must retain every registered rank/rate/noise condition')
    if not (selected.matrices == 10).all() or not np.isfinite(selected[['rmse', 'rmse_ci95_half']]).all().all():
        raise ValueError('Each comparison requires ten finite matrix results')
    return selected


def solver_curves(axis, selected, horizontal, style):
    methods = [('matrixformer', 'MatrixFormer', style.BLUE, 'o', '-'),
               ('softimpute', 'SoftImpute', '#555555', 's', '--'),
               ('uncentered_svd', 'Uncentered SVD', '#999999', '^', ':'),
               ('two_way_knn', 'Two-way K-NN', '#777777', 'x', '-.')]
    for method, name, color, marker, linestyle in methods:
        data = selected[selected.method == method].sort_values(horizontal)
        values = data[horizontal] * (100 if horizontal == 'observed_fraction' else 1)
        axis.errorbar(values, data.rmse, yerr=data.rmse_ci95_half, color=color, label=name,
                      marker=marker, markersize=3, linewidth=1.05, linestyle=linestyle,
                      capsize=1.7, elinewidth=.65)
    if horizontal == 'rank':
        axis.set_xscale('log', base=2)
        axis.set_xticks([2, 4, 8, 16, 32], ['2', '4', '8', '16', '32'])
        axis.set_xlabel('Matrix rank')
    else:
        axis.set_xticks([20, 40, 60, 80])
        axis.set_xlabel('Observed entries (%)')
    style.style_axes(axis, grid='y')


def solver_tradeoffs(scores, output, style):
    scores = solver_comparisons(scores)
    scores.to_csv(output / 'solver_sweep_summary.csv', index=False)
    gaussian = scores[(scores.family == 'low_rank') & (scores.observed_fraction == .4)]
    clusters = scores[(scores.family == 'clusters') & (scores.noise == .25)]
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.8), sharey=True)
    solver_curves(axes[0], gaussian, 'rank', style)
    solver_curves(axes[1], clusters, 'observed_fraction', style)
    axes[0].set_title('Exact Gaussian factors\n40% observed')
    axes[1].set_title('Noisy clustered matrices\nNoise SD 0.25')
    axes[0].set_ylabel('Missing-entry RMSE (lower is better)')
    upper = max((gaussian.rmse + gaussian.rmse_ci95_half).max(),
                (clusters.rmse + clusters.rmse_ci95_half).max())
    axes[0].set_ylim(-.025, upper * 1.04)
    finish(figure, output / 'synthetic_phase', style, legend=True)
    figure, axes = plt.subplots(2, 2, figsize=(style.WIDTH, 4.3), sharex=True, sharey=True)
    for axis, rate in zip(axes.flat, (.2, .4, .6, .8)):
        solver_curves(axis, scores[(scores.family == 'low_rank') & (scores.observed_fraction == rate)], 'rank', style)
        axis.set_title(f'{rate:.0%} observed')
        axis.set_ylim(bottom=-.025)
    for axis in axes[:, 0]:
        axis.set_ylabel('Missing-entry RMSE')
    finish(figure, output / 'synthetic_low_rank_sweep', style, legend=True)
    figure, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.8), sharex=True, sharey=True)
    for axis, noise in zip(axes, (0., .25, 1.)):
        solver_curves(axis, scores[(scores.family == 'clusters') & (scores.noise == noise)], 'observed_fraction', style)
        axis.set_title(f'Noise SD {noise:g}')
        axis.set_ylim(bottom=-.035)
    axes[0].set_ylabel('Missing-entry RMSE')
    finish(figure, output / 'synthetic_cluster_sweep', style, legend=True)


def simple_bars(data, names, title, output, style, reference=False):
    selected = data[data.method.isin(names)].sort_values('rmse').reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(style.WIDTH, 1.75 if reference else 2.25))
    colors = [style.BLUE if method == 'matrixformer' else '#cda046' if method.startswith('oracle')
              else style.NEUTRAL for method in selected.method]
    positions = np.arange(len(selected))
    axis.barh(positions, selected.rmse, color=colors, height=.58)
    axis.errorbar(selected.rmse, positions, xerr=selected.rmse_ci95_half, fmt='none',
                  color=style.ERROR, elinewidth=.8, capsize=2)
    for position, record in selected.iterrows():
        axis.text(record.rmse + record.rmse_ci95_half + .02 * selected.rmse.max(), position,
                  f'{record.rmse:.3f}', va='center', fontsize=7)
    axis.set_yticks(positions, [names[method] for method in selected.method])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.2 * (selected.rmse + selected.rmse_ci95_half).max())
    axis.set_xlabel('Missing-entry RMSE (lower is better)')
    axis.set_title(title)
    if reference:
        axis.axvline(1, color='#555555', linewidth=.8, linestyle='--', label='Known-zero-mean population RMSE')
    style.style_axes(axis)
    finish(figure, output, style, legend=reference)


def neighbor_comparisons(scores):
    selected = scores[(scores.family == 'clusters') & (scores.noise == .25)
                      & scores.observed_fraction.isin(NEIGHBOR_RATES)
                      & scores.method.isin(NEIGHBOR_NAMES)].copy()
    keys = ['observed_fraction', 'method']
    expected = {(rate, method) for rate in NEIGHBOR_RATES for method in NEIGHBOR_NAMES}
    if selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('The neighbor comparison requires every method at both observation rates')
    values = selected[['rmse', 'rmse_ci95_half']]
    if not (selected.matrices == 10).all() or not np.isfinite(values).all().all() or (values < 0).any().any():
        raise ValueError('The neighbor comparison requires ten finite matrix scores and valid intervals')
    return selected


def neighbor_bars(scores, output, style):
    selected = neighbor_comparisons(scores)
    selected.to_csv(output / 'neighbor_comparison_summary.csv', index=False)
    order = selected.groupby('method').rmse.mean().sort_values(kind='stable').index
    colors = [style.BLUE if method == 'matrixformer' else '#cda046' if method.startswith('oracle')
              else style.NEUTRAL for method in order]
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.15), sharex=True, sharey=True)
    positions = np.arange(len(order))
    upper = (selected.rmse + selected.rmse_ci95_half).max()
    for axis, rate in zip(axes, NEIGHBOR_RATES):
        data = selected[selected.observed_fraction == rate].set_index('method').loc[order]
        axis.barh(positions, data.rmse, color=colors, height=.58)
        axis.errorbar(data.rmse, positions, xerr=data.rmse_ci95_half, fmt='none',
                      color=style.ERROR, elinewidth=.8, capsize=2)
        for position, record in enumerate(data.itertuples()):
            axis.text(record.rmse + record.rmse_ci95_half + .02 * upper, position,
                      f'{record.rmse:.3f}', va='center', fontsize=7)
        axis.set_yticks(positions, [NEIGHBOR_NAMES[method] for method in order])
        axis.set_xlim(0, 1.25 * upper)
        axis.set_xticks([0, .2, .4, .6])
        axis.set_title(f'{rate:.0%} observed')
        style.style_axes(axis)
    axes[0].invert_yaxis()
    figure.supxlabel('Missing-entry RMSE (lower is better)', fontsize=style.FONT_SIZE, y=.025)
    figure.tight_layout(pad=.6, w_pad=.8, rect=(0, .08, 1, 1))
    style.save(figure, output / 'synthetic_neighbors')
    plt.close(figure)


def attention_alignment(parent, output, style):
    frame = pd.read_csv(parent / 'report/attention_summary.csv')
    selected = frame[(frame.family == 'clusters') & (frame.noise == .25) & (frame.observed_fraction == .4)]
    figure, axis = plt.subplots(figsize=(style.WIDTH, 2.7))
    for direction, color, marker in [('row', style.BLUE, 'o'), ('column', '#777777', 's')]:
        data = selected[selected.axis == direction].sort_values('layer')
        axis.errorbar(data.layer + 1, 100 * data.group_excess, yerr=100 * data.group_excess_ci95_half,
                      color=color, label=f'{direction.capitalize()} attention', marker=marker,
                      markersize=3, linewidth=1, capsize=2, elinewidth=.7)
    axis.axhline(0, color='#aaaaaa', linestyle='--', linewidth=.7)
    axis.set_xlabel('Transformer layer')
    axis.set_xticks(range(1, 9))
    axis.set_ylabel('Same-group attention above chance (pp)')
    axis.set_title('Later attention favors same-group donors')
    style.style_axes(axis, grid='y')
    finish(figure, output / 'synthetic_attention', style, legend=True)


def embedding_links(clusters):
    before = np.asarray(clusters['mean_row_0_pca'])
    after = np.asarray(clusters['mean_row_8_pca'])
    if before.ndim != 2 or before.shape[1] != 2 or before.shape != after.shape:
        raise ValueError('Embedding views must contain the same rows in two dimensions')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Embedding coordinates must be finite')
    rows = np.arange(min(8, len(before)))
    return rows, before[rows].copy(), after[rows].copy()


def embedding_example(data, clusters, example_metrics, output, style):
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 3.05))
    labels = data['row_labels']
    linked_rows, before, after = embedding_links(clusters)
    for axis, layer, name in zip(axes, [0, 8], ['Encoder', 'After 8 blocks']):
        projection = clusters[f'mean_row_{layer}_pca']
        score = example_metrics[(example_metrics.representation == 'mean') &
                                (example_metrics.axis == 'row') & (example_metrics.layer == layer)].ari.iloc[0]
        for group, color in enumerate(COLORS):
            selected = labels == group
            axis.scatter(projection[selected, 0], projection[selected, 1],
                         facecolors='none' if layer == 0 else color,
                         edgecolors=color if layer == 0 else 'white',
                         label=f'Group {group + 1}', s=16, alpha=.8, linewidths=.6 if layer == 0 else .25)
        axis.scatter(projection[linked_rows, 0], projection[linked_rows, 1],
                     facecolors='none' if layer == 0 else np.asarray(COLORS)[labels[linked_rows]],
                     edgecolors=np.asarray(COLORS)[labels[linked_rows]], s=24, linewidths=1, zorder=4)
        axis.set_title(f'{name}: ARI {score:.2f}')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('Encoder PC1' if layer == 0 else 'Final PC1')
        axis.set_ylabel('Encoder PC2' if layer == 0 else 'Final PC2')
        axis.set_facecolor('none')
        axis.set_zorder(2)
        style.style_axes(axis)
    for row, start, end in zip(linked_rows, before, after):
        arrow = ConnectionPatch(xyA=start, xyB=end, coordsA='data', coordsB='data',
                                axesA=axes[0], axesB=axes[1], arrowstyle='-|>',
                                color=COLORS[labels[row]], alpha=.45, linewidth=.75,
                                mutation_scale=7, shrinkA=3, shrinkB=3, clip_on=False, zorder=1)
        figure.add_artist(arrow)
    handles, names = axes[1].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', ncol=4, frameon=False, fontsize=7)
    figure.suptitle('The same rows, before and after MatrixFormer', fontsize=8, y=.985)
    figure.text(.5, .105, f'Arrows link the first {len(linked_rows)} rows; all {len(labels)} rows are shown',
                ha='center', fontsize=6.5)
    figure.tight_layout(pad=.5, rect=(0, .18, 1, .94))
    style.save(figure, output / 'synthetic_embedding_pca')
    plt.close(figure)
    records = [dict(row=int(row), true_group=int(labels[row]), encoder_pc1=float(start[0]),
                    encoder_pc2=float(start[1]), final_pc1=float(end[0]), final_pc2=float(end[1]))
               for row, start, end in zip(linked_rows, before, after)]
    pd.DataFrame(records).to_csv(output / 'embedding_identity_links.csv', index=False)


def matrix_example(data, clusters, output, style):
    rows = np.argsort(clusters['profile_row_8'], kind='stable')
    columns = np.argsort(clusters['profile_column_8'], kind='stable')
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.5))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e8e8e8')
    limit = np.nanmax(np.abs(data['context']))
    for axis, matrix, title in zip(axes, [data['context'], data['context'][np.ix_(rows, columns)]],
                                   ['Original shuffled order', 'Ordered by embedding clusters']):
        image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest')
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.subplots_adjust(left=.03, right=.89, top=.89, bottom=.03, wspace=.12)
    colorbar = figure.colorbar(image, cax=figure.add_axes([.92, .17, .018, .56]))
    colorbar.ax.set_title('Value', fontsize=7)
    style.save(figure, output / 'synthetic_cluster_example')
    plt.close(figure)


def noise_recovery(frame, output, style):
    selected = frame[(frame.observed_fraction == .4) &
                     (((frame.representation.isin(['mean', 'profile'])) & (frame.layer == 8)) |
                      (frame.representation == 'observed'))]
    group_keys = ['case_id', 'seed', 'noise', 'representation']
    if not (selected.groupby(group_keys).size() == 2).all():
        raise ValueError('Both axes are required for each matrix average')
    per_matrix = selected.groupby(group_keys, as_index=False).ari.mean()
    summary = summarize(per_matrix, ['noise', 'representation'], ['ari'])
    if not (summary.matrices == 10).all():
        raise ValueError('Expected ten matrices, not twenty axis replicates')
    per_matrix.to_csv(output / 'cluster_noise_by_matrix.csv', index=False)
    summary.to_csv(output / 'cluster_noise_summary.csv', index=False)
    figure, axis = plt.subplots(figsize=(style.WIDTH, 2.65))
    for representation, label, color, marker in [('mean', 'Mean embeddings', '#72a6d8', 'o'),
                                                ('profile', 'Profile embeddings', style.BLUE, '^'),
                                                ('observed', 'Observed values + mask', '#777777', 's')]:
        data = summary[summary.representation == representation].sort_values('noise')
        axis.errorbar(data.noise, data.ari, yerr=data.ari_ci95_half, marker=marker, color=color,
                      label=label, linewidth=1.2, markersize=3.5, capsize=2, elinewidth=.8)
    axis.set_title('Cluster recovery depends on noise and pooling')
    axis.set_xlabel('Noise standard deviation')
    axis.set_ylabel('Cluster recovery (adjusted Rand index)')
    axis.set_xticks([0, .25, 1], ['0', '0.25', '1'])
    axis.set_ylim(-.04, 1.08)
    axis.axhline(0, color='#aaaaaa', linestyle='--', linewidth=.6)
    style.style_axes(axis, grid='y')
    finish(figure, output / 'synthetic_cluster_recovery', style, legend=True)


def pooling_control(frame, output, style):
    selected = frame[(frame.noise == 0) & (frame.observed_fraction == .4) & (frame.axis == 'column') &
                     (((frame.representation.isin(['mean', 'profile'])) & (frame.layer == 8)) |
                      frame.representation.isin(['observed', 'completed']))]
    summary = summarize(selected, ['representation'], ['ari']).sort_values('ari', ascending=False).reset_index(drop=True)
    names = {'mean': 'Mean embeddings', 'profile': 'Profile embeddings',
             'observed': 'Observed values + mask', 'completed': 'Completed values'}
    colors = {'mean': '#72a6d8', 'profile': style.BLUE, 'observed': style.NEUTRAL, 'completed': style.NEUTRAL}
    figure, axis = plt.subplots(figsize=(style.WIDTH, 2.25))
    positions = np.arange(len(summary))
    axis.barh(positions, summary.ari, color=[colors[name] for name in summary.representation], height=.58)
    axis.errorbar(summary.ari, positions, xerr=summary.ari_ci95_half, fmt='none',
                  color=style.ERROR, elinewidth=.8, capsize=2)
    axis.set_yticks(positions, [names[name] for name in summary.representation])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.15)
    axis.set_xlabel('Column-cluster ARI (higher is better)')
    axis.set_title('Mean pooling hides column-cluster structure')
    for position, record in summary.iterrows():
        axis.text(record.ari + record.ari_ci95_half + .02, position, f'{record.ari:.3f}', va='center', fontsize=7)
    style.style_axes(axis)
    finish(figure, output / 'synthetic_cluster_pooling', style)


def attention_example(data, attention, output, style):
    values = attention['context_mean_row_8'].mean(0)
    norm = PowerNorm(gamma=.5, vmin=0, vmax=float(values[12:, 12:].max()))
    figure = plt.figure(figsize=(style.WIDTH, 3.3))
    axis = figure.add_axes([.16, .15, .5, .78])
    image = attention_image(axis, values, data['row_labels'], norm, 'Final-layer row attention')
    axis.set_title('Final-layer row attention')
    axis.set_xlabel('Attended-to row group')
    axis.set_ylabel('Query row group')
    axis.tick_params(labelsize=7)
    figure.colorbar(image, cax=figure.add_axes([.70, .15, .025, .78]),
                    ticks=[0, .01, .02, .03, .04], label='Attention probability')
    style.save(figure, output / 'synthetic_attention_example')
    plt.close(figure)


def attention_heatmaps(data, attention, output, style):
    selected = {(direction, layer): attention[f'context_mean_{direction}_{layer}'].mean(0)
                for direction in ('row', 'column') for layer in DISPLAY_LAYERS}
    maximum = max(float(values[12:, 12:].max()) for values in selected.values())
    norm = PowerNorm(gamma=.5, vmin=0, vmax=maximum)
    figure, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 4.0))
    for row, direction in enumerate(('row', 'column')):
        for column, layer in enumerate(DISPLAY_LAYERS):
            axis = axes[row, column]
            image = attention_image(axis, selected[direction, layer], data[f'{direction}_labels'],
                                    norm, f'{direction.capitalize()} attention: layer {layer}')
            if column == 0:
                axis.set_ylabel('Query group')
            if row == 1:
                axis.set_xlabel('Key group')
    figure.subplots_adjust(left=.075, right=.985, bottom=.22, top=.94, wspace=.18, hspace=.34)
    figure.colorbar(image, cax=figure.add_axes([.28, .11, .44, .015]), orientation='horizontal',
                    label='Attention probability (square-root color scale)')
    style.save(figure, output / 'synthetic_attention_maps')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('/root/MatrixFormerPaper'))
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location('simple_synthetic_style', args.paper / 'scripts/figure_style.py')
    style = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(style)
    style.apply()
    protocol = json.loads((args.output / 'protocol.json').read_text())
    parent = Path(protocol['parent'])
    output = args.output / 'paper_figures'
    output.mkdir(exist_ok=True)
    scores = pd.read_csv(parent / 'report/scores_summary.csv')
    solver_tradeoffs(scores, output, style)
    neighbor_bars(scores, output, style)
    simple_bars(scores[(scores.family == 'noise') & (scores.observed_fraction == .4)],
                {'matrixformer': 'MatrixFormer', 'column_mean': 'Column mean'},
                'No advantage on independent noise', output / 'synthetic_null', style, reference=True)
    attention_alignment(parent, output, style)
    frame = pd.read_csv(args.output / 'metrics.csv')
    case = next(case for case in protocol['cases'] if case['seed'] == 0 and case['noise'] == .25 and case['observed_fraction'] == .4)
    data, prediction, clusters, attention = load_example(args.output, case)
    embedding_example(data, clusters, frame[frame.case_id == case['case_id']], output, style)
    matrix_example(data, clusters, output, style)
    noise_recovery(frame, output, style)
    pooling_control(frame, output, style)
    attention_example(data, attention, output, style)
    attention_heatmaps(data, attention, output, style)
    write_json(output / 'manifest.json', dict(source=str(Path(__file__).resolve()), source_sha256=sha256(Path(__file__)),
               selection='Main nearest-neighbor comparison: clustered noise SD .25 at 20% and 40% observed; '
                         'MatrixFormer, two-way K-NN, oracle neighbors, oracle block mean, ten matrices per rate. '
                         'Broad solver/noise sweeps remain archived but are not included in the manuscript. '
                         'Attention: both axes, first/middle/final layers 1/4/8, all heads/data contexts; '
                         'seed 0, noise .25, 40% observed, unchanged from the earlier fixed example. '
                         'Embedding arrows: first eight row indices in the unchanged independent PCA views; '
                         'identity correspondences, not displacement in a common projection or intermediate paths.',
               complete_results=str(args.output / 'report'), full_diagnostic_figures=str(args.output / 'figures'),
               artifacts={path.name: sha256(path) for path in output.iterdir() if path.name != 'manifest.json'},
               inputs={str(path): sha256(path) for path in [parent / 'report/scores_summary.csv',
                       parent / 'report/attention_summary.csv', args.output / 'metrics.csv',
                       parent / 'cases' / case['case_id'] / 'inputs.npz',
                       args.output / 'cases' / case['case_id'] / 'clustering.npz',
                       args.output / 'cases' / case['case_id'] / 'attention.npz']}))
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
