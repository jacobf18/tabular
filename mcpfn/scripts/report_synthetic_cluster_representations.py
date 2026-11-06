"""Plot all registered cluster diagnostics and fixed examples without head selection."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from recsys_evaluation_common import atomic_npz, sha256, write_json
from report_matrixformer_synthetic_study import summarize


CONDITIONS = ['noise', 'observed_fraction']
COLORS = ['#2A6FBB', '#dc9c36', '#2e946b', '#ae567c']
MARKERS = ['o', 's', '^', 'D']
NAMES = {'mean': 'Mean embedding', 'profile': 'Cell-profile embedding',
         'observed': 'Observed values + mask', 'completed': 'Completed values',
         'mask': 'Mask only', 'oracle_complete': 'Complete values (oracle)'}


def series_style(style):
    return {'profile': (style.BLUE, 'o', '-'), 'mean': ('#72a6d8', 's', '-'),
            'observed': ('#555555', '^', '--'), 'completed': ('#92969b', 'D', ':'),
            'oracle_complete': ('#c78f28', 'v', '--'), 'mask': ('#b9bec8', 'x', ':')}


def recovery_plot(summary, figures, style, by_layer):
    figure, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 3.45), sharex=True, sharey=True)
    configuration = series_style(style)
    for row, direction in enumerate(('row', 'column')):
        for column, noise in enumerate((0., .25, 1.)):
            axis = axes[row, column]
            selected = summary[(summary.axis == direction) & (summary.noise == noise)]
            for representation, (color, marker, linestyle) in configuration.items():
                data = selected[selected.representation == representation]
                if by_layer:
                    data = data[data.observed_fraction == .4].sort_values('layer')
                    if representation in ('mean', 'profile'):
                        horizontal = data.layer.to_numpy()
                        values, half = data.ari.to_numpy(), data.ari_ci95_half.to_numpy()
                    else:
                        horizontal = np.array([0, 8])
                        values, half = np.repeat(data.ari.iloc[0], 2), np.repeat(data.ari_ci95_half.iloc[0], 2)
                else:
                    if representation in ('mean', 'profile'):
                        data = data[data.layer == 8]
                    data = data.sort_values('observed_fraction')
                    horizontal = data.observed_fraction.to_numpy() * 100
                    values, half = data.ari.to_numpy(), data.ari_ci95_half.to_numpy()
                axis.plot(horizontal, values, color=color, marker=marker, markersize=2.4,
                          linewidth=.85, linestyle=linestyle, label=NAMES[representation])
                axis.fill_between(horizontal, values - half, values + half, color=color, alpha=.10, linewidth=0)
            axis.axhline(0, color='#999999', linewidth=.5)
            axis.set_ylim(-.08, 1.07)
            axis.set_title(f'{direction.capitalize()} groups; noise SD {noise:g}')
            axis.set_xticks([0, 2, 4, 6, 8] if by_layer else [20, 40, 60, 80])
            if column == 0:
                axis.set_ylabel('Adjusted Rand index')
            if row == 1:
                axis.set_xlabel('Layer (0 = encoder)' if by_layer else 'Observed entries (%)')
            style.style_axes(axis, grid='y')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=6.4,
                  columnspacing=.9, handlelength=1.5)
    figure.tight_layout(pad=.4, rect=(0, .115, 1, 1))
    style.save(figure, figures / ('synthetic_embedding_layers' if by_layer else 'synthetic_cluster_recovery'))
    plt.close(figure)


def load_example(output, case):
    protocol = json.loads((output / 'protocol.json').read_text())
    old = Path(protocol['parent']) / 'cases' / case['case_id']
    new = output / 'cases' / case['case_id']
    with np.load(old / 'inputs.npz') as saved:
        data = dict(saved)
    with np.load(old / 'predictions.npz') as saved:
        prediction = saved['matrixformer']
    with np.load(new / 'clustering.npz') as saved:
        clusters = dict(saved)
    with np.load(new / 'attention.npz') as saved:
        attention = dict(saved)
    return data, prediction, clusters, attention


def matrix_example(data, prediction, clusters, figures, style):
    true_rows, true_columns = np.argsort(data['row_labels'], kind='stable'), np.argsort(data['column_labels'], kind='stable')
    fitted_rows = np.argsort(clusters['profile_row_8'], kind='stable')
    fitted_columns = np.argsort(clusters['profile_column_8'], kind='stable')
    figure, axes = plt.subplots(1, 4, figsize=(style.WIDTH, 1.65))
    panels = [(data['context'], 'Observed (shuffled)'),
              (data['context'][np.ix_(true_rows, true_columns)], 'True grouping'),
              (data['context'][np.ix_(fitted_rows, fitted_columns)], 'Embedding grouping'),
              (prediction[np.ix_(fitted_rows, fitted_columns)], 'Completion (same order)')]
    limit = np.max(np.abs(data['truth']))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e8e8e8')
    for axis, (matrix, title) in zip(axes, panels):
        image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest')
        axis.set_title(title, fontsize=6.5)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.subplots_adjust(left=.015, right=.92, bottom=.08, top=.87, wspace=.12)
    colorbar = figure.colorbar(image, cax=figure.add_axes([.94, .19, .012, .55]))
    colorbar.ax.set_title('Value', fontsize=6.5)
    style.save(figure, figures / 'synthetic_cluster_example')
    plt.close(figure)


def pca_example(data, clusters, metrics, figures, style):
    figure, axes = plt.subplots(2, 4, figsize=(style.WIDTH, 3.0))
    representations = [('observed', -1, 'Observed values + mask'), ('mean', 0, 'Encoder mean'),
                       ('mean', 8, 'Final mean'), ('profile', 8, 'Final cell profile')]
    for row, direction in enumerate(('row', 'column')):
        labels = data[f'{direction}_labels']
        for column, (representation, layer, title) in enumerate(representations):
            axis = axes[row, column]
            name = f'{representation}_{direction}_{layer}'
            projection, fitted = clusters[f'{name}_pca'], clusters[name]
            for group, marker in enumerate(MARKERS):
                selected = fitted == group
                axis.scatter(projection[selected, 0], projection[selected, 1],
                             c=np.asarray(COLORS)[labels[selected]], marker=marker, s=9,
                             linewidths=.2, edgecolors='white', alpha=.85)
            score = metrics[(metrics.representation == representation) & (metrics.axis == direction) & (metrics.layer == layer)].ari.iloc[0]
            variance = clusters[f'{name}_pca_variance']
            axis.set_title(f'{title}\nARI {score:.2f}; 2D var. {100 * sum(variance):.0f}%', fontsize=6.4)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f'{direction.capitalize()} groups\nPC2')
            if row == 1:
                axis.set_xlabel('PC1')
            style.style_axes(axis)
    color_handles = [Line2D([], [], marker='o', linestyle='', color=color, markersize=3, label=str(index + 1))
                     for index, color in enumerate(COLORS)]
    shape_handles = [Line2D([], [], marker=marker, linestyle='', color='#666666', markersize=3, label=str(index + 1))
                     for index, marker in enumerate(MARKERS)]
    figure.legend(color_handles, [handle.get_label() for handle in color_handles], title='Color: true group',
                  loc='lower left', bbox_to_anchor=(.08, 0), ncol=4, frameon=False, fontsize=6.2, title_fontsize=6.2,
                  columnspacing=.4, handletextpad=.2)
    figure.legend(shape_handles, [handle.get_label() for handle in shape_handles], title='Shape: fitted group (unordered)',
                  loc='lower right', bbox_to_anchor=(.96, 0), ncol=4, frameon=False, fontsize=6.2, title_fontsize=6.2,
                  columnspacing=.4, handletextpad=.2)
    figure.tight_layout(pad=.4, rect=(0, .16, 1, 1))
    style.save(figure, figures / 'synthetic_embedding_pca')
    plt.close(figure)


def attention_image(axis, values, labels, norm, title, ticks=True):
    special = values.shape[-1] - len(labels)
    order = np.argsort(labels, kind='stable') + special
    image = axis.imshow(values[np.ix_(order, order)], cmap='Blues', norm=norm, interpolation='nearest')
    for boundary in (15.5, 31.5, 47.5):
        axis.axhline(boundary, linewidth=.35, color='#d49136', alpha=.8)
        axis.axvline(boundary, linewidth=.35, color='#d49136', alpha=.8)
    axis.set_title(title, fontsize=6.4)
    axis.set_xticks([7.5, 23.5, 39.5, 55.5] if ticks else [], ['1', '2', '3', '4'] if ticks else [])
    axis.set_yticks([7.5, 23.5, 39.5, 55.5] if ticks else [], ['1', '2', '3', '4'] if ticks else [])
    axis.tick_params(length=0, pad=1, labelsize=5.7)
    return image


def attention_layers(data, attention, noise, figures, style):
    arrays = [attention[f'context_mean_{axis}_{layer}'].mean(0)[12:, 12:]
              for axis in ('row', 'column') for layer in range(1, 9)]
    norm = PowerNorm(gamma=.5, vmin=0, vmax=max(float(array.max()) for array in arrays))
    figure, axes = plt.subplots(4, 4, figsize=(style.WIDTH, 5.7))
    for direction_index, direction in enumerate(('row', 'column')):
        for layer in range(1, 9):
            row, column = direction_index * 2 + (layer - 1) // 4, (layer - 1) % 4
            image = attention_image(axes[row, column], attention[f'context_mean_{direction}_{layer}'].mean(0),
                                    data[f'{direction}_labels'], norm, f'{direction.capitalize()} attention: layer {layer}')
            if column == 0:
                axes[row, column].set_ylabel('Query group')
            if row == 3:
                axes[row, column].set_xlabel('Key group')
    figure.tight_layout(pad=.4, rect=(0, .12, 1, .97))
    figure.suptitle(f'All heads and all contexts; noise SD {noise:g}; seed 0', y=.995, fontsize=7.5)
    figure.colorbar(image, cax=figure.add_axes([.25, .075, .5, .012]), orientation='horizontal',
                    label='Attention probability (square-root color scale)')
    suffix = '' if noise == .25 else f'_noise{noise:g}'
    style.save(figure, figures / f'synthetic_attention_layers{suffix}')
    plt.close(figure)


def head_gallery(data, attention, figures, style):
    with PdfPages(figures / 'synthetic_attention_all_heads.pdf', metadata={'CreationDate': None, 'ModDate': None}) as document:
        for kind, kind_name in (('context_mean', 'Mean over all 64 data contexts'),
                                ('fixed_context', f"Fixed missing-query context {tuple(attention['target'])}")):
            maximum = max(float(attention[f'{kind}_{axis}_{layer}'][:, 12:, 12:].max())
                          for axis in ('row', 'column') for layer in range(1, 9))
            norm = PowerNorm(gamma=.5, vmin=0, vmax=maximum)
            for direction in ('row', 'column'):
                for layer in range(1, 9):
                    figure, axes = plt.subplots(4, 4, figsize=(style.WIDTH, 5.9))
                    for head, axis in enumerate(axes.flat):
                        image = attention_image(axis, attention[f'{kind}_{direction}_{layer}'][head],
                                                data[f'{direction}_labels'], norm, f'Head {head + 1}')
                    figure.suptitle(f'{direction.capitalize()} attention, layer {layer}: every head\n{kind_name}', fontsize=8, y=.99)
                    figure.tight_layout(pad=.5, rect=(0, .11, 1, .94))
                    figure.colorbar(image, cax=figure.add_axes([.25, .065, .5, .012]), orientation='horizontal',
                                    label='Attention probability (square-root color scale)')
                    document.savefig(figure, dpi=180)
                    if kind == 'context_mean' and layer == 8:
                        style.save(figure, figures / f'synthetic_attention_final_{direction}_heads')
                    plt.close(figure)


def report(output, paper):
    specification = importlib.util.spec_from_file_location('cluster_figure_style', paper / 'scripts/figure_style.py')
    style = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(style)
    style.apply()
    protocol = json.loads((output / 'protocol.json').read_text())
    audited = json.loads((output / 'audit.json').read_text())
    if protocol['profile'] != 'main' or audited['cases'] != 120:
        raise ValueError('The report requires all 120 registered cases')
    frame = pd.read_csv(output / 'metrics.csv')
    keys = ['case_id', 'representation', 'axis', 'layer']
    if frame.duplicated(keys).any() or len(frame) != 120 * 44:
        raise ValueError('Invalid clustering inventory')
    destination, figures = output / 'report', output / 'figures'
    destination.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    metrics = ['ari', 'nmi', 'knn5_purity', 'knn5_excess', 'shuffled_ari']
    summary = summarize(frame, CONDITIONS + ['representation', 'axis', 'layer'], metrics)
    if not (summary.matrices == 10).all():
        raise ValueError('Incomplete condition coverage')
    summary.to_csv(destination / 'clustering_summary.csv', index=False)
    frame.to_csv(destination / 'clustering_by_matrix.csv', index=False)
    baseline = frame[frame.representation == 'observed'][['case_id', 'axis', 'ari']].rename(columns={'ari': 'observed_ari'})
    paired = frame.merge(baseline, on=['case_id', 'axis'], validate='many_to_one')
    paired['ari_difference'] = paired.ari - paired.observed_ari
    summarize(paired, CONDITIONS + ['representation', 'axis', 'layer'], ['ari_difference']).to_csv(
        destination / 'paired_vs_observed.csv', index=False)
    recovery_plot(summary, figures, style, by_layer=False)
    recovery_plot(summary, figures, style, by_layer=True)
    for case in protocol['cases']:
        if case['seed'] != 0 or case['observed_fraction'] != .4:
            continue
        data, prediction, clusters, attention = load_example(output, case)
        attention_layers(data, attention, case['noise'], figures, style)
        if case['noise'] == .25:
            matrix_example(data, prediction, clusters, figures, style)
            pca_example(data, clusters, frame[frame.case_id == case['case_id']], figures, style)
            head_gallery(data, attention, figures, style)
            atomic_npz(destination / 'fixed_example.npz', **data, prediction=prediction, **clusters,
                       **{name: values.mean(0) for name, values in attention.items() if name != 'target'},
                       target=attention['target'])
    for name in ('protocol.json', 'manifest.json', 'audit.json'):
        shutil.copyfile(output / name, destination / name)
    write_json(destination / 'export_manifest.json', dict(output=str(output),
               report_source=str(Path(__file__).resolve()), report_sha256=sha256(Path(__file__)),
               figure_style_sha256=sha256(paper / 'scripts/figure_style.py'),
               artifacts={str(path.relative_to(output)): sha256(path)
                          for folder in (destination, figures) for path in folder.iterdir()
                          if path.is_file() and path.name != 'export_manifest.json'}))
    print(json.dumps(dict(figures=str(figures), report=str(destination))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('/root/MatrixFormerPaper'))
    args = parser.parse_args()
    report(args.output, args.paper)
