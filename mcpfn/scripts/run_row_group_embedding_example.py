"""Probe one frozen row-clustered matrix without latent column groups."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch

from plot_synthetic_embedding_filmstrip import COLORS, LAYERS, MARKERS, digest, figure_for_axes, validate_bounds
from recsys_evaluation_common import atomic_npz, write_json
from run_matrixformer_synthetic_study import CHECKPOINT, CHECKPOINT_SHA, Predictor, source_paths
import synthetic_cluster_representations as analysis


CASE = dict(family='row_profiles', rows=64, columns=64, row_groups=4, seed=0,
            noise=.25, observed_fraction=.4, case_id='row_profiles_seed0_noise025_observed040')


def generate():
    generator = np.random.default_rng(CASE['seed'])
    profiles = generator.normal(size=(CASE['row_groups'], CASE['columns']))
    row_labels = generator.permutation(np.repeat(np.arange(CASE['row_groups']), 16))
    signal = profiles[row_labels]
    noise = generator.normal(size=signal.shape)
    truth = (signal + CASE['noise'] * noise).astype(np.float32)
    observed = np.zeros(truth.size, dtype=bool)
    observed[generator.permutation(truth.size)[:round(CASE['observed_fraction'] * truth.size)]] = True
    observed = observed.reshape(truth.shape)
    if not observed.any(0).all() or not observed.any(1).all():
        raise ValueError('The fixed matrix must have observed context in every row and column')
    context = np.where(observed, truth, np.nan)
    return dict(profiles=profiles, row_labels=row_labels, signal=signal, truth=truth,
                noise=noise, observed=observed, context=context)


def score_features(features, row_labels):
    records, assignments, points = [], {}, []
    for representation in ('mean', 'profile'):
        for layer in range(9):
            key = f'{representation}_row_{layer}'
            values = features[key]
            if values.shape[0] != 64:
                raise ValueError('Expected all 64 rows in every representation')
            fitted, projection, variance = analysis.fit_clusters(values, clusters=4)
            metrics = analysis.cluster_metrics(values, fitted, row_labels)
            records.append(dict(**CASE, representation=representation, axis='row', layer=layer, **metrics))
            assignments[key] = fitted
            assignments[f'{key}_pca'] = projection
            assignments[f'{key}_pca_variance'] = variance
            if representation == 'mean' and layer in LAYERS:
                points.extend(dict(axis='row', layer=layer, object_index=index, true_group=int(group),
                                   color=COLORS[group], marker=MARKERS[group],
                                   pc1=float(point[0]), pc2=float(point[1]), full_space_ari=metrics['ari'])
                              for index, (group, point) in enumerate(zip(row_labels, projection)))
    return pd.DataFrame(records), assignments, pd.DataFrame(points)


def construction_figure(data, colorbar_side='bottom'):
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    order = np.argsort(data['row_labels'], kind='stable')
    panels = (data['profiles'], data['truth'][order], data['context'])
    titles = ('Four row groups', 'Noisy matrix, grouped rows', 'Shuffled, masked model input')
    if colorbar_side not in ('bottom', 'left'):
        raise ValueError('colorbar_side must be bottom or left')
    # A color scale on the left needs room before the first panel's row labels.
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65),
                                gridspec_kw={'left': .07 if colorbar_side == 'bottom' else .118,
                                             'right': .985, 'bottom': .29, 'top': .83,
                                             'wspace': .70 if colorbar_side == 'bottom' else .66})
    limit = np.ceil(np.max(np.abs(data['truth'])))
    colormap = plt.get_cmap('RdBu_r').copy()
    colormap.set_bad('#d5d5d5')
    for index, (panel, values, title) in enumerate(zip(axes, panels, titles)):
        image = panel.imshow(np.ma.masked_invalid(values), cmap=colormap,
                             norm=Normalize(-limit, limit), interpolation='nearest', aspect='auto')
        panel.set_title(title, fontsize=8)
        panel.set_xlabel('Column', fontsize=7)
        panel.set_ylabel('Row group' if index == 0 else 'Row', fontsize=7)
        panel.set_xticks([0, 31, 63], ['1', '32', '64'])
        if index == 0:
            panel.set_yticks(range(4), range(1, 5))
        else:
            panel.set_yticks([0, 31, 63], ['1', '32', '64'])
        panel.tick_params(length=2, labelsize=6)
        if index == 1:
            for boundary in (16, 32, 48):
                panel.axhline(boundary - .5, color='white', linewidth=.7)
    for left_panel, right_panel, explanation in zip(axes[:-1], axes[1:],
                                                    ('repeat rows\n+ noise', 'shuffle rows\n+ mask')):
        left = left_panel.get_position().x1 + .012
        right = right_panel.get_position().x0 - .06
        middle = (left_panel.get_position().y0 + left_panel.get_position().y1) / 2
        figure.add_artist(FancyArrowPatch((left, middle), (right, middle),
                          transform=figure.transFigure, arrowstyle='-|>',
                          mutation_scale=10, linewidth=1.1, color='#333333'))
        figure.text((left + right) / 2, middle + .055, explanation,
                    ha='center', va='bottom', fontsize=7)
    first_panel = axes[0].get_position()
    if colorbar_side == 'bottom':
        color_width = first_panel.width * .65
        color_left = first_panel.x0 + (first_panel.width - color_width) / 2
        color_axis = figure.add_axes([color_left, .12, color_width, .016])
        colorbar = figure.colorbar(image, cax=color_axis, orientation='horizontal', ticks=[-limit, 0, limit])
        colorbar.set_label('Value', fontsize=5.5, labelpad=.5)
        colorbar.ax.tick_params(length=1.5, pad=.5, labelsize=5)
    else:
        color_axis = figure.add_axes([.042, first_panel.y0, .0065, first_panel.height])
        colorbar = figure.colorbar(image, cax=color_axis, orientation='vertical', ticks=[-limit, 0, limit])
        colorbar.ax.yaxis.set_ticks_position('left')
        colorbar.ax.tick_params(length=1.5, pad=1, labelsize=6)
    colorbar.outline.set_linewidth(.4)
    figure.text(.82, .10, 'Gray = missing\n40% observed', ha='center', va='center', fontsize=7)
    validate_bounds(figure)
    return figure


def render(output, matrix_only=False, colorbar_side='bottom'):
    output = Path(output)
    with np.load(output / 'inputs.npz', allow_pickle=False) as saved:
        data = dict(saved)
    scores = pd.read_csv(output / 'metrics.csv')
    points = pd.read_csv(output / 'plot_points.csv')
    if matrix_only:
        figures = [('row_group_matrix_arrows', construction_figure(data, colorbar_side))]
    else:
        figures = [('row_group_embeddings', figure_for_axes(scores[scores.representation == 'mean'], points, ('row',))),
                   ('row_group_matrix', construction_figure(data))]
    for name, figure in figures:
        try:
            for extension in ('pdf', 'png'):
                figure.savefig(output / f'{name}.{extension}', dpi=300,
                               metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
        finally:
            plt.close(figure)
    if matrix_only:
        paths = [output / f'row_group_matrix_arrows.{extension}' for extension in ('pdf', 'png')]
        write_json(output / 'row_group_matrix_arrows.json', dict(
            source_sha256={str(output / 'inputs.npz'): digest(output / 'inputs.npz')},
            script_sha256=digest(__file__), new_inference=False,
            artifacts_sha256={str(path): digest(path) for path in paths}))


def run(output, device):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; original examples must not be overwritten')
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Frozen checkpoint hash mismatch')
    paths = set(source_paths()) | {Path(__file__).resolve(), Path(analysis.__file__).resolve(),
                                  Path(__file__).with_name('plot_synthetic_embedding_filmstrip.py').resolve(),
                                  CHECKPOINT.parent / 'config.json'}
    hashes = {str(path): digest(path) for path in paths}
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / 'protocol.json', dict(case=CASE, formula='X_ij = P_{g_i,j} + 0.25 E_ij',
               profiles='4 x 64 independent standard-normal entries; no column labels or column groups',
               checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
               device=device, precision='float32' if device == 'cpu' else 'amp_bfloat16',
               selection='One fixed seed and condition, chosen before inference; no seed search',
               normalization='Observed-only per-column mean and population standard deviation',
               representation='Mean of cell embeddings across columns, excluding special tokens',
               clustering=f'Full-space K-means, K=4, n_init=20, seed={analysis.CLUSTER_SEED}',
               pca='Independent label-blind PCA per layer, display only',
               interpretation='Illustrative single-matrix probe; no confidence intervals or population claim',
               source_sha256=hashes))
    data = generate()
    atomic_npz(output / 'inputs.npz', **data)
    torch.set_num_threads(2)
    torch.manual_seed(CASE['seed'])
    predictor = Predictor(device=device)
    context = data['context']
    means = np.nanmean(context.astype(float), axis=0)
    scales = np.nanstd(context.astype(float), axis=0)
    baseline = predictor.predict(context, means, scales)
    prediction, features, attention = analysis.capture(predictor, context, means, scales)
    np.testing.assert_array_equal(prediction, baseline)
    if attention:
        raise ValueError('This experiment does not capture attention probabilities')
    predictor.check_unchanged()
    row_features = {key: values for key, values in features.items() if '_row_' in key}
    scores, assignments, points = score_features(row_features, data['row_labels'])
    atomic_npz(output / 'features.npz', **row_features)
    atomic_npz(output / 'clustering.npz', **assignments)
    atomic_npz(output / 'prediction.npz', prediction=prediction, means=means, scales=scales)
    scores.to_csv(output / 'metrics.csv', index=False)
    points.to_csv(output / 'plot_points.csv', index=False)
    render(output)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Source changed during the experiment: {path}')
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint changed during the experiment')
    write_json(output / 'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(),
               case=CASE, capture_bit_identical=True, model_parameters_unchanged=True,
               checkpoint_unchanged=True, seconds=time.perf_counter() - started,
               artifacts_sha256={path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}))
    print(scores[['representation', 'layer', 'ari']].to_string(index=False), flush=True)
    print(f'Saved new row-only example to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'render', 'render-matrix'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--colorbar', choices=('bottom', 'left'), default='bottom',
                        help='Side of the first matrix for the color scale (render-matrix only)')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.output, args.device)
    else:
        render(args.output, matrix_only=args.command == 'render-matrix', colorbar_side=args.colorbar)


if __name__ == '__main__':
    main()
