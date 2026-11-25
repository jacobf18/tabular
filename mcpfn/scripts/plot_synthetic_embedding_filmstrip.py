"""Plot cluster recovery and layer snapshots of one archived matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


CASE_ID = '7f7a39e928ff6b43'
LAYERS = (0, 2, 4, 6, 8)
AXES = ('row', 'column')
COLORS = ('#2A6FBB', '#dc9c36', '#2e946b', '#ae567c')
MARKERS = ('o', 's', '^', 'D')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def example_ari(saved, scores):
    selected = scores[(scores.case_id == CASE_ID) & (scores.representation == 'mean')
                      & (scores.noise == .25) & (scores.observed_fraction == .4)].copy()
    if selected.duplicated(['case_id', 'axis', 'layer']).any():
        raise ValueError('Duplicate matrix-axis-layer scores')
    if not np.isfinite(selected.ari).all() or not selected.ari.between(-1, 1).all():
        raise ValueError('ARI scores must be finite and within [-1, 1]')
    if set(zip(selected.axis, selected.layer)) != {(axis, layer) for axis in AXES for layer in range(9)}:
        raise ValueError('Expected both axes at every layer from zero to eight')
    if not selected.seed.eq(0).all():
        raise ValueError('Expected the original seed-zero example score')
    for axis in AXES:
        labels = saved[f'{axis}_labels']
        if labels.shape != (64,) or not np.array_equal(np.bincount(labels), [16] * 4):
            raise ValueError('Expected 64 objects in four balanced groups')
        for layer in range(9):
            assignments = saved[f'mean_{axis}_{layer}']
            if assignments.shape != labels.shape:
                raise ValueError('Expected one archived cluster assignment per object')
            ari = adjusted_rand_score(labels, assignments)
            matching = (selected.axis == axis) & (selected.layer == layer)
            np.testing.assert_allclose(ari, selected.loc[matching, 'ari'].iloc[0], rtol=0, atol=1e-12)
            selected.loc[matching, 'ari'] = ari
    return selected.sort_values(['axis', 'layer']).reset_index(drop=True)


def example_points(saved, scores):
    scores = example_ari(saved, scores)
    records = []
    for axis in AXES:
        labels = saved[f'{axis}_labels']
        if labels.shape != (64,) or not np.array_equal(np.bincount(labels), [16] * 4):
            raise ValueError('Expected 64 objects in four balanced groups')
        for layer in LAYERS:
            key = f'mean_{axis}_{layer}'
            projection, assignments = saved[f'{key}_pca'], saved[key]
            if projection.shape != (64, 2) or not np.isfinite(projection).all():
                raise ValueError('Expected all 64 finite archived PCA coordinates')
            if assignments.shape != labels.shape:
                raise ValueError('Expected one archived cluster assignment per object')
            selected = scores[(scores.case_id == CASE_ID) & (scores.axis == axis)
                              & (scores.layer == layer)]
            if len(selected) != 1 or selected.seed.iloc[0] != 0:
                raise ValueError('Expected the original seed-zero example score')
            ari = adjusted_rand_score(labels, assignments)
            np.testing.assert_allclose(ari, selected.ari.iloc[0], rtol=0, atol=1e-12)
            records.extend(dict(axis=axis, layer=layer, object_index=index, true_group=int(group),
                                color=COLORS[group], marker=MARKERS[group],
                                pc1=float(point[0]), pc2=float(point[1]), full_space_ari=ari)
                           for index, (group, point) in enumerate(zip(labels, projection)))
    return pd.DataFrame(records)


def validate_bounds(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = [axis.get_tightbbox(renderer) for axis in figure.axes]
    bounds.extend(legend.get_window_extent(renderer) for legend in figure.legends)
    for bounds_item in bounds:
        if (bounds_item.x0 < 0 or bounds_item.y0 < 0
                or bounds_item.x1 > figure.bbox.width or bounds_item.y1 > figure.bbox.height):
            raise ValueError('Figure elements would be clipped')


def figure_for_axes(layer_scores, points, axes=AXES):
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.bbox': None, 'axes.facecolor': 'white',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figure = plt.figure(figsize=(5.5, 2.8 if len(axes) == 2 else 1.35))
    grid = figure.add_gridspec(len(axes), 7, width_ratios=[1, .45, 1, 1, 1, 1, 1],
                              left=.065, right=.995, top=.90 if len(axes) == 2 else .86,
                              bottom=.17 if len(axes) == 2 else .22, wspace=.10, hspace=.85)
    panels = [[figure.add_subplot(grid[row, column]) for column in (0, 2, 3, 4, 5, 6)]
              for row in range(len(axes))]
    for row, axis_name in enumerate(axes):
        recovery = panels[row][0]
        curve = layer_scores[layer_scores.axis == axis_name].sort_values('layer')
        recovery.plot(curve.layer, curve.ari, color=COLORS[0], marker='o',
                      markersize=2.4, linewidth=1.1)
        if len(axes) == 2:
            recovery.set_title(f'{axis_name.capitalize()} groups', fontsize=6.5, pad=2)
        recovery.set_xlabel('Layers', fontsize=6.5, labelpad=1.5)
        recovery.set_ylabel('ARI', fontsize=6.5, labelpad=1.5)
        recovery.set_xticks(LAYERS)
        recovery.set_yticks([0, .5, 1])
        recovery.set_xlim(-.4, 8.4)
        recovery.set_ylim(-.12, 1.14)
        recovery.set_box_aspect(1)
        recovery.tick_params(length=2, pad=1.5, labelsize=7.5)
        recovery.grid(axis='y', color='#e3e3e3', linewidth=.5)
        recovery.set_axisbelow(True)
        for column, layer in enumerate(LAYERS, start=1):
            panel = panels[row][column]
            selected = points[(points.axis == axis_name) & (points.layer == layer)]
            for group, (color, marker) in enumerate(zip(COLORS, MARKERS)):
                members = selected[selected.true_group == group]
                panel.scatter(members.pc1, members.pc2, color=color, marker=marker,
                              s=8, alpha=.85, edgecolor='white', linewidth=.2)
            panel.set_title(f'Layer {layer}', fontsize=6.5, pad=2)
            panel.text(.06, .94, f'ARI = {selected.full_space_ari.iloc[0]:.2f}',
                       transform=panel.transAxes, fontsize=6.5, va='top', ha='left', color='#333333')
            panel.set_xlabel('PC1', fontsize=6.5, labelpad=0)
            if column == 1:
                panel.set_ylabel('PC2', fontsize=6.5, labelpad=3)
            panel.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], prune='both'))
            panel.tick_params(axis='x', length=1.5, pad=.5, labelsize=5)
            panel.tick_params(axis='y', direction='in', length=1.5, pad=-2.5, labelsize=5)
            panel.set_box_aspect(1)
            left, right = selected.pc1.min(), selected.pc1.max()
            width = max(right - left, 1e-12)
            panel.set_xlim(left - .4 * width, right + .15 * width)
            lower, upper = selected.pc2.min(), selected.pc2.max()
            extent = max(upper - lower, 1e-12)
            panel.set_ylim(lower - .15 * extent, upper + .4 * extent)
            minimum, maximum = panel.get_ylim()
            tick_lower = minimum + .12 * (maximum - minimum)
            tick_upper = minimum + .74 * (maximum - minimum)
            ticks = MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(tick_lower, tick_upper)
            panel.set_yticks(ticks[(ticks >= tick_lower) & (ticks <= tick_upper)])
            for label in panel.get_yticklabels():
                label.set_horizontalalignment('left')
        for panel in panels[row]:
            panel.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], linestyle='none', marker=marker, color=color, markersize=3.5,
                      label=f'Group {group + 1}')
               for group, (marker, color) in enumerate(zip(MARKERS, COLORS))]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.58, .005),
                  ncol=4, frameon=False, handletextpad=.25, columnspacing=1.2, fontsize=7)
    validate_bounds(figure)
    return figure


def render(source, output):
    source, output = Path(source), Path(output)
    source_paths = [source / name for name in
                    ('fixed_example.npz', 'clustering_by_matrix.csv', 'paper_export.json')]
    source_hashes = {str(path.resolve()): digest(path) for path in source_paths}
    registry = json.loads(source_paths[2].read_text())
    for path in source_paths[:2]:
        if digest(path) != registry['results'][path.name]:
            raise ValueError(f'Source does not match its archived hash: {path.name}')
    with np.load(source_paths[0], allow_pickle=False) as saved:
        selected = example_ari(saved, pd.read_csv(source_paths[1]))
        points = example_points(saved, selected)
    layer_scores = selected[['case_id', 'seed', 'axis', 'layer', 'ari']]
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for suffix, axes in (('', AXES), ('_rows', ('row',)), ('_columns', ('column',))):
        figure = figure_for_axes(layer_scores, points, axes)
        try:
            for extension in ('pdf', 'png'):
                target = output.parent / f'{output.name}{suffix}.{extension}'
                figure.savefig(target, dpi=300,
                               metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
                artifacts.append(target)
        finally:
            plt.close(figure)
    for suffix, frame in (('layer_ari_by_matrix', selected), ('layer_ari', layer_scores), ('points', points)):
        target = output.parent / f'{output.name}_{suffix}.csv'
        frame.to_csv(target, index=False)
        artifacts.append(target)
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Source changed during rendering: {path}')
    manifest = dict(case_id=CASE_ID, noise=.25, observed_fraction=.4, seed=0, matrices=1,
                    axes=list(AXES), layers=list(LAYERS), curve_layers=list(range(9)),
                    representation='Mean pooling across the opposite matrix axis',
                    curve='Fixed-example full-space K-means ARI; one matrix, no averaging across matrices or confidence intervals',
                    scatter='Original independent label-blind PCA; all 64 objects per axis and layer',
                    scatter_ari='Upper-left inset: same single-matrix full-space ARI as the curve, not PCA-space clustering',
                    layout='Equal square panels; single-axis 5.5 x 1.35 inch canvas matches the latent-factor grid',
                    scatter_axes='PC1 horizontal; shared PC2 vertical label; sparse numeric ticks in each panel',
                    selection='Unchanged seed-zero example; no new inference, fitting, jitter, or point removal',
                    source_sha256=source_hashes, script_sha256=digest(__file__),
                    artifacts_sha256={str(path.resolve()): digest(path) for path in artifacts})
    output.with_name(f'{output.name}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved combined, row, and column filmstrips to {output.parent}; '
          f'{len(points)} archived points, {len(selected)} matrix-layer scores.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output path stem, without extension')
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == '__main__':
    main()
