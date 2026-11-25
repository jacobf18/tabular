import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot_synthetic_embedding_filmstrip import (
    AXES, CASE_ID, COLORS, LAYERS, MARKERS, digest, example_ari, example_points, figure_for_axes, render,
)


@pytest.fixture
def archived_data():
    generator = np.random.default_rng(17)
    labels = np.repeat(np.arange(4), 16)
    saved = {f'{axis}_labels': labels for axis in AXES}
    records = []
    for axis in AXES:
        for layer in range(9):
            key = f'mean_{axis}_{layer}'
            saved[key] = labels.copy()
            saved[f'{key}_pca'] = generator.normal(size=(64, 2))
            for seed in range(10):
                records.append(dict(case_id=CASE_ID if seed == 0 else f'case-{seed}', seed=seed,
                                    representation='mean', noise=.25, observed_fraction=.4,
                                    axis=axis, layer=layer, ari=1 - seed / 20))
    return saved, pd.DataFrame(records)


def test_curve_uses_only_the_fixed_matrix_without_intervals(archived_data):
    saved, scores = archived_data
    selected = example_ari(saved, scores)
    assert len(selected) == 18
    assert selected.case_id.eq(CASE_ID).all()
    assert selected.seed.eq(0).all()
    assert selected.ari.eq(1).all()
    assert 'ari_ci95_half' not in selected
    pd.testing.assert_frame_equal(selected, example_ari(saved, scores[scores.case_id == CASE_ID]))


def test_other_matrices_noise_levels_and_representations_do_not_change_curve(archived_data):
    saved, scores = archived_data
    expected = example_ari(saved, scores)
    scores.loc[scores.seed != 0, 'ari'] = np.nan
    other_noise = scores.assign(noise=1., ari=0.)
    other_rate = scores.assign(observed_fraction=.8, ari=0.)
    other_representation = scores.assign(representation='profile', ari=0.)
    actual = example_ari(saved, pd.concat([scores, other_noise, other_rate, other_representation]))
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('problem', ['duplicate', 'missing_matrix', 'missing_layer', 'nan', 'out_of_range',
                                     'wrong_seed', 'wrong_noise', 'wrong_rate'])
def test_invalid_scores_are_rejected(archived_data, problem):
    saved, scores = archived_data
    if problem == 'duplicate':
        scores = pd.concat([scores, scores.iloc[:1]])
    elif problem == 'missing_matrix':
        scores = scores.iloc[1:]
    elif problem == 'missing_layer':
        scores = scores[scores.layer != 2]
    elif problem == 'nan':
        scores.loc[0, 'ari'] = np.nan
    elif problem == 'out_of_range':
        scores.loc[0, 'ari'] = 2.
    elif problem == 'wrong_seed':
        scores.loc[0, 'seed'] = 1
    elif problem == 'wrong_noise':
        scores.loc[0, 'noise'] = 1.
    else:
        scores.loc[0, 'observed_fraction'] = .8
    with pytest.raises(ValueError):
        example_ari(saved, scores)


def test_snapshots_preserve_all_archived_points_and_group_encodings(archived_data):
    saved, scores = archived_data
    points = example_points(saved, scores)
    assert len(points) == 640
    assert set(points.layer) == set(LAYERS)
    for (axis, layer), group in points.groupby(['axis', 'layer']):
        np.testing.assert_array_equal(group[['pc1', 'pc2']], saved[f'mean_{axis}_{layer}_pca'])
        np.testing.assert_array_equal(group.true_group, saved[f'{axis}_labels'])
        assert group.object_index.tolist() == list(range(64))
        assert group.color.tolist() == [COLORS[label] for label in saved[f'{axis}_labels']]
        assert group.marker.tolist() == [MARKERS[label] for label in saved[f'{axis}_labels']]
        assert group.full_space_ari.eq(1).all()


@pytest.mark.parametrize('layer', [1, 2, 8])
def test_saved_assignments_must_match_archived_ari(archived_data, layer):
    saved, scores = archived_data
    saved[f'mean_row_{layer}'] = np.zeros(64, dtype=int)
    with pytest.raises(AssertionError):
        example_points(saved, scores)


@pytest.mark.parametrize('axes', [AXES, ('row',), ('column',)])
def test_filmstrip_layout_and_plotted_points(archived_data, axes):
    saved, scores = archived_data
    selected = example_ari(saved, scores)
    points = example_points(saved, selected)
    figure = figure_for_axes(selected, points, axes)
    try:
        assert len(figure.axes) == 6 * len(axes)
        for index, axis_name in enumerate(axes):
            curve_axis, *scatter_axes = figure.axes[index * 6:(index + 1) * 6]
            np.testing.assert_array_equal(curve_axis.lines[0].get_xdata(), np.arange(9))
            np.testing.assert_array_equal(curve_axis.lines[0].get_ydata(), np.ones(9))
            assert len(curve_axis.lines) == 1
            assert not curve_axis.containers
            assert not curve_axis.collections
            for layer, panel in zip(LAYERS, scatter_axes):
                scatter_ari = points[(points.axis == axis_name) & (points.layer == layer)].full_space_ari.iloc[0]
                assert curve_axis.lines[0].get_ydata()[layer] == scatter_ari
                assert panel.get_title() == f'Layer {layer}'
                assert panel.get_xlabel() == 'PC1'
                assert panel.get_ylabel() == ('PC2' if layer == LAYERS[0] else '')
                assert len(panel.get_xticks()) >= 2
                assert len(panel.get_yticks()) >= 2
                assert all(label.get_visible() for label in panel.get_xticklabels())
                assert all(label.get_visible() for label in panel.get_yticklabels())
                assert len(panel.texts) == 1
                annotation = panel.texts[0]
                assert annotation.get_text() == 'ARI = 1.00'
                assert annotation.get_position() == (.06, .94)
                assert annotation.get_transform() == panel.transAxes
                renderer = figure.canvas.get_renderer()
                bounds = annotation.get_window_extent(renderer)
                assert panel.bbox.contains(bounds.x0, bounds.y0)
                assert panel.bbox.contains(bounds.x1, bounds.y1)
                for tick_label in panel.get_yticklabels():
                    tick_bounds = tick_label.get_window_extent(renderer)
                    assert tick_bounds.y0 > panel.bbox.y0
                    assert tick_bounds.y1 < bounds.y0
                plotted = panel.transData.transform(saved[f'mean_{axis_name}_{layer}_pca'])
                assert plotted[:, 1].max() < bounds.y0
                assert len(panel.collections) == 4
                assert sum(len(artist.get_offsets()) for artist in panel.collections) == 64
                for group, artist in enumerate(panel.collections):
                    expected = saved[f'mean_{axis_name}_{layer}_pca'][saved[f'{axis_name}_labels'] == group]
                    np.testing.assert_array_equal(artist.get_offsets(), expected)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('axis_name', AXES)
def test_single_axis_grid_matches_original_latent_factor_layout(archived_data, axis_name):
    saved, scores = archived_data
    selected = example_ari(saved, scores)
    points = example_points(saved, selected)
    figure = figure_for_axes(selected, points, (axis_name,))
    reference = plt.figure(figsize=(5.5, 1.35))
    grid = reference.add_gridspec(1, 7, width_ratios=[1, .45, 1, 1, 1, 1, 1],
                                  wspace=.10, left=.065, right=.995, top=.86, bottom=.22)
    try:
        for index, column in enumerate((0, 2, 3, 4, 5, 6)):
            panel = reference.add_subplot(grid[0, column])
            if index == 0:
                panel.set_xlim(-.4, 8.4)
                panel.set_ylim(-.03, 1.03)
                panel.set_aspect(8.8 / 1.06)
            else:
                panel.set_xlim(-3.3, 3.3)
                panel.set_ylim(-3.3, 3.3)
                panel.set_aspect('equal')
        reference.canvas.draw()
        np.testing.assert_allclose(figure.get_size_inches(), reference.get_size_inches())
        for actual, expected in zip(figure.axes, reference.axes):
            np.testing.assert_allclose(actual.get_position().bounds, expected.get_position().bounds,
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual.bbox.width, actual.bbox.height)
    finally:
        plt.close(figure)
        plt.close(reference)


def test_render_exports_auditable_unchanged_inputs(archived_data, tmp_path):
    saved, scores = archived_data
    source = tmp_path / 'source'
    source.mkdir()
    np.savez(source / 'fixed_example.npz', **saved)
    scores.to_csv(source / 'clustering_by_matrix.csv', index=False)
    hashes = {name: digest(source / name) for name in ('fixed_example.npz', 'clustering_by_matrix.csv')}
    (source / 'paper_export.json').write_text(json.dumps({'results': hashes}))
    output = tmp_path / 'output' / 'filmstrip'
    render(source, output)
    manifest = json.loads(output.with_name('filmstrip_manifest.json').read_text())
    assert manifest['layers'] == list(LAYERS)
    assert manifest['curve_layers'] == list(range(9))
    assert manifest['matrices'] == 1
    for path, expected in manifest['artifacts_sha256'].items():
        assert digest(path) == expected
    for name, expected in hashes.items():
        assert digest(source / name) == expected
    assert len(pd.read_csv(output.with_name('filmstrip_points.csv'))) == 640
    for suffix in ('layer_ari_by_matrix', 'layer_ari'):
        exported = pd.read_csv(output.with_name(f'filmstrip_{suffix}.csv'))
        assert len(exported) == 18
        assert exported.case_id.eq(CASE_ID).all()
        assert exported.seed.eq(0).all()
        assert exported.ari.eq(1).all()
        assert 'ari_ci95_half' not in exported
    (source / 'clustering_by_matrix.csv').write_text('changed')
    with pytest.raises(ValueError, match='archived hash'):
        render(source, output)
