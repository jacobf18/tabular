import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from run_row_group_embedding_example import CASE, construction_figure, generate, score_features
import run_row_group_embedding_example as probe
from run_matrixformer_synthetic_study import Predictor


def test_predictor_uses_supplied_checkpoint_and_adjacent_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'final.pth'
    config = {'model': {'example_setting': 8}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    received = {}

    def imputer_factory(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))

    monkeypatch.setitem(sys.modules, 'tabimpute.interface', SimpleNamespace(ImputePFN=imputer_factory))
    Predictor(device='cpu', checkpoint=checkpoint)
    assert received['checkpoint_path'] == str(checkpoint)
    assert received['json_config'] == {'config': config['model']}
    assert received['prediction_stat'] == 'mean'


def test_probe_rejects_wrong_checkpoint_hash_before_creating_output(tmp_path):
    checkpoint = tmp_path / 'final.pth'
    checkpoint.write_bytes(b'checkpoint fixture')
    output = tmp_path / 'probe'
    with pytest.raises(ValueError, match='checkpoint hash mismatch'):
        probe.run(output, 'cpu', checkpoint=checkpoint, checkpoint_sha256='incorrect')
    assert not output.exists()


def test_probe_records_and_uses_supplied_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'final.pth'
    checkpoint.write_bytes(b'checkpoint fixture')
    (tmp_path / 'config.json').write_text('{}')
    checkpoint_hash = probe.digest(checkpoint)
    data = generate()
    calls = {}

    def make_predictor(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(predict=lambda *args: data['truth'].copy(), check_unchanged=lambda: None)

    features = {f'{representation}_row_{layer}': data['signal']
                for representation in ('mean', 'profile') for layer in range(9)}
    monkeypatch.setattr(probe, 'Predictor', make_predictor)
    monkeypatch.setattr(probe, 'source_paths', lambda: [])
    monkeypatch.setattr(probe.analysis, 'capture', lambda *args: (data['truth'].copy(), features, {}))
    monkeypatch.setattr(probe, 'render', lambda *args: None)
    output = tmp_path / 'probe'
    probe.run(output, 'cpu', checkpoint=checkpoint, checkpoint_sha256=checkpoint_hash)
    assert calls == {'device': 'cpu', 'checkpoint': checkpoint.resolve()}
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['checkpoint'] == str(checkpoint.resolve())
    assert protocol['checkpoint_sha256'] == checkpoint_hash
    assert protocol['case'] == CASE
    assert str(tmp_path / 'config.json') in protocol['source_sha256']
    assert json.loads((output / 'manifest.json').read_text())['checkpoint_unchanged']


def test_generator_has_four_row_profiles_and_no_column_groups():
    data = generate()
    assert data['profiles'].shape == (4, 64)
    assert data['truth'].shape == (64, 64)
    np.testing.assert_array_equal(np.bincount(data['row_labels']), [16] * 4)
    np.testing.assert_array_equal(data['signal'], data['profiles'][data['row_labels']])
    assert np.unique(data['signal'], axis=0).shape[0] == 4
    assert np.unique(data['profiles'], axis=1).shape[1] == 64
    assert 'column_labels' not in data
    np.testing.assert_array_equal(data['truth'], (data['signal'] + .25 * data['noise']).astype(np.float32))
    np.testing.assert_array_equal(np.isnan(data['context']), ~data['observed'])
    np.testing.assert_array_equal(data['context'][data['observed']], data['truth'][data['observed']])
    assert data['observed'].sum() == 1638
    assert data['observed'].any(0).all() and data['observed'].any(1).all()
    for key, value in data.items():
        np.testing.assert_array_equal(value, generate()[key])


def test_clustering_and_plotted_points_use_only_the_fixed_row_groups():
    data = generate()
    features = {f'{representation}_row_{layer}': data['signal']
                for representation in ('mean', 'profile') for layer in range(9)}
    scores, assignments, points = score_features(features, data['row_labels'])
    assert len(scores) == 18
    assert scores.seed.eq(0).all()
    assert scores.ari.eq(1).all()
    assert scores.case_id.eq(CASE['case_id']).all()
    assert scores.axis.eq('row').all()
    assert len(points) == 320
    for layer in (0, 2, 4, 6, 8):
        selected = points[points.layer == layer]
        np.testing.assert_array_equal(selected.true_group, data['row_labels'])
        np.testing.assert_array_equal(selected[['pc1', 'pc2']], assignments[f'mean_row_{layer}_pca'])
        assert selected.full_space_ari.eq(1).all()
    with pytest.raises(ValueError, match='64 rows'):
        score_features({'mean_row_0': data['signal'][:16]}, data['row_labels'])


def test_construction_uses_the_actual_matrix_and_mask():
    data = generate()
    figure = construction_figure(data)
    try:
        panels = figure.axes[:3]
        assert panels[0].get_title() == 'Four row groups'
        np.testing.assert_array_equal(panels[0].images[0].get_array(), data['profiles'])
        order = np.argsort(data['row_labels'], kind='stable')
        np.testing.assert_array_equal(panels[1].images[0].get_array(), data['truth'][order])
        context = panels[2].images[0].get_array()
        np.testing.assert_array_equal(context.mask, ~data['observed'])
        np.testing.assert_array_equal(context.compressed(), data['context'][data['observed']])
        assert len([artist for artist in figure.artists if isinstance(artist, FancyArrowPatch)]) == 2
        labels = [text.get_text() for text in figure.texts]
        assert 'repeat rows\n+ noise' in labels
        assert 'shuffle rows\n+ mask' in labels
        first_panel = panels[0].get_position()
        color_axis = figure.axes[3]
        color_bounds = color_axis.get_position()
        assert first_panel.x0 < color_bounds.x0 < color_bounds.x1 < first_panel.x1
        assert color_bounds.width == pytest.approx(first_panel.width * .65)
        assert color_bounds.height == pytest.approx(.016)
        assert color_bounds.y1 < first_panel.y0
        assert color_axis.get_xlabel() == 'Value'
        assert len({panel.images[0].get_clim() for panel in panels}) == 1
    finally:
        plt.close(figure)
