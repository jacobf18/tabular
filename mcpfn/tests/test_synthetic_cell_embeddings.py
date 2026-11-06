import numpy as np
import pytest
import torch

import synthetic_cell_embeddings as analysis


def test_data_grid_and_slices_preserve_individual_cells():
    output = torch.arange(1 * 7 * 8 * 6).reshape(1, 7, 8, 6).float()
    original = output.clone()
    grid = analysis.data_grid(output, 2, 3)
    np.testing.assert_array_equal(grid, output[0, 2:, 3:])
    np.testing.assert_array_equal(analysis.slice_features(grid, 'row', 1), output[0, 2:, 4, :])
    np.testing.assert_array_equal(analysis.slice_features(grid, 'column', 1), output[0, 3, 3:, :])
    torch.testing.assert_close(output, original)


def test_representative_indices_are_first_in_original_order():
    labels = np.array([3, 1, 3, 0, 2, 0, 1, 2])
    assert analysis.predetermined_indices(labels) == [3, 1, 4, 0]


def test_clusters_use_features_not_group_labels():
    labels = np.repeat(np.arange(4), 16)
    generator = np.random.default_rng(42)
    features = np.eye(4)[labels] * 10 + generator.normal(size=(64, 4)) * .01
    assignments = analysis.fit_slice(features)
    observed = np.arange(64) % 2 == 0
    first = analysis.score_slice(assignments, labels, observed)
    second = analysis.score_slice(assignments, np.array([3, 0, 2, 1])[labels], observed)
    assert first == second
    assert all(record['ari'] == 1. for record in first)
    assert [record['cells'] for record in first] == [64, 32]


def test_masked_score_uses_same_assignments_and_excludes_degenerate_subsets():
    labels = np.array([0, 0, 1, 1])
    scores = analysis.score_slice(np.array([0, 0, 1, 1]), labels, np.array([False, False, True, True]))
    assert scores[0]['ari'] == 1.
    assert not scores[1]['eligible'] and np.isnan(scores[1]['ari'])


def test_failed_capture_removes_hooks():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = torch.nn.Identity()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)])

    class Predictor:
        def __init__(self):
            self.imputer = type('Imputer', (), {'model': Model()})()

        def predict(self, *arguments):
            raise RuntimeError('Deliberate failure')

    predictor = Predictor()
    with pytest.raises(RuntimeError, match='Deliberate'):
        analysis.capture(predictor, np.zeros((4, 4)), np.zeros(4), np.ones(4))
    assert not predictor.imputer.model.feature_encoder._forward_hooks
    assert not predictor.imputer.model.transformer_blocks[7]._forward_hooks


def test_invalid_features_rejected():
    with pytest.raises(ValueError, match='finite'):
        analysis.fit_slice(np.full((4, 5), np.nan))


def test_summary_uses_matrix_means_not_cells_or_slices_as_replicates():
    import pandas as pd
    from run_synthetic_cell_embeddings import summarize

    records = []
    for seed in range(2):
        for layer in (0, 8):
            for context_index in range(64):
                records.append(dict(case_id=f'case-{seed}', noise=.25, observed_fraction=.4,
                                    seed=seed, axis='row', layer=layer, scope='all',
                                    ari=.1 * seed + .2 * (layer == 8) + context_index / 1000,
                                    cells=1 + context_index))
    per_matrix, summary, paired = summarize(pd.DataFrame(records))
    assert len(per_matrix) == 4 and (per_matrix.slices == 64).all()
    assert (summary.matrices == 2).all()
    np.testing.assert_allclose(summary.sort_values('layer')['mean'], [.0815, .2815])
    np.testing.assert_allclose(paired.difference, .2)
