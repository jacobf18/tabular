import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch

import synthetic_cluster_representations as analysis


def test_profile_preserves_full_cell_distances():
    grid = torch.randn(8, 6, 12, generator=torch.Generator().manual_seed(19))
    for axis in ('row', 'column'):
        full = grid if axis == 'row' else grid.transpose(0, 1)
        full = full.reshape(full.shape[0], -1).numpy()
        compact = analysis.compact_profile(grid, axis)
        np.testing.assert_allclose(cdist(compact, compact), cdist(full, full), atol=2e-6)


def test_pooling_excludes_special_tokens_and_preserves_input():
    grid = torch.full((1, 6, 7, 4), 10000.)
    grid[:, 2:, 3:, :] = torch.arange(64).reshape(4, 4, 4).float()
    original = grid.clone()
    result = analysis.pool_grid(grid, 2, 3)
    np.testing.assert_allclose(result['mean_row'], grid[0, 2:, 3:].mean(1))
    np.testing.assert_allclose(result['mean_column'], grid[0, 2:, 3:].mean(0))
    assert result['mean_row'].shape == (4, 4)
    torch.testing.assert_close(grid, original)


def test_pooling_opposite_axis_permutation_preserves_distances():
    grid = torch.randn(1, 7, 8, 4, generator=torch.Generator().manual_seed(18))
    permuted = grid[:, :, [2, 5, 1, 6, 3, 0, 7, 4]]
    first, second = analysis.pool_grid(grid, 0, 0), analysis.pool_grid(permuted, 0, 0)
    for pooling in analysis.POOLINGS:
        np.testing.assert_allclose(cdist(first[f'{pooling}_row'], first[f'{pooling}_row']),
                                   cdist(second[f'{pooling}_row'], second[f'{pooling}_row']), atol=2e-6)


def test_clustering_label_blind_and_label_permutation_invariant():
    generator = np.random.default_rng(4)
    labels = np.repeat(np.arange(4), 16)
    centers = generator.normal(size=(4, 8)) * 10
    features = centers[labels] + generator.normal(size=(64, 8)) * .01
    assignments, projection, variance = analysis.fit_clusters(features)
    metric = analysis.cluster_metrics(features, assignments, labels)
    permuted = analysis.cluster_metrics(features, assignments, np.array([3, 0, 2, 1])[labels])
    assert metric['ari'] == 1
    assert metric['knn5_purity'] == 1
    assert abs(metric['shuffled_ari']) < .03
    assert metric == permuted
    np.testing.assert_array_equal(assignments, analysis.fit_clusters(features)[0])
    assert projection.shape == (64, 2) and variance.shape == (2,)


def test_hidden_values_only_enter_privileged_baseline():
    context = np.array([[1., np.nan], [2., 8.]])
    prediction = np.nan_to_num(context)
    first = analysis.baseline_features(context, prediction, np.ones((2, 2)), np.zeros(2), np.ones(2))
    second = analysis.baseline_features(context, prediction, np.full((2, 2), 999.), np.zeros(2), np.ones(2))
    for key in first:
        if key.startswith('oracle_complete'):
            assert not np.array_equal(first[key], second[key])
        else:
            np.testing.assert_array_equal(first[key], second[key])


def test_invalid_features_rejected():
    with pytest.raises(ValueError, match='finite'):
        analysis.fit_clusters(np.array([[np.nan]]))
    with pytest.raises(ValueError, match='single'):
        analysis.pool_grid(torch.zeros(2, 4, 4, 4), 0, 0)


def test_probabilities_match_native_float32_capture_with_rope():
    from tabimpute.model.model_new_stable import AxialRoPESelfAttention
    module = AxialRoPESelfAttention(embedding_size=16, nhead=4, rope_fraction=.5).eval()
    source = torch.randn(3, 9, 16, generator=torch.Generator().manual_seed(23))
    projected = {}

    def save_query(layer, arguments, output):
        projected['query'] = output.detach()

    def save_key(layer, arguments, output):
        projected['key'] = output.detach()

    handles = [module.q_proj.register_forward_hook(save_query), module.k_proj.register_forward_hook(save_key)]
    try:
        module.set_attention_capture(True)
        with torch.inference_mode():
            module(source, rope_start_index=2)
            probabilities = analysis.attention_probabilities(module, projected['query'], projected['key'], 2)
        torch.testing.assert_close(probabilities, module.last_attention_weights, atol=1e-7, rtol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


def test_hooks_removed_after_failure():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = torch.nn.Identity()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])

    class FakePredictor:
        def __init__(self):
            self.imputer = type('Imputer', (), {'model': FakeModel()})()

        def predict(self, *args):
            raise RuntimeError('Deliberate failure')

    predictor = FakePredictor()
    with pytest.raises(RuntimeError, match='Deliberate'):
        analysis.capture(predictor, np.zeros((4, 4)), np.zeros(4), np.ones(4))
    assert not predictor.imputer.model.feature_encoder._forward_hooks
    assert not predictor.imputer.model.transformer_blocks[0]._forward_hooks


def test_figure_interval_averages_axes_within_matrix(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from plot_synthetic_paper_figures import noise_recovery

    records = []
    for seed in range(10):
        for noise in (0., .25, 1.):
            for representation, layer in [('mean', 8), ('profile', 8), ('observed', -1)]:
                for direction, score in [('row', 0.), ('column', 1.)]:
                    records.append(dict(case_id=f'{noise}_{seed}', seed=seed, noise=noise,
                                        observed_fraction=.4, representation=representation,
                                        layer=layer, axis=direction, ari=score))
    style = SimpleNamespace(WIDTH=5.5, BLUE='#2A6FBB', style_axes=lambda *args, **kwargs: None,
                            save=lambda *args: None)
    noise_recovery(pd.DataFrame(records), tmp_path, style)
    summary = pd.read_csv(tmp_path / 'cluster_noise_summary.csv')
    assert (summary.matrices == 10).all()
    np.testing.assert_allclose(summary.ari, .5)
    np.testing.assert_allclose(summary.ari_ci95_half, 0)


def solver_summary_fixture():
    import pandas as pd
    from plot_synthetic_paper_figures import SOLVERS
    records = []
    conditions = [('low_rank', rank, 0.) for rank in (2, 4, 8, 16, 32)]
    conditions += [('clusters', 4, noise) for noise in (0., .25, 1.)]
    for family, rank, noise in conditions:
        for rate in (.2, .4, .6, .8):
            for method in SOLVERS:
                records.append(dict(family=family, rank=rank, noise=noise, observed_fraction=rate,
                                    method=method, matrices=10, rmse=rank / 100 + noise + rate, rmse_ci95_half=.01))
    return pd.DataFrame(records)


def test_solver_comparison_retains_complete_sweeps_and_original_scores():
    from plot_synthetic_paper_figures import solver_comparisons
    frame = solver_summary_fixture()
    selected = solver_comparisons(frame)
    assert len(selected) == 128
    assert (selected.groupby('method').size() == 32).all()
    np.testing.assert_array_equal(selected.rmse, frame.rmse)
    np.testing.assert_array_equal(selected.rmse_ci95_half, frame.rmse_ci95_half)


def test_solver_comparison_rejects_removed_unfavorable_condition():
    from plot_synthetic_paper_figures import solver_comparisons
    frame = solver_summary_fixture()
    with pytest.raises(ValueError, match='every registered'):
        solver_comparisons(frame.iloc[1:])
    frame.loc[0, 'matrices'] = 9
    with pytest.raises(ValueError, match='ten finite'):
        solver_comparisons(frame)


def neighbor_summary_fixture():
    import pandas as pd
    from plot_synthetic_paper_figures import NEIGHBOR_NAMES
    return pd.DataFrame([dict(family='clusters', noise=noise, observed_fraction=rate,
                              method=method, matrices=10, rmse=rate + noise, rmse_ci95_half=.01)
                         for noise in (0., .25, 1.) for rate in (.2, .4, .6, .8)
                         for method in NEIGHBOR_NAMES])


def test_neighbor_comparison_retains_both_settings_and_original_intervals():
    from plot_synthetic_paper_figures import neighbor_comparisons
    frame = neighbor_summary_fixture()
    selected = neighbor_comparisons(frame)
    assert len(selected) == 8
    assert (selected.groupby('method').size() == 2).all()
    assert set(selected.observed_fraction) == {.2, .4}
    np.testing.assert_array_equal(selected.rmse, frame.loc[selected.index, 'rmse'])
    np.testing.assert_array_equal(selected.rmse_ci95_half, frame.loc[selected.index, 'rmse_ci95_half'])


def test_neighbor_comparison_rejects_missing_duplicate_or_invalid_results():
    import pandas as pd
    from plot_synthetic_paper_figures import neighbor_comparisons
    selected = neighbor_comparisons(neighbor_summary_fixture())
    for frame in (selected.iloc[1:], pd.concat([selected, selected.iloc[:1]])):
        with pytest.raises(ValueError, match='every method'):
            neighbor_comparisons(frame)
    for column, value in [('matrices', 9), ('rmse', np.nan), ('rmse_ci95_half', -.01)]:
        frame = selected.copy()
        frame.loc[frame.index[0], column] = value
        with pytest.raises(ValueError, match='ten finite'):
            neighbor_comparisons(frame)


def test_embedding_links_match_same_fixed_rows_without_changing_coordinates():
    from plot_synthetic_paper_figures import embedding_links
    generator = np.random.default_rng(41)
    before, after = generator.normal(size=(64, 2)), generator.normal(size=(64, 2))
    original_before, original_after = before.copy(), after.copy()
    rows, starts, ends = embedding_links({'mean_row_0_pca': before, 'mean_row_8_pca': after})
    np.testing.assert_array_equal(rows, np.arange(8))
    np.testing.assert_array_equal(starts, before[:8])
    np.testing.assert_array_equal(ends, after[:8])
    np.testing.assert_array_equal(before, original_before)
    np.testing.assert_array_equal(after, original_after)
    starts[:] = 99
    np.testing.assert_array_equal(before, original_before)


def test_embedding_links_reject_mismatched_or_nonfinite_views():
    from plot_synthetic_paper_figures import embedding_links
    with pytest.raises(ValueError, match='same rows'):
        embedding_links({'mean_row_0_pca': np.zeros((64, 2)), 'mean_row_8_pca': np.zeros((63, 2))})
    with pytest.raises(ValueError, match='finite'):
        embedding_links({'mean_row_0_pca': np.full((64, 2), np.nan), 'mean_row_8_pca': np.zeros((64, 2))})
