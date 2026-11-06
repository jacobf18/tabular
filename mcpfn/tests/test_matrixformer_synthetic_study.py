import numpy as np
import pytest

import synthetic_matrixformer_study as study


def example(family='clusters', **updates):
    case = next(case for case in study.protocol('smoke')['cases'] if case['family'] == family).copy()
    case.update(updates)
    return case


def test_fixed_protocol_inventory():
    cases = study.protocol()['cases']
    assert len(cases) == 360
    assert len({case['case_id'] for case in cases}) == 360
    assert sum(study.selected_attention_case(case) for case in cases) == 60
    assert len(study.protocol('smoke')['cases']) == 3


def test_generation_repeats_and_nested_masks_without_support_repair():
    case = example('low_rank')
    first = study.generate(case)
    for name, array in first.items():
        np.testing.assert_array_equal(array, study.generate(case)[name])
    second = study.generate({**case, 'observed_fraction': .8})
    np.testing.assert_array_equal(first['truth'], second['truth'])
    assert np.all(second['observed'][first['observed']])
    assert first['observed'].sum() == round(64 ** 2 * .4)
    assert np.linalg.matrix_rank(first['signal']) == 2


def test_noise_and_mask_streams_are_separate():
    first = study.generate(example(noise=0))
    second = study.generate(example(noise=1))
    np.testing.assert_array_equal(first['signal'], second['signal'])
    np.testing.assert_array_equal(first['observed'], second['observed'])
    assert not np.array_equal(first['truth'], second['truth'])
    assert np.any(np.diff(first['row_labels']) < 0)


def test_oracles_use_only_supplied_donors_not_hidden_truth():
    case = example(noise=0)
    data = study.generate(case)
    prediction = study.oracle_prediction(data['context'], data, 'clusters', block=True)
    np.testing.assert_allclose(prediction, data['truth'], atol=1e-6)
    contaminated = {**data, 'truth': np.full_like(data['truth'], 999), 'signal': np.full_like(data['signal'], -99)}
    for block in (True, False):
        np.testing.assert_array_equal(study.oracle_prediction(data['context'], data, 'clusters', block=block),
                                      study.oracle_prediction(data['context'], contaminated, 'clusters', block=block))
    with pytest.raises(ValueError, match='no oracle'):
        study.oracle_prediction(data['context'], data, 'noise')


def test_donor_filter_excludes_missing_targets_and_self():
    matrix = np.array([[1., np.nan], [2., 10.], [3., np.nan], [4., 30.]])
    prediction = study.donor_predict(matrix, labels=np.array([0, 0, 0, 1]))
    assert prediction[0, 1] == prediction[2, 1] == 10
    np.testing.assert_array_equal(prediction[np.isfinite(matrix)], matrix[np.isfinite(matrix)])


def test_scores_reject_corrupt_observations_and_keep_all_missing_cells():
    data = study.generate(example())
    prediction = np.nan_to_num(data['context'])
    assert study.scores(prediction, data)['test_cells'] == (~data['observed']).sum()
    prediction[data['observed']] += 1
    with pytest.raises(ValueError, match='Observed'):
        study.scores(prediction, data)


def test_attention_special_tokens_and_uniform_group_chance():
    data = study.generate(example())
    targets = np.argwhere(~data['observed'])[:3]
    weights = np.full((3, 2, 76), 1 / 76)
    for axis in ('row', 'column'):
        records = study.attention_statistics(weights, targets, data, axis)
        assert len(records) == 6
        for record in records:
            assert record['special_mass'] == pytest.approx(12 / 76)
            assert record['group_excess'] == pytest.approx(0)


def test_matched_random_removal_preserves_counts_and_position_bands():
    eligible = np.arange(20)
    selected = np.array([0, 4, 8])
    chosen = study.matched_random_donors(eligible, selected, 10, np.random.default_rng(0))
    assert len(set(chosen)) == 3
    np.testing.assert_array_equal(np.floor(np.log2(1 + np.abs(chosen - 10))),
                                  np.floor(np.log2(1 + np.abs(selected - 10))))


def test_uncentered_svd_recovers_a_well_supported_rank_two_matrix():
    from synthetic_uncentered_svd_control import complete
    data = study.generate(example('low_rank', observed_fraction=.8))
    prediction, iterations = complete(data['context'], rank=2)
    assert iterations <= 500
    assert study.scores(prediction, data)['rmse'] < 1e-4


def test_read_only_capture_matches_explicit_attention_without_changing_output():
    import torch
    from run_matrixformer_synthetic_study import Predictor
    from tabimpute.model.model_new_stable import TabImputeModel

    class Imputer:
        def __init__(self):
            self.model = TabImputeModel(embedding_size=16, num_attention_heads=4,
                                       mlp_hidden_size=32, num_layers=2, num_outputs=8, num_cls=2).eval()

        def impute(self, context, means=None, stds=None):
            values = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            return self.model(values).mean(-1).squeeze(0).detach().numpy()

    predictor = Predictor.__new__(Predictor)
    predictor.imputer = Imputer()
    context = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    context[1, 3] = context[5, 4] = np.nan
    targets = np.array([[1, 3], [5, 4]])
    expected = predictor.predict(context)
    actual, weights, maps = predictor.capture(context, targets, None, None)
    np.testing.assert_array_equal(actual, expected)
    assert len(weights) == len(maps) == 4
    predictor.imputer.model.set_attention_capture(True, store_weights=True)
    predictor.predict(context)
    diagnostics = predictor.imputer.model.get_attention_diagnostics()
    for layer in range(2):
        for query, (row, column) in enumerate(targets):
            np.testing.assert_allclose(weights[layer, 'row'][query],
                                       diagnostics[layer]['datapoints']['weights'][column + 2, :, row + 2, :], atol=1e-6)
            np.testing.assert_allclose(weights[layer, 'column'][query],
                                       diagnostics[layer]['features']['weights'][row + 2, :, column + 2, :], atol=1e-6)


def test_summary_interval_uses_independent_matrix_scores():
    import pandas as pd
    from scipy.stats import t as student_t
    from report_matrixformer_synthetic_study import summarize
    frame = pd.DataFrame({'method': ['example'] * 10, 'rmse': np.arange(10, dtype=float)})
    result = summarize(frame, ['method'], ['rmse']).iloc[0]
    assert result.matrices == 10
    assert result.rmse == 4.5
    assert result.rmse_ci95_half == pytest.approx(student_t.ppf(.975, 9) * np.std(np.arange(10), ddof=1) / np.sqrt(10))
