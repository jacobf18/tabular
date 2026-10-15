import numpy as np
import pytest

from diagnose_dwip_bpr_saved_errors import compare_predictions


def predictions(values):
    return {'users': np.array([0, 0, 0, 2]), 'items': np.array([0, 1, 2, 0]),
            'truth': np.array([3., 3., 3., 3.]), 'prediction': np.asarray(values, dtype=float)}


def test_saved_comparison_preserves_macro_user_weighting():
    before = predictions([4., 4., 4., 1.])
    after = predictions([3., 3., 3., 2.])
    result = compare_predictions(before, after)
    assert result['before'] == {'MAE': 1.5, 'RMSE': 1.5, 'users': 2}
    assert result['after'] == {'MAE': .5, 'RMSE': .5, 'users': 2}
    assert result['user_mean_squared_error_change'] == -2
    assert result['bias_squared_component_change'] == -2
    assert result['within_user_centered_error_component_change'] == 0
    assert result['macro_prediction_shift'] == 0
    assert result['macro_prediction_change_rms'] == 1
    assert result['users_with_improved_rmse_fraction'] == 1


def test_saved_decomposition_separates_bias_and_within_user_errors():
    result = compare_predictions(predictions([3., 3., 3., 3.]), predictions([4., 2., 3., 4.]))
    assert result['user_mean_squared_error_change'] == pytest.approx(5 / 6)
    assert result['bias_squared_component_change'] == .5
    assert result['within_user_centered_error_component_change'] == pytest.approx(1 / 3)
    assert result['macro_prediction_shift'] == .5


def test_identical_saved_predictions_have_zero_change_without_altering_inputs():
    before = predictions([4., 3., 2., 4.])
    original = {name: values.copy() for name, values in before.items()}
    result = compare_predictions(before, before)
    assert result['macro_prediction_shift'] == 0
    assert result['macro_prediction_change_rms'] == 0
    assert result['query_prediction_change_quantiles'] == [0] * 5
    assert all(np.array_equal(before[name], values) for name, values in original.items())


@pytest.mark.parametrize('name', ['users', 'items', 'truth'])
def test_changed_query_identities_or_targets_are_rejected(name):
    before, after = predictions([4., 3., 2., 4.]), predictions([4., 3., 2., 4.])
    after[name][0] += 1
    with pytest.raises(ValueError, match='identities or targets'):
        compare_predictions(before, after)


@pytest.mark.parametrize('name', ['prediction', 'truth'])
def test_nonfinite_saved_values_are_rejected(name):
    before = predictions([4., 3., 2., 4.])
    before[name][0] = np.nan
    with pytest.raises(ValueError):
        compare_predictions(before, before)


def test_misaligned_saved_vectors_are_rejected():
    before, after = predictions([4., 3., 2., 4.]), predictions([4., 3., 2.])
    with pytest.raises(ValueError, match='aligned'):
        compare_predictions(before, after)
