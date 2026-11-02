from types import SimpleNamespace

import numpy as np
import pytest

from evaluate_single_matrixformer_orientation import (
    aligned_completion, combine, interface_normalization, predict_llm, predict_tabular, rating_queries,
)
from tabimpute.interface import ImputePFN


class Model:
    def __init__(self):
        self.inputs = []
        self.preprocessors = None

    def get_imputation(self, values, num_repeats=1):
        self.inputs.append(values.copy())
        completed = np.nan_to_num(values, nan=0.25)
        return completed, completed.copy()

    impute = ImputePFN.impute


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_manual_normalization_matches_original_interface(dtype):
    observed = np.array([[1, 1000, 3, np.nan], [3, np.nan, 3, np.nan], [np.nan, 3000, 3, np.nan]], dtype=dtype)
    model = Model()
    with pytest.warns(RuntimeWarning):
        expected = model.impute(observed.copy())
    actual = predict_tabular(model, observed, 'missbench', False)
    np.testing.assert_array_equal(actual[np.isnan(observed)], expected[np.isnan(observed)])
    np.testing.assert_array_equal(actual[~np.isnan(observed)], observed[~np.isnan(observed)])


def test_transpose_preserves_original_feature_normalization():
    observed = np.array([[1., 1000.], [3., np.nan], [np.nan, 3000.]])
    normalized, means, scales = interface_normalization(observed)
    model = Model()
    prediction = aligned_completion(model, normalized, means, scales, transpose=True)
    np.testing.assert_array_equal(model.inputs[0], normalized.T)
    np.testing.assert_allclose(prediction[1, 1], 2250.)
    np.testing.assert_allclose(prediction[2, 0], 2.25)
    assert model.max_num_rows is None and model.max_num_cols is None


def test_tabular_matches_original_c_order_reduction_on_fortran_arrays():
    generator = np.random.default_rng(42)
    observed = np.asfortranarray(generator.normal(size=(150, 5)).astype(np.float32))
    observed[generator.random(observed.shape) < 0.4] = np.nan
    model = Model()
    expected = model.impute(observed.copy())
    actual = predict_tabular(model, observed, 'missbench', False)
    np.testing.assert_array_equal(actual[np.isnan(observed)], expected[np.isnan(observed)])


def test_uci_windows_keep_training_axis_limits_after_transpose():
    normalized = np.zeros((2050, 3))
    model = Model()
    aligned_completion(model, normalized, np.zeros(3), np.ones(3), transpose=True, windowed=True)
    assert model.max_num_rows is None and model.max_num_cols == 1000


def test_average_is_predictions_not_errors_and_rejects_invalid_arrays():
    predictions = combine(np.array([-1., 4.]), np.array([1., 2.]))
    np.testing.assert_array_equal(predictions['average'], [0, 3])
    with pytest.raises(ValueError):
        combine(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        combine(np.zeros(2), np.array([np.nan, 0]))


def test_llm_preserves_batches_and_never_observes_queries():
    context = np.array([[1., 1000.], [3., 3000.]])
    visible = np.full((401, 2), np.nan)
    visible[:, 0] = 2.
    mask = np.isnan(visible)
    model = Model()
    actual = predict_llm(model, context, visible, mask, np.zeros(2), np.ones(2), transpose=True)
    np.testing.assert_array_equal(actual, np.full(401, 2250.))
    assert [values.shape for values in model.inputs] == [(2, 202), (2, 202), (2, 3)]
    for values in model.inputs:
        assert np.isnan(values[1, 2:]).all()


def test_transposed_rating_indices_normalization_and_overlap():
    users = np.array([0, 999, 1050, 2049])
    items = np.array([0, 1, 2, 1])
    observed = np.ones((2050, 3))
    observed[users, items] = np.nan
    normalizer = {'means': np.array([2., 3., 4.]), 'scales': np.array([0.5, 1., 2.]),
                  'offsets': np.full(2050, 0.1), 'row_scales': np.full(2050, 2.)}
    model = Model()
    actual, counts = rating_queries(model, observed, users, items, normalizer, (0, 10), True)
    expected = 0.25 * 2 * normalizer['scales'][items] + normalizer['means'][items] + 0.1
    np.testing.assert_allclose(actual, expected)
    assert (counts > 1).any()
    assert all(values.shape[0] == 3 for values in model.inputs)


def test_transposed_rating_resume_is_hash_checked(tmp_path):
    model = Model()
    observed = np.array([[1., np.nan], [np.nan, 3.]])
    users, items = np.array([0, 1]), np.array([1, 0])
    normalizer = {'means': np.zeros(2), 'scales': np.ones(2), 'offsets': np.zeros(2), 'row_scales': np.ones(2)}
    expected, counts = rating_queries(model, observed, users, items, normalizer, (0, 5), True, tmp_path, 'frozen')
    never_called = SimpleNamespace(impute=lambda *args, **kwargs: pytest.fail('Completed windows rerun'))
    actual, resumed = rating_queries(never_called, observed, users, items, normalizer, (0, 5), True, tmp_path, 'frozen')
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(resumed, counts)
    with pytest.raises(ValueError, match='identity'):
        rating_queries(model, observed, users, items, normalizer, (0, 5), True, tmp_path, 'changed')
