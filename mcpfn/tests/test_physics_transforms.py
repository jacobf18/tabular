import numpy as np
import pytest

import evaluate_physics_transforms as experiment


def test_row_transform_is_observed_only_and_invertible():
    matrix = np.array([[1., np.nan, 3.], [4., 4., np.nan], [np.nan, np.nan, np.nan]])
    for scale_rows in (False, True):
        transformed, means, scales = experiment.row_transform(matrix, scale_rows)
        np.testing.assert_allclose(transformed * scales + means, matrix)
        assert np.isfinite(means).all() and (scales > 0).all()
        assert means[0, 0] == 2.


def test_lag_transform_preserves_masked_cells_in_all_copies():
    matrix = np.arange(21, dtype=float).reshape(3, 7)
    matrix[1, 3] = np.nan
    restored = experiment.undo_lags(experiment.lag_transform(matrix), 7)
    np.testing.assert_allclose(restored, matrix)
    assert np.isnan(restored[1, 3])


def test_input_transforms_round_trip_with_identity_predictor():
    matrix = np.arange(40, dtype=float).reshape(5, 8)
    for transform in experiment.FLOW_TRANSFORMS:
        np.testing.assert_allclose(experiment.predict_flow(matrix, transform, lambda values: values), matrix)


def test_low_rank_refinement_preserves_observations_and_recovers_rank_one():
    truth = np.outer(np.arange(1., 8.), np.arange(1., 6.))
    matrix = truth.copy()
    matrix[1, 3] = np.nan
    result = experiment.low_rank_refine(matrix, truth, 1)
    np.testing.assert_array_equal(result[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    np.testing.assert_allclose(result, truth, atol=1e-10)


def test_quantum_unfoldings_are_inverted_and_identity_is_not_input():
    matrix = np.arange(256, dtype=float).reshape(16, 16)
    inputs = []

    def predict(values):
        inputs.append(values.copy())
        return np.nan_to_num(values)

    result = experiment.predict_quantum(matrix, 'identity_free_unfoldings', predict)
    np.testing.assert_array_equal(result, matrix)
    assert len(inputs) == 3 and all(np.isnan(values.flat[0]) for values in inputs)


def test_physical_refinement_zero_strength_ignores_neural_prior():
    _, operators = experiment.quantum.pauli_basis(2)
    matrix = np.zeros((4, 4))
    matrix.flat[0] = 1.
    matrix.flat[1::2] = np.nan
    first, diagnostics = experiment.physical_refine(matrix, np.zeros_like(matrix), 0., operators)
    second, _ = experiment.physical_refine(matrix, np.ones_like(matrix), 0., operators)
    np.testing.assert_allclose(first, second, atol=1e-10)
    assert np.linalg.eigvalsh(first).min() >= -1e-10
    assert np.trace(first) == pytest.approx(1.)
    assert diagnostics['status'] in ('optimal', 'optimal_inaccurate')


def test_validation_score_does_not_read_original_missing_targets():
    matrix = np.array([[1., np.nan], [2., 3.]])
    validation = np.array([[False, False], [True, False]])
    first = np.array([[1., 100.], [2.5, 3.]])
    second = first.copy()
    second[0, 1] = -1000.
    assert experiment.validation_error(first, matrix, validation) == experiment.validation_error(second, matrix, validation) == .5
