import numpy as np
import pytest

from evaluate_matched_llm import gaussian_prediction, score_standardized, target_mask, visible_input


def test_visible_inputs_ignore_hidden_targets():
    context = np.array([[1., 2.], [3., 4.]])
    context_observed = np.ones_like(context, dtype=bool)
    targets = np.array([[5., 6.], [7., 8.]])
    observed = np.array([[True, True], [False, True]])
    before = visible_input(context, context_observed, targets, observed, [0])
    changed = targets.copy()
    changed[target_mask(observed, [0])] = 999.
    after = visible_input(context, context_observed, changed, observed, [0])
    for original, modified in zip(before, after):
        np.testing.assert_array_equal(original, modified)
    assert np.isnan(before[1][1]).all()


def test_mask_keeps_zero_context_rows_and_excludes_natural_missingness():
    observed = np.array([[True, True, False], [False, True, True]])
    np.testing.assert_array_equal(target_mask(observed, [0]), [[False, True, False], [False, True, True]])


def test_gaussian_empty_context_uses_training_mean():
    visible = np.array([[np.nan, np.nan], [4., np.nan]])
    prediction = gaussian_prediction(visible, np.array([[1., .5], [.5, 1.]]), np.array([2., 3.]), np.ones(2))
    np.testing.assert_array_equal(prediction[0], [0., 0.])
    assert prediction[1, 1] == pytest.approx(1. / 1.01)


def test_common_clipping_and_training_mean_denominator():
    metrics = score_standardized(np.array([20., 2.]), np.array([-20., 3.]))
    assert metrics == {"r2": 1. - 401. / 104., "rmse": np.sqrt(401. / 2), "n_eval": 2, "ss_res": 401., "ss_tot": 104.}
    assert score_standardized(np.zeros(2), np.zeros(2))["r2"] is None


def test_scoring_rejects_missing_predictions_and_shape_mismatch():
    with pytest.raises(ValueError):
        score_standardized(np.ones(2), np.array([1., np.nan]))
    with pytest.raises(ValueError):
        score_standardized(np.ones(2), np.ones((2, 1)))
