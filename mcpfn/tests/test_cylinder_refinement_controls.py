import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import evaluate_cylinder_refinement_controls as controls
import report_cylinder_refinement_controls as reporting


def test_soft_prediction_handles_only_unused_lag_boundary_columns(monkeypatch):
    matrix = np.array([[np.nan, 1., np.nan, 3.], [np.nan, 2., 4., np.nan]])
    seen = []

    def fit(values, method, parameter, seed):
        seen.append(values.copy())
        return np.nan_to_num(values, nan=5.)

    monkeypatch.setattr(controls.experiment.shared, 'fit_baseline', fit)
    result = controls.soft_prediction(matrix, .005, 0)
    assert result.shape == matrix.shape and len(seen) == 1
    assert seen[0].shape == (2, 3)
    np.testing.assert_array_equal(result[:, 0], 0.)
    np.testing.assert_array_equal(result[np.isfinite(matrix)], matrix[np.isfinite(matrix)])


def test_soft_selection_hides_validation_before_fitting_and_keeps_direct_shrinkage(monkeypatch):
    matrix = np.arange(30, dtype=float).reshape(6, 5)
    matrix[0, 1] = np.nan
    validation = np.zeros_like(matrix, dtype=bool)
    validation[1, 2] = True
    calls = []

    def predict(values, shrinkage, seed):
        calls.append(values.copy())
        return np.nan_to_num(values, nan=7.)

    monkeypatch.setattr(controls, 'soft_prediction', predict)
    monkeypatch.setattr(controls.experiment, 'FLOW_TRANSFORMS', ('global',))
    monkeypatch.setattr(controls.experiment, 'RANKS', (2,))
    predictions, decisions = controls.select_soft(matrix, validation, 0, .02)
    for values in calls[:len(controls.SHRINKAGES)]:
        assert np.isnan(values[validation]).all() and np.isnan(values[0, 1])
    assert decisions['selected']['SoftImpute + refinement']['shrinkage'] == .02
    assert all(np.isfinite(values).all() for values in predictions.values())


def test_intervals_use_mask_seed_means_not_seed_condition_pairs():
    frame = pd.DataFrame([dict(method='method', seed=seed, rate=rate, nrmse=seed + offset)
                          for seed in range(3) for rate, offset in ((.2, 1.), (.8, 2.))])
    summary = reporting.seed_intervals(frame, ['method']).iloc[0]
    assert summary['count'] == 3
    np.testing.assert_allclose(summary['mean'], 2.5)
    np.testing.assert_allclose(summary.ci95_half_width, student_t.ppf(.975, 2) / np.sqrt(3))
