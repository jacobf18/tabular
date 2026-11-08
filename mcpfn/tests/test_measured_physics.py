import numpy as np
import pytest

import evaluate_measured_physics as experiment


@pytest.mark.parametrize('transform', experiment.TRANSFORMS)
def test_transforms_roundtrip_observations(transform):
    matrix = np.arange(1., 31.).reshape(6, 5)
    matrix[2, 3] = np.nan
    normalized, inverse = experiment.transform_observed(matrix, transform)
    np.testing.assert_allclose(inverse(normalized), matrix, equal_nan=True)


def test_nuclear_prox_and_batched_complex():
    diagonal = np.diag([4., 2., .5])
    np.testing.assert_allclose(experiment.nuclear_prox(diagonal, 1), np.diag([3., 1., 0.]))
    batch = np.stack([diagonal.astype(complex), 1j * diagonal])
    expected = np.stack([np.diag([3., 1., 0.]), 1j * np.diag([3., 1., 0.])])
    np.testing.assert_allclose(experiment.nuclear_prox(batch, 1), expected)


@pytest.mark.parametrize('method', ['SoftImpute', 'HaLRTC', 'TNN'])
def test_completion_preserves_observed_and_recovers_constant(method):
    random = np.random.default_rng(99)
    cube = np.ones((8, 9, 10)) * 2
    missing = random.random(cube.shape) < .3
    values = np.where(missing, np.nan, cube)
    if method == 'SoftImpute':
        filled, details = experiment.soft_complete(values.reshape(-1, 10), .001)
        filled = filled.reshape(cube.shape)
    else:
        filled, details = experiment.tensor_complete(values, method, .1)
    np.testing.assert_allclose(filled[~missing], cube[~missing])
    assert np.sqrt(np.mean((filled[missing] - cube[missing]) ** 2)) < .02


def test_select_hides_validation_before_every_trial(monkeypatch):
    matrix = np.arange(1., 31.).reshape(6, 5)
    matrix[2, 2] = np.nan
    validation = np.zeros_like(matrix, dtype=bool)
    validation[1, 1] = True
    calls = []

    def prediction(values, shape, method, setting, models, distribution):
        calls.append(values.copy())
        return np.nan_to_num(values, nan=7.), {}

    monkeypatch.setattr(experiment, 'predict', prediction)
    experiment.select(matrix, validation, (6, 5), 'MatrixFormer', {}, None)
    assert len(calls) == len(experiment.TRANSFORMS) + 1
    assert all(np.isnan(values[validation]).all() for values in calls[:-1])
    assert np.isfinite(calls[-1][validation]).all()
    assert all(np.isnan(values[2, 2]) for values in calls)
