import numpy as np
import pytest
from scipy.special import ndtri
import torch

import evaluate_cylinder_completion as evaluation


@pytest.mark.parametrize("pattern", ["mcar", "sensor_outage"])
@pytest.mark.parametrize("rate", [.2, .5, .8])
def test_masks_preserve_support_and_repeat(pattern, rate):
    missing = evaluation.make_mask((256, 151), rate, pattern, 42)
    np.testing.assert_array_equal(missing, evaluation.make_mask(missing.shape, rate, pattern, 42))
    assert not missing.all(axis=0).any()
    assert not missing.all(axis=1).any()
    assert abs(missing.mean() - rate) < .02
    if pattern == "sensor_outage":
        assert np.all(np.count_nonzero(missing != np.roll(missing, 1, axis=1), axis=1) == 2)


def test_normalization_only_uses_observations():
    observed = np.array([[1., np.nan], [3., 5.]])
    normalized, mean, scale = evaluation.normalize_observed(observed)
    assert mean == 3
    assert scale == pytest.approx(np.std([1., 3., 5.]))
    assert np.isnan(normalized[0, 1])


def test_full_support_quantile_tails_are_not_truncated():
    borders = torch.tensor([-2., -1., 1., 2.], dtype=torch.float64)
    levels = torch.tensor([.025, .25, .5, .75, .975], dtype=torch.float64)
    left = evaluation.full_support_quantiles(torch.tensor([[1., 0., 0.]], dtype=torch.float64), borders, levels)
    right = evaluation.full_support_quantiles(torch.tensor([[0., 0., 1.]], dtype=torch.float64), borders, levels)
    np.testing.assert_allclose(left.numpy()[0], -1 - ndtri(1 - levels.numpy() / 2) / .6744897501960817)
    np.testing.assert_allclose(right.numpy()[0], 1 + ndtri((1 + levels.numpy()) / 2) / .6744897501960817)
    assert left[0, 0] < borders[0]
    assert right[0, -1] > borders[-1]


def test_full_support_quantiles_uniform_interior():
    probabilities = torch.tensor([[.1, .8, .1]])
    borders = torch.tensor([-2., -1., 1., 2.])
    levels = torch.tensor([.1, .5, .9])
    result = evaluation.full_support_quantiles(probabilities, borders, levels)
    torch.testing.assert_close(result, torch.tensor([[-1., 0., 1.]]), atol=1e-6, rtol=1e-6)


def test_quantile_crps_matches_normal_integral():
    levels = (torch.arange(199, dtype=torch.float64) + .5) / 199
    quantiles = torch.from_numpy(ndtri(levels.numpy())).repeat(3, 1)
    target = torch.tensor([-2., 0., 2.], dtype=torch.float64)
    numerical = evaluation.quantile_crps(quantiles, target, levels)
    analytic = [evaluation.gaussian_metrics(np.array([value]), np.array([0.]), 1., 1.)["crps"] for value in target.numpy()]
    np.testing.assert_allclose(numerical.numpy(), analytic, atol=2e-4)


def test_point_scores_ignore_observations():
    truth = np.array([[1., 2.], [3., 4.]])
    missing = np.array([[True, False], [False, True]])
    estimate = np.array([[2., -100.], [-100., 3.]])
    scores = evaluation.point_metrics(truth, estimate, missing, 2.)
    assert scores["rmse"] == 1
    assert scores["nrmse"] == .5


def test_baseline_tuning_receives_no_outer_targets(monkeypatch):
    missing = evaluation.make_mask((30, 20), .5, "mcar", 10)
    matrix = np.where(missing, np.nan, np.random.default_rng(10).normal(size=missing.shape))
    calls = []

    def fake_fit(source, method, parameter, seed):
        assert np.isnan(source[missing]).all()
        calls.append(source)
        return np.nan_to_num(source)

    monkeypatch.setattr(evaluation, "fit_baseline", fake_fit)
    _, tuning, validation = evaluation.classical_predictions(matrix, 10)
    assert not (missing & validation).any()
    assert len(calls) == 10
    assert all(details["validation_cells"] == validation.sum() for details in tuning.values())


def test_acquisition_selection_has_no_truth_input():
    from evaluate_cylinder_acquisition import choose_acquisitions

    pool = np.array([1, 3, 5, 7, 9])
    uncertainty = np.array([1., 5., 3., 4., 2.])
    prediction = np.arange(10).reshape(2, 5)
    choices = choose_acquisitions(pool, uncertainty, prediction, 2, 7)
    np.testing.assert_array_equal(choices["uncertainty"], [3, 7])
    np.testing.assert_array_equal(choices["magnitude"], [7, 9])
    for selected in choices.values():
        assert len(selected) == 2 and len(np.unique(selected)) == 2
        assert np.isin(selected, pool).all()
