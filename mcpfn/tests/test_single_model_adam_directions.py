import pytest
import torch

import probe_single_model_adam_directions as probe


def group():
    return {"lr": 5e-7, "betas": (.9,.999), "eps": 1e-8, "weight_decay": .02}


def test_first_adam_matches_closed_form_and_never_mutates_originals():
    parameter = torch.tensor([1., -2., .3, 0.], dtype=torch.float64)
    gradient = torch.tensor([.2, -.5, 0., 1e-9], dtype=torch.float64)
    before_parameter, before_gradient = parameter.clone(), gradient.clone()
    actual, analytic = probe.first_adam(parameter, gradient, group())
    expected = group()["lr"] * (gradient/(gradient.abs()+group()["eps"]) + group()["weight_decay"]*parameter)
    torch.testing.assert_close(analytic, expected, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-15)
    assert torch.equal(parameter, before_parameter) and torch.equal(gradient, before_gradient)
    repeated, _ = probe.first_adam(parameter, gradient, group())
    assert torch.equal(repeated, actual)


def test_sgd_common_descent_does_not_imply_adam_common_descent():
    imputation = torch.tensor([10., -2.], dtype=torch.float64)
    rating = torch.tensor([-2., 1.], dtype=torch.float64)
    gradient = probe.method_gradients(imputation, rating, [1,1])["domain_clip_then_mean"]
    sgd = probe.directional_statistics(gradient, imputation, rating)
    assert all(sgd[domain]["dot_with_descent_direction"] > 0 for domain in ("imputation", "recsys"))
    actual, analytic = probe.first_adam(torch.zeros_like(gradient), gradient, group())
    for direction in (actual, analytic):
        result = probe.directional_statistics(direction, imputation, rating)
        assert result["recsys"]["dot_with_descent_direction"] < 0


def test_global_guard_and_domain_caps_match_the_training_recipe():
    imputation, rating = torch.tensor([10.,0.,1.]), torch.tensor([0.,2.,1.])
    methods = probe.method_gradients(imputation, rating, [2,1])
    for vector in methods.values():
        assert vector.norm() <= .5 + 1e-6
    expected = probe.clipped(.5*imputation,[2,1],.25) + probe.clipped(.5*rating,[2,1],.25)
    torch.testing.assert_close(methods["domain_clip_then_mean"], probe.clipped(expected,[2,1],.5))


def test_zero_gradient_retains_weight_decay_without_inventing_loss_descent():
    actual, analytic = probe.first_adam(torch.ones(3,dtype=torch.float64), torch.zeros(3,dtype=torch.float64), group())
    assert actual.abs().sum() > 0
    result = probe.directional_statistics(analytic, torch.zeros(3), torch.zeros(3))
    assert result["recsys"]["cosine"] is None
    assert result["recsys"]["linearized_loss_delta"] == 0


def test_invalid_or_nonfinite_coordinates_cannot_be_omitted():
    with pytest.raises(ValueError):
        probe.clipped(torch.ones(3), [2], .5)
    with pytest.raises(ValueError):
        probe.method_gradients(torch.ones(2), torch.ones(3), [2])
    with pytest.raises(ValueError):
        probe.first_adam(torch.ones(1), torch.tensor([float("nan")]), group())
