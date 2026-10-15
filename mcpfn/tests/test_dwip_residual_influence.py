import pytest
import torch

from diagnose_dwip_residual_influence import residual_influence


def test_huber_definition_and_derivative_energy_match_autograd():
    errors = torch.tensor([-.5, 0., 2., 100.], dtype=torch.float64, requires_grad=True)
    result = residual_influence(errors)
    loss = torch.nn.functional.huber_loss(errors, torch.zeros_like(errors), delta=1.)
    derivative, = torch.autograd.grad(loss, errors)
    assert result["huber_delta1"] == pytest.approx(float(loss.detach()))
    assert result["huber_delta1_derivative_norm"] == pytest.approx(float(derivative.norm()))
    assert result["mse_top1percent_derivative_energy_fraction"] > .99
    assert result["huber_delta1_top1percent_derivative_energy_fraction"] < .5
    assert result["fraction_absolute_error_above1"] == .5


def test_zero_residuals_have_finite_zero_energy():
    result = residual_influence(torch.zeros(10))
    assert result["mse"] == 0
    assert result["huber_delta1_derivative_norm"] == 0
    assert result["huber_delta1_top5percent_derivative_energy_fraction"] == 0


@pytest.mark.parametrize("errors", [torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_residuals_reject_empty_or_nonfinite_values(errors):
    with pytest.raises(ValueError):
        residual_influence(errors)
