from copy import deepcopy

import pytest
import torch

from single_model_finite_guarded_update import StepEvaluationError, metric_decision, protected_update


def fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
    gradients = torch.tensor([[1., 1.], [1., 0.]])
    return model, optimizer, gradients


def assert_optimizer_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert set(actual["state"]) == set(expected["state"])
    for key, state in expected["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(actual["state"][key][field], value, rtol=0, atol=0)


def test_accepts_actual_metric_improvement_and_keeps_adam_moments():
    model, optimizer, gradients = fixture()
    record, saved = protected_update(model, optimizer, gradients, [1.], .01, {"rmse": 1.},
                                     lambda current: {"rmse": float(current.weight[0, 0].detach())})
    assert record["accepted"] and record["reason"] == "accepted"
    assert record["finite"]["all_retained"]
    assert optimizer.state
    assert not torch.equal(saved["before"], saved["after"])


@pytest.mark.parametrize("existing_moments", [False, True])
def test_rejection_restores_parameters_and_moments_exactly(existing_moments):
    model, optimizer, gradients = fixture()
    if existing_moments:
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    parameters = model.weight.detach().clone()
    optimizer.param_groups[0]["lr"] = .003
    state = deepcopy(optimizer.state_dict())
    record, saved = protected_update(model, optimizer, gradients, [1.], .01, {"rmse": 1.}, lambda current: {"rmse": 1.1})
    assert not record["accepted"] and record["reason"] == "finite_metrics_rejected"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    assert_optimizer_equal(optimizer.state_dict(), state)
    torch.testing.assert_close(saved["before"], saved["after"], rtol=0, atol=0)


def test_evaluation_error_preserves_snapshot_and_rolls_back():
    model, optimizer, gradients = fixture()
    parameters = model.weight.detach().clone()
    state = deepcopy(optimizer.state_dict())

    def fail(current):
        raise ValueError("deliberate scorer failure")

    with pytest.raises(StepEvaluationError) as failure:
        protected_update(model, optimizer, gradients, [1.], .01, {"rmse": 1.}, fail)
    assert failure.value.snapshot["before"].numel() == 2
    assert failure.value.record["reason"] == "evaluation_failed"
    torch.testing.assert_close(model.weight, parameters, rtol=0, atol=0)
    assert_optimizer_equal(optimizer.state_dict(), state)


def test_opposed_constraints_reject_without_metric_evaluation():
    model, optimizer, _ = fixture()
    gradients = torch.tensor([[1., 0.], [-1., 0.]])

    def forbidden(current):
        raise AssertionError("No finite evaluation of an infeasible projection")

    record, saved = protected_update(model, optimizer, gradients, [1.], .01, {"rmse": 1.}, forbidden)
    assert not record["accepted"] and record["reason"] == "projection_infeasible"
    assert not optimizer.state
    torch.testing.assert_close(saved["before"], saved["after"], rtol=0, atol=0)


def test_metrics_retain_strict_signs_and_truth_defined_coverage():
    assert not metric_decision({"rmse": 1.}, {"rmse": 1. + 1e-14})["accepted"]
    assert not metric_decision({"rmse": 1.}, {"rmse": 1.})["accepted"]
    assert metric_decision({"rmse": 1., "common": None}, {"rmse": .9, "common": None})["accepted"]
    with pytest.raises(ValueError, match="coverage"):
        metric_decision({"common": None}, {"common": 0.})
    with pytest.raises(ValueError, match="every original"):
        metric_decision({"rmse": 1., "mae": 1.}, {"rmse": .9})
