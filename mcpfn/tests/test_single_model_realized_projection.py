import pytest
import torch

from single_model_realized_projection import MARGINS, realized_projection


def test_feasible_original_proposal_is_retained():
    before = torch.tensor([1., 1.])
    tentative = before - .01
    gradients = torch.eye(2)
    result = realized_projection(before, tentative, gradients)
    assert result["accepted_index"] == 0
    torch.testing.assert_close(result["attempts"][0]["parameters"], tentative, rtol=0, atol=0)


def test_rounding_requires_larger_inward_margin():
    before = torch.tensor([1., 1.])
    tentative = torch.tensor([1., 1. + 800 * torch.finfo(torch.float32).eps])
    gradients = torch.tensor([[1., 1.]])
    result = realized_projection(before, tentative, gradients)
    assert result["accepted_index"] is not None and result["accepted_index"] > 0
    for attempt in result["attempts"][:-1]:
        assert not attempt["accepted"] and attempt["parameters"] is not None
        assert not any(product > 0 for product in attempt["actual_products"])
    selected = result["attempts"][result["accepted_index"]]
    assert selected["accepted"] and all(product > 0 for product in selected["actual_products"])


def test_unrepresentable_small_correction_is_not_accepted():
    before = torch.tensor([1., 0.])
    tentative = torch.tensor([1., -1e-8])
    gradients = torch.tensor([[1., -1.]])
    result = realized_projection(before, tentative, gradients)
    assert result["accepted_index"] is None and len(result["attempts"]) == len(MARGINS)
    assert all(not attempt["accepted"] and attempt["adverse_tasks"] == 1 for attempt in result["attempts"])


def test_infeasible_opposed_gradients_do_not_silently_drop_constraints():
    before = torch.zeros(2)
    tentative = torch.tensor([.1, .1])
    gradients = torch.tensor([[1., 0.], [-1., 0.]])
    result = realized_projection(before, tentative, gradients)
    assert result["accepted_index"] is None and len(result["attempts"]) == len(MARGINS)
    assert all(not attempt["accepted"] for attempt in result["attempts"])


def test_zero_proposal_is_not_accepted():
    result = realized_projection(torch.zeros(2), torch.zeros(2), torch.eye(2))
    assert result["accepted_index"] is None
    assert all("solver_error" in attempt for attempt in result["attempts"])


def test_nonfinite_parameters_are_rejected():
    with pytest.raises(ValueError, match="finite FP32"):
        realized_projection(torch.tensor([torch.nan]), torch.ones(1), torch.ones(1, 1))
