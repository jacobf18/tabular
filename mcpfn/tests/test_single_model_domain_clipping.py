from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_domain_clipping as clipping


def backward(model, criterion, episode, teacher, weight):
    loss = (model.weight * episode.gradient).sum()
    (weight * loss).backward()
    return float(loss.detach())


def episode(domain, gradient):
    return SimpleNamespace(metadata={"domain": domain, "gradient_data": "synthetic"}, gradient=torch.tensor([gradient]))


@pytest.mark.parametrize("first,second", [([10., 0.], [0., 1.]), ([10., 0.], [-1., 1.]), ([.1, .2], [.2, .1]),
                                         ([0., 0.], [0., 0.]), ([1., 0.], [-1., 0.])])
def test_actual_accumulation_matches_isolated_clipped_vectors(first, second):
    model = torch.nn.Linear(2, 1, bias=False)
    initial = deepcopy(model.state_dict())
    pair = clipping.DomainGradientPair(backward)
    episodes = [episode("imputation", first), episode("recsys", second)]
    expected = torch.zeros_like(model.weight)
    for current in episodes:
        vector = .5 * current.gradient
        expected += vector * min(1., .25 / (float(vector.norm()) + 1e-6))
        pair(model, None, current, None, .5)
    torch.testing.assert_close(model.weight.grad, expected)
    assert pair.pending is None
    assert model.weight.grad.norm() <= .5 + 1e-7
    assert torch.equal(initial["weight"], model.weight)
    assert all(current.metadata["domain_gradient_clipping"]["gradient_isolated_before_backward"] for current in episodes)


@pytest.mark.parametrize("change", ["wrong_order", "wrong_weight", "real_data", "dirty_gradients", "wrong_model"])
def test_pairing_invariants_are_enforced(change):
    model = torch.nn.Linear(2, 1, bias=False)
    pair = clipping.DomainGradientPair(backward)
    current, weight = episode("imputation", [1., 2.]), .5
    if change == "wrong_order":
        current.metadata["domain"] = "recsys"
    elif change == "wrong_weight":
        weight = 1.
    elif change == "real_data":
        current.metadata["gradient_data"] = "real"
    elif change == "dirty_gradients":
        model.weight.grad = torch.ones_like(model.weight)
    elif change == "wrong_model":
        pair(model, None, current, None, weight)
        current, model = episode("recsys", [1., 2.]), torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError):
        pair(model, None, current, None, weight)


def test_context_restores_original_callback_on_failure_and_incomplete_pair():
    backend = SimpleNamespace(backward_episode=backward)
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError, match="before its rating"):
        with clipping.runtime(backend):
            backend.backward_episode(model, None, episode("imputation", [1., 2.]), None, .5)
    assert backend.backward_episode is backward
    with pytest.raises(RuntimeError):
        with clipping.runtime(backend):
            raise RuntimeError("test")
    assert backend.backward_episode is backward


def test_nonfinite_domain_gradient_is_rejected():
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        clipping.DomainGradientPair(backward)(model, None, episode("imputation", [float("nan"), 1.]), None, .5)


def test_clipping_uses_one_norm_across_all_parameter_tensors():
    model = torch.nn.Linear(2, 1, bias=True)
    def both_parameters(student, criterion, current, teacher, weight):
        loss = (student.weight * current.gradient).sum() + student.bias.sum()
        (weight * loss).backward()
        return float(loss.detach())
    pair = clipping.DomainGradientPair(both_parameters)
    expected = torch.zeros(3)
    for domain, values in (("imputation", [10., 0.]), ("recsys", [-1., 1.])):
        weighted = .5 * torch.tensor([*values, 1.])
        expected += weighted * min(1., .25 / (float(weighted.norm()) + 1e-6))
        pair(model, None, episode(domain, values), None, .5)
    actual = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    torch.testing.assert_close(actual, expected)
