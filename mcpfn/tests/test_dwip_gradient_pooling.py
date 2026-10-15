import copy

import pytest
import torch

from diagnose_dwip_gradient_pooling import clipped, flatten_gradients, frozen_adam_direction, pooling_vectors, projection
from diagnose_dwip_training_objective import adam_descent


def test_pooling_exposes_clipping_order_without_mutation():
    vectors = [torch.tensor([10., 0.]), torch.tensor([0., 1.]), torch.tensor([0., 1.]), torch.tensor([0., 1.])]
    original = [value.clone() for value in vectors]
    pooled = pooling_vectors(vectors)
    assert torch.allclose(pooled["clip_mean_gradient"], clipped(torch.tensor([2.5, .75])))
    assert torch.allclose(pooled["mean_clipped_gradient"], torch.stack([clipped(value) for value in vectors]).mean(0))
    assert projection(pooled["clip_mean_gradient"], pooled["mean_clipped_gradient"])["descent_cosine"] < .6
    assert all(torch.equal(before, after) for before, after in zip(original, vectors))


def test_pooling_identical_gradients_and_zero_direction():
    vector = torch.tensor([3., 4.])
    pooled = pooling_vectors([vector] * 4)
    assert torch.equal(pooled["clip_mean_gradient"], pooled["mean_clipped_gradient"])
    assert projection(torch.zeros(2), vector)["descent_cosine"] is None
    assert projection(-vector, vector)["linearized_loss_delta"] == 25


@pytest.mark.parametrize("vectors", [[torch.ones(2)] * 3, [torch.ones(2), torch.ones(3), torch.ones(2), torch.ones(2)],
    [torch.tensor([float('nan')])] * 4])
def test_pooling_rejects_invalid_inputs(vectors):
    with pytest.raises(ValueError):
        pooling_vectors(vectors)


def test_flatten_preserves_unused_parameter_activity():
    parameters = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]
    vector, active = flatten_gradients(parameters, [torch.tensor([2., 3.]), None])
    assert torch.equal(vector, torch.tensor([2., 3., 0.]))
    assert active == (True, False)
    with pytest.raises(ValueError):
        flatten_gradients(parameters, [torch.ones(1), None])


def test_frozen_adam_matches_existing_analytical_update_and_skips_unused():
    parameters = [torch.nn.Parameter(torch.tensor([1., -2.])), torch.nn.Parameter(torch.tensor([3.]))]
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=.1)
    parameters[0].grad = torch.tensor([.3, .4])
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    before = [parameter.detach().clone() for parameter in parameters]
    gradient = torch.tensor([.1, -.2, 0.])
    expected = adam_descent(parameters[0], gradient[:2], state["state"][0], state["param_groups"][0])
    direction = frozen_adam_direction(parameters, gradient, (True, False), state)
    assert torch.equal(direction[:2], expected)
    assert direction[-1] == 0
    assert all(torch.equal(parameter, value) for parameter, value in zip(parameters, before))
    assert torch.equal(state["state"][0]["exp_avg"], optimizer.state_dict()["state"][0]["exp_avg"])
    state["param_groups"][0]["params"].reverse()
    with pytest.raises(ValueError):
        frozen_adam_direction(parameters, gradient, (True, False), state)


def test_projection_checks_shapes_and_finiteness():
    with pytest.raises(ValueError):
        projection(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        projection(torch.tensor([float('inf')]), torch.ones(1))
