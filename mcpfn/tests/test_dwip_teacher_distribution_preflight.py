import math

import pytest
import torch

from preflight_dwip_teacher_distribution import gradient_dot, gradient_sum, gradient_gram, direction_metrics


def test_gradient_combination_preserves_unused_parameters_and_weights():
    first = (torch.tensor([1., 2.]), None, None)
    second = (torch.tensor([3., 4.]), torch.tensor([5.]), None)
    result = gradient_sum([first, second], [2., .5])
    assert torch.equal(result[0], torch.tensor([3.5, 6.]))
    assert torch.equal(result[1], torch.tensor([2.5]))
    assert result[2] is None
    assert torch.equal(first[0], torch.tensor([1., 2.]))


def test_parameter_direction_metrics_match_explicit_flat_vectors():
    direction = (torch.tensor([1., 2.]), None, torch.tensor([3.]))
    reference = (torch.tensor([-1., 4.]), torch.tensor([5.]), None)
    result = direction_metrics(direction, {"reference": reference})
    assert result["unused_parameters"] == 1
    assert result["norm"] == math.sqrt(14.)
    metrics = result["references"]["reference"]
    assert metrics["dot"] == 7.
    assert metrics["reference_norm"] == math.sqrt(42.)
    assert metrics["cosine"] == pytest.approx(7 / math.sqrt(14 * 42))
    assert metrics["unit_descent_reference_decrease"] == 7 / math.sqrt(14)


def test_negative_dot_means_descent_worsens_reference_and_zero_is_not_nan():
    result = direction_metrics((torch.ones(2),), {"opposite": (-torch.ones(2),), "unused": (None,)})
    assert result["references"]["opposite"]["cosine"] == pytest.approx(-1.)
    assert result["references"]["opposite"]["unit_descent_reference_decrease"] < 0
    assert result["references"]["unused"]["cosine"] is None
    zero = direction_metrics((None,), {"nonzero": (torch.ones(2),)})
    assert zero["norm"] == 0 and zero["references"]["nonzero"]["unit_descent_reference_decrease"] is None


def test_streamed_parameter_gram_matches_pairwise_double_precision_dots():
    gradients = {"first": (torch.tensor([1., 2.]), None, torch.tensor([3.])),
                 "second": (torch.tensor([-1., 4.]), torch.tensor([5.]), None),
                 "unused": (None, None, None)}
    gram = gradient_gram(gradients)
    for first_name, first in gradients.items():
        for second_name, second in gradients.items():
            assert gram[first_name][second_name] == gradient_dot(first, second)


@pytest.mark.parametrize("gradients", ({}, {"first": (None,), "second": ()},
                                      {"first": (torch.ones(2),), "second": (torch.ones(3),)},
                                      {"first": (torch.ones(2, dtype=torch.float64),)},
                                      {"first": (torch.tensor([float("nan")]),)}))
def test_streamed_gram_rejects_invalid_gradients(gradients):
    with pytest.raises(ValueError):
        gradient_gram(gradients)


@pytest.mark.parametrize("change", ("length", "shape", "dtype", "nonfinite"))
def test_gradient_dot_rejects_misaligned_or_invalid_parameter_tensors(change):
    first, second = (torch.ones(2),), (torch.ones(2),)
    if change == "length":
        second = ()
    elif change == "shape":
        second = (torch.ones(3),)
    elif change == "dtype":
        second = (torch.ones(2, dtype=torch.float64),)
    else:
        first = (torch.tensor([float("inf"), 1.]),)
    with pytest.raises(ValueError):
        gradient_dot(first, second)


@pytest.mark.parametrize("gradients,weights", (([], []), ([(None,)], []), ([(None,), ()], [1., 1.]),
                                             ([(torch.ones(2),)], [float("nan")])))
def test_gradient_sum_rejects_ambiguous_lists(gradients, weights):
    with pytest.raises(ValueError):
        gradient_sum(gradients, weights)
