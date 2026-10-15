from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dwip_teacher_distribution_training import (
    FrozenDistributionTeachers, categorical_kl, equal_probability_mixture,
    objective_terms, shared_bar_means, validate_probabilities,
)
from dwip_teacher_mean_training import objective_terms as mean_objective
from rowcol_fixed_validation import preserved_rng
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def distribution():
    return FullSupportBarDistribution(torch.linspace(-4., 4., 9))


class TinyModel(torch.nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(8).float() * offset)
        self.calls = 0

    def forward(self, source, decode_mask):
        self.calls += 1
        torch.rand(1)
        return self.logits.expand(int(decode_mask.sum()), -1)


def test_equal_probability_mixture_is_detached_fp64_arithmetic():
    first = torch.tensor([[1000., -1000., 1.]], requires_grad=True)
    second = torch.tensor([[-1000., 1000., 1.]], requires_grad=True)
    mixture = equal_probability_mixture([first, second])
    assert torch.equal(mixture, ((first.detach().softmax(-1).double() + second.detach().softmax(-1).double()) / 2).float())
    assert torch.equal(mixture, torch.tensor([[.5, .5, 0.]]))
    assert mixture.dtype == torch.float32 and not mixture.requires_grad


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("chunk_size", (1, 3, 2048))
def test_kl_gradient_matches_soft_target_cross_entropy_including_zeros(dtype, chunk_size):
    with preserved_rng(23):
        logits = torch.randn(7, 8, dtype=dtype, requires_grad=True)
        probabilities = torch.randn(7, 8).softmax(-1)
    probabilities[:, -1] = 0
    probabilities /= probabilities.sum(-1, keepdim=True)
    probabilities.requires_grad_(True)
    loss = categorical_kl(logits, probabilities, chunk_size)
    expected = -(probabilities.detach() * logits.float().log_softmax(-1)).sum(-1).mean()
    actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=.01 if dtype == torch.bfloat16 else 1e-6, atol=1e-7)
    loss.backward()
    assert probabilities.grad is None and torch.isfinite(logits.grad).all()


def test_identical_probabilities_have_zero_kl_and_gradient():
    logits = torch.zeros(5, 8, requires_grad=True)
    loss = categorical_kl(logits, torch.full((5, 8), 1/8))
    assert abs(float(loss.detach())) < 1e-7
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad), rtol=0., atol=1e-8)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_substitution_preserves_exact_true_target_nll_and_cdf(dtype):
    with preserved_rng(17):
        logits = torch.randn(7, 8, dtype=dtype, requires_grad=True)
        truth = torch.randn(7)
        teachers = [torch.randn(7, 8), torch.randn(7, 8)]
    criterion = distribution()
    probabilities = equal_probability_mixture(teachers)
    teacher_mean = probabilities @ criterion._full_bucket_means
    old = mean_objective(SimpleNamespace(), criterion, logits, truth, teacher_mean)
    new = objective_terms(SimpleNamespace(), criterion, logits, truth, probabilities)
    for name in ("nll", "cdf", "true_rmse", "teacher_rmse"):
        assert torch.equal(old[name], new[name]), name
    assert torch.equal(new["loss"], new["nll"] + 2 * new["kl"] + .05 * new["cdf"])


def test_frozen_teachers_preserve_rng_state_and_predictive_mean():
    first, second, student = TinyModel(.25), TinyModel(-.5), TinyModel()
    original = deepcopy(student.state_dict())
    teachers = FrozenDistributionTeachers([first, second], distribution())
    source = torch.tensor([[[1., float("nan")], [float("nan"), 2.]]])
    mask = torch.isnan(source)
    state = torch.get_rng_state()
    targets = teachers.distribution_target(source, mask)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(targets["probabilities"] @ distribution()._full_bucket_means, targets["mean"])
    assert not targets["probabilities"].requires_grad and not targets["mean"].requires_grad
    assert targets["probabilities"].shape == (2, 8)
    assert student.calls == 0 and first.calls == second.calls == 1
    assert all(torch.equal(value, original[name]) for name, value in student.state_dict().items())
    assert all(not model.training and all(not parameter.requires_grad for parameter in model.parameters()) for model in teachers.models)


@pytest.mark.parametrize("change", ("nonfinite", "negative", "unnormalized", "dtype", "empty", "dimensions"))
def test_invalid_probabilities_are_rejected(change):
    probabilities = torch.full((2, 8), 1/8)
    if change == "nonfinite":
        probabilities[0, 0] = float("nan")
    elif change == "negative":
        probabilities[0, 0] = -.1
    elif change == "unnormalized":
        probabilities *= .5
    elif change == "dtype":
        probabilities = probabilities.double()
    elif change == "empty":
        probabilities = probabilities[:0]
    else:
        probabilities = probabilities.flatten()
    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


@pytest.mark.parametrize("teachers", ([], [torch.zeros(2, 8)], [torch.zeros(2, 8), torch.zeros(3, 8)],
                                       [torch.zeros(2, 8), torch.full((2, 8), float("inf"))]))
def test_invalid_teacher_logits_fail_closed(teachers):
    with pytest.raises(ValueError):
        equal_probability_mixture(teachers)


def test_translation_and_mismatched_bars_are_rejected():
    with pytest.raises(ValueError, match="Translated"):
        shared_bar_means(SimpleNamespace(last_prediction_location=torch.zeros(2)), distribution(), torch.zeros(2, 8))
    with pytest.raises(ValueError, match="bar means"):
        shared_bar_means(SimpleNamespace(), distribution(), torch.zeros(2, 9))


@pytest.mark.parametrize("chunk_size", (0, -1, 1.5, True))
def test_invalid_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk size"):
        categorical_kl(torch.zeros(2, 8), torch.full((2, 8), 1/8), chunk_size)


def test_teacher_mode_and_observed_cell_guards_remain_required():
    teachers = FrozenDistributionTeachers([TinyModel(), TinyModel()], distribution())
    source = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match="unobserved"):
        teachers.distribution_target(source, torch.ones_like(source, dtype=torch.bool))
    source.fill_(float("nan"))
    teachers.models[0].train()
    with pytest.raises(ValueError, match="eval-only"):
        teachers.distribution_target(source, torch.ones_like(source, dtype=torch.bool))
