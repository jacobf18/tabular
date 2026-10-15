from contextlib import nullcontext
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

import single_model_column_anchor_retention as retention
import test_single_model_anchor_retention as inherited
from single_model_distillation import Episode
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import example


def column_example():
    target = torch.tensor([[[0., 10., 7., 2., float("nan")], [2., 14., 7., 3., float("nan")],
                            [4., 18., 7., 4., float("nan")], [6., 22., 7., 5., float("nan")]]])
    mask = torch.tensor([[[True, True, True, True, False], [True, False, True, False, False],
                          [True, False, True, False, False], [True, True, True, False, False]]])
    return target, mask


def test_coefficients_match_equal_column_formula_and_keep_constant_single_query_columns():
    target, mask = column_example()
    weights = retention.column_coefficients(target.requires_grad_(), mask)
    expected = torch.zeros_like(target)
    for column_index in range(target.shape[2]):
        queried = mask[0, :, column_index]
        if queried.any():
            variance = target.detach()[0, queried, column_index].double().var(unbiased=False)
            expected[0, queried, column_index] = float(mask.sum()) / (4 * int(queried.sum()) * max(float(variance), 1e-6))
    torch.testing.assert_close(weights, expected)
    assert not weights.requires_grad and not weights[~mask].any()
    assert torch.all(weights[mask] > 0) and torch.isfinite(weights).all()
    assert weights[0, 0, 2] > weights[0, 0, 0]
    assert weights[0, 0, 3] > 0


@pytest.mark.parametrize("mutation", ["shape", "batch", "mask_dtype", "mask_shape", "target_dtype", "empty", "nonfinite"])
def test_invalid_column_inputs_fail_without_omission(mutation):
    target, mask = column_example()
    if mutation == "shape":
        target, mask = target[0], mask[0]
    elif mutation == "batch":
        target, mask = target.repeat(2, 1, 1), mask.repeat(2, 1, 1)
    elif mutation == "mask_dtype":
        mask = mask.float()
    elif mutation == "mask_shape":
        mask = mask[:, :1]
    elif mutation == "target_dtype":
        target = torch.zeros_like(target, dtype=torch.int64)
    elif mutation == "empty":
        mask.zero_()
    else:
        target[0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        retention.column_coefficients(target, mask)


def test_replication_and_column_permutation_preserve_equal_column_loss():
    target, mask = column_example()
    prediction = torch.nan_to_num(target) + .2
    reference = torch.nan_to_num(target) - .3
    expected = retention.independent_penalty(target, mask, prediction, reference)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted = [value[:, :, permutation] for value in (target, mask, prediction, reference)]
    assert retention.independent_penalty(*permuted) == pytest.approx(expected)
    repeated = [value.repeat_interleave(3, dim=1) for value in (target, mask, prediction, reference)]
    assert retention.independent_penalty(*repeated) == pytest.approx(expected)
    weights = retention.column_coefficients(target, mask)
    actual = retention.SPECIFICATION["weight"] * ((prediction - reference).square() * weights)[mask].mean()
    assert float(actual) == pytest.approx(expected, rel=1e-6)
    assert retention.independent_penalty(target, mask, reference, reference) == 0.


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("weight", [.25, 1.])
@pytest.mark.parametrize("unequal_columns", [False, True])
def test_actual_callback_matches_independent_gradient_and_detaches_targets(monkeypatch, matched, weight, unequal_columns):
    current = inherited.episode()
    if unequal_columns:
        current.target = torch.tensor([[[0., 1., 7., 3.], [2., 1.05, 7., 4.],
                                        [4., 1.1, 7., 5.], [6., 1.15, 7., 6.]]])
        current.mask = torch.tensor([[[True, True, True, True], [True, False, True, False],
                                      [True, False, True, False], [True, True, False, False]]])
        current.source = current.target.masked_fill(current.mask, float("nan"))
    student = torch.nn.Parameter(current.target + .3)
    frozen = (student.detach().clone() if matched else current.target - .4).requires_grad_()
    teacher = (current.target + .1).requires_grad_()
    criterion = SimpleNamespace(mean=lambda logits: logits[:, 0])

    def stream(model, source, target, mask, callback, *, auxiliary_targets, **options):
        order = torch.arange(int(mask.sum()) - 1, -1, -1)
        selected = auxiliary_targets[mask][order]
        return callback(model.reshape(-1)[selected, None], target.reshape(-1)[selected], selected).mean()

    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "streamed_training_loss", stream)
    monkeypatch.setattr(retention.column.balanced, "balanced_loss",
        lambda criterion, logits, truth, auxiliary, **options: (criterion.mean(logits) - auxiliary.detach()).square())
    coefficients, offset, _ = retention.query_budget_majorizer(current.target, current.mask, teacher, seed=128)
    original = ((student - teacher.detach()).square() + coefficients * (student - current.target).square())[current.mask].mean() + offset
    contributions = []
    for column_index in range(current.target.shape[2]):
        selected = current.mask[0, :, column_index]
        labels = current.target[0, selected, column_index]
        variance = labels.double().var(unbiased=False).clamp_min(1e-6)
        contributions.append((student[0, selected, column_index] - frozen.detach()[0, selected, column_index]).square().mean() / variance)
    expected = original + 10. * torch.stack(contributions).mean()
    gradient = torch.autograd.grad(weight * expected, student, retain_graph=True)[0]
    actual = retention._backward_episode(student, criterion, current, teacher, weight,
        query_seed_offset=5, retention_target=frozen)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6)
    torch.testing.assert_close(student.grad, gradient)
    assert not student.grad[~current.mask].any() and teacher.grad is None and frozen.grad is None


@pytest.mark.parametrize("transposed", [False, True])
def test_full_streamed_network_matches_dense_loss_and_all_parameter_gradients(monkeypatch, transposed):
    dense, source, truth, mask = example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    model = deepcopy(dense)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth, requires_grad=True)
    frozen = torch.randn_like(truth, requires_grad=True)
    weights, offset, _ = retention.query_budget_majorizer(truth, mask, teacher, seed=128)
    logits = dense(source, decode_mask=mask)
    means = criterion.mean(logits.float())
    losses = retention.column.balanced.balanced_loss(criterion, logits, truth[mask], teacher[mask],
        domain="imputation", probability_scales=retention.column.balanced.PROBABILITY_SCALES)
    retention_weights = retention.column_coefficients(truth, mask)
    expected = (losses + weights[mask] * (means - truth[mask]).square()
        + 10. * retention_weights[mask] * (means - frozen.detach()[mask]).square()).mean() + offset
    monkeypatch.setattr(retention, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
        "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    actual = retention._backward_episode(model, criterion, Episode(source, truth, mask, inherited.metadata()), teacher, .5,
        query_seed_offset=5, retention_target=frozen)
    (.5 * expected).backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    for (name, expected_parameter), (actual_name, actual_parameter) in zip(dense.named_parameters(), model.named_parameters()):
        assert name == actual_name
        expected_gradient, actual_gradient = expected_parameter.grad, actual_parameter.grad
        assert expected_gradient is not None and actual_gradient is not None
        assert torch.isfinite(actual_gradient).all()
        scale = float(expected_gradient.abs().max())
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=max(3e-7, 2e-5 * scale), msg=name)
        assert float((actual_gradient - expected_gradient).norm()) <= 1e-5 * float(expected_gradient.norm()) + 3e-7, name
    assert frozen.grad is None and teacher.grad is None


@pytest.mark.parametrize("failure", [False, True])
def test_full_reference_binding_and_callback_restoration(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(inherited, "retention", retention)
    inherited.test_runtime_keeps_original_teacher_saves_all_tasks_and_restores_callback(tmp_path, monkeypatch, failure)


@pytest.mark.parametrize("domain,component", [("imputation", 0), ("recsys", 6)])
def test_spectral_and_ratings_keep_original_callback(tmp_path, monkeypatch, domain, component):
    monkeypatch.setattr(inherited, "retention", retention)
    inherited.test_nonanchor_runtime_uses_exact_original_callback_without_reference_inference(tmp_path, monkeypatch, domain, component)


def test_prospective_recipe_uses_original_eligibility_and_changes_only_training_penalty():
    assert retention.eligible is retention.predecessor.eligible
    assert retention.SPECIFICATION["weight"] == 10.
    assert retention.SPECIFICATION["variance_floor"] == 1e-6
    assert retention.SPECIFICATION["reference_sha256"] == retention.predecessor.REFERENCE_SHA256
    assert retention.SPECIFICATION["changes_original_teacher"] is False
    assert retention.SPECIFICATION["changes_point_metrics"] is False
    assert retention.SPECIFICATION["inference_retention_teacher"] is False
    assert retention.RUNTIME_SOURCE == retention.inspect.getsource(retention.predecessor.runtime)
    assert retention.BACKWARD_SOURCE_SHA256 == hashlib.sha256(retention.BACKWARD_SOURCE.encode()).hexdigest()
    assert retention._backward_episode.__globals__ is vars(retention)
    assert not torch.cuda.is_initialized()
