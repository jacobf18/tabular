from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import single_model_shrunk_column_anchor_retention as retention
import test_single_model_anchor_retention as original_tests
import test_single_model_column_anchor_retention as column_tests


def test_exact_shrinkage_formula_keeps_empty_single_and_constant_columns():
    target, mask = column_tests.column_example()
    target.requires_grad_()
    actual = retention.column_coefficients(target, mask)
    expected = torch.zeros_like(target)
    queried_columns = int(mask.any(dim=1).sum())
    for column_index in range(target.shape[2]):
        selected = mask[0, :, column_index]
        count = int(selected.sum())
        if count:
            variance = float(target.detach()[0, selected, column_index].double().var(unbiased=False))
            denominator = (count * variance + 1.) / (count + 1.)
            expected[0, selected, column_index] = int(mask.sum()) / (queried_columns * count * max(denominator, 1e-6))
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and not actual[~mask].any()
    assert (actual[mask] > 0).all() and torch.isfinite(actual).all()
    assert retention.shrunk_variance(0., 1) == .5
    assert retention.shrunk_variance(0., 4) == .2
    assert retention.shrunk_variance(1., 500) == 1.


def test_permutation_invariant_but_replication_intentionally_changes_count_shrinkage():
    target = torch.tensor([[[2., 3.], [2., 3.]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    reference = torch.zeros_like(target)
    prediction = torch.ones_like(target)
    expected = retention.independent_penalty(target, mask, prediction, reference)
    assert expected == 25.
    permutation = torch.tensor([1, 0])
    assert retention.independent_penalty(*(value[:, :, permutation] for value in
        (target, mask, prediction, reference))) == expected
    repeated = [value.repeat_interleave(3, dim=1) for value in (target, mask, prediction, reference)]
    assert retention.independent_penalty(*repeated) == 55.
    coefficients = retention.column_coefficients(target, mask)
    assert float(10. * coefficients[mask].mean()) == pytest.approx(expected)
    assert retention.independent_penalty(target, mask, reference, reference) == 0.


@pytest.mark.parametrize("mutation", ["shape", "batch", "mask_dtype", "mask_shape", "target_dtype", "empty", "nonfinite"])
def test_invalid_original_query_inputs_are_rejected(monkeypatch, mutation):
    monkeypatch.setattr(column_tests, "retention", retention)
    column_tests.test_invalid_column_inputs_fail_without_omission(mutation)


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("weight", [.25, 1.])
def test_callback_matches_independent_dense_gradient_with_detached_targets(monkeypatch, matched, weight):
    episode = original_tests.episode()
    episode.target = torch.tensor([[[0., 1., 7., 3.], [2., 1.05, 7., 4.],
                                    [4., 1.1, 7., 5.], [6., 1.15, 7., 6.]]])
    episode.mask = torch.tensor([[[True, True, True, True], [True, False, True, False],
                                  [True, False, True, False], [True, True, False, False]]])
    episode.source = episode.target.masked_fill(episode.mask, float("nan"))
    student = torch.nn.Parameter(episode.target + .3)
    frozen = (student.detach().clone() if matched else episode.target - .4).requires_grad_()
    teacher = (episode.target + .1).requires_grad_()
    criterion = SimpleNamespace(mean=lambda logits: logits[:, 0])

    def stream(model, source, target, mask, callback, *, auxiliary_targets, **options):
        order = torch.arange(int(mask.sum()) - 1, -1, -1)
        selected = auxiliary_targets[mask][order]
        return callback(model.reshape(-1)[selected, None], target.reshape(-1)[selected], selected).mean()

    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "streamed_training_loss", stream)
    monkeypatch.setattr(retention.column.balanced, "balanced_loss", lambda current, logits, truth, target, **options:
        (logits[:, 0] - target.detach()).square())
    coefficients, offset, _ = retention.query_budget_majorizer(episode.target, episode.mask, teacher, seed=128)
    original = ((student[episode.mask] - teacher.detach()[episode.mask]).square()
        + coefficients[episode.mask] * (student[episode.mask] - episode.target[episode.mask]).square()).mean() + offset
    contributions = []
    for column_index in range(episode.target.shape[2]):
        selected = episode.mask[0, :, column_index]
        count = int(selected.sum())
        variance = episode.target[0, selected, column_index].double().var(unbiased=False)
        denominator = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6)
        contributions.append((student[0, selected, column_index]
            - frozen.detach()[0, selected, column_index]).square().mean() / denominator)
    expected = original + 10. * torch.stack(contributions).mean()
    gradient = torch.autograd.grad(weight * expected, student, retain_graph=True)[0]
    actual = retention._backward_episode(student, criterion, episode, teacher, weight,
        query_seed_offset=5, retention_target=frozen)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6)
    torch.testing.assert_close(student.grad, gradient)
    assert not student.grad[~episode.mask].any() and teacher.grad is None and frozen.grad is None


@pytest.mark.parametrize("transposed", [False, True])
def test_full_streamed_network_and_every_parameter_gradient_match_dense(monkeypatch, transposed):
    monkeypatch.setattr(column_tests, "retention", retention)
    column_tests.test_full_streamed_network_matches_dense_loss_and_all_parameter_gradients(monkeypatch, transposed)


@pytest.mark.parametrize("failure", [False, True])
def test_full_reference_binding_and_runtime_restoration(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(original_tests, "retention", retention)
    original_tests.test_runtime_keeps_original_teacher_saves_all_tasks_and_restores_callback(tmp_path, monkeypatch, failure)


@pytest.mark.parametrize("domain,component", [("imputation", 0), ("recsys", 6)])
def test_all_ineligible_tasks_keep_the_exact_original_callback(tmp_path, monkeypatch, domain, component):
    monkeypatch.setattr(original_tests, "retention", retention)
    original_tests.test_nonanchor_runtime_uses_exact_original_callback_without_reference_inference(
        tmp_path, monkeypatch, domain, component)


def test_only_prospective_training_variance_estimator_changes():
    assert retention.eligible is retention.predecessor.eligible
    for name in ("weight", "reference_sha256", "legacy_components", "structured_component", "variance_floor",
                 "changes_original_teacher", "changes_point_metrics", "inference_retention_teacher", "new_normalization"):
        assert retention.SPECIFICATION[name] == retention.predecessor.SPECIFICATION[name]
    assert retention.SPECIFICATION["prior_count"] == retention.SPECIFICATION["prior_variance"] == 1.
    assert retention.BACKWARD_SOURCE == retention.predecessor.BACKWARD_SOURCE
    assert retention.RUNTIME_SOURCE == retention.predecessor.RUNTIME_SOURCE
    assert retention._backward_episode.__globals__ is vars(retention)
    assert retention.runtime.__wrapped__.__globals__ is vars(retention)
    assert not torch.cuda.is_initialized()
