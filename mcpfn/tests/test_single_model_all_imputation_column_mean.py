from copy import deepcopy
import inspect

import pytest
import torch
from torch.nn import functional

import single_model_all_imputation_column_mean as objective
import single_model_native_parent_retention as original
import test_single_model_anchor_retention as fixtures
import test_single_model_column_anchor_retention as columns
import test_single_model_native_parent_retention as parent_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
def test_native_weights_and_references_remain_exact(component):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(component,
        prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    episode.target, episode.mask = columns.column_example()
    with objective.stronger.training_objective():
        expected = original.teacher_mean_weights(episode)
        actual = objective.teacher_mean_weights(episode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert objective.reference_role(episode.metadata) == original.reference_role(episode.metadata) == "v29"
    assert objective.retention_weighting(episode.metadata) == original.retention_weighting(episode.metadata)


@pytest.mark.parametrize("family", objective.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_all_structured_tasks_use_same_column_rule_without_reference_changes(family, component):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(component, structured=True, family=family)
    episode.target, episode.mask = columns.column_example()
    with objective.stronger.training_objective():
        assert original.teacher_mean_weights(episode) is None
        actual = objective.teacher_mean_weights(episode)
        expected = original.column_coefficients(episode.target, episode.mask).reshape(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        episode.metadata.update(seed=987654, error=1e12, failed=True)
        torch.testing.assert_close(objective.teacher_mean_weights(episode), actual, rtol=0, atol=0)
    assert not actual.requires_grad
    assert objective.reference_role(episode.metadata) == "structured_mean"
    assert objective.retention_weighting(episode.metadata) == original.retention_weighting(episode.metadata)


def test_rating_callback_does_not_require_or_change_mean_override():
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(domain="recsys")
    assert objective.teacher_mean_weights(episode) is None
    assert objective.reference_role(episode.metadata) is None
    assert objective.RUNTIME_SOURCE == original.RUNTIME_SOURCE


@pytest.mark.parametrize("metadata", [fixtures.metadata(), fixtures.metadata("low_rank_mcar", structured=True, family="mar")])
def test_imputation_requires_active_mean20(metadata):
    episode = fixtures.episode()
    episode.metadata = metadata
    with pytest.raises(ValueError, match="active mean20"):
        objective.teacher_mean_weights(episode)


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1),
    dict(domain="unknown"), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior="wrong"), dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True)])
def test_invalid_metadata_cannot_choose_an_objective(changes):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(**changes)
    with objective.stronger.training_objective(), pytest.raises(ValueError):
        objective.teacher_mean_weights(episode)


def independent_teacher_mean(prediction, teacher, truth, mask):
    terms = []
    for column_index in range(truth.shape[-1]):
        selected = mask[0, :, column_index]
        count = int(selected.sum())
        if count:
            variance = truth.detach()[0, selected, column_index].double().var(unbiased=False)
            denominator = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6)
            losses = functional.huber_loss(prediction[0, selected, column_index],
                teacher.detach()[0, selected, column_index], delta=1., reduction="none")
            terms.append(losses.mean() / denominator)
    return 20. * torch.stack(terms).mean()


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", *objective.FAMILIES])
def test_actual_streamed_loss_and_every_parameter_gradient_match_dense_formula(monkeypatch, transposed, cohort):
    dense, source, truth, mask = columns.example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    streamed = deepcopy(dense)
    criterion = columns.FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    metadata = fixtures.metadata() if cohort == "native" else fixtures.metadata("low_rank_mcar", structured=True, family=cohort)
    teacher = torch.randn_like(truth, requires_grad=True)
    reference = torch.randn_like(truth, requires_grad=True)
    coefficients, offset, _diagnostics = objective.query_budget_majorizer(truth, mask, teacher, seed=128)
    monkeypatch.setattr(objective, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
        "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    with objective.stronger.training_objective():
        logits = dense(source, decode_mask=mask)
        means = criterion.mean(logits.float())
        losses = objective.stronger.balanced_loss(criterion, logits, truth[mask], teacher[mask],
            domain="imputation", probability_scales=objective.column.balanced.PROBABILITY_SCALES)
        expected = (losses + coefficients[mask] * (means - truth[mask]).square()
            + 10. * objective.retention_weighting(metadata)["multiplier"] * objective.column_coefficients(truth, mask)[mask]
            * (means - reference.detach()[mask]).square()).mean() + offset
        grid = torch.zeros_like(truth).masked_scatter(mask, means)
        expected = expected - 20. * functional.huber_loss(means, teacher.detach()[mask], delta=1., reduction="mean")
        expected = expected + independent_teacher_mean(grid, teacher, truth, mask)
        actual = objective._backward_episode(streamed, criterion, columns.Episode(source, truth, mask, metadata),
            teacher, 1., query_seed_offset=5, retention_target=reference)
    expected.backward()
    assert actual == pytest.approx(float(expected.detach()), rel=2e-6, abs=2e-6)
    for (name, parameter), (streamed_name, streamed_parameter) in zip(dense.named_parameters(), streamed.named_parameters()):
        assert name == streamed_name and parameter.grad is not None and streamed_parameter.grad is not None
        scale = float(parameter.grad.abs().max())
        torch.testing.assert_close(streamed_parameter.grad, parameter.grad, rtol=2e-4,
            atol=max(3e-7, 2e-5 * scale), msg=name)
        assert float((streamed_parameter.grad - parameter.grad).norm()) <= 1e-5 * float(parameter.grad.norm()) + 3e-7
    assert reference.grad is teacher.grad is None


def test_singletons_constants_and_unqueried_columns_are_preserved():
    truth, mask = columns.column_example()
    teacher = torch.nan_to_num(truth) - .25
    prediction = (torch.nan_to_num(truth) + .3).requires_grad_()
    weights = objective.column_coefficients(truth, mask)
    actual = 20. * (weights[mask] * functional.huber_loss(prediction[mask], teacher[mask], reduction="none")).mean()
    expected = independent_teacher_mean(prediction, teacher, truth, mask)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(weights).all() and (weights[mask] > 0).all()
    assert (weights[~mask] == 0).all()


def test_runtime_globals_and_unchanged_scientific_components():
    assert objective.SPECIFICATION == original.SPECIFICATION
    assert objective.REFERENCE_SHA256S == original.REFERENCE_SHA256S
    assert objective.BACKWARD_SOURCE == original.BACKWARD_SOURCE
    assert objective.BACKWARD_SOURCE_SHA256 == original.BACKWARD_SOURCE_SHA256
    assert objective.RUNTIME_SOURCE == original.RUNTIME_SOURCE
    for name in ("teacher_mean_huber_weight", "huber_delta", "residual_units", "coefficient", "shrunk_variance",
            "moments", "coverage", "retention", "teacher_detached", "architecture_and_inference_unchanged"):
        assert objective.MEAN_SPECIFICATION[name] == original.MEAN_SPECIFICATION[name]
    for name in ("reference_role", "reference_weights", "target_identity", "combine_predictions", "_backward_episode", "runtime", "teacher_mean_weights"):
        function = inspect.unwrap(getattr(objective, name))
        assert function.__globals__ is vars(objective) and function.__closure__ is None
        assert unresolved_globals(function) == []


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(parent_tests, "retention", objective)
    return parent_tests.runtime_fixture.__wrapped__(monkeypatch)


def test_exact_rating_callback_and_all_reference_components(tmp_path, runtime_fixture):
    parent_tests.test_runtime_preserves_parent_native_components_and_does_not_infer_general(tmp_path, runtime_fixture)


@pytest.mark.parametrize("failure", ["duplicate", "changed_reference", "student_alias", "reference_alias", "wrong_identity"])
def test_runtime_errors_restore_original_callback(tmp_path, runtime_fixture, failure):
    parent_tests.test_invalid_runtime_restores_original_callback(tmp_path, runtime_fixture, failure)
