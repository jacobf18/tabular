from copy import deepcopy
import inspect
import math

import pytest
import torch
from torch.nn import functional

import single_model_standardized_teacher_huber as objective
import single_model_all_imputation_column_mean as original
import test_single_model_anchor_retention as fixtures
import test_single_model_column_anchor_retention as columns
import test_single_model_native_parent_retention as parent_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def independent_teacher_mean(prediction, teacher, truth, mask):
    terms = []
    for column_index in range(truth.shape[-1]):
        selected = mask[0, :, column_index]
        count = int(selected.sum())
        if count:
            variance = truth.detach()[0, selected, column_index].double().var(unbiased=False)
            scale = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6).sqrt().float()
            residual = (prediction[0, selected, column_index] - teacher.detach()[0, selected, column_index]) / scale
            terms.append(functional.huber_loss(residual, torch.zeros_like(residual), reduction='mean'))
    return 20. * torch.stack(terms).mean()


@pytest.mark.parametrize('component', range(8))
def test_native_geometry_and_references(component):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(component,
        prior='anchor_latent_profile_mixed' if component == 7 else 'anchor_latent_mcar')
    episode.target, episode.mask = columns.column_example()
    with objective.stronger.training_objective():
        actual = objective.teacher_mean_geometry(episode)
    expected = objective.column_geometry(episode.target, episode.mask)
    for actual_grid, expected_grid in zip(actual, expected):
        torch.testing.assert_close(actual_grid, expected_grid.reshape(-1), rtol=0, atol=0)
        assert not actual_grid.requires_grad
    assert objective.reference_role(episode.metadata) == original.reference_role(episode.metadata) == 'v29'
    assert objective.retention_weighting(episode.metadata) == original.retention_weighting(episode.metadata)


@pytest.mark.parametrize('family', objective.FAMILIES)
@pytest.mark.parametrize('component', ('low_rank_mcar', 'slow_tail_mcar', 'steep_tail_mcar', 'anchor_latent_mcar'))
def test_uniform_structured_rule_no_case_or_error_selection(family, component):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(component, structured=True, family=family)
    episode.target, episode.mask = columns.column_example()
    with objective.stronger.training_objective():
        expected = objective.teacher_mean_geometry(episode)
        episode.metadata.update(seed=987654, error=1e12, failed=True)
        actual = objective.teacher_mean_geometry(episode)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert objective.reference_role(episode.metadata) == 'structured_mean'
    assert objective.retention_weighting(episode.metadata) == original.retention_weighting(episode.metadata)


def test_rating_path_and_runtime_unchanged():
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(domain='recsys')
    assert objective.teacher_mean_geometry(episode) == (None, None)
    assert objective.RUNTIME_SOURCE == original.RUNTIME_SOURCE


def test_mean20_override_is_required():
    with pytest.raises(ValueError, match='active mean20'):
        objective.teacher_mean_geometry(fixtures.episode())


@pytest.mark.parametrize('changes', [dict(gradient_data='real'), dict(seed=True), dict(seed=-1),
    dict(domain='unknown'), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior='wrong'), dict(structured=True, component='new_prior'), dict(domain='recsys', structured=True)])
def test_invalid_metadata_rejected(changes):
    episode = fixtures.episode()
    episode.metadata = fixtures.metadata(**changes)
    with objective.stronger.training_objective(), pytest.raises(ValueError):
        objective.teacher_mean_geometry(episode)


@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('cohort', ['native', *objective.FAMILIES])
def test_actual_streamed_loss_and_every_parameter_gradient_match_dense_formula(monkeypatch, transposed, cohort):
    dense, source, truth, mask = columns.example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    streamed = deepcopy(dense)
    criterion = columns.FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    metadata = fixtures.metadata() if cohort == 'native' else fixtures.metadata('low_rank_mcar', structured=True, family=cohort)
    teacher = torch.randn_like(truth, requires_grad=True)
    reference = torch.randn_like(truth, requires_grad=True)
    coefficients, offset, _diagnostics = objective.query_budget_majorizer(truth, mask, teacher, seed=128)
    monkeypatch.setattr(objective, 'TRAIN_OPTIONS', {'attention_batch_size': 2, 'cell_batch_size': 13,
        'encoder_row_batch_size': 3, 'decoder_cell_batch_size': 5})
    with objective.stronger.training_objective():
        logits = dense(source, decode_mask=mask)
        means = criterion.mean(logits.float())
        losses = objective.stronger.balanced_loss(criterion, logits, truth[mask], teacher[mask],
            domain='imputation', probability_scales=objective.column.balanced.PROBABILITY_SCALES)
        expected = (losses + coefficients[mask] * (means - truth[mask]).square()
            + 10. * objective.retention_weighting(metadata)['multiplier'] * objective.column_coefficients(truth, mask)[mask]
            * (means - reference.detach()[mask]).square()).mean() + offset
        grid = torch.zeros_like(truth).masked_scatter(mask, means)
        expected = expected - 20. * functional.huber_loss(means, teacher.detach()[mask], reduction='mean')
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


def test_constant_singleton_empty_columns_and_independent_formula():
    truth, mask = columns.column_example()
    truth.requires_grad_()
    teacher = (torch.nan_to_num(truth.detach()) - .25).requires_grad_()
    prediction = (torch.nan_to_num(truth.detach()) + .3).requires_grad_()
    weights, scales = objective.column_geometry(truth, mask)
    actual = 20. * (weights[mask] * functional.huber_loss((prediction[mask] - teacher.detach()[mask]) / scales[mask],
        torch.zeros_like(prediction[mask]), reduction='none')).mean()
    expected = independent_teacher_mean(prediction, teacher, truth, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert float(actual.detach()) == pytest.approx(objective.independent_teacher_mean(truth, mask, prediction, teacher), rel=1e-6)
    assert (weights[~mask] == 0).all() and (scales[~mask] == 1).all()
    assert float(weights[mask].mean()) == pytest.approx(1.)
    assert not weights.requires_grad and not scales.requires_grad
    actual.backward()
    assert teacher.grad is truth.grad is None and prediction.grad is not None


@pytest.mark.parametrize('scale', [.01, .1, 1., 3.])
def test_quadratic_region_matches_prior_objective(scale):
    point = torch.tensor([min(1., scale) / 4], requires_grad=True)
    teacher = torch.zeros_like(point)
    mass = torch.tensor([2.])
    base = 20. * functional.huber_loss(point, teacher, reduction='none')
    actual = objective.reweight_teacher_mean(base, point, teacher, mass, torch.full_like(point, scale))
    expected = original.reweight_teacher_mean(base, point, teacher, mass / scale**2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), point, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), point)[0], rtol=1e-6, atol=1e-6)


def test_small_scale_large_error_has_bounded_teacher_gradient():
    point = torch.tensor([.5], requires_grad=True)
    teacher = torch.zeros_like(point, requires_grad=True)
    base = 20. * functional.huber_loss(point, teacher.detach(), reduction='none')
    actual = objective.reweight_teacher_mean(base, point, teacher, torch.ones_like(point), torch.full_like(point, .1))
    expected = original.reweight_teacher_mean(base, point, teacher, torch.full_like(point, 100.))
    current_gradient = torch.autograd.grad(actual.sum(), point, retain_graph=True)[0]
    prior_gradient = torch.autograd.grad(expected.sum(), point)[0]
    assert float(current_gradient) == pytest.approx(200.)
    assert float(prior_gradient) == pytest.approx(1000.)
    assert teacher.grad is None


@pytest.mark.parametrize('field,value', [('mass', 0.), ('mass', -1.), ('mass', math.nan),
    ('scale', 0.), ('scale', -1.), ('scale', math.inf)])
def test_invalid_geometry_rejected(field, value):
    point = torch.ones(3)
    mass, scales = torch.ones(3), torch.ones(3)
    (mass if field == 'mass' else scales)[0] = value
    with pytest.raises(ValueError, match='detached finite positive'):
        objective.reweight_teacher_mean(point, point, point, mass, scales)


@pytest.mark.parametrize('field', ['mass', 'scale'])
def test_geometry_gradients_rejected(field):
    point = torch.ones(3)
    mass, scales = torch.ones(3), torch.ones(3)
    (mass if field == 'mass' else scales).requires_grad_()
    with pytest.raises(ValueError, match='detached finite positive'):
        objective.reweight_teacher_mean(point, point, point, mass, scales)


def test_globals_and_unchanged_retention():
    assert objective.SPECIFICATION == original.SPECIFICATION
    assert objective.REFERENCE_SHA256S == original.REFERENCE_SHA256S
    for name in ('reference_role', 'reference_weights', 'target_identity', 'combine_predictions',
            '_backward_episode', 'runtime', 'teacher_mean_geometry', 'column_geometry'):
        function = inspect.unwrap(getattr(objective, name))
        assert function.__globals__ is vars(objective) and function.__closure__ is None
        assert unresolved_globals(function) == []


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(parent_tests, 'retention', objective)
    return parent_tests.runtime_fixture.__wrapped__(monkeypatch)


def test_exact_rating_callback_and_all_reference_components(tmp_path, runtime_fixture):
    parent_tests.test_runtime_preserves_parent_native_components_and_does_not_infer_general(tmp_path, runtime_fixture)


@pytest.mark.parametrize('failure', ['duplicate', 'changed_reference', 'student_alias', 'reference_alias', 'wrong_identity'])
def test_runtime_errors_restore_original_callback(tmp_path, runtime_fixture, failure):
    parent_tests.test_invalid_runtime_restores_original_callback(tmp_path, runtime_fixture, failure)
