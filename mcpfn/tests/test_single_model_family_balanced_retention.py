from contextlib import nullcontext
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

import single_model_family_balanced_retention as retention
import test_single_model_anchor_retention as original_tests
import test_single_model_column_anchor_retention as column_tests
import test_single_model_cohort_anchor_retention as cohort_tests
import test_single_model_shrunk_column_anchor_retention as shrunk_tests


@pytest.mark.parametrize('family', retention.FAMILIES)
def test_every_family_receives_same_retention_mass_without_changing_reference(family):
    metadata = original_tests.metadata('low_rank_mcar', structured=True, family=family)
    weight = .05 if family in ('mar', 'mnar_skip_logic') else .0025
    actual = retention.retention_weighting(metadata)
    assert actual['effective_task_weight'] == .25 / 24
    assert actual['multiplier'] == pytest.approx(5. / 24 if weight == .05 else 25. / 6)
    assert retention.reference_role(metadata) == ('v8' if family == 'mnar_two_phase_subset' else 'v29')
    changed = {**metadata, 'seed': 123456, 'validation_failure': True, 'error': 1e20, 'queries': 1}
    assert retention.retention_weighting(changed) == actual


@pytest.mark.parametrize('component', range(8))
def test_legacy_task_weighting_is_exactly_unchanged(component):
    prior = 'anchor_latent_profile_mixed' if component == 7 else 'anchor_latent_mcar' if component == 6 else 'random'
    metadata = original_tests.metadata(component, prior=prior)
    if component < 6:
        metadata = original_tests.metadata(component)
    assert retention.retention_weighting(metadata) == {
        'task_weight': .25, 'multiplier': 1., 'effective_task_weight': .25}


def test_ratings_acquire_no_retention_weight():
    assert retention.retention_weighting(original_tests.metadata(domain='recsys')) == {
        'task_weight': .125, 'multiplier': 0., 'effective_task_weight': 0.}


@pytest.mark.parametrize('changes', [{'structured': True, 'component': 'low_rank_mcar', 'family': 'unknown'},
    {'domain': 'real'}, {'gradient_data': 'real'}, {'seed': True}])
def test_invalid_cohort_metadata_cannot_choose_a_weight(changes):
    with pytest.raises(ValueError):
        retention.retention_weighting(original_tests.metadata(**changes))


@pytest.mark.parametrize('family,weight', [('mar', .05), ('mnar_skip_logic', .05),
    ('mar_block_neural', .0025), ('mnar_two_phase_subset', .0025)])
@pytest.mark.parametrize('matched', [False, True])
@pytest.mark.parametrize('gradient_scale', [1., .125])
def test_structured_callback_changes_only_retention_and_matches_dense_gradient(
        monkeypatch, family, weight, matched, gradient_scale):
    episode = original_tests.episode()
    episode.metadata = original_tests.metadata('low_rank_mcar', structured=True, family=family)
    episode.target, episode.mask = column_tests.column_example()
    episode.source = episode.target.masked_fill(episode.mask, float('nan'))
    student = torch.nn.Parameter(episode.target + .3)
    reference = (student.detach().clone() if matched else episode.target - .4).requires_grad_()
    teacher = (episode.target + .1).requires_grad_()
    criterion = SimpleNamespace(mean=lambda logits: logits[:, 0])

    def stream(model, source, target, mask, callback, *, auxiliary_targets, **options):
        selected = auxiliary_targets[mask].flip(0)
        return callback(model.reshape(-1)[selected, None], target.reshape(-1)[selected], selected).mean()

    monkeypatch.setattr(retention, 'compute_context', lambda device: nullcontext())
    monkeypatch.setattr(retention, 'streamed_training_loss', stream)
    monkeypatch.setattr(retention.column.balanced, 'balanced_loss',
        lambda current, logits, truth, target, **options: (logits[:, 0] - target.detach()).square())
    coefficients, offset, _ = retention.query_budget_majorizer(episode.target, episode.mask, teacher, seed=128)
    base = ((student[episode.mask] - teacher.detach()[episode.mask]).square()
        + coefficients[episode.mask] * (student[episode.mask] - episode.target[episode.mask]).square()).mean() + offset
    terms = []
    for column_index in range(episode.target.shape[2]):
        selected = episode.mask[0, :, column_index]
        count = int(selected.sum())
        if count:
            variance = episode.target[0, selected, column_index].double().var(unbiased=False)
            denominator = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6)
            terms.append((student[0, selected, column_index]
                - reference.detach()[0, selected, column_index]).square().mean() / denominator)
    expected = base + 10. * ((.25 / 24) / weight) * torch.stack(terms).mean()
    expected_gradient = torch.autograd.grad(gradient_scale * expected, student)[0]
    actual = retention._backward_episode(student, criterion, episode, teacher, gradient_scale,
        query_seed_offset=5, retention_target=reference)
    assert actual == pytest.approx(float(expected.detach()), rel=3e-6)
    torch.testing.assert_close(student.grad, expected_gradient, rtol=3e-5, atol=3e-6)
    assert reference.grad is teacher.grad is None


@pytest.mark.parametrize('matched', [False, True])
@pytest.mark.parametrize('weight', [.25, 1.])
def test_legacy_callback_dense_gradients_stay_unchanged(monkeypatch, matched, weight):
    monkeypatch.setattr(shrunk_tests, 'retention', retention)
    shrunk_tests.test_callback_matches_independent_dense_gradient_with_detached_targets(monkeypatch, matched, weight)


@pytest.mark.parametrize('transposed', [False, True])
def test_real_streamed_network_legacy_gradients_stay_unchanged(monkeypatch, transposed):
    monkeypatch.setattr(column_tests, 'retention', retention)
    column_tests.test_full_streamed_network_matches_dense_loss_and_all_parameter_gradients(monkeypatch, transposed)


@pytest.mark.parametrize('family', ['mar', 'mar_block_neural', 'mnar_two_phase_subset'])
@pytest.mark.parametrize('transposed', [False, True])
def test_real_streamed_structured_model_matches_all_dense_parameter_gradients(monkeypatch, family, transposed):
    dense, source, truth, mask = column_tests.example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    model = deepcopy(dense)
    criterion = column_tests.FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    metadata = original_tests.metadata('low_rank_mcar', structured=True, family=family)
    teacher = torch.randn_like(truth, requires_grad=True)
    reference = torch.randn_like(truth, requires_grad=True)
    coefficients, offset, _ = retention.query_budget_majorizer(truth, mask, teacher, seed=128)
    logits = dense(source, decode_mask=mask)
    means = criterion.mean(logits.float())
    losses = retention.column.balanced.balanced_loss(criterion, logits, truth[mask], teacher[mask],
        domain='imputation', probability_scales=retention.column.balanced.PROBABILITY_SCALES)
    multiplier = (5. / 24) if family == 'mar' else (25. / 6)
    expected = (losses + coefficients[mask] * (means - truth[mask]).square()
        + 10. * multiplier * retention.column_coefficients(truth, mask)[mask]
        * (means - reference.detach()[mask]).square()).mean() + offset
    monkeypatch.setattr(retention, 'TRAIN_OPTIONS', {'attention_batch_size': 2, 'cell_batch_size': 13,
        'encoder_row_batch_size': 3, 'decoder_cell_batch_size': 5})
    actual = retention._backward_episode(model, criterion, column_tests.Episode(source, truth, mask, metadata),
        teacher, 1., query_seed_offset=5, retention_target=reference)
    expected.backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    for (name, expected_parameter), (actual_name, actual_parameter) in zip(dense.named_parameters(), model.named_parameters()):
        assert name == actual_name and expected_parameter.grad is not None and actual_parameter.grad is not None
        scale = float(expected_parameter.grad.abs().max())
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad,
            rtol=2e-4, atol=max(3e-7, 2e-5 * scale), msg=name)
        assert float((actual_parameter.grad - expected_parameter.grad).norm()) <= (
            1e-5 * float(expected_parameter.grad.norm()) + 3e-7), name
    assert reference.grad is teacher.grad is None


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(cohort_tests, 'retention', retention)
    return cohort_tests.runtime_fixture.__wrapped__(monkeypatch)


def test_runtime_keeps_exact_rating_callback_and_records_declared_weights(tmp_path, runtime_fixture):
    cohort_tests.test_runtime_preserves_complete_roles_sources_and_exact_rating_callback(tmp_path, runtime_fixture)
    manifest = json.loads((tmp_path / 'references/manifest.json').read_text())
    for record in manifest['records']:
        assert record['weighting'] == retention.retention_weighting(record['metadata'])
        assert record['gradient_scale'] == .25
        if record['eligible']:
            payload = torch.load(record['artifact'], weights_only=True)
            assert payload['weighting'] == record['weighting']


@pytest.mark.parametrize('failure', ['reused_seed', 'student_alias', 'unused_reference_changed', 'callback'])
def test_runtime_restores_callbacks_and_preserves_both_frozen_references(tmp_path, runtime_fixture, failure):
    cohort_tests.test_runtime_fails_closed_and_restores_callback(tmp_path, runtime_fixture, failure)


def test_task_mixture_cannot_drop_or_reweight_families():
    tasks = [{'cohort': cohort, 'weight': weight} for cohort, weight in
        [('legacy', .25), ('recsys', .125)] + [(family, .05 if family in ('mar', 'mnar_skip_logic') else .0025)
            for family in retention.FAMILIES] for replica in range(2)]
    retention.validate_task_mixture(tasks)
    with pytest.raises(ValueError):
        retention.validate_task_mixture(tasks[:-1])
    changed = deepcopy(tasks)
    changed[-1]['weight'] *= 2
    with pytest.raises(ValueError):
        retention.validate_task_mixture(changed)


def test_no_closures_or_changes_to_original_reference_math():
    for name in ('_backward_episode', 'runtime', 'independent_penalty', 'column_coefficients'):
        function = inspect.unwrap(getattr(retention, name))
        assert function.__globals__ is vars(retention) and function.__closure__ is None
    for name, value in retention.predecessor.SPECIFICATION.items():
        assert retention.SPECIFICATION[name] == value
    assert not torch.cuda.is_initialized()
