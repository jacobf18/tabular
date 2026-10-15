from copy import deepcopy
import inspect
import math
import os
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_family_balanced_retention as probe
import preflight_single_model_cohort_anchor_retention as previous
import test_single_model_cohort_anchor_retention_preflight as previous_tests
from single_model_distillation import Episode
from single_model_registered_adaptation import replace_once


def completed_reports():
    training = {'complete': True, 'complete_training_audit': True, 'through_step': 64,
        'attempted_batches': 64, 'accepted_updates': 58, 'training_episodes': 1792,
        'prediction_grids': 1368, 'full_network_training': True, 'fixed_half_training': True,
        'displacement_fraction': .5, 'independently_regenerated_training_tasks': 1792,
        'independently_regenerated_rating_contexts': 128, 'independently_regenerated_structured_draws': 1536,
        'confirmation_evaluated': False, 'full_optimizer_trajectory_replayed': False, 'autograd_replayed': False,
        'best': {'step': 0}, 'admission_attempts': {'data_rejected_steps': [], 'optimizer_attempts': 64},
        'raw_endpoint': {'adam': {'gradient_vectors': 84}, 'optimizer_attempt_step': 64}}
    replay = {'complete': True, 'endpoint_grids': 56, 'selected_grids': 0}
    gate = {'endpoint_reproduced': True, 'selected_validation_passed': False, 'selected_step': 0}
    pipeline = {'complete': True, 'gate': {'passed': False, 'confirmation_evaluated': False},
        'campaign': None, 'review': None}
    diagnostic = {'complete': True, 'endpoint_step': 64,
        'tasks': [{'reference_role': 'v8'}] * 26 + [{'reference_role': None}] * 2,
        'cuda_initialized': False, 'training_performed': False, 'model_inference_performed': False,
        'checkpoint_selected': False}
    return training, replay, gate, pipeline, diagnostic


@pytest.mark.parametrize('entry,path,value', [
    (0, ('complete',), False), (0, ('through_step',), 63), (0, ('accepted_updates',), 57),
    (0, ('training_episodes',), 1764), (0, ('prediction_grids',), 1216), (0, ('full_network_training',), False),
    (0, ('displacement_fraction',), 1.), (0, ('best', 'step'), 64),
    (0, ('admission_attempts', 'data_rejected_steps'), [4]), (0, ('raw_endpoint', 'adam', 'gradient_vectors'), 28),
    (0, ('full_optimizer_trajectory_replayed',), True), (0, ('confirmation_evaluated',), True),
    (1, ('endpoint_grids',), 55), (1, ('selected_grids',), 152), (2, ('endpoint_reproduced',), False),
    (2, ('selected_validation_passed',), True), (3, ('gate', 'passed'), True),
    (3, ('gate', 'confirmation_evaluated'), True), (3, ('campaign',), {}), (4, ('tasks',), []),
    (4, ('cuda_initialized',), True), (4, ('training_performed',), True),
    (4, ('model_inference_performed',), True), (4, ('checkpoint_selected',), True)])
def test_require_complete_negative_v42_and_descriptive_diagnosis(entry, path, value):
    reports = completed_reports()
    probe.validate_completed(*reports)
    target = reports[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)


def test_only_retention_weighting_changes_while_seeds_are_fresh():
    current, original = probe.batch(), previous.batch()
    assert len(current['tasks']) == len(original['tasks']) == 28
    for actual, expected in zip(current['tasks'], original['tasks']):
        restored = deepcopy(actual)
        assert restored['gradient_seed'] != expected['gradient_seed']
        restored['gradient_seed'] = expected['gradient_seed']
        specification = restored['specification']
        specification.get('structured', specification)['seed'] = expected['specification'].get(
            'structured', expected['specification'])['seed']
        assert restored == expected
    assert probe.reference_identities() == previous.reference_identities()
    assert probe.ARITHMETIC == {'blas_threads': 24, 'torch_threads': 4}
    assert probe.TEACHER_REPLAY_SOURCE == previous.TEACHER_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_AUDIT_SOURCE == previous.TEACHER_REPLAY_AUDIT_SOURCE
    for name in ('run', 'verify', 'audit', 'audit_original', 'register_trial', 'replay_admitted',
            'audit_teacher_replay', 'replay_cohort_targets'):
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe) and function.__closure__ is None
    assert 'require_completed_v42' in probe.require_predecessor.__code__.co_names
    assert 'original_arithmetic' in probe.main.__code__.co_names
    assert not torch.cuda.is_initialized()


FIXTURE_SOURCE = previous_tests.FIXTURE_SOURCE
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '        truth = torch.tensor(',
    '        weighting = probe.retention.retention_weighting(metadata)\n'
    '        cohort = "recsys" if metadata["domain"] == "recsys" else metadata.get("family", "legacy")\n'
    '        truth = torch.tensor(')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '"weight": 1 / 28}',
    '"weight": weighting["task_weight"], "cohort": cohort}')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE,
    '"queries": int(mask.sum()), "metadata": dict(metadata)}',
    '"queries": int(mask.sum()), "metadata": dict(metadata), "weighting": weighting, "gradient_scale": 1.}')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE,
    '"specification": probe.retention.SPECIFICATION, "metadata": dict(metadata)}, path)',
    '"specification": probe.retention.SPECIFICATION, "metadata": dict(metadata), "weighting": weighting}, path)')
exec(compile(FIXTURE_SOURCE, __file__, 'exec'), globals())


@pytest.mark.parametrize('mutation', [None, 'record_weight', 'record_multiplier', 'payload_weight',
    'gradient_scale', 'omit', 'task_mixture'])
def test_all_reference_and_weighting_decisions_are_bound(tmp_path, mutation):
    retention_fixture(tmp_path)
    path = tmp_path / 'retention_targets/manifest.json'
    manifest = probe.read(path)
    record = manifest['records'][1]
    if mutation == 'record_weight':
        record['weighting']['task_weight'] *= 2
    elif mutation == 'record_multiplier':
        record['weighting']['multiplier'] *= 2
    elif mutation == 'payload_weight':
        payload = torch.load(record['artifact'], weights_only=True)
        payload['weighting']['effective_task_weight'] *= 2
        torch.save(payload, record['artifact'])
        record['sha256'] = probe.sha256(record['artifact'])
    elif mutation == 'gradient_scale':
        record['gradient_scale'] = .0025
    elif mutation == 'omit':
        manifest['records'].pop()
    elif mutation == 'task_mixture':
        config = probe.read(tmp_path / 'config.json')
        config['batch']['tasks'][1]['weight'] *= 2
        probe.write_json(tmp_path / 'config.json', config)
    probe.write_json(path, manifest)
    if mutation is None:
        _, records = probe.retention_records(tmp_path)
        assert len(records) == 26
    else:
        with pytest.raises(ValueError):
            probe.retention_records(tmp_path)


def test_independent_cpu_audit_checks_new_scaled_penalties(tmp_path, monkeypatch):
    retention_fixture(tmp_path)
    monkeypatch.setattr(probe, 'audit_original', lambda output: None)
    monkeypatch.setattr(probe, 'verify', lambda output: probe.read(output / 'config.json'))
    monkeypatch.setattr(probe, 'audit_teacher_replay', lambda output: None)
    probe.audit(tmp_path)
    report = probe.read(tmp_path / 'retention_audit.json')
    assert report['complete'] and report['reference_grids'] == 26
    assert report['reference_role_counts'] == {'v8': 4, 'v29': 22}
    tasks = probe.read(tmp_path / 'training_batch/manifest.json')['tasks']
    for row in report['endpoint_penalties']:
        assert row['before'] == row['before_mixture_weighted'] == 0.
        task = next(task for task in tasks if task['name'] == row['task'])
        episode = torch.load(task['artifact'], weights_only=True)['episode']
        terms = []
        for column_index in range(episode['target'].shape[2]):
            selected = episode['mask'][0, :, column_index]
            count = int(selected.sum())
            if count:
                variance = float(episode['target'][0, selected, column_index].double().var(unbiased=False))
                terms.append(.25 ** 2 / max((count * variance + 1.) / (count + 1.), 1e-6))
        base = 10. * math.fsum(terms) / len(terms)
        assert row['candidate'] == pytest.approx(base * row['weighting']['multiplier'])
        assert row['candidate_mixture_weighted'] == pytest.approx(
            row['candidate'] * row['weighting']['task_weight'])
