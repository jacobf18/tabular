from collections import Counter
from copy import deepcopy
import inspect

import pytest

import preflight_single_model_balanced_native_quarter as probe
import preflight_single_model_balanced_native_coverage as predecessor
import single_model_fixed_half_protection as half
import single_model_fixed_quarter_protection as quarter
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_native_probe_linearization import closed_reports


def reports_fixture():
    reports = dict(zip(('status', 'result', 'gate', 'audit', 'retention', 'review'), closed_reports()))
    reports['audit']['raw_endpoint'].update(fixed_half_step={'fraction': .5, 'parameters': 77998496},
        rejected_candidate_requires_gpu_reconstruction=True, finite_point_scores_require_neural_replay=True)
    reports['audit'].update(autograd_replayed=False, frozen={'changed_trainable_tensors': []})
    reports['mean'] = {'complete': True, 'tasks': 28, 'columns': 6875, 'imputation_reweighted_tasks': 26,
        'unchanged_rating_tasks': 2, 'adverse_teacher_mean_tasks': 11, 'cuda_initialized': False}
    reports['decomposition'] = {'complete': True, 'tasks': 28, 'point_slots': 56, 'grouped_guards': 30,
        'original_accepted': False, 'original_accepted_updates': 0,
        'adverse_points': ['first', 'second', 'third'], 'adverse_groups': ['first'],
        **{key: False for key in ('model_inference_performed', 'cuda_initialized', 'parameter_autograd_replayed',
            'new_scientific_seed', 'training_performed', 'checkpoint_selection_performed',
            'confirmation_repeated', 'real_data_used', 'guard_changed', 'full_trial_authorized', 'goal_achieved')}}
    return reports


def test_fresh_quarter_probe_changes_only_optimizer_fraction_and_streams():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1451100901, 1452100901)
    assert probe.OUTPUT.name == 'balanced_native_quarter_preflight_v1'
    expected = deepcopy(predecessor.batch())
    for task in expected['tasks']:
        task['gradient_seed'] += 10000000
        specification = task['specification']
        specification.get('structured', specification)['seed'] += 10000000
    assert probe.batch() == expected
    native = [task['specification'] for task in expected['tasks'] if task['cohort'] == 'legacy']
    assert [(task['shape'], task['component'], task['transpose']) for task in native] == [
        ([2000, 1000], 0, False), ([2000, 1000], 4, True)]
    assert Counter(task['specification']['structured']['component'] for task in expected['tasks']
        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS}
    fresh = probe.training.alignment.fixed.seeds_in(expected)
    previous = probe.training.alignment.fixed.seeds_in(predecessor.batch())
    assert len(fresh) == 56 and not fresh.intersection(previous)


def test_losses_references_model_resources_and_retention_remain_identical():
    for name in ('retention', 'cohort_retention', 'teacher_policy', 'stronger', 'mixture', 'replicated'):
        assert getattr(probe, name) is getattr(predecessor, name)
    for name in ('PARENT', 'PARENT_SHA256', 'RATE', 'TRAINABLE_PARAMETERS', 'TRAINABLE_TENSORS', 'FROZEN_TENSORS',
            'RESERVES', 'ARITHMETIC', 'WRITE_BUDGETS', 'STORAGE_ACCOUNTING', 'BLEND_COMPONENT_ALLOWANCE',
            'RETENTION_REPLAY_SOURCE', 'TEACHER_REPLAY_SOURCE_SHA256', 'TEACHER_REPLAY_AUDIT_SOURCE_SHA256'):
        assert getattr(probe, name) == getattr(predecessor, name), name
    assert probe.reference_identities() == predecessor.reference_identities()
    assert probe.blend_replay.storage_accounting(probe.batch())['reserved_checkpoint_bytes'] == probe.BLEND_COMPONENT_ALLOWANCE
    assert 'independently audit every actual fresh physical column' in probe.POLICY_TEXT
    assert 'all22 development guards,32 fresh confirmation' in probe.POLICY_TEXT
    assert 'complete frozen480-comparison real review' in probe.POLICY_TEXT
    assert 'fixed-half V8/V29 structured' in probe.POLICY_TEXT


def test_all_compiled_run_audit_helpers_use_quarter_without_mutating_half():
    assert probe.paired is quarter and quarter.FRACTION == .25
    assert predecessor.paired is half and half.FRACTION == .5
    assert 'single_model_fixed_half_protection as paired' not in probe.QUARTER_PROBE_SOURCE
    assert 'single_model_fixed_quarter_protection as paired' in probe.QUARTER_PROBE_SOURCE
    assert 'require_completed_native_probe' in probe.require_predecessor.__code__.co_names
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__
            if 'paired' in inspect.unwrap(function).__code__.co_names:
                assert inspect.unwrap(function).__globals__['paired'] is quarter
    for name in ('run', 'audit_original', 'original_verify', 'register_trial'):
        assert inspect.unwrap(getattr(probe, name)).__globals__ is vars(probe)
    assert 'fixed_quarter_step' in quarter.AUDIT_SOURCE
    assert 'fixed_half_step' not in quarter.AUDIT_SOURCE
    assert 'fixed_half_step' in half.AUDIT_SOURCE


@pytest.mark.parametrize('key', ['native_coverage', 'native_probe_shape', 'structured_component_balance'])
def test_verify_preserves_separate_native_and_structured_metadata(monkeypatch, key):
    config = {'native_coverage': probe.native.SPECIFICATION, 'native_probe_shape': probe.NATIVE_SHAPE,
        'structured_component_balance': probe.components.SPECIFICATION}
    monkeypatch.setattr(probe, 'original_verify', lambda output: config)
    assert probe.verify(probe.OUTPUT) == config
    config[key] = None
    with pytest.raises(ValueError, match='distinct native coverage'):
        probe.verify(probe.OUTPUT)


@pytest.mark.parametrize('mutation', [None,
    ('status.state', 'running'), ('result.record.accepted', True), ('result.record.accepted_updates', 1),
    ('gate.passed', True), ('retention.component_grids', 49), ('review.full_trial_authorized', True),
    ('audit.raw_endpoint.fixed_half_step.fraction', .25), ('audit.raw_endpoint.fixed_half_step.parameters', 1),
    ('audit.raw_endpoint.rejected_candidate_requires_gpu_reconstruction', False),
    ('audit.raw_endpoint.finite_point_scores_require_neural_replay', False),
    ('audit.autograd_replayed', True), ('audit.frozen.changed_trainable_tensors', ['changed']),
    ('mean.complete', False), ('mean.tasks', 27), ('mean.columns', 6874),
    ('mean.imputation_reweighted_tasks', 25), ('mean.unchanged_rating_tasks', 1),
    ('mean.adverse_teacher_mean_tasks', 0), ('mean.cuda_initialized', True),
    ('decomposition.complete', False), ('decomposition.tasks', 27), ('decomposition.point_slots', 55),
    ('decomposition.grouped_guards', 29), ('decomposition.original_accepted', True),
    ('decomposition.original_accepted_updates', 1), ('decomposition.adverse_points', []),
    ('decomposition.adverse_groups', []),
    *((f'decomposition.{key}', True) for key in ('model_inference_performed', 'cuda_initialized',
        'parameter_autograd_replayed', 'new_scientific_seed', 'training_performed',
        'checkpoint_selection_performed', 'confirmation_repeated', 'real_data_used', 'guard_changed',
        'full_trial_authorized', 'goal_achieved'))])
def test_previous_negative_probe_diagnosis_and_audit_limits_cannot_be_relabeled(mutation):
    reports = reports_fixture()
    if mutation is None:
        probe.validate_previous(reports)
    else:
        key, value = mutation
        destination = reports
        path = key.split('.')
        for field in path[:-1]:
            destination = destination[field]
        destination[path[-1]] = value
        with pytest.raises(ValueError):
            probe.validate_previous(reports)


@pytest.mark.parametrize('mutation', [None, 'running', 'live', 'conflict', 'missing_pid', 'negative_pid'])
def test_prerequisite_closure_binds_evidence_without_rerunning_analysis(monkeypatch, mutation):
    reports = reports_fixture()
    documents = {probe.PREVIOUS / filename: reports[name] for name, filename in (
        ('status', 'status.json'), ('result', 'result.json'), ('gate', 'gate.json'), ('audit', 'audit.json'),
        ('retention', 'retention_audit.json'), ('review', 'external_recheck.json'))}
    documents[probe.PREVIOUS / 'status.json']['pid'] = 1234
    documents[probe.MEAN / 'audit.json'] = reports['mean']
    documents[probe.DECOMPOSITION / 'summary.json'] = reports['decomposition']
    documents[probe.DECOMPOSITION / 'status.json'] = {'state': 'complete', 'pid': 1235}
    for directory in (probe.PREVIOUS, probe.MEAN, probe.DECOMPOSITION):
        documents[directory / 'config.json'] = {'source_sha256': {'bound_source': 'unchanged'}}
    if mutation == 'running':
        documents[probe.DECOMPOSITION / 'status.json']['state'] = 'analyzing'
    elif mutation == 'conflict':
        documents[probe.DECOMPOSITION / 'config.json']['source_sha256']['bound_source'] = 'changed'
    elif mutation == 'missing_pid':
        documents[probe.DECOMPOSITION / 'status.json'].pop('pid')
    elif mutation == 'negative_pid':
        documents[probe.DECOMPOSITION / 'status.json']['pid'] = -1
    verified = []
    closure = []
    monkeypatch.setattr(probe, 'read', lambda path: documents[path])
    monkeypatch.setattr(probe, 'sha256', lambda path: 'digest')
    monkeypatch.setattr(probe, 'verify_hashes', lambda inventory: verified.append(dict(inventory)))
    monkeypatch.setattr(probe.native_diagnosis, 'require_closed', lambda: closure.append(True))
    monkeypatch.setattr(probe.probe_blueprint.closed.original, 'process_identity',
        lambda pid: {'alive': True} if mutation == 'live' else None)
    if mutation is None:
        evidence = probe.require_completed_native_probe()
        assert evidence['bound_source'] == 'unchanged'
        assert verified == [probe.CLOSURE_HASHES, evidence] and closure == [True]
        for directory in (probe.PREVIOUS, probe.DECOMPOSITION):
            assert evidence[str(directory / 'status.json')] == 'digest'
    else:
        with pytest.raises(ValueError):
            probe.require_completed_native_probe()


def test_registration_binds_all_new_tests_auditors_and_explicit_fraction():
    for path in ('test_single_model_fixed_quarter_protection.py',
            'test_single_model_balanced_native_quarter_preflight.py',
            'audit_single_model_balanced_native_quarter_probe.py',
            'test_single_model_balanced_native_quarter_probe_audit.py',
            'recheck_single_model_balanced_native_quarter_probe.py',
            'test_single_model_balanced_native_quarter_probe_recheck.py'):
        assert path in probe.REGISTER_SOURCE
    assert '"displacement_fraction": paired.FRACTION' in probe.REGISTER_SOURCE
    assert 'fresh.intersection(used)' in probe.REGISTER_SOURCE
