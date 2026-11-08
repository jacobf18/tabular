from collections import Counter
from copy import deepcopy
import inspect

import pytest

import preflight_single_model_balanced_native_coverage as probe
import preflight_single_model_standardized_teacher_huber as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def diagnoses():
    attribution = {'complete': True, 'prediction_grids': 284,
        'comparisons': [{'candidate_failed': index < 7} for index in range(32)],
        'post_hoc_consumed_bank': True, 'goal_success_not_assessed': True,
        **{key: False for key in ('cuda_initialized', 'checkpoint_selection_performed',
            'confirmation_repeated', 'training_performed', 'real_benchmarks_evaluated')}}
    columns = {'complete': True, 'cases': 284, 'imputation_cases': 252, 'rating_cases': 32,
        'roles': ['general', 'teacher', 'baseline', 'parent', 'candidate'], 'queried_columns': 35227,
        'common_eligible_columns': 35152, 'unique_imputation_metric_checks': 5040,
        'attribution_comparisons': 32, 'rating_case_metric_rows': 64,
        **{key: False for key in ('cuda_initialized', 'model_inference_performed', 'training_performed',
            'checkpoint_selection_performed', 'confirmation_repeated', 'gate_changed', 'real_data_used')}}
    return attribution, columns


def test_fresh_probe_only_changes_native_schedule_context_size_and_seeds():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1441100901, 1442100901)
    assert probe.OUTPUT.name == 'balanced_native_coverage_preflight_v1'
    expected = deepcopy(predecessor.batch())
    for task in expected['tasks']:
        task['gradient_seed'] += 30000000
        specification = task['specification']
        specification.get('structured', specification)['seed'] += 30000000
        if task['cohort'] == 'legacy':
            specification.update(component=4 * task['replicate'], transpose=bool(task['replicate']), shape=[2000, 1000])
    assert probe.batch() == expected
    assert Counter(task['specification']['structured']['component'] for task in expected['tasks']
        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS}
    fresh = probe.training.alignment.fixed.seeds_in(expected)
    previous = probe.training.alignment.fixed.seeds_in(predecessor.batch())
    assert len(fresh) == 56 and not fresh.intersection(previous)


def test_loss_references_original_audits_and_optimizer_guards_are_unchanged():
    assert probe.retention is predecessor.retention and probe.cohort_retention is predecessor.cohort_retention
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger
    assert probe.PARENT == predecessor.PARENT and probe.PARENT_SHA256 == predecessor.PARENT_SHA256
    assert probe.RATE == predecessor.RATE == 5e-6
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    assert probe.reference_identities() == predecessor.reference_identities()
    assert probe.RETENTION_REPLAY_SOURCE == predecessor.RETENTION_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_SOURCE_SHA256 == predecessor.TEACHER_REPLAY_SOURCE_SHA256
    assert probe.TEACHER_REPLAY_AUDIT_SOURCE_SHA256 == predecessor.TEACHER_REPLAY_AUDIT_SOURCE_SHA256
    assert probe.RESERVES == predecessor.RESERVES and probe.ARITHMETIC == predecessor.ARITHMETIC
    assert 'require_completed_v54' in probe.require_predecessor.__code__.co_names
    assert 'native_coverage' in probe.REGISTER_SOURCE and 'structured_component_balance' in probe.REGISTER_SOURCE
    assert '7/32' in probe.POLICY_TEXT and 'No retry' in probe.POLICY_TEXT
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__


def test_storage_explicitly_budgets_both_full_native_component_archives():
    assert probe.WRITE_BUDGETS == {**predecessor.WRITE_BUDGETS, 'checkpoint': 576 * 1024**2}
    difference = 2 * (2000 * 1000 - 256 * 128) * 2 * 4
    assert probe.BLEND_COMPONENT_ALLOWANCE == predecessor.BLEND_COMPONENT_ALLOWANCE + difference
    expected = {**predecessor.STORAGE_ACCOUNTING,
        'checkpoint_artifact_allowance_bytes': predecessor.STORAGE_ACCOUNTING['checkpoint_artifact_allowance_bytes'] + difference}
    assert probe.STORAGE_ACCOUNTING == expected
    assert expected['checkpoint_artifact_allowance_bytes'] < probe.WRITE_BUDGETS['checkpoint']
    assert probe.blend_replay.storage_accounting(probe.batch())['reserved_checkpoint_bytes'] == probe.BLEND_COMPONENT_ALLOWANCE


@pytest.mark.parametrize('change', [None, (0, 'complete', False), (0, 'prediction_grids', 283),
    (0, 'comparisons', [{'candidate_failed': False}] * 32), (0, 'post_hoc_consumed_bank', False),
    (0, 'goal_success_not_assessed', False), (0, 'cuda_initialized', True),
    (0, 'checkpoint_selection_performed', True), (0, 'confirmation_repeated', True),
    (0, 'training_performed', True), (0, 'real_benchmarks_evaluated', True),
    (1, 'complete', False), (1, 'cases', 283), (1, 'imputation_cases', 251), (1, 'rating_cases', 31),
    (1, 'roles', ['duplicate'] * 5), (1, 'queried_columns', 35226), (1, 'common_eligible_columns', 35151),
    (1, 'unique_imputation_metric_checks', 5039), (1, 'attribution_comparisons', 31),
    (1, 'rating_case_metric_rows', 63), (1, 'cuda_initialized', True),
    (1, 'model_inference_performed', True), (1, 'training_performed', True),
    (1, 'checkpoint_selection_performed', True), (1, 'confirmation_repeated', True),
    (1, 'gate_changed', True), (1, 'real_data_used', True)])
def test_complete_negative_diagnostics_cannot_be_relabeled(change):
    reports = diagnoses()
    if change is None:
        probe.validate_diagnoses(*reports)
    else:
        index, key, value = change
        reports[index][key] = value
        with pytest.raises(ValueError):
            probe.validate_diagnoses(*reports)


@pytest.mark.parametrize('key', ['native_coverage', 'native_probe_shape', 'structured_component_balance'])
def test_verify_rejects_modified_schedule_metadata(monkeypatch, key):
    config = {'native_coverage': probe.native.SPECIFICATION, 'native_probe_shape': probe.NATIVE_SHAPE,
        'structured_component_balance': probe.components.SPECIFICATION}
    monkeypatch.setattr(probe, 'original_verify', lambda output: config)
    assert probe.verify(probe.OUTPUT) == config
    config[key] = None
    with pytest.raises(ValueError, match='distinct native coverage'):
        probe.verify(probe.OUTPUT)


@pytest.mark.parametrize('mutation', [None, 'state', 'hash', 'extra_pid', 'live'])
def test_column_terminal_receipt_has_no_pid_but_requires_no_worker(monkeypatch, mutation):
    state = {'state': 'complete', 'summary_sha256': probe.CLOSURE_HASHES[str(probe.COLUMNS / 'summary.json')]}
    monkeypatch.setattr(probe, 'column_worker_running', lambda: mutation == 'live')
    if mutation == 'state':
        state['state'] = 'analyzing'
    elif mutation == 'hash':
        state['summary_sha256'] = '0' * 64
    elif mutation == 'extra_pid':
        state['pid'] = 1234
    if mutation is None:
        probe.require_terminal(probe.COLUMNS, state)
    else:
        with pytest.raises(ValueError, match='PID-free'):
            probe.require_terminal(probe.COLUMNS, state)


@pytest.mark.parametrize('pid', [None, True, 0, -1, '1234', 1234])
def test_every_other_terminal_owner_still_requires_a_stopped_pid(monkeypatch, pid):
    monkeypatch.setattr(probe.closed.original, 'process_identity', lambda value: None)
    state = {'state': 'complete', 'pid': pid}
    if type(pid) is int and pid > 0:
        probe.require_terminal(probe.DIAGNOSIS, state)
        monkeypatch.setattr(probe.closed.original, 'process_identity', lambda value: {'alive': True})
    with pytest.raises(ValueError, match='must be terminal'):
        probe.require_terminal(probe.DIAGNOSIS, state)
