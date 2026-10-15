from collections import Counter
from copy import deepcopy
import inspect

import pytest

import preflight_single_model_all_imputation_column_mean as predecessor
import preflight_single_model_balanced_structured_components as probe
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_all_imputation_column_mean_preflight import closed_reports as predecessor_reports


def closed_reports():
    result = predecessor_reports()
    result['training']['accepted_updates'] = 57
    result['external']['accepted_updates'] = 57
    result['external']['rejected_attempts'] = [1, 12, 45, 51, 52, 54, 60]
    return result


def test_fresh_probe_changes_only_seed_block_and_structured_components():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1381100901, 1382100901)
    assert probe.OUTPUT.name == 'balanced_structured_components_preflight_v1'
    expected = deepcopy(predecessor.batch())
    for task in expected['tasks']:
        task['gradient_seed'] += 30000000
        specification = task['specification']
        specification.get('structured', specification)['seed'] += 30000000
    expected = probe.components.balance(expected)
    assert probe.batch() == expected
    assert Counter(task['specification']['structured']['component'] for task in expected['tasks']
        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS}
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    assert probe.PARENT == predecessor.PARENT and probe.RATE == predecessor.RATE == 5e-6
    for name in ('RESERVES', 'WRITE_BUDGETS', 'ARITHMETIC', 'BLEND_COMPONENT_ALLOWANCE'):
        assert getattr(probe, name) == getattr(predecessor, name)


def test_losses_references_replay_and_all_original_guards_are_preserved():
    assert probe.retention is predecessor.retention
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger
    assert probe.reference_identities() == predecessor.reference_identities()
    assert probe.RETENTION_REPLAY_SOURCE == predecessor.RETENTION_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_SOURCE_SHA256 == predecessor.TEACHER_REPLAY_SOURCE_SHA256
    assert 'require_completed_v52' in probe.require_predecessor.__code__.co_names
    assert 'No retry' in probe.POLICY_TEXT and 'all22 development guards,32 fresh confirmation' in probe.POLICY_TEXT
    assert '"structured_component_balance": components.SPECIFICATION' in probe.REGISTER_SOURCE
    for name in ('balanced_structured_components', 'balanced_structured_components_preflight'):
        assert f'test_single_model_{name}.py' in probe.REGISTER_SOURCE
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__


@pytest.mark.parametrize('change', [None, ('training', 'accepted_updates', 58),
    ('training', 'independently_regenerated_training_tasks', 1764), ('training', 'autograd_replayed', True),
    ('training', 'confirmation_evaluated', True), ('endpoint', 'endpoint_grids', 55),
    ('endpoint', 'selected_grids', 152), ('endpoint', 'cuda_initialized', True),
    ('pipeline', 'complete', False), ('pipeline', 'campaign', '/real'), ('pipeline', 'review', {}),
    ('gate', 'passed', True), ('gate', 'confirmation_evaluated', True), ('gate', 'selected_step', 64),
    ('external', 'goal_achieved', True), ('external', 'outcome', 'qualified'),
    ('external', 'rejected_attempts', []), ('external', 'original_workers_terminal', False),
    ('external', 'gate_changed', True), ('retention', 'component_grids', 1664),
    ('retention', 'component_role_counts', {'general': 128, 'v8': 1536, 'v29': 1536}),
    ('blend', 'raw_component_grids', 3327), ('blend', 'query_truth_used_for_routing', True)])
def test_probe_requires_complete_exact_negative_v52_closure(change):
    reports = closed_reports()
    if change is None:
        probe.validate_closed_v52(reports)
    else:
        report, key, value = change
        reports[report][key] = value
        with pytest.raises(ValueError):
            probe.validate_closed_v52(reports)
