from collections import Counter
from copy import deepcopy
import inspect

import pytest

import preflight_single_model_balanced_structured_components as predecessor
import preflight_single_model_standardized_teacher_huber as probe
import single_model_standardized_teacher_huber as objective
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_balanced_structured_components_preflight import closed_reports as predecessor_reports


def closed_reports():
    reports = predecessor_reports()
    reports['training']['accepted_updates'] = 59
    reports['closure'] = {'complete': True, 'unqualified': True, 'gate': deepcopy(reports['gate']),
        'campaign': None, 'review': None, 'confirmation_cases_generated': 0, 'scientific_stages_repeated': False,
        'normal_empty_lock_verified': True,
        'original_failed_pipeline': str(probe.ROOT / 'balanced_structured_components_v53_pipeline'),
        'failed_capacity_recovery': str(probe.ROOT / 'balanced_structured_components_v53_capacity_recovery')}
    reports['analysis'] = {'complete': True, 'task_role_rows': 56, 'imputation_rows': 52, 'rating_rows': 4,
        'retention_coefficients_bitwise_unchanged': True, 'rating_terms_and_coordinate_gradients_bitwise_unchanged': True,
        **{key: False for key in ('model_inference_performed', 'new_scientific_seeds', 'training_performed',
            'checkpoint_selection_performed', 'confirmation_evaluated', 'real_data_used', 'cuda_initialized')}}
    return reports


def test_fresh_probe_only_changes_loss_and_seed_block():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1411100901, 1412100901)
    assert probe.OUTPUT.name == 'standardized_teacher_huber_preflight_v1'
    expected = deepcopy(predecessor.batch())
    for task in expected['tasks']:
        task['gradient_seed'] += 30000000
        specification = task['specification']
        specification.get('structured', specification)['seed'] += 30000000
    assert probe.batch() == expected
    assert Counter(task['specification']['structured']['component'] for task in expected['tasks']
        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS}
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    assert probe.PARENT == predecessor.PARENT and probe.RATE == predecessor.RATE == 5e-6
    for name in ('RESERVES', 'WRITE_BUDGETS', 'ARITHMETIC', 'BLEND_COMPONENT_ALLOWANCE'):
        assert getattr(probe, name) == getattr(predecessor, name)


def test_new_loss_and_complete_original_protections():
    assert probe.retention is objective and probe.cohort_retention is objective
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger
    assert probe.reference_identities() == predecessor.reference_identities()
    assert probe.RETENTION_REPLAY_SOURCE == predecessor.RETENTION_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_SOURCE_SHA256 == predecessor.TEACHER_REPLAY_SOURCE_SHA256
    assert 'require_completed_v53' in probe.require_predecessor.__code__.co_names
    assert 'No retry' in probe.POLICY_TEXT and 'all22 development guards,32 fresh confirmation' in probe.POLICY_TEXT
    assert '"structured_component_balance": components.SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'test_single_model_standardized_teacher_huber_probe_audit.py' in probe.REGISTER_SOURCE
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__


@pytest.mark.parametrize('change', [None, ('training', 'accepted_updates', 58),
    ('training', 'independently_regenerated_training_tasks', 1764), ('training', 'autograd_replayed', True),
    ('training', 'confirmation_evaluated', True), ('endpoint', 'endpoint_grids', 55),
    ('endpoint', 'selected_grids', 152), ('endpoint', 'cuda_initialized', True),
    ('closure', 'complete', False), ('closure', 'unqualified', False), ('closure', 'campaign', '/real'),
    ('closure', 'review', {}), ('closure', 'confirmation_cases_generated', 284),
    ('closure', 'scientific_stages_repeated', True), ('closure', 'normal_empty_lock_verified', False),
    ('closure', 'original_failed_pipeline', '/wrong'), ('closure', 'failed_capacity_recovery', '/wrong'),
    ('gate', 'passed', True), ('gate', 'confirmation_evaluated', True), ('gate', 'selected_step', 64),
    ('retention', 'component_grids', 1664), ('retention', 'component_role_counts', {'general': 128}),
    ('blend', 'raw_component_grids', 3327), ('blend', 'query_truth_used_for_routing', True),
    ('analysis', 'task_role_rows', 55), ('analysis', 'imputation_rows', 50), ('analysis', 'rating_rows', 2),
    ('analysis', 'model_inference_performed', True), ('analysis', 'new_scientific_seeds', True),
    ('analysis', 'retention_coefficients_bitwise_unchanged', False), ('analysis', 'real_data_used', True)])
def test_probe_requires_exact_negative_v53_closure_and_complete_diagnosis(change):
    reports = closed_reports()
    if change is None:
        probe.validate_closed_v53(reports)
    else:
        report, key, value = change
        reports[report][key] = value
        with pytest.raises(ValueError):
            probe.validate_closed_v53(reports)
