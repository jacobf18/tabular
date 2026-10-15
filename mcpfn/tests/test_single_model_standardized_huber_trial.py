from collections import Counter
from copy import deepcopy
import math
import types

import pytest

import test_single_model_balanced_structured_trial as template_tests
import test_single_model_all_imputation_column_mean_trial as parent_tests
import train_single_matrixformer_balanced_structured as previous_training
import train_single_matrixformer_standardized_huber as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


fixed_preflight = types.FunctionType(template_tests.fixed_preflight.__code__, globals(),
    'fixed_preflight', template_tests.fixed_preflight.__defaults__)
TEST_SOURCE = template_tests.TEST_SOURCE
for old, new in {
    'train_single_matrixformer_balanced_structured': 'train_single_matrixformer_standardized_huber',
    '1391100901': '1421100901', '1392100901': '1422100901',
    '1401100901': '1431100901', '1402100901': '1432100901',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing standardized-Huber lifecycle test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())
PROBE_TEST_SOURCE = template_tests.PROBE_TEST_SOURCE.replace('require_completed_v52', 'require_completed_v53').replace('closed_v52', 'closed_v53')
exec(compile(PROBE_TEST_SOURCE, __file__, 'exec'), globals())


def test_actual_batches_only_change_streams_not_balanced_priors():
    assert training.OUTPUT.name == 'standardized_teacher_huber_v54'
    current, original = training.batches(), deepcopy(previous_training.batches())
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([current, training.confirmation_cases()])
    assert len(fresh) == 3868
    assert not fresh.intersection(seeds([original, previous_training.confirmation_cases(), training.preflight.batch()]))
    for batch, before in zip(current, original):
        for task, prior in zip(batch['tasks'], before['tasks']):
            copied = deepcopy(task)
            copied['gradient_seed'] = prior['gradient_seed']
            specification = copied['specification'].get('structured', copied['specification'])
            previous = prior['specification'].get('structured', prior['specification'])
            specification['seed'] = previous['seed']
            assert copied == prior
        assert Counter(task['specification']['structured']['component'] for task in batch['tasks']
            if 'structured' in task['specification']) == dict.fromkeys(training.components.COMPONENTS, 6)
    assert previous_training.batches() == original


def test_loss_changes_but_initializer_references_optimizer_and_all_gates_do_not():
    for name in ('PARENT', 'PARENT_SHA256', 'RATE', 'PARENT_UPDATES', 'BOUNDARIES', 'STEPS',
            'TRIAL_QUALIFICATION', 'STORAGE_ACCOUNTING', 'teacher_policy', 'stronger', 'paired'):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.retention is training.preflight.retention
    assert training.retention is not previous_training.retention
    assert training.retention.SPECIFICATION == previous_training.retention.SPECIFICATION
    assert training.retention.MEAN_SPECIFICATION != previous_training.retention.MEAN_SPECIFICATION
    assert training.retention_training.retention is training.retention
    assert training.capacity.probe is training.preflight
    assert training.diagnostics is training.all_trial.component_storage
    assert 'require_completed_v53' in training.all_trial.PROBE_EVIDENCE_SOURCE
    assert training.all_trial.EXPECTED_MEAN_COLUMNS == 4053
    assert training.all_trial.EXPECTED_MEAN_ADVERSE == 8


def test_registered_source_inventory_includes_new_audits_and_recheck():
    for path in ('test_single_model_standardized_huber_training.py',
            'test_single_model_standardized_huber_full_audit.py',
            'test_single_model_standardized_huber_storage.py',
            'test_single_model_standardized_teacher_huber_probe_recheck.py',
            'test_single_model_standardized_huber_capacity.py'):
        assert path in training.REGISTER_SOURCE
