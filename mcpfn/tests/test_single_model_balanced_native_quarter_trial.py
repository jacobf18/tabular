from collections import Counter
from copy import deepcopy
import inspect
import math
import types

import pytest

import test_single_model_standardized_huber_trial as template_tests
import train_single_matrixformer_standardized_huber as previous_training
import train_single_matrixformer_balanced_native_quarter as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from single_model_registered_adaptation import replace_once


def fixed_preflight():
    records = original_tests.valid_preflight()
    records[0]['record'].update(proposal_policy=training.paired.POLICY, displacement_fraction=.25)
    records[1]['raw_endpoint']['fixed_quarter_step'] = {'fraction': .25, 'evaluated': True,
        'parameters': 77998496, 'actual_saved_displacements_checked': True}
    return records


TEST_SOURCE = template_tests.TEST_SOURCE
for old, new in {
    'train_single_matrixformer_standardized_huber': 'train_single_matrixformer_balanced_native_quarter',
    '1421100901': '1461100901', '1422100901': '1462100901',
    '1431100901': '1471100901', '1432100901': '1472100901',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing native-quarter lifecycle test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
TEST_SOURCE = replace_once(TEST_SOURCE, '            assert copied == prior',
    '            if copied["cohort"] == "legacy":\n'
    '                for field in ("component", "transpose"):\n'
    '                    copied["specification"][field] = prior["specification"][field]\n'
    '            assert copied == prior')
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())
PROBE_TEST_SOURCE = template_tests.PROBE_TEST_SOURCE.replace(
    'require_completed_v53', 'require_completed_native_probe').replace('closed_v53', 'closed_native_probe')
PROBE_TEST_SOURCE = replace_once(PROBE_TEST_SOURCE, '"balance", "batch"',
    '"balance", "batch", "native_balance", "native_shape", "quarter_fraction"')
PROBE_TEST_SOURCE = replace_once(PROBE_TEST_SOURCE, '    if mutation == "balance":',
    '    if mutation in ("native_balance", "native_shape", "quarter_fraction"):\n'
    '        documents["config.json"][{"native_balance": "native_coverage", "native_shape": "native_probe_shape",\n'
    '            "quarter_fraction": "displacement_fraction"}[mutation]] = None\n'
    '    elif mutation == "balance":')
exec(compile(PROBE_TEST_SOURCE, __file__, 'exec'), globals())


def test_full_native_schedule_changes_only_explicit_fields_and_streams():
    assert training.OUTPUT.name == 'balanced_native_quarter_v55'
    current, original = training.batches(), deepcopy(previous_training.batches())
    seeds = training.original.alignment.fixed.seeds_in
    fresh = seeds([current, training.confirmation_cases()])
    assert len(fresh) == 3868
    assert not fresh.intersection(seeds([original, previous_training.confirmation_cases(), training.preflight.batch()]))
    strata = Counter()
    for batch, before in zip(current, original):
        for task, prior in zip(batch['tasks'], before['tasks']):
            copied = deepcopy(task)
            copied['gradient_seed'] = prior['gradient_seed']
            specification = copied['specification'].get('structured', copied['specification'])
            previous = prior['specification'].get('structured', prior['specification'])
            specification['seed'] = previous['seed']
            if task['cohort'] == 'legacy':
                assert specification['component'] == (batch['step'] - 1 + 4 * task['replicate']) % 8
                assert specification['transpose'] == bool(((batch['step'] - 1) // 8 + task['replicate']) % 2)
                strata[(specification['component'], tuple(specification['shape']), specification['transpose'])] += 1
                for field in ('component', 'transpose'):
                    specification[field] = previous[field]
            assert copied == prior
        assert Counter(task['specification']['structured']['component'] for task in batch['tasks']
            if 'structured' in task['specification']) == dict.fromkeys(training.components.COMPONENTS, 6)
    assert len(strata) == 32 and set(strata.values()) == {4}
    assert previous_training.batches() == original


def test_loss_references_model_geometry_and_every_original_gate_remain_unchanged():
    for name in ('PARENT', 'PARENT_SHA256', 'RATE', 'PARENT_UPDATES', 'BOUNDARIES', 'STEPS',
            'TRIAL_QUALIFICATION', 'STORAGE_ACCOUNTING', 'teacher_policy', 'stronger', 'retention', 'retention_training'):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.paired is training.fixed_trial.protection is training.preflight.paired
    assert training.paired.FRACTION == .25 and previous_training.paired.FRACTION == .5
    assert training.capacity.probe is training.preflight
    assert training.diagnostics is training.all_trial.component_storage
    assert training.all_trial.EXPECTED_MEAN_COLUMNS == 6870 and training.all_trial.EXPECTED_MEAN_ADVERSE == 12
    assert len(training.confirmation_cases()) == 284
    assert training.CAPACITY_RECEIPT_SHA256 == training.capacity.sha256(training.capacity.OUTPUT / 'external_recheck.json')


def test_all_generated_callbacks_bind_native_schedule_and_quarter_optimizer():
    assert '"native_coverage": native.SPECIFICATION' in training.REGISTER_SOURCE
    assert 'config["native_coverage"] != native.SPECIFICATION' in training.ADAPTED_SOURCE
    assert 'native.balance(batch)' in training.BATCH_WRAPPER
    assert training.SOURCE.index(training.BATCH_WRAPPER) < training.SOURCE.index('if __name__ == "__main__":')
    assert training.require_evidence.__globals__ is vars(training)
    for module in (training, training.all_trial, training.diagnostics):
        for function in vars(module).values():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert not unresolved_globals(function), function.__name__
                if 'paired' in inspect.unwrap(function).__code__.co_names:
                    assert inspect.unwrap(function).__globals__['paired'] is training.paired
    for path in ('test_single_model_balanced_native_quarter_trial.py',
            'test_single_model_balanced_native_quarter_full_audit.py', 'test_single_model_balanced_native_quarter_pipeline.py',
            'test_single_model_balanced_native_quarter_storage.py', 'test_single_model_fixed_quarter_trial.py',
            'test_single_model_balanced_native_quarter_capacity.py'):
        assert path in training.REGISTER_SOURCE


@pytest.mark.parametrize('fraction,key', [(.25, 'fixed_quarter_step'), (.5, 'fixed_quarter_step'),
    (.25, 'fixed_half_step'), (.5, 'fixed_half_step')])
def test_preflight_validation_requires_explicit_quarter_evidence(fraction, key):
    records = fixed_preflight()
    records[-1].pop('accepted')
    records[-1]['retention_audited'] = True
    records[0]['record']['displacement_fraction'] = fraction
    contraction = records[1]['raw_endpoint'].pop('fixed_quarter_step')
    contraction['fraction'] = fraction
    records[1]['raw_endpoint'][key] = contraction
    if fraction == .25 and key == 'fixed_quarter_step':
        training.validate_preflight(*records, 'digest')
    else:
        with pytest.raises(ValueError):
            training.validate_preflight(*records, 'digest')
