from collections import Counter, defaultdict
from copy import deepcopy
import inspect
import math
import types

import pytest

import test_single_model_all_imputation_column_mean_trial as parent_tests
import test_single_model_native_column_trial as mean_tests
import train_single_matrixformer_all_imputation_column_mean as previous_training
import train_single_matrixformer_balanced_structured as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from single_model_registered_adaptation import replace_once


fixed_preflight = types.FunctionType(parent_tests.fixed_preflight.__code__, globals(), 'fixed_preflight', parent_tests.fixed_preflight.__defaults__)
TEST_SOURCE = parent_tests.ADAPTED_TEST_SOURCE
for old, new in {
    'train_single_matrixformer_all_imputation_column_mean': 'train_single_matrixformer_balanced_structured',
    '1361100901': '1391100901', '1362100901': '1392100901',
    '1371100901': '1401100901', '1372100901': '1402100901',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing balanced-prior lifecycle test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
TEST_SOURCE = replace_once(TEST_SOURCE, '            assert copied == prior',
    '            if "structured" in copied["specification"]:\n'
    '                copied["specification"]["structured"]["component"] = prior["specification"]["structured"]["component"]\n'
    '            assert copied == prior')
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())

PROBE_TEST_SOURCE = inspect.getsource(mean_tests.test_full_trial_requires_complete_probe_and_supplementary_evidence)
PROBE_TEST_SOURCE = PROBE_TEST_SOURCE.replace('require_completed_v47', 'require_completed_v52').replace('closed_v47', 'closed_v52')
PROBE_TEST_SOURCE = replace_once(PROBE_TEST_SOURCE, '"partial_mean", "hidden_adverse", "mean_inference"',
    '"partial_mean", "hidden_adverse", "mean_inference", "balance", "batch"')
PROBE_TEST_SOURCE = replace_once(PROBE_TEST_SOURCE, '    if mutation == "reference":',
    '    if mutation == "balance":\n'
    '        documents["config.json"]["structured_component_balance"] = {}\n'
    '    elif mutation == "batch":\n'
    '        documents["config.json"]["batch"]["tasks"][1]["specification"]["structured"]["component"] = "changed"\n'
    '    elif mutation == "reference":')
exec(compile(PROBE_TEST_SOURCE, __file__, 'exec'), globals())


def test_actual_registered_batches_have_exact_component_balance_and_unchanged_other_fields():
    assert training.OUTPUT.name == 'balanced_structured_components_v53'
    current = training.batches()
    original = deepcopy(previous_training.batches())
    seed_inventory = training.original.alignment.fixed.seeds_in
    assert len(seed_inventory([current, training.confirmation_cases()])) == 3868
    assert not seed_inventory([current, training.confirmation_cases()]).intersection(
        seed_inventory([original, previous_training.confirmation_cases(), training.preflight.batch()]))
    families = training.components.FAMILIES
    for batch, before in zip(current, original):
        counts, weights = Counter(), defaultdict(float)
        for task, prior in zip(batch['tasks'], before['tasks']):
            copied = deepcopy(task)
            copied['gradient_seed'] = prior['gradient_seed']
            specification = copied['specification'].get('structured', copied['specification'])
            previous = prior['specification'].get('structured', prior['specification'])
            specification['seed'] = previous['seed']
            if task['cohort'] in families:
                component = specification['component']
                counts[component] += 1
                weights[component] += task['weight']
                specification['component'] = previous['component']
            assert copied == prior
        assert counts == dict.fromkeys(training.components.COMPONENTS, 6)
        assert all(math.isclose(value, .0625, rel_tol=0, abs_tol=1e-12) for value in weights.values())
    for first, second in zip(current, current[1:]):
        for family in families:
            selected = {task['specification']['structured']['component'] for batch in (first, second)
                for task in batch['tasks'] if task['cohort'] == family}
            assert selected == set(training.components.COMPONENTS)
    assert previous_training.batches() == original


def test_same_loss_reference_kernel_capacity_and_all_original_gates():
    for name in ('PARENT', 'PARENT_SHA256', 'RATE', 'PARENT_UPDATES', 'BOUNDARIES', 'STEPS',
            'TRIAL_QUALIFICATION', 'STORAGE_ACCOUNTING', 'teacher_policy', 'stronger', 'paired',
            'retention', 'retention_training'):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.retention is training.preflight.retention
    assert training.capacity.probe is training.preflight
    assert training.diagnostics is training.all_trial.component_storage
    assert training.diagnostics.RESERVES == previous_training.diagnostics.RESERVES
    assert training.diagnostics.WRITE_BUDGETS == previous_training.diagnostics.WRITE_BUDGETS
    assert len(training.confirmation_cases()) == 284
    assert '1348269191' in training.all_trial.STORAGE_EVIDENCE_SOURCE
    assert "'canonical_artifacts_checked'] != 3" in training.all_trial.STORAGE_EVIDENCE_SOURCE
    assert training.CAPACITY_RECEIPT_SHA256 == training.capacity.sha256(training.capacity.OUTPUT / 'external_recheck.json')


def test_cli_registration_and_verification_use_the_balanced_batches_before_main():
    assert '"structured_component_balance": components.SPECIFICATION' in training.REGISTER_SOURCE
    assert 'config["structured_component_balance"] != components.SPECIFICATION' in training.ADAPTED_SOURCE
    assert training.SOURCE.index(training.BATCH_WRAPPER) < training.SOURCE.index('if __name__ == "__main__":')
    assert 'result = [components.balance(batch) for batch in unvalidated_batches()]' in training.BATCH_WRAPPER
    assert training.require_evidence.__globals__ is vars(training)
    for module in (training, training.all_trial, training.diagnostics):
        for function in vars(module).values():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert unresolved_globals(function) == []
