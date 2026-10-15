from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

import test_single_model_cohort_anchor_retention_trial as complete_trial_blueprint
import train_single_matrixformer_all_imputation_blend as previous_training
import train_single_matrixformer_stronger_mean as training


function = complete_trial_blueprint.fixed_preflight
fixed_preflight = types.FunctionType(function.__code__, globals(), 'fixed_preflight', function.__defaults__)
ADAPTED_TEST_SOURCE = complete_trial_blueprint.ADAPTED_TEST_SOURCE.replace('cohort_anchor_retention', 'stronger_mean')
for old, new in {'1061100901': '1181100901', '1062100901': '1182100901',
        '1071100901': '1191100901', '1072100901': '1192100901'}.items():
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, 'exec'), globals())


def test_stronger_objective_is_only_scientific_change():
    assert training.OUTPUT.name == 'stronger_mean_distillation_v46'
    for name in ('PARENT', 'PARENT_SHA256', 'RATE', 'PARENT_UPDATES', 'BOUNDARIES', 'STEPS', 'TRIAL_QUALIFICATION',
            'STORAGE_ACCOUNTING', 'retention', 'retention_training', 'teacher_policy', 'diagnostics', 'paired'):
        assert getattr(training, name) == getattr(previous_training, name)
    assert training.capacity.probe is training.preflight
    assert training.stronger.MEAN_WEIGHT == 20.
    fresh = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    old = training.original.alignment.fixed.seeds_in([
        previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(fresh) == 3868 and not fresh.intersection(old)
    for batch, prior in zip(training.batches(), previous_training.batches()):
        training.retention.validate_task_mixture(batch['tasks'])
        for task, original in zip(batch['tasks'], prior['tasks']):
            restored = deepcopy(task)
            restored['gradient_seed'] = original['gradient_seed']
            restored['specification'].get('structured', restored['specification'])['seed'] = original['specification'].get(
                'structured', original['specification'])['seed']
            assert restored == original
    assert sum(training.STORAGE_ACCOUNTING['scratch'].values()) == 41224310912


def test_registration_and_real_callback_bind_exact_objective():
    assert '"mean_distillation": stronger.SPECIFICATION' in training.REGISTER_SOURCE
    assert 'config["mean_distillation"] != stronger.SPECIFICATION' in training.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in training.ADAPTED_SOURCE
    for name in ('train', 'verify_registration', 'register_trial', 'setup', 'require_evidence'):
        function = getattr(training, name)
        assert function.__globals__ is vars(training)
    assert {'all_trial', 'require_probe', 'preflight', 'validate_preflight', 'memory_evidence',
        'require_evidence', 'failed_entry_evidence', 'retention_evidence'} == set(training.require_evidence.__code__.co_names)
    for name in ('trial', 'full_audit', 'pipeline'):
        assert f'tests/test_single_model_stronger_mean_{name}.py' in training.REGISTER_SOURCE
    assert training.stronger.original.balanced_loss is training.stronger.BASE_LOSS


def test_capacity_receipt_must_be_bound(monkeypatch):
    monkeypatch.setattr(training, 'CAPACITY_RECEIPT_SHA256', None)
    with pytest.raises(ValueError, match='storage-preservation receipt'):
        training.retention_evidence()


def test_registration_keeps_all_original_and_recovered_evidence(monkeypatch):
    monkeypatch.setattr(training.all_trial, 'require_probe', lambda probe, validate: {'recovered_probe': 'hash'})
    monkeypatch.setattr(training.memory_evidence, 'require_evidence', lambda: {'allocator_proof': 'hash'})
    monkeypatch.setattr(training, 'failed_entry_evidence', lambda: {'original_entry_proof': 'hash'})
    monkeypatch.setattr(training, 'retention_evidence', lambda: {'exact_capacity': 'hash'})
    assert training.require_evidence() == dict.fromkeys(
        ('recovered_probe', 'allocator_proof', 'original_entry_proof', 'exact_capacity'), 'hash')


@pytest.mark.parametrize('field,value', [(None, None), ('complete', False), ('canonical_artifacts_checked', 4),
    ('bytes_relocated', 2138934893), ('config_sha256', 'wrong'), ('all_original_serialized_bytes_and_paths_preserved', False),
    ('selected_checkpoints_unchanged', False), ('provider_persistent_backup', True),
    ('training_admission_bytes', 1), ('checkpoint_available_bytes', 1)])
def test_capacity_preserves_exact_receipt_and_every_bound_input(field, value):
    output = Path('/nonexistent/capacity')
    receipt = {'complete': True, 'canonical_artifacts_checked': 5, 'bytes_relocated': 2138934894,
        'config_sha256': 'config-hash', 'all_original_serialized_bytes_and_paths_preserved': True,
        'selected_checkpoints_unchanged': True, 'provider_persistent_backup': False,
        'training_admission_bytes': 21877489664, 'checkpoint_available_bytes': 21991448576,
        'artifact_sha256': {'original_and_preserved': 'artifact-hash'}}
    configuration = {'policy': 'exact policy', 'input_sha256': {'closure': 'input-hash'},
        'source_sha256': {'implementation': 'source-hash'}}
    documents = {output / 'external_recheck.json': receipt, output / 'config.json': configuration,
        output / 'registration.json': {'config_sha256': 'config-hash'}, output / 'status.json': {'state': 'complete'}}
    events = []
    capacity = SimpleNamespace(OUTPUT=output, POLICY='exact policy', TRAINING_ADMISSION_BYTES=21877489664,
        read=lambda path: documents[path], verify_hashes=lambda hashes: events.append(hashes),
        sha256=lambda path: 'receipt-hash' if path.name == 'external_recheck.json' else 'config-hash')
    if field is not None:
        receipt[field] = value
        with pytest.raises(ValueError):
            training.all_trial.storage_evidence(capacity, 'receipt-hash')
    else:
        result = training.all_trial.storage_evidence(capacity, 'receipt-hash')
        assert result == {'closure': 'input-hash', str(output / 'external_recheck.json'): 'receipt-hash',
            str(output / 'config.json'): 'config-hash'}
        assert events == [receipt['artifact_sha256'], configuration['input_sha256'], configuration['source_sha256']]


@pytest.mark.parametrize('mutation', [None, 'failure_erased', 'wrong_error', 'missing_retention', 'gate_changed',
    'rerun', 'wrong_objective', 'missing_components', 'incomplete_recovery', 'wrong_recovery_hash'])
def test_full_trial_consumes_recovery_without_relabelling_failed_original(monkeypatch, mutation):
    recipe, recovery = training.all_trial, training.all_trial.recovery
    original_root = Path('/nonexistent/original')
    recovered_root = Path('/nonexistent/recovered')
    monkeypatch.setattr(recovery, 'OUTPUT', recovered_root)
    result = dict.fromkeys(('complete', 'accepted', 'original_failure_preserved', 'original_base_audit_reused',
        'original_inventory_unchanged'), True)
    result.update(dict.fromkeys(('training_repeated', 'model_or_reference_inference_repeated', 'new_seeds_consumed',
        'selection_or_benchmarks_evaluated', 'cuda_initialized', 'gate_changed'), False))
    result.update(recovery_config_sha256='config-hash', base_audit_sha256='audit-hash',
        base_gate_sha256='gate-hash', retention_audit_sha256='retention-hash')
    documents = {
        recovered_root / 'recovery_result.json': result,
        original_root / 'status.json': {'state': 'failed', 'failed_stage': 'audit', 'pid': 999999999,
            'error': 'NameError("name \'Counter\' is not defined")'},
        recovered_root / 'status.json': {'state': 'complete', 'pid': 999999998,
            'original_failed_status_preserved': True, 'retention_audited': True},
        recovered_root / 'recovery_config.json': {'source_sha256': {}, 'input_sha256': {}},
        original_root / 'result.json': {'original': True},
        recovered_root / 'audit.json': {'artifact_sha256': {}, 'teacher_replay': {
            'blend_tasks': 26, 'component_grids': 52, 'observed_weights_refitted': True}},
        recovered_root / 'gate.json': {'original_gate': True},
        recovered_root / 'retention_audit.json': {'complete': True, 'eligibility_decisions': 28,
            'reference_grids': 26, 'reference_role_counts': {'v8': 26}, 'artifact_sha256': {}},
        original_root / 'config.json': {'mean_distillation': deepcopy(training.stronger.SPECIFICATION)},
    }
    if mutation == 'failure_erased':
        documents[original_root / 'status.json']['state'] = 'complete'
    elif mutation == 'wrong_error':
        documents[original_root / 'status.json']['error'] = 'different failure'
    elif mutation == 'missing_retention':
        documents[recovered_root / 'retention_audit.json']['reference_grids'] = 25
    elif mutation == 'gate_changed':
        result['gate_changed'] = True
    elif mutation == 'rerun':
        result['training_repeated'] = True
    elif mutation == 'wrong_objective':
        documents[original_root / 'config.json']['mean_distillation']['teacher_mean_huber_weight'] = 2.
    elif mutation == 'missing_components':
        documents[recovered_root / 'audit.json']['teacher_replay']['component_grids'] = 50
    elif mutation == 'incomplete_recovery':
        result['complete'] = False
    events = []
    probe = SimpleNamespace(OUTPUT=original_root, verify=lambda output: events.append(output),
        require_completed_v45=lambda: {'completed_v45': 'hash'}, read=lambda path: documents[path],
        sha256=lambda path: ('wrong' if mutation == 'wrong_recovery_hash' else recipe.RECOVERY_SHA256),
        verify_hashes=lambda hashes: events.append(hashes))

    def validate(original_result, audit, gate, status, digest):
        assert original_result == {'original': True} and gate == {'original_gate': True}
        assert status['state'] == 'complete' and status['retention_audited'] is True
        assert digest == 'audit-hash'
        events.append('unchanged_acceptance_validator')

    if mutation:
        with pytest.raises(ValueError):
            recipe.require_probe(probe, validate)
    else:
        before = deepcopy(documents)
        proof = recipe.require_probe(probe, validate)
        assert documents == before and 'unchanged_acceptance_validator' in events
        assert proof['completed_v45'] == 'hash' and str(recovered_root / 'retention_audit.json') in proof


def test_every_dynamic_runtime_and_audit_dependency_resolves():
    import audit_single_model_stronger_mean as audit
    import run_single_model_stronger_mean_pipeline as pipeline
    from recover_single_model_stronger_mean_audit import missing_globals

    for module in (training, audit, pipeline, training.all_trial, training.stronger,
            training.retention_training, training.retention, training.blend_replay, training.teacher_policy):
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__globals__ is vars(module):
                assert not missing_globals(value, vars(module)), (module.__name__, name, missing_globals(value, vars(module)))
            elif inspect.isclass(value) and value.__module__ == module.__name__:
                for method_name, method in vars(value).items():
                    if inspect.isfunction(method):
                        assert not missing_globals(method, method.__globals__), (module.__name__, name, method_name)
