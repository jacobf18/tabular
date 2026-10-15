import types

import test_single_model_cohort_anchor_retention_trial as uniform_trial_blueprint
import train_single_matrixformer_family_balanced_retention as previous_training


function = uniform_trial_blueprint.fixed_preflight
fixed_preflight = types.FunctionType(function.__code__, globals(), 'fixed_preflight', function.__defaults__)
SOURCE = uniform_trial_blueprint.ADAPTED_TEST_SOURCE.replace('cohort_anchor_retention', 'uniform_v8_retention')
for old, new in {'1061100901': '1121100901', '1062100901': '1122100901',
        '1071100901': '1131100901', '1072100901': '1132100901'}.items():
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())


def test_only_retention_assignment_changes_in_the_full_trial():
    assert training.OUTPUT.name == 'uniform_v8_retention_v44'
    assert training.PARENT == previous_training.PARENT and training.PARENT_SHA256 == previous_training.PARENT_SHA256
    assert training.retention is training.preflight.retention and training.retention is not previous_training.retention
    assert training.retention_training is not previous_training.retention_training
    assert training.retention.BACKWARD_SOURCE_SHA256 == previous_training.retention.BACKWARD_SOURCE_SHA256
    assert training.teacher_policy is previous_training.teacher_policy and training.paired is previous_training.paired
    assert training.RATE == previous_training.RATE and training.TRIAL_QUALIFICATION == previous_training.TRIAL_QUALIFICATION
    assert training.STORAGE_ACCOUNTING == previous_training.STORAGE_ACCOUNTING
    assert training.diagnostics is previous_training.diagnostics and training.capacity.probe is training.preflight
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([
        previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, prior in zip(training.batches(), previous_training.batches()):
        training.retention.validate_task_mixture(batch['tasks'])
        for task, original in zip(batch['tasks'], prior['tasks']):
            restored = deepcopy(task)
            restored['gradient_seed'] = original['gradient_seed']
            specification = restored['specification']
            specification.get('structured', specification)['seed'] = original['specification'].get(
                'structured', original['specification'])['seed']
            assert restored == original


def test_registration_binds_exact_external_receipt_and_all_actual_runtime_helpers():
    assert training.CAPACITY_RECEIPT_SHA256 == '97b3d82c1ee9419d201b3f0b19db2c414599c11291aad958f75acdf42ae2fa8d'
    assert "'external_recheck.json'" in training.RETENTION_EVIDENCE_SOURCE
    assert 'independent_recheck.json' not in training.RETENTION_EVIDENCE_SOURCE
    assert 'selected_checkpoints_unchanged' in training.RETENTION_EVIDENCE_SOURCE
    assert 'source_sha256' in training.RETENTION_EVIDENCE_SOURCE
    assert 'require_completed_v43' in training.RETENTION_EVIDENCE_SOURCE
    assert 'require_probe' not in training.RETENTION_EVIDENCE_SOURCE
    assert 'config["arithmetic"] != all_trial.ARITHMETIC' in training.ADAPTED_SOURCE
    assert training.SOURCE.index('unvalidated_batches = batches') < training.SOURCE.index('\nif __name__ == "__main__":')
    for name in ('trial', 'audit', 'pipeline', 'training'):
        assert f'tests/test_single_model_uniform_v8_retention_{name}.py' in training.REGISTER_SOURCE
    for name in ('train', 'verify_registration', 'register_trial', 'setup'):
        assert getattr(training, name).__globals__['retention_training'] is training.retention_training


def test_registration_requires_the_completed_capacity_receipt(monkeypatch):
    monkeypatch.setattr(training, 'CAPACITY_RECEIPT_SHA256', None)
    with pytest.raises(ValueError, match='storage-preservation receipt'):
        training.retention_evidence()


@pytest.mark.parametrize('mutation', [None, 'complete', 'canonical_artifacts_checked', 'config_sha256',
    'all_original_serialized_bytes_and_paths_preserved', 'selected_checkpoints_unchanged',
    'provider_persistent_backup', 'bytes_relocated', 'training_admission_bytes', 'checkpoint_available_bytes',
    'policy', 'registration', 'status', 'receipt_digest'])
def test_capacity_binding_checks_actual_receipt_and_every_preserved_input(tmp_path, monkeypatch, mutation):
    from types import SimpleNamespace

    events = []
    admission = 21877489664
    receipt = {'complete': True, 'canonical_artifacts_checked': 3, 'config_sha256': 'config-digest',
        'all_original_serialized_bytes_and_paths_preserved': True, 'selected_checkpoints_unchanged': True,
        'provider_persistent_backup': False, 'bytes_relocated': 1652558524,
        'training_admission_bytes': admission, 'checkpoint_available_bytes': admission + 100,
        'artifact_sha256': {'artifact': 'original-bytes'}}
    configuration = {'policy': 'preservation-policy', 'input_sha256': {'input': 'input-digest'},
        'source_sha256': {'source': 'source-digest'}}
    records = {'external_recheck.json': receipt, 'config.json': configuration,
        'registration.json': {'config_sha256': 'config-digest'}, 'status.json': {'state': 'complete'}}
    if mutation in receipt:
        original = receipt[mutation]
        receipt[mutation] = not original if type(original) is bool else 0
    elif mutation == 'policy':
        configuration['policy'] = 'changed-policy'
    elif mutation == 'registration':
        records['registration.json']['config_sha256'] = 'changed'
    elif mutation == 'status':
        records['status.json']['state'] = 'failed'
    digest = 'changed' if mutation == 'receipt_digest' else training.CAPACITY_RECEIPT_SHA256
    capacity = SimpleNamespace(OUTPUT=tmp_path, TRAINING_ADMISSION_BYTES=admission, POLICY='preservation-policy',
        read=lambda path: records[path.name],
        sha256=lambda path: digest if path.name == 'external_recheck.json' else 'config-digest',
        verify_hashes=lambda identities: events.append(identities),
        probe=SimpleNamespace(OUTPUT=tmp_path / 'probe', verify=lambda path: events.append('probe'),
            retention_records=lambda path: events.append('references'),
            require_completed_v43=lambda: {'predecessor': 'verified'}))
    monkeypatch.setattr(training, 'capacity', capacity)
    if mutation is not None:
        with pytest.raises(ValueError):
            training.retention_evidence()
        assert events == []
    else:
        result = training.retention_evidence()
        assert events == [receipt['artifact_sha256'], configuration['input_sha256'],
            configuration['source_sha256'], 'probe', 'references']
        assert result['input'] == 'input-digest' and result['predecessor'] == 'verified'
        assert result[str(tmp_path / 'external_recheck.json')] == digest
