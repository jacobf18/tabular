import inspect

import test_single_model_cohort_anchor_retention_trial as previous_tests
import train_single_matrixformer_cohort_anchor_retention as previous_training


exec(compile(inspect.getsource(previous_tests.fixed_preflight), __file__, 'exec'), globals())
SOURCE = previous_tests.ADAPTED_TEST_SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
for old, new in {'1061100901': '1091100901', '1062100901': '1092100901',
        '1071100901': '1101100901', '1072100901': '1102100901'}.items():
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())


def test_only_retention_weights_change_in_the_registered_full_trial():
    assert training.OUTPUT.name == 'family_balanced_retention_v43'
    assert training.PARENT == previous_training.PARENT and training.PARENT_SHA256 == previous_training.PARENT_SHA256
    assert training.retention is training.preflight.retention and training.retention is not previous_training.retention
    assert training.retention_training is not previous_training.retention_training
    assert training.retention.BACKWARD_SOURCE_SHA256 != previous_training.retention.BACKWARD_SOURCE_SHA256
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


def test_registration_binds_new_sources_receipt_and_cli_mixture_validation():
    assert training.CAPACITY_RECEIPT_SHA256 == '43fc295406bb622ad06cc8c92ba021a045755c87f1be70c2eb80ff7dc16d64cb'
    assert 'receipt["canonical_artifacts_checked"] != 3' in training.ADAPTATION_SOURCE
    assert '"retention_references": retention_training.reference_identities(PARENT, base)' in training.REGISTER_SOURCE
    assert '"arithmetic": all_trial.ARITHMETIC' in training.REGISTER_SOURCE
    assert 'config["arithmetic"] != all_trial.ARITHMETIC' in training.ADAPTED_SOURCE
    assert training.SOURCE.index('unvalidated_batches = batches') < training.SOURCE.index('\nif __name__ == "__main__":')
    for name in ('trial', 'audit', 'pipeline', 'training'):
        assert f'tests/test_single_model_family_balanced_retention_{name}.py' in training.REGISTER_SOURCE
    for name in ('train', 'verify_registration', 'register_trial', 'setup'):
        assert getattr(training, name).__globals__['retention_training'] is training.retention_training


def test_registration_requires_the_completed_capacity_receipt(monkeypatch):
    monkeypatch.setattr(training, 'CAPACITY_RECEIPT_SHA256', None)
    with pytest.raises(ValueError, match='storage-preservation receipt'):
        training.retention_evidence()
