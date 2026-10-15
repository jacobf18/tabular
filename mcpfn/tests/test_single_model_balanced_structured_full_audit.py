import test_single_model_all_imputation_column_mean_full_audit as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'audit_single_model_all_imputation_column_mean': 'audit_single_model_balanced_structured',
    'all_imputation_column_mean_v52': 'balanced_structured_components_v53',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing balanced-prior full-audit test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_balanced_priors_keep_original_full_teacher_and_reference_replay():
    assert audit.retention_training is audit.training.retention_training
    assert audit.retention_training.expected_component_counts(64) == {'v8': 1536, 'v29': 1664}
    assert 'validate_saved_components' in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert 'require_report_components' in audit.retention_training.require_report.__code__.co_names
    assert audit.training.batches() != audit.training.unvalidated_batches()
