import test_single_model_balanced_structured_full_audit as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'audit_single_model_balanced_structured': 'audit_single_model_standardized_huber',
    'balanced_structured_components_v53': 'standardized_teacher_huber_v54',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing standardized-Huber full-audit test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_full_audit_keeps_original_teacher_and_reference_coverage():
    assert audit.retention_training is audit.training.retention_training
    assert audit.retention_training.expected_component_counts(64) == {'v8': 1536, 'v29': 1664}
    assert 'validate_saved_components' in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert 'require_report_components' in audit.retention_training.require_report.__code__.co_names
    assert audit.training.retention is audit.training.preflight.retention
