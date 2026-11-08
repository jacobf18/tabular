import test_single_model_standardized_huber_full_audit as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'audit_single_model_standardized_huber': 'audit_single_model_balanced_native_quarter',
    'standardized_teacher_huber_v54': 'balanced_native_quarter_v55',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing native-quarter audit-test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())
del test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention


def test_actual_full_audit_installs_component_replay_and_explicit_quarter_evidence():
    assert audit.training.teacher_policy is audit.blend_replay.teacher_policy
    assert audit.retention_training is audit.training.retention_training
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert 'active["legacy_blend"] = blend_replay.ComponentReplay(' in audit.RUN_AUDIT_SOURCE
    assert audit.RUN_AUDIT_SOURCE.index('ComponentReplay(') < audit.RUN_AUDIT_SOURCE.index('audit(output, training.STEPS)')
    assert '"fixed_quarter_training": True' in audit.ADAPTED_SOURCE
    assert '"displacement_fraction": training.paired.FRACTION' in audit.ADAPTED_SOURCE
    for name in ('run_audit', 'load_regenerated_task', 'retention_load_regenerated_task',
            'regeneration_evidence', 'retention_regeneration_evidence'):
        assert getattr(audit, name).__globals__ is vars(audit)
        assert getattr(audit, name).__closure__ is None


def test_full_audit_uses_quarter_schema_and_complete_unchanged_reference_coverage():
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert audit.checks.FRACTION == .25
    assert '"fixed_quarter_training": True' in audit.ADAPTED_SOURCE
    assert '"fixed_half_training": True' not in audit.ADAPTED_SOURCE
    assert audit.retention_training is audit.training.retention_training
    assert audit.retention_training.expected_component_counts(64) == {'v8': 1536, 'v29': 1664}
    assert 'validate_saved_components' in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert 'require_report_components' in audit.retention_training.require_report.__code__.co_names
