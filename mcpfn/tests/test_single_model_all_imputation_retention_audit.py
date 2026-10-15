import test_single_model_fixed_half_shrunk_retention_audit as previous_tests


SOURCE = previous_tests.SOURCE
assert "audit_single_model_fixed_half_shrunk_retention" in SOURCE
SOURCE = SOURCE.replace("audit_single_model_fixed_half_shrunk_retention", "audit_single_model_all_imputation_retention")
exec(compile(SOURCE, __file__, "exec"), globals())


def test_full_scope_retention_audit_and_fixed_half_remain_mandatory():
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert audit.retention_training is audit.training.retention_training
    assert audit.retention_training.retention.SPECIFICATION["legacy_components"] == list(range(8))
    assert '"fixed_half_training": True' in audit.ADAPTED_SOURCE
    assert "retention_training.RetentionReplay(" in audit.RUN_AUDIT_SOURCE
