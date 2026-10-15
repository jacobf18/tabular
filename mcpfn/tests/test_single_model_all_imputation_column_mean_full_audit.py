import test_single_model_native_parent_full_audit as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "audit_single_model_native_parent": "audit_single_model_all_imputation_column_mean",
    "native_parent_v51": "all_imputation_column_mean_v52",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit all-imputation full-audit test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_all_reference_components_and_new_mean_callback_are_bound():
    assert audit.retention_training.expected_component_counts(64) == {"v8": 1536, "v29": 1664}
    assert "validate_saved_components" in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert "require_report_components" in audit.retention_training.require_report.__code__.co_names
    assert audit.retention_training.retention is audit.training.retention
    assert audit.training.retention.MEAN_SPECIFICATION["scope"] == "every original native and structured imputation query; no ratings"
