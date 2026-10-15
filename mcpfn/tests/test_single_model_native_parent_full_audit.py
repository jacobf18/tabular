import test_single_model_dual_parent_full_audit as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "audit_single_model_dual_parent": "audit_single_model_native_parent",
    "dual_parent_v50": "native_parent_v51",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent full-audit test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_full_audit_keeps_two_components_and_replaces_only_native_reference():
    assert audit.retention_training.expected_component_counts(64) == {"v8": 1536, "v29": 1664}
    assert "validate_saved_components" in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert "require_report_components" in audit.retention_training.require_report.__code__.co_names
    base = audit.training.read(audit.training.original.alignment.fixed.PRODUCER / "config.json")
    assert set(audit.retention_training.reference_identities(
        audit.training.PARENT, base)) == {"general", "v29", "v8"}
