from pathlib import Path

import test_single_model_parent_structured_full_audit as parent_tests


TEST_SOURCE = Path(parent_tests.__file__).read_text()
for old, new in {
    "audit_single_model_parent_structured": "audit_single_model_dual_parent",
    "parent_structured_v49": "dual_parent_v50",
    '{"general", "v29"}': '{"general", "v29", "v8"}',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit dual-parent full-audit test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_full_audit_uses_the_complete_dual_component_replayer():
    assert audit.retention_training.expected_component_counts(64) == {"general": 128, "v8": 1536, "v29": 1536}
    assert "validate_saved_components" in audit.retention_training.RetentionReplay.check.__code__.co_names
    assert "require_report_components" in audit.retention_training.require_report.__code__.co_names
