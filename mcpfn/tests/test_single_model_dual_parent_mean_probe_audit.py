from pathlib import Path


TEST_SOURCE = Path(__file__).with_name("test_single_model_parent_structured_mean_probe_audit.py").read_text()
for old, new in {
    "audit_single_model_parent_structured_mean_probe": "audit_single_model_dual_parent_mean_probe",
    "single_model_parent_structured_retention": "single_model_dual_parent_retention",
    "parent_structured_retention_preflight_v1_mean_audit": "dual_parent_retention_preflight_v1_mean_audit",
    "test_single_model_parent_structured_mean_probe_audit.py": "test_single_model_dual_parent_mean_probe_audit.py",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit dual-parent mean-audit test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_all_components_and_original_mean_implementation_are_bound():
    assert 'Path(blueprint.blueprint.__file__).resolve()' in audit.AUDIT_SOURCE
    assert '"component_grids"] != 50' in audit.AUDIT_SOURCE
    assert '"reference_grids"] != 26' in audit.AUDIT_SOURCE
    assert audit.CONFIG_SHA256 == "040ee68892038461bc2c3f31492dec9ebdced9e03e0b285dcc98500dd282f9ac"
