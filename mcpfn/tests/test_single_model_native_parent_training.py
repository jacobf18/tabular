from pathlib import Path


TEST_SOURCE = Path(__file__).with_name("test_single_model_dual_parent_training.py").read_text()
for old, new in {
    "single_model_dual_parent_training as integration": "single_model_native_parent_training as integration",
    "preflight_single_model_dual_parent_retention as probe": "preflight_single_model_native_parent_retention as probe",
    '{"general": 2, "v8": 24, "v29": 24}': '{"v8": 24, "v29": 26}',
    '{"general": 2, "structured_mean": 24}': '{"v29": 2, "structured_mean": 24}',
    '{name: calls.count(name) for name in references}': '{name: calls.count(name) for name in references if calls.count(name)}',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent training test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_full_counts_keep_all_native_and_structured_components_and_zero_general():
    assert integration.expected_component_counts(0) == {}
    assert integration.expected_component_counts(1) == {"v8": 24, "v29": 26}
    assert integration.expected_component_counts(64) == {"v8": 1536, "v29": 1664}
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            integration.expected_component_counts(invalid)
