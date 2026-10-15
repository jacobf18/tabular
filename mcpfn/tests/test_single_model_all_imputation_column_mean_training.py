import test_single_model_native_parent_training as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "single_model_native_parent_training as integration": "single_model_all_imputation_column_mean_training as integration",
    "preflight_single_model_native_parent_retention as probe": "preflight_single_model_all_imputation_column_mean as probe",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit all-imputation full-trial replay adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_all_imputation_objective_and_native_reference_counts_are_bound():
    assert integration.retention is probe.retention
    assert integration.expected_component_counts(0) == {}
    assert integration.expected_component_counts(1) == {"v8": 24, "v29": 26}
    assert integration.expected_component_counts(64) == {"v8": 1536, "v29": 1664}
    assert integration.retention.MEAN_SPECIFICATION["scope"] == "every original native and structured imputation query; no ratings"
    assert integration.INITIALIZER == parent_tests.integration.INITIALIZER
    assert integration.EVALUATION_BASELINE == parent_tests.integration.EVALUATION_BASELINE
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            integration.expected_component_counts(invalid)
