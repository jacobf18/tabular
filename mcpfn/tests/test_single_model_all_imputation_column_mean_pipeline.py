import test_single_model_native_parent_pipeline as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "run_single_model_native_parent_pipeline": "run_single_model_all_imputation_column_mean_pipeline",
    "audit_single_model_native_parent": "audit_single_model_all_imputation_column_mean",
    "train_single_matrixformer_native_parent": "train_single_matrixformer_all_imputation_column_mean",
    "native_parent_v51": "all_imputation_column_mean_v52",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit all-imputation pipeline test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_pipeline_binds_unchanged_component_counts_and_all_imputation_mean():
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert "require_report_components" in pipeline.retention_training.require_report.__code__.co_names
    assert pipeline.training.retention.MEAN_SPECIFICATION["scope"] == "every original native and structured imputation query; no ratings"
