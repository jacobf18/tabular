import test_single_model_all_imputation_column_mean_pipeline as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'run_single_model_all_imputation_column_mean_pipeline': 'run_single_model_balanced_structured_pipeline',
    'audit_single_model_all_imputation_column_mean': 'audit_single_model_balanced_structured',
    'train_single_matrixformer_all_imputation_column_mean': 'train_single_matrixformer_balanced_structured',
    'all_imputation_column_mean_v52': 'balanced_structured_components_v53',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing balanced-prior coordinator test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_balanced_pipeline_binds_original_kernel_and_new_storage():
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.training.diagnostics is pipeline.cohort_trial.component_storage
    assert 'require_report_components' in pipeline.retention_training.require_report.__code__.co_names
