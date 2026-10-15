import test_single_model_balanced_structured_pipeline as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'run_single_model_balanced_structured_pipeline': 'run_single_model_standardized_huber_pipeline',
    'audit_single_model_balanced_structured': 'audit_single_model_standardized_huber',
    'train_single_matrixformer_balanced_structured': 'train_single_matrixformer_standardized_huber',
    'balanced_structured_components_v53': 'standardized_teacher_huber_v54',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing standardized-Huber coordinator test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_pipeline_binds_new_kernel_and_storage_without_changing_gates():
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.training.diagnostics is pipeline.cohort_trial.component_storage
    assert pipeline.training.retention is pipeline.training.preflight.retention
    assert 'require_report_components' in pipeline.retention_training.require_report.__code__.co_names
