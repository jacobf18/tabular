import test_single_model_standardized_huber_pipeline as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    'run_single_model_standardized_huber_pipeline': 'run_single_model_balanced_native_quarter_pipeline',
    'audit_single_model_standardized_huber': 'audit_single_model_balanced_native_quarter',
    'train_single_matrixformer_standardized_huber': 'train_single_matrixformer_balanced_native_quarter',
    'standardized_teacher_huber_v54': 'balanced_native_quarter_v55',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f'Missing native-quarter pipeline-test adaptation: {old}')
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, 'exec'), globals())


def test_pipeline_binds_explicit_quarter_reconstruction_and_all_original_gates():
    assert pipeline.fixed_trial is pipeline.training.fixed_trial is pipeline.training_audit.fixed_trial
    assert pipeline.fixed_trial.protection is pipeline.training.paired
    assert pipeline.training.paired.FRACTION == .25
    assert 'fixed_trial.reconstruct_candidate(training, payload, "cuda")' in pipeline.ADAPTED_SOURCE
    assert 'fixed_trial.check_replayed_candidate(payload, candidate["parameters"])' in pipeline.ADAPTED_SOURCE
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.training.diagnostics is pipeline.cohort_trial.component_storage
