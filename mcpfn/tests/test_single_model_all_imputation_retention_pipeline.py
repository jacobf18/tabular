import test_single_model_fixed_half_shrunk_retention_pipeline as previous_tests


SOURCE = previous_tests.SOURCE
for old, new in {
    "run_single_model_fixed_half_shrunk_retention_pipeline": "run_single_model_all_imputation_retention_pipeline",
    "train_single_matrixformer_fixed_half_shrunk_retention": "train_single_matrixformer_all_imputation_retention",
    "fixed_half_shrunk_retention_v39": "all_imputation_retention_v40",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_pipeline_keeps_full_comparisons_and_new_storage_identity():
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.diagnostics is pipeline.training.diagnostics
    assert pipeline.retention_training is pipeline.training.retention_training
    assert 'fixed_trial.reconstruct_candidate(training, payload, "cuda")' in pipeline.ADAPTED_SOURCE
    assert 'fixed_trial.require_report(training, audit)' in pipeline.ADAPTED_SOURCE
    assert "run_single_matrixformer_cached_benchmarks.py" in pipeline.ADAPTED_SOURCE
    assert "review_single_matrixformer_campaign.py" in pipeline.ADAPTED_SOURCE
    assert len(pipeline.training.confirmation_cases()) == 284
