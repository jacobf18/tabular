import inspect

import test_single_model_anchor_retention_full_pipeline as previous_tests


SOURCE = inspect.getsource(previous_tests)
for old, new in {
    "run_single_model_anchor_retention_full_pipeline": "run_single_model_column_anchor_retention_full_pipeline",
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_column_anchor_retention",
    "anchor_retention_v37": "column_anchor_retention_v38",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_entire_qualification_pipeline_binds_column_retention_and_complete_frozen_protocols():
    assert pipeline.training.retention.SPECIFICATION["variance_floor"] == 1e-6
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.training_audit.retention_training is pipeline.retention_training
    assert pipeline.OUTPUT.name == "column_anchor_retention_v38_pipeline"
    assert pipeline.training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert pipeline.training.STEPS == 64
    assert 'run_single_matrixformer_cached_benchmarks.py' in pipeline.ADAPTED_SOURCE
    assert 'review_single_matrixformer_campaign.py' in pipeline.ADAPTED_SOURCE
