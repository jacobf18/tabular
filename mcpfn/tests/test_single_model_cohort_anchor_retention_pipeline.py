import inspect

import test_single_model_legacy_blend_teacher_pipeline as legacy_pipeline_tests


SOURCE = legacy_pipeline_tests.SOURCE
for old, new in {
    "run_single_model_legacy_blend_teacher_pipeline": "run_single_model_cohort_anchor_retention_pipeline",
    "train_single_matrixformer_legacy_blend_teacher": "train_single_matrixformer_cohort_anchor_retention",
    "legacy_blend_teacher_v41": "cohort_anchor_retention_v42",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",
        "test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged"):
    exec(compile(inspect.getsource(getattr(legacy_pipeline_tests, name)), __file__, "exec"), globals())


def test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads():
    assert '"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "24", "MKL_NUM_THREADS": "24"' in pipeline.ADAPTED_SOURCE
    assert '"OPENBLAS_NUM_THREADS": "4"' not in pipeline.ADAPTED_SOURCE
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in pipeline.SOURCE
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.training.retention is pipeline.training.preflight.retention
