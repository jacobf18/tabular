import types

import test_single_model_uniform_v8_retention_pipeline as pipeline_test_blueprint


SOURCE = pipeline_test_blueprint.SOURCE.replace("run_single_model_uniform_v8_retention_pipeline", "run_single_model_all_imputation_blend_pipeline")
SOURCE = SOURCE.replace("audit_single_model_uniform_v8_retention", "audit_single_model_all_imputation_blend")
SOURCE = SOURCE.replace("train_single_matrixformer_uniform_v8_retention", "train_single_matrixformer_all_imputation_blend")
SOURCE = SOURCE.replace("uniform_v8_retention_v44", "all_imputation_blend_v45")
exec(compile(SOURCE, __file__, "exec"), globals())
import run_single_model_all_imputation_blend_pipeline as pipeline

for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",
        "test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged",
        "test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads"):
    function = getattr(pipeline_test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_qualification_rechecks_all_imputation_components_and_original_reference_weights():
    assert pipeline.training.OUTPUT.name == "all_imputation_blend_v45"
    assert pipeline.retention_training.retention.__name__ == "single_model_uniform_v8_retention"
    assert pipeline.blend_replay is pipeline.training.blend_replay
    assert pipeline.blend_replay.teacher_policy.POLICY["structured"] == "observed_blend"
    for text in ('row["weighting"] != retention.retention_weighting(row["metadata"])',
            'row["gradient_scale"] != 1.', 'registered_weights[row["task"]]'):
        assert text in pipeline.retention_training.SOURCE
