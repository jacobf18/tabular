import types

import test_single_model_all_imputation_blend_pipeline as test_blueprint


SOURCE = test_blueprint.SOURCE
for old, new in {
    'run_single_model_all_imputation_blend_pipeline': 'run_single_model_stronger_mean_pipeline',
    'audit_single_model_all_imputation_blend': 'audit_single_model_stronger_mean',
    'train_single_matrixformer_all_imputation_blend': 'train_single_matrixformer_stronger_mean',
    'all_imputation_blend_v45': 'stronger_mean_distillation_v46',
}.items():
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, 'exec'), globals())
import run_single_model_stronger_mean_pipeline as pipeline

for name in ('test_actual_prerequisite_requires_all_original_and_added_evidence',
        'test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged',
        'test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads'):
    function = getattr(test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_original_full_qualification_is_bound_to_new_training_and_original_retention():
    assert pipeline.training.OUTPUT.name == 'stronger_mean_distillation_v46'
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.blend_replay is pipeline.training.blend_replay
    assert pipeline.training.require_evidence.__globals__ is vars(pipeline.training)
