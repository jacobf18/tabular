import types

import test_single_model_family_balanced_retention_pipeline as uniform_pipeline_blueprint


SOURCE = uniform_pipeline_blueprint.SOURCE.replace('family_balanced_retention', 'uniform_v8_retention')
SOURCE = SOURCE.replace('uniform_v8_retention_v43', 'uniform_v8_retention_v44')
exec(compile(SOURCE, __file__, 'exec'), globals())
import run_single_model_uniform_v8_retention_pipeline as pipeline

for name in ('test_actual_prerequisite_requires_all_original_and_added_evidence',
        'test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged',
        'test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads'):
    function = getattr(uniform_pipeline_blueprint, name)
    assert function.__closure__ is None
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_actual_qualification_binds_uniform_v8_and_all_original_weight_checks():
    assert pipeline.training.OUTPUT.name == 'uniform_v8_retention_v44'
    assert pipeline.retention_training.retention.__name__ == 'single_model_uniform_v8_retention'
    for text in ('row["weighting"] != retention.retention_weighting(row["metadata"])',
            'row["gradient_scale"] != 1.', 'registered_weights[row["task"]]'):
        assert text in pipeline.retention_training.SOURCE
