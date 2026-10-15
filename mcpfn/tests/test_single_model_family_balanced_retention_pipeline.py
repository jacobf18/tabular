import inspect
import types

import test_single_model_cohort_anchor_retention_pipeline as family_reference_test_blueprint


SOURCE = family_reference_test_blueprint.SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
SOURCE = SOURCE.replace('family_balanced_retention_v42', 'family_balanced_retention_v43')
exec(compile(SOURCE, __file__, 'exec'), globals())
import run_single_model_family_balanced_retention_pipeline as pipeline

for name in ('test_actual_prerequisite_requires_all_original_and_added_evidence',
        'test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged',
        'test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads'):
    function = getattr(family_reference_test_blueprint, name)
    assert function.__closure__ is None
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_full_qualification_requires_metadata_and_registered_weight_agreement():
    assert pipeline.__name__ == 'run_single_model_family_balanced_retention_pipeline'
    source = pipeline.retention_training.SOURCE
    for text in ('row["weighting"] != retention.retention_weighting(row["metadata"])',
            'row["gradient_scale"] != 1.', 'registered_weights[row["task"]]'):
        assert text in source
    assert pipeline.training.OUTPUT.name == 'family_balanced_retention_v43'
