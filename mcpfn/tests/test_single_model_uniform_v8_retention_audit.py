import types

import test_single_model_family_balanced_retention_audit as uniform_audit_blueprint


SOURCE = uniform_audit_blueprint.SOURCE.replace('family_balanced_retention', 'uniform_v8_retention')
exec(compile(SOURCE, __file__, 'exec'), globals())
import audit_single_model_uniform_v8_retention as audit

for name in ('test_component_audit_is_additional_to_original_task_and_reference_replay',
        'test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention',
        'test_actual_audit_loads_and_replays_both_frozen_references_in_the_original_full_task_loop'):
    function = getattr(uniform_audit_blueprint, name)
    assert function.__closure__ is None
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_actual_full_audit_binds_uniform_v8_with_original_weight_validation():
    assert audit.training.OUTPUT.name == 'uniform_v8_retention_v44'
    assert audit.retention_training.retention.__name__ == 'single_model_uniform_v8_retention'
    for text in ('decision["weighting"] != retention.retention_weighting(episode.metadata)',
            'cached["weighting"] != decision["weighting"]', 'decision["gradient_scale"] != 1.'):
        assert text in audit.retention_training.SOURCE
