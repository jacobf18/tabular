import inspect
import types

import test_single_model_cohort_anchor_retention_audit as family_reference_test_blueprint


SOURCE = family_reference_test_blueprint.SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
exec(compile(SOURCE, __file__, 'exec'), globals())
import audit_single_model_family_balanced_retention as audit

for name in ('test_component_audit_is_additional_to_original_task_and_reference_replay',
        'test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention',
        'test_actual_audit_loads_and_replays_both_frozen_references_in_the_original_full_task_loop'):
    function = getattr(family_reference_test_blueprint, name)
    assert function.__closure__ is None
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_full_audit_uses_the_weight_binding_helper():
    assert audit.__name__ == 'audit_single_model_family_balanced_retention'
    source = audit.retention_training.SOURCE
    for text in ('decision["weighting"] != retention.retention_weighting(episode.metadata)',
            'cached["weighting"] != decision["weighting"]', 'decision["gradient_scale"] != 1.'):
        assert text in source
    assert audit.training.OUTPUT.name == 'family_balanced_retention_v43'
