import types

import test_single_model_all_imputation_blend_audit as test_blueprint


SOURCE = test_blueprint.SOURCE.replace('audit_single_model_all_imputation_blend', 'audit_single_model_stronger_mean')
exec(compile(SOURCE, __file__, 'exec'), globals())
import audit_single_model_stronger_mean as audit

for name in ('test_component_audit_is_additional_to_original_task_and_reference_replay',
        'test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention',
        'test_actual_audit_loads_and_replays_both_frozen_references_in_the_original_full_task_loop'):
    function = getattr(test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_full_original_audit_is_bound_to_new_training():
    assert audit.training.OUTPUT.name == 'stronger_mean_distillation_v46'
    assert audit.retention_training is audit.training.retention_training
    assert audit.blend_replay is audit.training.blend_replay
    assert audit.training.require_evidence.__globals__ is vars(audit.training)
