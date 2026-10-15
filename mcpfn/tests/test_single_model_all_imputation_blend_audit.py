import types

import test_single_model_uniform_v8_retention_audit as audit_test_blueprint


SOURCE = audit_test_blueprint.SOURCE.replace("audit_single_model_uniform_v8_retention", "audit_single_model_all_imputation_blend")
exec(compile(SOURCE, __file__, "exec"), globals())
import audit_single_model_all_imputation_blend as audit

for name in ("test_component_audit_is_additional_to_original_task_and_reference_replay",
        "test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention",
        "test_actual_audit_loads_and_replays_both_frozen_references_in_the_original_full_task_loop"):
    function = getattr(audit_test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_every_imputation_task_has_independent_component_replay_with_unchanged_retention():
    assert audit.training.OUTPUT.name == "all_imputation_blend_v45"
    assert audit.retention_training.retention.__name__ == "single_model_uniform_v8_retention"
    assert audit.blend_replay is audit.training.blend_replay
    assert audit.blend_replay.teacher_policy.POLICY["structured"] == "observed_blend"
    for text in ('decision["weighting"] != retention.retention_weighting(episode.metadata)',
            'cached["weighting"] != decision["weighting"]', 'decision["gradient_scale"] != 1.'):
        assert text in audit.retention_training.SOURCE
