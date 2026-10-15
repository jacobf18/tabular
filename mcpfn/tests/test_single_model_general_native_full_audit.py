import types

import test_single_model_stronger_mean_full_audit as test_blueprint


SOURCE = test_blueprint.SOURCE.replace("audit_single_model_stronger_mean", "audit_single_model_general_native_retention")
exec(compile(SOURCE, __file__, "exec"), globals())
import audit_single_model_general_native_retention as audit

for name in ("test_component_audit_is_additional_to_original_task_and_reference_replay",
        "test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention"):
    function = getattr(test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_full_audit_binds_new_training_and_both_reference_roles():
    assert audit.training.OUTPUT.name == "general_native_retention_v47"
    assert audit.retention_training is audit.training.retention_training
    assert audit.blend_replay is audit.training.blend_replay
    assert set(audit.retention_training.retention.REFERENCE_SHA256S) == {"general", "v8"}
    assert 'references = retention_training.frozen_reference(model, retention_training.identity(config))' in audit.RUN_AUDIT_SOURCE
    assert 'training, config, counts, references, criterion, active["backend"])' in audit.RUN_AUDIT_SOURCE
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in audit.FULL_AUDIT_SOURCE
    assert audit.RUN_AUDIT_SOURCE.index('references = retention_training.frozen_reference') < audit.RUN_AUDIT_SOURCE.index(
        'retention_training.RetentionReplay(')
