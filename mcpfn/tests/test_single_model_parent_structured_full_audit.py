import types

import test_single_model_native_column_full_audit as blueprint


SOURCE = blueprint.SOURCE.replace("audit_single_model_native_column_mean", "audit_single_model_parent_structured")
exec(compile(SOURCE, __file__, "exec"), globals())
import audit_single_model_parent_structured as audit

for name in ("test_component_audit_is_additional_to_original_task_and_reference_replay",
        "test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention"):
    function = getattr(blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_full_audit_uses_parent_structured_references_and_unchanged_native_mean():
    assert audit.training.OUTPUT.name == "parent_structured_v49"
    assert audit.retention_training is audit.training.retention_training
    assert audit.retention_training.retention is audit.training.retention
    assert set(audit.retention_training.retention.REFERENCE_SHA256S) == {"general", "v29"}
    assert 'references = retention_training.frozen_reference(model, retention_training.identity(config))' in audit.RUN_AUDIT_SOURCE
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in audit.FULL_AUDIT_SOURCE
    assert 'config["native_teacher_mean"] != retention.MEAN_SPECIFICATION' in audit.training.ADAPTED_SOURCE
