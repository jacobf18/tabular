import inspect

import test_single_model_legacy_blend_teacher_audit as legacy_audit_tests


SOURCE = legacy_audit_tests.SOURCE.replace("audit_single_model_legacy_blend_teacher", "audit_single_model_cohort_anchor_retention")
assert SOURCE != legacy_audit_tests.SOURCE
SOURCE = SOURCE.replace('def test_actual_run_audit_uses_frozen_v29_reference_before_complete_original_audit():',
    'def test_actual_run_audit_uses_both_frozen_references_before_complete_original_audit():')
SOURCE = SOURCE.replace("assert 'del model' not in audit.RUN_AUDIT_SOURCE",
    "assert audit.RUN_AUDIT_SOURCE.index('frozen_reference(') < audit.RUN_AUDIT_SOURCE.index('del model') < audit.RUN_AUDIT_SOURCE.index('RetentionReplay(')")
exec(compile(SOURCE, __file__, "exec"), globals())
for name in ("test_component_audit_is_additional_to_original_task_and_reference_replay",
        "test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention"):
    exec(compile(inspect.getsource(getattr(legacy_audit_tests, name)), __file__, "exec"), globals())


def test_actual_audit_loads_and_replays_both_frozen_references_in_the_original_full_task_loop():
    assert audit.retention_training is audit.training.retention_training
    assert 'references = retention_training.frozen_reference(model, retention_training.identity(config))' in audit.RUN_AUDIT_SOURCE
    assert 'training, config, counts, references, criterion, active["backend"])' in audit.RUN_AUDIT_SOURCE
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in audit.SOURCE
    assert 'del model' in audit.RUN_AUDIT_SOURCE
    assert audit.RUN_AUDIT_SOURCE.index('references = retention_training.frozen_reference') < audit.RUN_AUDIT_SOURCE.index('retention_training.RetentionReplay(')
