import test_single_model_all_imputation_retention_audit as original_template


SOURCE = original_template.SOURCE
assert "audit_single_model_all_imputation_retention" in SOURCE
SOURCE = SOURCE.replace("audit_single_model_all_imputation_retention", "audit_single_model_legacy_blend_teacher")
SOURCE = SOURCE.replace('token = audit.REPLAY.set({"anchor_retention": reference})',
    'token = audit.REPLAY.set({"anchor_retention": reference, "legacy_blend": '
    'SimpleNamespace(check=lambda *args: None, finish=lambda: {})})')
exec(compile(SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("failure", [None, "check", "finish"])
def test_component_audit_is_additional_to_original_task_and_reference_replay(monkeypatch, failure):
    calls, payload = [], {"teacher": "original"}
    monkeypatch.setattr(audit, "retention_load_regenerated_task", lambda path: calls.append("original_and_retention") or payload)
    monkeypatch.setattr(audit, "retention_regeneration_evidence", lambda step: calls.append("original_and_retention_finish") or {"retention": "hash"})

    def check(path, current):
        assert current is payload
        calls.append("components")
        if failure == "check":
            raise ValueError("component check")

    def finish():
        calls.append("component_finish")
        if failure == "finish":
            raise ValueError("component finish")
        return {"components": "hash"}

    token = audit.REPLAY.set({"legacy_blend": SimpleNamespace(check=check, finish=finish)})
    try:
        if failure == "check":
            with pytest.raises(ValueError, match="component check"):
                audit.load_regenerated_task("task")
            assert calls == ["original_and_retention", "components"]
            return
        assert audit.load_regenerated_task("task") is payload
        if failure == "finish":
            with pytest.raises(ValueError, match="component finish"):
                audit.regeneration_evidence(64)
        else:
            assert audit.regeneration_evidence(64) == {"retention": "hash", "components": "hash"}
        assert calls == ["original_and_retention", "components", "original_and_retention_finish", "component_finish"]
    finally:
        audit.REPLAY.reset(token)


def test_actual_full_audit_installs_component_replay_without_changing_fixed_half_or_retention():
    assert audit.training.teacher_policy is audit.blend_replay.teacher_policy
    assert audit.retention_training is audit.training.retention_training
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert 'active["legacy_blend"] = blend_replay.ComponentReplay(' in audit.RUN_AUDIT_SOURCE
    assert audit.RUN_AUDIT_SOURCE.index("ComponentReplay(") < audit.RUN_AUDIT_SOURCE.index("audit(output, training.STEPS)")
    assert '"fixed_half_training": True' in audit.ADAPTED_SOURCE
    assert '"displacement_fraction": training.paired.FRACTION' in audit.ADAPTED_SOURCE
    for name in ("run_audit", "load_regenerated_task", "retention_load_regenerated_task", "regeneration_evidence", "retention_regeneration_evidence"):
        assert getattr(audit, name).__globals__ is vars(audit)
        assert getattr(audit, name).__closure__ is None
