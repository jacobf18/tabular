from types import SimpleNamespace

import pytest

import audit_single_model_anchor_retention as audit
import test_single_model_full_network_audit as previous_tests


@pytest.mark.parametrize("eligible", [False, True])
def test_original_scheduled_admissions_still_replay_including_noops(tmp_path, monkeypatch, eligible):
    monkeypatch.setattr(previous_tests, "audit", audit)
    previous_tests.test_every_scheduled_admission_is_independently_replayed(tmp_path, monkeypatch, eligible)


@pytest.mark.parametrize("mutation", [None, "terminal_as_actual", "model", "record"])
def test_original_endpoint_checks_preserve_every_optimizer_coordinate(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "audit", audit)
    previous_tests.test_terminal_audit_uses_last_actual_attempt(tmp_path, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", ["mask", "hash", "path", "fabricated_task", "fabricated_metric"])
def test_original_rejected_draw_evidence_cannot_be_repaired(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "audit", audit)
    previous_tests.test_rejected_draw_evidence_cannot_be_repaired_or_fabricated(tmp_path, monkeypatch, mutation)


def test_added_reference_checks_do_not_replace_original_regeneration(monkeypatch):
    calls, payload = [], {"teacher": "original"}
    monkeypatch.setattr(audit, "original_load_regenerated_task", lambda path: calls.append("original") or payload)
    monkeypatch.setattr(audit, "original_regeneration_evidence", lambda step: calls.append("original_finish") or {"original": "hash"})
    reference = SimpleNamespace(check=lambda path, value: calls.append((path, value)),
                                finish=lambda: calls.append("retention_finish") or {"retention": "hash"})
    token = audit.REPLAY.set({"anchor_retention": reference})
    try:
        assert audit.load_regenerated_task("task") is payload
        assert audit.regeneration_evidence(64) == {"original": "hash", "retention": "hash"}
    finally:
        audit.REPLAY.reset(token)
    assert calls == ["original", ("task", payload), "original_finish", "retention_finish"]


def test_actual_run_audit_uses_frozen_v29_reference_before_complete_original_audit():
    assert 'model.load_state_dict(torch.load(training.PARENT' in audit.RUN_AUDIT_SOURCE
    assert 'model.eval().requires_grad_(False)' in audit.RUN_AUDIT_SOURCE
    assert 'active["anchor_retention"] = retention_training.RetentionReplay(' in audit.RUN_AUDIT_SOURCE
    assert 'audit(output, training.STEPS)' in audit.RUN_AUDIT_SOURCE
    assert 'del model' not in audit.RUN_AUDIT_SOURCE
    assert audit.RUN_AUDIT_SOURCE.index('RetentionReplay(') < audit.RUN_AUDIT_SOURCE.index('audit(output, training.STEPS)')
    assert 'len(decision["comparisons"]) != 22' in audit.ADAPTED_SOURCE
    assert 'frozen["frozen_tensors"] != 16' in audit.ADAPTED_SOURCE
    for name in ("run_audit", "load_regenerated_task", "original_load_regenerated_task", "regeneration_evidence", "original_regeneration_evidence"):
        assert getattr(audit, name).__globals__ is vars(audit)
        assert getattr(audit, name).__closure__ is None
