from copy import deepcopy

import pytest

import run_single_model_finite_guarded_pipeline as pipeline
from run_single_model_finite_guarded_pipeline import no_confirmation_reason


def test_initialization_never_consumes_confirmation():
    assert no_confirmation_reason({"step": 0}, {"selected_validation_passed": True}) is not None


def test_selected_replay_failure_has_no_fallback():
    assert "no fallback" in no_confirmation_reason({"step": 10}, {"selected_validation_passed": False})


def test_audited_trained_selection_may_reach_fresh_confirmation():
    assert no_confirmation_reason({"step": 10}, {"selected_validation_passed": True}) is None


@pytest.mark.parametrize("confirmation", [False, True])
def test_reservation_binds_policy_cases_and_parent_configs(confirmation):
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": pipeline.training.confirmation_cases() if confirmation else None}
    pipeline.verify_reservation(reservation, "pipeline", "run", confirmation)
    for key in reservation:
        changed = deepcopy(reservation)
        changed[key] = "changed"
        with pytest.raises(ValueError, match="reservation"):
            pipeline.verify_reservation(changed, "pipeline", "run", confirmation)


@pytest.mark.parametrize("confirmation_evaluated,passed", [(False, False), (True, False), (True, True)])
def test_pipeline_runs_real_campaign_only_after_independent_confirmation(monkeypatch, tmp_path, confirmation_evaluated, passed):
    calls = []
    gate = {"passed": passed, "confirmation_evaluated": confirmation_evaluated}
    reports = {pipeline.training.OUTPUT / "status.json": {"state": "training_complete"},
               pipeline.CONFIRMATION / "status.json": {"state": "awaiting_independent_audit" if confirmation_evaluated else "complete"},
               pipeline.CONFIRMATION / "gate.json": gate}
    original_read = pipeline.read
    monkeypatch.setattr(pipeline, "read", lambda path: reports[path])
    monkeypatch.setattr(pipeline, "verify", lambda output: None)
    monkeypatch.setattr(pipeline, "sha256", lambda path: "audit-digest")
    monkeypatch.setattr(pipeline, "run_child", lambda output, label, script, arguments, cpu=False: calls.append((label, cpu)))
    pipeline.run(tmp_path)
    expected = [("training_audit", True), ("replay", False), ("audit-replay", True), ("confirm", False)]
    if confirmation_evaluated:
        expected.append(("audit-confirmation", True))
    if passed:
        expected.extend([("benchmarks", False), ("review", True)])
    assert calls == expected
    result = original_read(tmp_path / "result.json")
    assert result["gate"] == gate
    assert result["campaign"] == (str(pipeline.CAMPAIGN) if passed else None)
    assert result["goal_success_not_assessed"] is True


def test_terminal_training_failure_never_starts_qualification(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "verify", lambda output: None)
    monkeypatch.setattr(pipeline, "read", lambda path: {"state": "failed"})
    monkeypatch.setattr(pipeline, "run_child", lambda *args, **kwargs: pytest.fail("Stopped training must never qualify"))
    with pytest.raises(RuntimeError, match="Training stopped"):
        pipeline.run(tmp_path)
