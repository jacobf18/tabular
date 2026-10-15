from pathlib import Path

import pytest

import run_single_model_general_native_retention_pipeline as pipeline


@pytest.mark.parametrize("selected_step,replay_passed", [(0, False), (0, True), (10, False)])
def test_actual_failed_qualification_consumes_no_confirmation(tmp_path, monkeypatch, selected_step, replay_passed):
    replay = tmp_path / "replay"
    confirmation = tmp_path / "confirmation"
    replay.mkdir()
    confirmation.mkdir()
    pipeline.write_json(replay / "status.json", {"state": "complete"})
    pipeline.write_json(replay / "gate.json", {"endpoint_reproduced": True,
                                             "selected_validation_passed": replay_passed})
    pipeline.write_json(confirmation / "status.json", {"state": "registered"})
    monkeypatch.setattr(pipeline, "REPLAY", replay)
    monkeypatch.setattr(pipeline, "CONFIRMATION", confirmation)
    monkeypatch.setattr(pipeline, "prerequisite", lambda output: {"best": {"step": selected_step}})

    def forbidden(*args, **kwargs):
        pytest.fail("Failed qualification must not load, initialize or evaluate any neural model")

    monkeypatch.setattr(pipeline, "wait_for_gpu", forbidden)
    monkeypatch.setattr(pipeline.torch, "load", forbidden)
    pipeline.confirm(tmp_path)
    gate = pipeline.read(confirmation / "gate.json")
    assert gate["passed"] is False and gate["confirmation_evaluated"] is False
    assert gate["selected_step"] == selected_step and gate["real_benchmark_success"] is False
    assert pipeline.read(confirmation / "status.json")["state"] == "complete"
    assert set(path.name for path in confirmation.iterdir()) == {"status.json", "gate.json"}
    assert not pipeline.torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="Never repeat"):
        pipeline.confirm(tmp_path)


def test_actual_coordinator_stops_before_real_benchmarks_on_failed_gate(tmp_path, monkeypatch):
    run = tmp_path / "training"
    audit = tmp_path / "audit"
    confirmation = tmp_path / "confirmation"
    for directory in (run, audit, confirmation):
        directory.mkdir()
    pipeline.write_json(run / "status.json", {"state": "training_complete"})
    pipeline.write_json(audit / "audit.json", {"complete": True})
    monkeypatch.setattr(pipeline.training, "OUTPUT", run)
    monkeypatch.setattr(pipeline, "AUDIT", audit)
    monkeypatch.setattr(pipeline, "CONFIRMATION", confirmation)
    monkeypatch.setattr(pipeline, "CAMPAIGN", tmp_path / "benchmarks")
    monkeypatch.setattr(pipeline, "REVIEW", tmp_path / "review")
    monkeypatch.setattr(pipeline, "verify", lambda output: {})
    events = []
    monkeypatch.setattr(pipeline, "wait_for_training_audit", lambda output: events.append("full-training-audit"))

    def child(output, label, script, arguments, cpu=False):
        events.append((label, cpu))
        assert Path(output) == tmp_path
        if label == "confirm":
            pipeline.write_json(confirmation / "status.json", {"state": "complete"})
            pipeline.write_json(confirmation / "gate.json", {"passed": False, "selected_step": 0,
                                                            "confirmation_evaluated": False})
        elif label not in {"replay", "audit-replay"}:
            pytest.fail("A failed selection must never dispatch confirmation inference or real benchmarks")

    monkeypatch.setattr(pipeline, "run_child", child)
    pipeline.run(tmp_path)
    assert events == ["full-training-audit", ("replay", False), ("audit-replay", True), ("confirm", False)]
    result = pipeline.read(tmp_path / "result.json")
    assert result["complete"] is True and result["gate"]["passed"] is False
    assert result["campaign"] is None and result["review"] is None
    assert result["no_inference_ensemble"] is True and result["goal_success_not_assessed"] is True
    assert not pipeline.CAMPAIGN.exists() and not pipeline.REVIEW.exists()
    assert not pipeline.torch.cuda.is_initialized()
