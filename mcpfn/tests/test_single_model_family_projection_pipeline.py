import pytest
import torch

import run_single_model_family_projection_pipeline as pipeline


def test_unselected_arm_never_consumes_confirmation_or_publishes_alias(tmp_path, monkeypatch):
    directory = tmp_path / "confirmation"
    directory.mkdir()
    arm = tmp_path / "arm"
    arm.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    pipeline.write_json(audit / "audit.json", {"complete": True})
    pipeline.status(directory, "registered")
    monkeypatch.setattr(pipeline, "AUDIT", audit)
    monkeypatch.setattr(pipeline, "CONFIRMATIONS", {"control": directory})
    monkeypatch.setattr(pipeline.training, "ARMS", {"control": arm})
    monkeypatch.setattr(pipeline, "confirmation_registration", lambda *args: ({}, {}, {"step": 0, "score": 1.}))

    def forbidden(*args, **kwargs):
        pytest.fail("Unselected arm must never initialize or evaluate a model")

    monkeypatch.setattr(pipeline, "wait_for_gpu", forbidden)
    monkeypatch.setattr(pipeline.training.mixture.backend, "setup", forbidden)
    pipeline.confirm(tmp_path, "control")
    gate = pipeline.training.read(directory / "gate.json")
    assert not gate["passed"] and not gate["confirmation_evaluated"]
    assert gate["selected_step"] == 0
    assert pipeline.training.read(directory / "status.json")["state"] == "complete"
    assert not (arm / "best_validation_checkpoint.pth").exists()
    with pytest.raises(ValueError, match="Never repeat"):
        pipeline.confirm(tmp_path, "control")


@pytest.mark.parametrize("passed", [{"projected": True, "control": True}, {"projected": False, "control": True}, {"projected": False, "control": False}])
def test_full_audit_then_both_confirmations_then_serial_campaigns(tmp_path, monkeypatch, passed):
    pair, audit = tmp_path / "pair", tmp_path / "audit"
    pair.mkdir()
    audit.mkdir()
    pipeline.status(pair, "training_complete")
    pipeline.write_json(audit / "audit.json", {"complete": True})
    confirmations = {name: tmp_path / f"{name}_confirmation" for name in pipeline.ORDER}
    for directory in confirmations.values():
        directory.mkdir()
    monkeypatch.setattr(pipeline, "AUDIT", audit)
    monkeypatch.setattr(pipeline.training, "PAIR", pair)
    monkeypatch.setattr(pipeline, "CONFIRMATIONS", confirmations)
    monkeypatch.setattr(pipeline, "verify", lambda *args: {})
    calls = []

    def child(output, label, script, arguments, *, cpu=False):
        calls.append((label, cpu))
        for name in pipeline.ORDER:
            if label == f"{name}_confirmation":
                pipeline.status(confirmations[name], "awaiting_independent_audit")
            elif label == f"{name}_confirmation_audit":
                pipeline.write_json(confirmations[name] / "gate.json", {"passed": passed[name], "selected_step": 25, "confirmation_evaluated": True})
                pipeline.status(confirmations[name], "complete")

    monkeypatch.setattr(pipeline, "run_child", child)
    pipeline.run(tmp_path)
    expected = [("training_audit", True), ("projected_confirmation", False), ("projected_confirmation_audit", True),
                ("control_confirmation", False), ("control_confirmation_audit", True)]
    for name in pipeline.ORDER:
        if passed[name]:
            expected += [(f"{name}_benchmarks", False), (f"{name}_review", True)]
    assert calls == expected
    result = pipeline.training.read(tmp_path / "result.json")
    assert result["complete"] and result["goal_success_not_assessed"] and result["no_inference_ensemble"]
    assert set(result["campaigns"]) == {name for name in pipeline.ORDER if passed[name]}


@pytest.mark.parametrize("state", ["failed", "interrupted"])
def test_training_failure_stops_pipeline_without_restart_or_confirmation(tmp_path, monkeypatch, state):
    pair = tmp_path / "pair"
    pair.mkdir()
    pipeline.status(pair, state)
    monkeypatch.setattr(pipeline.training, "PAIR", pair)
    monkeypatch.setattr(pipeline, "verify", lambda *args: {})

    def forbidden(*args, **kwargs):
        pytest.fail("Failed or interrupted training must never be restarted or promoted")

    monkeypatch.setattr(pipeline, "run_child", forbidden)
    with pytest.raises(RuntimeError, match="Paired training stopped"):
        pipeline.run(tmp_path)


def test_pipeline_never_reregisters_an_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.training, "verify_registration", lambda *args: {})
    monkeypatch.setattr(pipeline, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError):
        pipeline.register(tmp_path)


def test_confirmation_auditor_requires_cpu_before_reading_any_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        pipeline.audit_confirmation(tmp_path, "control")


def test_changed_pipeline_hash_fails_before_access_to_training(tmp_path):
    pipeline.write_json(tmp_path / "config.json", {"tampered": True})
    pipeline.write_json(tmp_path / "registration.json", {"config_sha256": "wrong"})
    with pytest.raises(ValueError, match="registration changed"):
        pipeline.verify(tmp_path)


@pytest.mark.parametrize("passed", [False, True])
def test_confirmation_publishes_byte_identical_alias_only_after_passed_cpu_audit(tmp_path, monkeypatch, passed):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    directory, arm, audit = tmp_path / "confirmation", tmp_path / "arm", tmp_path / "audit"
    for path in (directory, arm, audit):
        path.mkdir()
    pipeline.write_json(audit / "audit.json", {"complete": True})
    cases = [{"seed": index} for index in range(284)]
    bank = [{"metadata": case} for case in cases]
    torch.save(bank, directory / "synthetic_bank.pt")
    score, floors = {"fixture": 1.}, {"floor": 1.}
    reference = {"cases": cases, "general": score, "teacher": score, "references": floors,
                 "bank_sha256": pipeline.sha256(directory / "synthetic_bank.pt")}
    decision = {"passed": passed, "real_benchmark_success": False}
    for filename, payload in (("config.json", {}), ("reference.json", reference), ("decision.json", decision),
                              ("initialization.json", {"cases": cases, "student": score}), ("student.json", {"cases": cases, "student": score})):
        pipeline.write_json(directory / filename, payload)
    checkpoint = arm / "selected_checkpoint.pth"
    checkpoint.write_bytes(b"immutable selected checkpoint fixture")
    pipeline.write_json(directory / "manifest.json", {"selected_step": 25, "checkpoint": str(checkpoint), "checkpoint_sha256": pipeline.sha256(checkpoint)})
    pipeline.status(directory, "awaiting_independent_audit")
    monkeypatch.setattr(pipeline, "CONFIRMATIONS", {"projected": directory})
    monkeypatch.setattr(pipeline.training, "ARMS", {"projected": arm})
    monkeypatch.setattr(pipeline, "AUDIT", audit)
    monkeypatch.setattr(pipeline, "verify", lambda *args: {})
    monkeypatch.setattr(pipeline, "confirmation_registration", lambda *args: ({}, {"cases": cases}, {"step": 25, "score": 1.}))
    monkeypatch.setattr(pipeline.training.mixture, "aggregate", lambda *args: score)
    monkeypatch.setattr(pipeline.training.mixture, "confirmation_gate", lambda *args: decision)
    monkeypatch.setattr(pipeline.training.fixed, "reference_values", lambda *args: floors)
    monkeypatch.setattr(pipeline.training.fixed, "raw_audit", lambda *args: (1136, {}))
    pipeline.audit_confirmation(tmp_path, "projected")
    result = pipeline.training.read(directory / "gate.json")
    assert result["passed"] == passed and result["confirmation_evaluated"]
    assert result["audit_sha256"] == pipeline.sha256(directory / "audit.json")
    alias = arm / "best_validation_checkpoint.pth"
    assert alias.exists() == passed
    if passed:
        assert alias.read_bytes() == checkpoint.read_bytes()
        assert result["checkpoint_sha256"] == pipeline.sha256(alias)
    with pytest.raises(ValueError, match="fresh completed"):
        pipeline.audit_confirmation(tmp_path, "projected")
