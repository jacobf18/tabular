from copy import deepcopy
import inspect

import numpy as np
import pytest

import compare_single_model_point_proposals as diagnostic


def prerequisite_records():
    return {"status.json": {"state": "complete", "pid": 999999999},
            "audit.json": {"complete": True, "training_tasks": 28, "prediction_grids": 56,
                           "raw_endpoint": {"gradient_vectors": 84, "accepted_updates": 0},
                           "teacher_replay": {"complete": True, "full_grids": 28, "bitwise_equal": True},
                           "artifact_sha256": {"original": "hash"}},
            "gate.json": {"passed": False, "reason": "finite_metrics_rejected", "audit_sha256": "audit-hash"}}


def test_original_complete_preflight_evidence_is_verified(monkeypatch):
    records = prerequisite_records()
    checked = []
    monkeypatch.setattr(diagnostic.preflight, "verify", lambda path: {"base": "original"})
    monkeypatch.setattr(diagnostic, "read", lambda path: records[path.name])
    monkeypatch.setattr(diagnostic, "sha256", lambda path: "audit-hash")
    monkeypatch.setattr(diagnostic, "verify_hashes", checked.append)
    config, audit = diagnostic.prerequisite()
    assert config == {"base": "original"} and checked == [{"original": "hash"}]


@pytest.mark.parametrize("path,field,value", [
    ("status.json", "state", "awaiting_independent_audit"), ("status.json", "pid", 1),
    ("audit.json", "complete", False), ("audit.json", "training_tasks", 27),
    ("audit.json", "prediction_grids", 55), ("gate.json", "passed", True),
    ("gate.json", "reason", "accepted"), ("gate.json", "audit_sha256", "other"),
])
def test_incomplete_live_or_changed_source_cannot_be_reused(monkeypatch, path, field, value):
    records = prerequisite_records()
    records[path][field] = value
    monkeypatch.setattr(diagnostic.preflight, "verify", lambda path: {})
    monkeypatch.setattr(diagnostic, "read", lambda path: records[path.name])
    monkeypatch.setattr(diagnostic, "sha256", lambda path: "audit-hash")
    with pytest.raises(ValueError, match="completed independently audited"):
        diagnostic.prerequisite()


def test_fixed_two_by_two_grid_reuses_original_failure_once():
    cases = diagnostic.registered_cases()
    assert len(cases) == 4 and sum(case["reused"] for case in cases) == 1
    assert {(case["method"], case["rate"]) for case in cases} == {
        (method, rate) for method in ("composite", "point_inclusive") for rate in (2e-5, 1e-5)}
    assert [case for case in cases if case["reused"]] == [
        {"name": "point_2e-5", "method": "point_inclusive", "rate": 2e-5, "reused": True}]
    cases[0]["rate"] = 1
    assert diagnostic.CASES[0]["rate"] == 2e-5


def test_registration_has_no_predictions_or_optimizer_updates():
    source = inspect.getsource(diagnostic.register)
    assert "prerequisite()" in source and "storage_allowance" in source
    assert "new_control_predictions_performed" in source
    assert "protected_update" not in source and "preflight.predictions" not in source


def test_before_grid_check_catches_any_changed_prediction(tmp_path, monkeypatch):
    original, current = tmp_path / "original.npz", tmp_path / "current.npz"
    np.savez(original, prediction=np.ones((1, 2, 3), dtype=np.float32))
    np.savez(current, prediction=np.ones((1, 2, 3), dtype=np.float32))
    rows = [{"task": f"task_{index}", "index": index, "prediction": str(current)} for index in range(28)]
    expected = {"rows": [{**row, "prediction": str(original)} for row in rows], "values": {"metric": 1.}}
    report = {"rows": rows, "values": {"metric": 1.}}
    monkeypatch.setattr(diagnostic, "read", lambda path: report)
    diagnostic.check_before(tmp_path, expected)
    np.savez(current, prediction=np.zeros((1, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="bitwise"):
        diagnostic.check_before(tmp_path, expected)


@pytest.mark.parametrize("mutation", ["partial", "order", "metrics"])
def test_before_check_requires_all_original_identities(tmp_path, monkeypatch, mutation):
    path = tmp_path / "grid.npz"
    np.savez(path, prediction=np.ones(2, dtype=np.float32))
    expected = {"rows": [{"task": f"task_{index}", "index": index, "prediction": str(path)} for index in range(28)],
                "values": {"metric": 1.}}
    report = deepcopy(expected)
    if mutation == "partial":
        report["rows"].pop()
    elif mutation == "order":
        report["rows"].reverse()
    else:
        report["values"]["metric"] += .1
    monkeypatch.setattr(diagnostic, "read", lambda path: report)
    with pytest.raises(ValueError):
        diagnostic.check_before(tmp_path, expected)


def test_all_controls_restore_state_and_preserve_frozen_protocols():
    source = inspect.getsource(diagnostic.run)
    assert 'model.load_state_dict(parent["model"], strict=True)' in source
    assert 'optimizer.load_state_dict(initial_optimizer)' in source
    assert 'evidence.compact' in source and 'storage.save' in source
    assert 'before_grids_bitwise_reproduced' in source
    assert 'failed_endpoint.pt' in source
    assert 'best_validation_checkpoint' not in source
    assert 'backward(' not in source


def test_independent_audit_checks_optimizer_and_all_prediction_grids():
    source = inspect.getsource(diagnostic.audit)
    assert 'method.check_record' in source and 'method.check_endpoint' in source
    assert 'evidence.expand' in source and 'storage.audit' in source
    assert 'audit_predictions(SOURCE' in source and 'audit_predictions(directory' in source
    assert 'read(SOURCE / "result.json")["record"]' in source
    assert 'torch.nextafter' in source and 'direction_products' in source
    assert 'selection_performed' in source
