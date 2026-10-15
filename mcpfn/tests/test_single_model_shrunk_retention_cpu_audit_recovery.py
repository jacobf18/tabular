from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

import recover_single_model_shrunk_retention_cpu_audit as recovery


def evidence():
    failed = {"state": "failed", "failed_stage": "audit", "error": "ValueError('Audit must be CPU only')"}
    result = {"data_admission_only": False, "preflight_only": True, "trainable_parameters": 77998496,
        "record": {"accepted": False, "accepted_updates": 0, "reason": "finite_metrics_rejected"}}
    manifest = {"complete": True, "specification": recovery.probe.retention.SPECIFICATION,
        "records": [{"eligible": index < 17} for index in range(28)], "reference": {"sha256": "reference"}}
    replay = {"complete": True, "records": [{} for index in range(17)], "reference": deepcopy(manifest["reference"])}
    return failed, result, replay, manifest, "ValueError: Audit must be CPU only", set()


@pytest.mark.parametrize("mutation", [None, "wrong_error", "wrong_stage", "partial_audit", "accepted",
    "updates", "reason", "incomplete_replay", "missing_reference", "missing_task", "changed_loss"])
def test_only_exact_pre_audit_failure_and_completed_rejected_probe_are_admitted(mutation):
    failed, result, replay, manifest, log, reports = evidence()
    if mutation == "wrong_error":
        failed["error"] = "different failure"
    elif mutation == "wrong_stage":
        failed["failed_stage"] = "run"
    elif mutation == "partial_audit":
        reports.add("audit.json")
    elif mutation == "accepted":
        result["record"]["accepted"] = True
    elif mutation == "updates":
        result["record"]["accepted_updates"] = 1
    elif mutation == "reason":
        result["record"]["reason"] = "another reason"
    elif mutation == "incomplete_replay":
        replay["complete"] = False
    elif mutation == "missing_reference":
        replay["records"].pop()
    elif mutation == "missing_task":
        manifest["records"].pop()
    elif mutation == "changed_loss":
        manifest["specification"] = {}
    if mutation:
        with pytest.raises(ValueError):
            recovery.validate_failure(failed, result, replay, manifest, log, reports)
    else:
        recovery.validate_failure(failed, result, replay, manifest, log, reports)


def test_view_preserves_all_absolute_input_paths_and_routes_only_new_audit_artifacts(tmp_path):
    original, destination = tmp_path / "original", tmp_path / "recovery"
    view = recovery.AuditView(original, destination)
    assert os.fspath(view) == str(original) and str(view) == str(original)
    for name in recovery.AUDIT_FILES:
        assert view / name == destination / name
    for name in ("config.json", "result.json", "raw_update.pt", "training_batch/task_00.pt", "retention_replay/manifest.json"):
        assert view / name == original / name
    for name in ("/absolute", "../escape"):
        with pytest.raises(ValueError):
            view / name


def test_isolated_audit_code_and_globals_preserve_registered_functions_and_reject_original_writes(tmp_path):
    original, destination = tmp_path / "original", tmp_path / "recovery"
    original.mkdir()
    destination.mkdir()
    status = original / "status.json"
    status.write_text('{"state":"failed"}')
    before = dict(vars(recovery.probe))
    current = recovery.namespace(destination)
    for name in ("audit", "audit_original", "audit_admitted", "audit_raw", "audit_teacher_replay", "audit_probe_checkpoint", "verify", "status"):
        assert current[name].__code__ is before[name].__code__
        assert current[name].__globals__ is current
        assert getattr(recovery.probe, name) is before[name]
    with pytest.raises(ValueError):
        current["write_json"](status, {"state": "complete"})
    current["write_json"](destination / "audit.json", {"complete": True})
    with pytest.raises(FileExistsError):
        current["write_json"](destination / "audit.json", {"complete": True})
    assert status.read_text() == '{"state":"failed"}'
    assert recovery.probe.write_json is before["write_json"]


def test_cpu_visibility_and_initialization_are_both_required(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError):
        recovery.require_cpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    recovery.require_cpu()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError):
        recovery.require_cpu()
