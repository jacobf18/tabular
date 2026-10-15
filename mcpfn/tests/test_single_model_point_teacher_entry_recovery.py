import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import recover_single_model_point_teacher_entry as recovery
import run_single_model_point_teacher_replay as gateway


def failure_fixture():
    audit = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
             "accepted_updates": 63, "training_episodes": 1792, "teacher_identity_checked_tasks": 1792,
             "prediction_grids": 1216, "cuda_initialized": False, "best": {"step": 0},
             "raw_endpoint": {"actual_final_parameters_checked": True}}
    log = 'replay_single_model_point_inclusive_teachers.py", line 14, in <module>\nNameError: name \'main\' is not defined'
    return [{"state": "failed", "error": recovery.ORIGINAL_ERROR}, {"state": "registered"},
            set(recovery.RESERVATION_FILES), {"state": "registered"}, {"state": "registered"}, log, audit]


def test_exact_pre_execution_failure_and_complete_nonselection_are_accepted():
    recovery.validate_failure(*failure_fixture())


@pytest.mark.parametrize("change", ["error", "teacher_state", "teacher_file", "endpoint", "confirmation", "log", "audit", "selected", "endpoint_proof"])
def test_started_or_unverified_scientific_work_is_never_retried(change):
    arguments = failure_fixture()
    if change == "error":
        arguments[0]["error"] = "CUDA out of memory"
    elif change == "teacher_state":
        arguments[1]["state"] = "replaying"
    elif change == "teacher_file":
        arguments[2].add("grids.jsonl")
    elif change == "endpoint":
        arguments[3]["state"] = "replaying_endpoint"
    elif change == "confirmation":
        arguments[4]["state"] = "confirming"
    elif change == "log":
        arguments[5] = "unrelated error"
    elif change == "audit":
        arguments[6]["complete_training_audit"] = False
    elif change == "selected":
        arguments[6]["best"]["step"] = 64
    else:
        arguments[6]["raw_endpoint"]["actual_final_parameters_checked"] = False
    with pytest.raises(ValueError):
        recovery.validate_failure(*arguments)


@pytest.mark.parametrize("pid", [0, -1, None, os.getpid()])
def test_live_or_invalid_original_process_prevents_recovery(pid):
    with pytest.raises(ValueError, match="absent"):
        recovery.require_absent(pid)


def test_cli_help_executes_the_real_entry_point_without_model_calls():
    result = subprocess.run([sys.executable, str(Path(gateway.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--stage {replay,audit}" in result.stdout


@pytest.mark.parametrize("stage,state", [("replay", "registered"), ("audit", "awaiting_independent_audit")])
def test_dispatch_calls_the_unchanged_original_stage_once(tmp_path, monkeypatch, stage, state):
    monkeypatch.setattr(gateway.teacher, "OUTPUT", tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"state": state}))
    calls = []
    monkeypatch.setattr(gateway, "verify_recovery", lambda: calls.append("verify"))
    monkeypatch.setattr(gateway.teacher, "replay", lambda output: calls.append(("replay", output)))
    monkeypatch.setattr(gateway.teacher, "audit", lambda output: calls.append(("audit", output)))
    gateway.main(["--stage", stage])
    assert calls == ["verify", (stage, tmp_path), "verify"]


@pytest.mark.parametrize("stage,state", [("replay", "failed"), ("replay", "complete"), ("audit", "registered"), ("audit", "complete")])
def test_dispatch_cannot_restart_or_relabel_a_prior_outcome(tmp_path, monkeypatch, stage, state):
    monkeypatch.setattr(gateway.teacher, "OUTPUT", tmp_path)
    original = json.dumps({"state": state})
    (tmp_path / "status.json").write_text(original)
    monkeypatch.setattr(gateway, "verify_recovery", lambda: None)
    with pytest.raises(ValueError, match="Never restart"):
        gateway.dispatch(stage)
    assert (tmp_path / "status.json").read_text() == original


def test_existing_recovery_is_never_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError, match="Never restart"):
        recovery.recover()


def test_source_functions_and_output_reservations_are_not_replaced():
    assert recovery.pipeline.teacher_replay is gateway.teacher
    assert recovery.pipeline.TEACHERS == gateway.teacher.OUTPUT
    assert recovery.OUTPUT == gateway.RECOVERY
    assert not hasattr(gateway.teacher, "main")
