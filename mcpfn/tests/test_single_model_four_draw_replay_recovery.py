from copy import deepcopy
import fcntl
import inspect

import pytest

import recover_single_model_four_draw_replay as recovery
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def reports():
    diagnosis = {"pid": 257967, "parent_pid": 255579, "teacher_replay_directory_exists": False,
        "retention_replay_directory_exists": False, "training_repeated": False, "registered_source_modified": False,
        "new_scientific_seed": False, "installed_replay_admitted_code_name": "replay_targets",
        "installed_replay_admitted_global_calls": ["replay_cohort_targets"],
        "registered_primitive_header": "def replay_admitted(output):"}
    result = {"data_admission_only": False, "record": {"accepted": True, "accepted_updates": 1,
        "composite_tasks": 56, "protected_tasks": 167, "finite": {"defined_slots": 34}}}
    failed = {"state": "failed", "pid": 255579, "completed_stages": [
        {"stage": "run", "returncode": 0}, {"stage": "replay", "returncode": -15}]}
    return diagnosis, result, failed


def test_installed_primitive_has_its_registered_name_and_the_original_namespace_stays_broken():
    namespace = recovery.replay_namespace()
    primitive = inspect.unwrap(namespace["replay_admitted"])
    assert primitive.__code__.co_name == "replay_admitted"
    assert "original_setup" in primitive.__code__.co_names
    assert "replay_cohort_targets" not in primitive.__code__.co_names
    assert inspect.unwrap(recovery.probe.replay_admitted).__code__.co_name == "replay_targets"
    for name in ("replay_admitted", "replay_original_teachers", "replay_cohort_targets"):
        function = inspect.unwrap(namespace[name])
        assert function.__globals__ is namespace
        assert not unresolved_globals(namespace[name])
    assert namespace["verify"] is recovery.probe.verify
    assert namespace["retention_records"] is recovery.probe.retention_records
    assert namespace["teacher_predictions"] is recovery.probe.teacher_predictions


def test_outer_wrapper_enters_the_actual_primitive_once_without_sampling_or_inference():
    namespace = recovery.replay_namespace()
    callers = []

    class ReachedPrimitive(Exception):
        pass

    def verify(output):
        callers.append(inspect.currentframe().f_back.f_code.co_name)
        if len(callers) == 2:
            raise ReachedPrimitive
        return {}

    namespace["verify"] = verify
    namespace["read"] = lambda path: {"eligible": True}
    with pytest.raises(ReachedPrimitive):
        namespace["replay_cohort_targets"](recovery.probe.OUTPUT)
    assert callers == ["replay_targets", "replay_admitted"]


def test_exact_diagnosed_failure_and_training_evidence_are_admitted():
    recovery.validate_original(*reports())


@pytest.mark.parametrize("location,path,value", [
    (0, "teacher_replay_directory_exists", True), (0, "retention_replay_directory_exists", True),
    (0, "training_repeated", True), (0, "registered_source_modified", True), (0, "new_scientific_seed", True),
    (0, "installed_replay_admitted_code_name", "replay_admitted"), (0, "pid", 1),
    (1, "data_admission_only", True), (1, "record.accepted", False), (1, "record.accepted_updates", 2),
    (1, "record.composite_tasks", 28), (1, "record.protected_tasks", 84), (1, "record.finite.defined_slots", 30),
    (2, "state", "running"), (2, "pid", 1)])
def test_scope_cannot_expand_into_a_scientific_retry(location, path, value):
    current = deepcopy(reports())
    target = current[location]
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError):
        recovery.validate_original(*current)


def test_recovery_cli_has_no_training_stage_and_every_callback_resolves():
    source = inspect.getsource(recovery.main)
    assert 'choices=("register", "pipeline", "replay")' in source
    assert '"run"' not in source
    for function in vars(recovery).values():
        if inspect.isfunction(function) and function.__globals__ is vars(recovery):
            assert not unresolved_globals(function), function.__name__


def test_coordinator_holds_exclusive_lock_for_the_complete_pipeline(monkeypatch, tmp_path):
    calls = []

    def pipeline():
        with (tmp_path / "run.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append("once")

    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    monkeypatch.setattr(recovery, "pipeline", pipeline)
    monkeypatch.setattr(recovery.sys, "argv", ["recovery", "--stage", "pipeline"])
    recovery.main()
    assert calls == ["once"]
    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
