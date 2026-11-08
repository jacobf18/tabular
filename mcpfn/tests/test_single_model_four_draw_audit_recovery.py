from copy import deepcopy
import fcntl
import inspect
import os
from types import SimpleNamespace

import pytest

import recover_single_model_four_draw_audit as recovery
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def reports():
    failed = {"state": "failed", "completed_stages": [{"stage": "replay", "returncode": 0}, {"stage": "audit", "returncode": 1}]}
    state = {"state": "failed", "failed_stage": "audit",
        "error": "ValueError('Independently reproduce the complete registered admission attempt')"}
    teacher = {"complete": True, "rows": [{}] * 56, "full_grid_bitwise_replay": True,
        "blend_tasks": 52, "component_grids": 104, "student_or_validation_evaluated": False}
    retention = {"complete": True, "records": [{}] * 52, "component_grids": 100, "component_role_counts": {"v8": 48, "v29": 52}}
    result = {"data_admission_only": False, "record": {"accepted": True, "accepted_updates": 1, "composite_tasks": 56, "protected_tasks": 167}}
    return failed, state, teacher, retention, result


def test_source_changes_exactly_three_counts_and_preserves_original():
    source = recovery.corrected_source()
    for before, after in reversed(recovery.REPLACEMENTS):
        assert source.count(after) == 1
        source = source.replace(after, before)
    assert source == inspect.getsource(recovery.probe.audit_admission)
    for fragment in ('replay.record != saved', 'result["admission_sha256"] != sha256',
            'result["training_pid"] == os.getpid()', 'admission.episode_digest(episode) != row["episode_content_sha256"]'):
        assert fragment in recovery.corrected_source()


def test_full_outer_audit_chain_reaches_corrected_helper():
    namespace = recovery.audit_namespace()
    callers = []

    class ReachedAdmission(Exception):
        pass

    def read(path):
        callers.append(inspect.currentframe().f_back.f_code.co_name)
        if path.name == "status.json":
            return {"state": "awaiting_independent_audit"}
        if path.name == "result.json":
            return {"data_admission_only": False}
        if path.name == "data_admission.json":
            if callers[-1] == "audit_admission":
                raise ReachedAdmission
            return {"eligible": True}
        raise AssertionError(path)

    namespace["verify"] = lambda output: {}
    namespace["read"] = read
    with pytest.raises(ReachedAdmission):
        namespace["audit"](recovery.probe.OUTPUT)
    assert callers[-1] == "audit_admission"
    for name in ("audit_admission", "audit_admitted", "audit_original", "retention_audit", "audit"):
        function = namespace[name]
        assert function.__globals__ is namespace
        assert not unresolved_globals(function)
    assert recovery.probe.audit_admission.__globals__ is not namespace


def admission_fixture(tmp_path):
    namespace = recovery.audit_namespace()
    saved = {"task_count": 56, "structured_draws": 48, "deferred_task_indices": [0, 13, 14, 27, 28, 41, 42, 55],
        "eligible": True, "tasks": [], "rejected_task_indices": []}
    result = {"admission_sha256": "digest", "training_pid": os.getpid() + 1}
    regenerated = deepcopy(saved)
    namespace["read"] = lambda path: saved if path.name == "data_admission.json" else result
    namespace["sha256"] = lambda path: "digest"
    namespace["StructuredSampler"] = lambda base, device: None
    namespace["admission"] = SimpleNamespace(TrainingBatchAdmission=lambda sampler, batch: SimpleNamespace(record=regenerated))
    return namespace, saved, result, regenerated


def test_corrected_helper_accepts_only_exact_regenerated_admission(tmp_path):
    namespace, saved, result, regenerated = admission_fixture(tmp_path)
    report = namespace["audit_admission"](tmp_path, {"base": {}, "batch": {}})
    assert report["complete"] and report["structured_draws"] == 48
    assert report["deferred_task_indices"] == saved["deferred_task_indices"]


@pytest.mark.parametrize("mutation", ["identity", "hash", "pid", "tasks", "structured", "deferred", "captured"])
def test_corrected_admission_rejects_identity_and_count_faults(tmp_path, mutation):
    namespace, saved, result, regenerated = admission_fixture(tmp_path)
    if mutation == "identity":
        regenerated["eligible"] = False
    elif mutation == "hash":
        result["admission_sha256"] = "wrong"
    elif mutation == "pid":
        result["training_pid"] = os.getpid()
    elif mutation in ("tasks", "structured", "deferred"):
        field, value = {"tasks": ("task_count", 28), "structured": ("structured_draws", 24),
            "deferred": ("deferred_task_indices", [0, 13, 14, 27])}[mutation]
        saved[field] = regenerated[field] = value
    else:
        task = {"index": 1}
        saved["tasks"] = regenerated["tasks"] = [{"state": "structured_draw_inspected", "task": task, "episode_content_sha256": "expected"}]
        namespace["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
            load=lambda *args, **kwargs: {"task": task, "episode": {}})
        namespace["training"] = SimpleNamespace(alignment=SimpleNamespace(mixture=SimpleNamespace(
            backend=SimpleNamespace(restore_episode=lambda episode, device: episode))))
        namespace["admission"].episode_digest = lambda episode: "changed"
    with pytest.raises(ValueError):
        namespace["audit_admission"](tmp_path, {"base": {}, "batch": {}})


def test_evidence_is_complete_and_failures_preserved():
    recovery.validate_evidence(*reports())


@pytest.mark.parametrize("location,field,value", [(0, "state", "complete"), (1, "failed_stage", "run"),
    (1, "error", "other error"), (2, "rows", [{}] * 28), (2, "full_grid_bitwise_replay", False),
    (2, "blend_tasks", 26), (2, "component_grids", 52), (2, "student_or_validation_evaluated", True),
    (3, "records", [{}] * 26), (3, "component_grids", 50), (3, "component_role_counts", {"v8": 24, "v29": 26}),
    (4, "data_admission_only", True)])
def test_incomplete_or_different_replay_cannot_be_recovered(location, field, value):
    current = deepcopy(reports())
    current[location][field] = value
    with pytest.raises(ValueError):
        recovery.validate_evidence(*current)


def test_no_training_or_neural_replay_entry_and_no_unresolved_globals():
    source = inspect.getsource(recovery.main)
    assert 'choices=("register", "pipeline", "audit")' in source
    for function in vars(recovery).values():
        if inspect.isfunction(function) and function.__globals__ is vars(recovery):
            assert not unresolved_globals(function), function.__name__


def test_coordinator_holds_exclusive_lock(monkeypatch, tmp_path):
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


def test_any_prior_audit_artifact_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery.probe, "OUTPUT", tmp_path)
    (tmp_path / "audit.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        recovery.require_unentered()
