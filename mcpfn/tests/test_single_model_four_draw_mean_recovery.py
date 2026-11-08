import ast
from copy import deepcopy
import fcntl
import inspect
from types import SimpleNamespace

import pytest

import recover_single_model_four_draw_mean_audit as recovery
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_exactly_one_stale_component_constant_changes_no_arithmetic():
    source = recovery.corrected_source()
    assert source.count(recovery.AFTER) == 1
    assert source.replace(recovery.AFTER, recovery.BEFORE) == recovery.mean.AUDIT_SOURCE
    constants = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and type(node.value) is int}
    assert {56, 52, 48, 100, 4}.issubset(constants)
    assert constants.isdisjoint({28, 26, 24, 50})


@pytest.mark.parametrize("components,admitted", [(50, False), (99, False), (100, True), (101, False)])
def test_actual_supplementary_prerequisite_accepts_only_the_complete_audited_count(tmp_path, components, admitted):
    namespace = recovery.namespace("registered")

    class ReachedTasks(Exception):
        pass

    def read(path):
        if path.name == "registration.json":
            return {"config_sha256": "registered"}
        if path.name == "status.json":
            return {"state": "complete", "pid": 1}
        if path.name == "audit.json":
            return {"complete": True}
        if path.name == "retention_audit.json":
            return {"complete": True, "component_grids": components, "component_role_counts": {"v8": 48, "v29": 52}, "reference_grids": 52}
        if path.name == "manifest.json":
            raise ReachedTasks
        raise AssertionError(path)

    namespace["OUTPUT"] = tmp_path / "not_created"
    namespace["shutil"] = SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=100 * 1024**3))
    namespace["verify_hashes"] = lambda hashes: None
    namespace["probe"] = SimpleNamespace(ROOT=tmp_path, OUTPUT=tmp_path, read=read,
        verify=lambda output: {"native_teacher_mean": {}}, retention=SimpleNamespace(MEAN_SPECIFICATION={}),
        diagnosis=SimpleNamespace(original=SimpleNamespace(process_identity=lambda pid: None)))
    with pytest.raises(ReachedTasks if admitted else ValueError):
        namespace["main"]()


def test_corrected_callback_and_original_mean_math_are_bound_without_replay():
    namespace = recovery.namespace("registered")
    assert namespace["main"].__globals__ is namespace
    assert namespace["describe"] is recovery.mean.describe
    assert namespace["imputation_task"] is recovery.mean.imputation_task
    assert not unresolved_globals(namespace["main"])
    for function in vars(recovery).values():
        if inspect.isfunction(function) and function.__globals__ is vars(recovery):
            assert not unresolved_globals(function), function.__name__
    assert 'choices=("register", "pipeline", "audit")' in inspect.getsource(recovery.main)
    source = inspect.getsource(recovery.pipeline)
    assert '"--stage", "audit"' in source and "neural_replay_repeated" in source


def completed_reports():
    return ({"data_admission_only": False, "record": {"accepted": True, "accepted_updates": 1, "protected_tasks": 167},
        "gradient_transfer_sha256": "transfer", "raw_sha256": "raw"},
        {"complete": True, "training_tasks": 56, "prediction_grids": 112,
            "raw_endpoint": {"gradient_vectors": 167, "accepted": True, "accepted_updates": 1}},
        {"complete": True, "eligibility_decisions": 56, "reference_grids": 52, "component_grids": 100,
            "component_role_counts": {"v8": 48, "v29": 52}, "reference_role_counts": {"v29": 4, "structured_mean": 48},
            "endpoint_roles": ["before", "candidate"]},
        {"complete": True, "transfer_sha256": "transfer", "raw_sha256": "raw"})


def test_completed_science_is_required_without_pretending_the_mean_audit_already_exists():
    recovery.validate_completed(*completed_reports())


@pytest.mark.parametrize("location,field,value", [(0, "data_admission_only", True), (1, "complete", False),
    (1, "training_tasks", 28), (1, "prediction_grids", 56), (2, "reference_grids", 26), (2, "component_grids", 50),
    (2, "component_role_counts", {"v8": 24, "v29": 26}), (2, "endpoint_roles", ["before"]),
    (3, "transfer_sha256", "changed"), (3, "raw_sha256", "changed")])
def test_missing_or_changed_completed_evidence_is_rejected(location, field, value):
    reports = deepcopy(completed_reports())
    reports[location][field] = value
    with pytest.raises(ValueError):
        recovery.validate_completed(*reports)


def test_coordinator_holds_exclusive_lock(monkeypatch, tmp_path):
    called = []

    def pipeline():
        with (tmp_path / "run.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        called.append(True)

    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    monkeypatch.setattr(recovery, "pipeline", pipeline)
    monkeypatch.setattr(recovery.sys, "argv", ["recovery", "--stage", "pipeline"])
    recovery.main()
    assert called == [True]
