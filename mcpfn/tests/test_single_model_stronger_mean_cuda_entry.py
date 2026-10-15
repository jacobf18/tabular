import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import recover_single_model_stronger_mean_cuda_entry as recovery


def test_clones_change_only_audit_location_and_preserve_original_science():
    before = dict(vars(recovery.original))
    namespace = recovery.execution_namespace()
    assert all(vars(recovery.original)[name] is value for name, value in before.items())
    assert vars(recovery.original).keys() == before.keys()
    for name, function in recovery.owned_functions().items():
        assert inspect.unwrap(namespace[name]).__code__ is inspect.unwrap(function).__code__
        assert inspect.unwrap(namespace[name]).__globals__ is namespace
    for name, value in before.items():
        if name not in recovery.owned_functions() and name != "AUDIT":
            assert namespace[name] is value
    assert namespace["AUDIT"] == recovery.AUDIT != recovery.original.AUDIT
    assert namespace["training"].BOUNDARIES == (0, 10, 25, 50, 64)
    assert len(namespace["training"].confirmation_cases()) == 284


@pytest.mark.parametrize("name", ["replay_actual", "confirm"])
def test_no_grad_wrappers_are_preserved(name):
    namespace = recovery.execution_namespace()

    def stop(output):
        assert not torch.is_grad_enabled()
        raise RuntimeError("prerequisite sentinel")

    namespace["prerequisite"] = stop
    with torch.enable_grad(), pytest.raises(RuntimeError, match="prerequisite sentinel"):
        namespace[name](Path("unused"))
    assert torch.is_grad_enabled()


def test_actual_auditor_and_qualification_globals_are_resolved():
    recovery.require_static_globals()


def test_static_check_includes_nested_code():
    namespace = {}
    exec("def missing():\n    return [undeclared(value) for value in range(2)]", namespace)
    assert recovery.unresolved_globals(namespace["missing"]) == ["undeclared"]


def test_original_cuda_failure_precedes_model_setup_and_replay():
    setup = recovery.auditor.training.original.alignment.mixture.original_setup
    source = inspect.getsource(setup)
    assert source.index("torch.cuda.set_per_process_memory_fraction") < source.index("initialize_student")
    source = recovery.auditor.RUN_AUDIT_SOURCE
    assert source.index('backend.setup(config["base"])') < source.index("RetentionReplay(")
    assert source.index('backend.setup(config["base"])') < source.index("ComponentReplay(")
    assert source.index('backend.setup(config["base"])') < source.index("audit(output, training.STEPS)")
    tree = ast.parse(source)
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "setup" for node in ast.walk(tree)) == 1


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    for name in ("OUTPUT", "REPLAY", "CONFIRMATION", "AUDIT", "CAMPAIGN", "REVIEW"):
        path = tmp_path / name.lower()
        monkeypatch.setattr(recovery.original, name, path)
        if name not in {"AUDIT", "CAMPAIGN", "REVIEW"}:
            path.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    monkeypatch.setattr(recovery.original.training, "OUTPUT", training)
    monkeypatch.setattr(recovery, "FAILED_REPLAY", tmp_path / "failed_replay")
    monkeypatch.setattr(recovery, "AUDIT", tmp_path / "recovered_audit")
    monkeypatch.setattr(recovery, "ENTRY", tmp_path / "entry")
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path / "coordinator")
    monkeypatch.setattr(recovery.original, "verify", lambda output: None)
    monkeypatch.setattr(recovery, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(recovery, "require_stopped", lambda state: None)
    recovery.FAILED_REPLAY.mkdir()
    recovery.write_json(training / "status.json", {"state": "training_complete", "pid": 1,
                        "step": 64, "accepted_updates": 55, "best": {"step": 10}})
    recovery.write_json(recovery.FAILED_REPLAY / "status.json", {"state": "failed", "pid": 2, "error": recovery.ERROR})
    recovery.write_json(recovery.FAILED_REPLAY / "config.json", {"source_sha256": {},
                        "training_config_sha256": recovery.CONFIG_SHA256,
                        "adapted_source_sha256": recovery.auditor.ADAPTED_SOURCE_SHA256,
                        "training_performed": False, "qualification_changed": False})
    (recovery.FAILED_REPLAY / "tasks.jsonl").touch()
    recovery.write_json(recovery.original.OUTPUT / "status.json", {"state": "failed", "pid": 3,
                        "error": "The original one-shot training audit failed: No CUDA GPUs are available"})
    (training / "train_and_audit.log").write_text('Traceback\nin setup\n'
        'torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])\n'
        'RuntimeError: No CUDA GPUs are available\n')
    retention = tmp_path / "retention"
    retention.mkdir()
    recovery.write_json(training / "config.json", {"retention_storage": {"replay": str(retention)},
                        "all_imputation_blend_replay": {"directory": str(tmp_path / "components")}})
    for directory in (recovery.original.REPLAY, recovery.original.CONFIRMATION):
        recovery.write_json(directory / "status.json", {"state": "registered"})
        for name in ("config.json", "registration.json"):
            recovery.write_json(directory / name, {})
    return tmp_path


def test_exact_failed_prefix_and_reservations_are_admitted(prefix):
    recovery.require_original_failure()
    recovery.require_pristine()


@pytest.mark.parametrize("mutation", ["task", "extra_file", "audit", "trace", "selection", "accepted", "error"])
def test_changed_or_started_original_audit_is_rejected(prefix, mutation):
    if mutation == "task":
        (recovery.FAILED_REPLAY / "tasks.jsonl").write_text("{}\n")
    elif mutation == "extra_file":
        (recovery.FAILED_REPLAY / "result.json").touch()
    elif mutation == "audit":
        recovery.original.AUDIT.mkdir()
    elif mutation == "trace":
        (recovery.original.training.OUTPUT / "train_and_audit.log").write_text("failure after prediction\n")
    elif mutation in {"selection", "accepted"}:
        path = recovery.original.training.OUTPUT / "status.json"
        current = recovery.original.read(path)
        if mutation == "selection":
            current["best"]["step"] = 64
        else:
            current["accepted_updates"] = 54
        recovery.write_json(path, current)
    else:
        path = recovery.FAILED_REPLAY / "status.json"
        current = recovery.original.read(path)
        current["error"] = "RuntimeError('CUDA out of memory')"
        recovery.write_json(path, current)
    with pytest.raises(ValueError):
        recovery.require_original_failure()


@pytest.mark.parametrize("mutation", ["retention", "components", "endpoint", "confirmation", "recovered_audit", "campaign", "publication"])
def test_any_prior_scientific_work_blocks_entry(prefix, mutation):
    if mutation == "retention":
        (prefix / "retention" / "tasks.jsonl").touch()
    elif mutation == "components":
        (prefix / "components").mkdir()
    elif mutation in {"endpoint", "confirmation"}:
        directory = recovery.original.REPLAY if mutation == "endpoint" else recovery.original.CONFIRMATION
        (directory / "run.lock").touch()
    elif mutation == "recovered_audit":
        recovery.task_replay(recovery.AUDIT).mkdir()
    elif mutation == "campaign":
        recovery.original.CAMPAIGN.mkdir()
    else:
        (recovery.original.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    with pytest.raises(ValueError):
        recovery.require_pristine()


@pytest.mark.parametrize("visible,available", [("", True), (None, False)])
def test_cuda_environment_is_checked_before_model_work(monkeypatch, visible, available):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    with pytest.raises(ValueError, match="visible CUDA"):
        recovery.require_cuda()


def test_unchanged_auditor_is_launched_with_cuda_and_original_arithmetic(prefix, monkeypatch):
    recovery.ENTRY.mkdir()
    monkeypatch.setattr(recovery, "verify", lambda directory: None)
    monkeypatch.setattr(recovery, "require_cuda", lambda: {})
    monkeypatch.setattr(recovery, "execution_namespace", lambda: {"prerequisite": lambda output: None})
    monkeypatch.setattr(recovery, "sha256", lambda path: "digest")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    launched = []

    def launch(command, **options):
        launched.append((command, options))
        return SimpleNamespace(pid=123, wait=lambda: 0)

    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    recovery.run_audit()
    assert len(launched) == 1
    command, options = launched[0]
    assert command[2] == str(Path(recovery.auditor.__file__).resolve())
    assert command[3:] == ["--output", str(recovery.AUDIT)]
    assert "CUDA_VISIBLE_DEVICES" not in options["env"]
    assert options["env"]["OPENBLAS_NUM_THREADS"] == "24"
    assert options["env"]["MKL_NUM_THREADS"] == "24"
    assert options["env"]["OMP_NUM_THREADS"] == "4"
    assert recovery.original.read(recovery.ENTRY / "result.json")["training_repeated"] is False


def test_scientific_stage_calls_unchanged_function_with_original_config(monkeypatch):
    called = []
    monkeypatch.setattr(recovery, "verify", lambda directory: called.append(("verify", directory)))
    monkeypatch.setattr(recovery, "require_completed_audit", lambda: called.append(("audit", None)))
    monkeypatch.setattr(recovery, "execution_namespace", lambda: {
        "audit_replay": lambda output: called.append(("stage", output))})
    recovery.scientific_stage("audit-replay")
    assert called == [("verify", recovery.OUTPUT), ("audit", None), ("stage", recovery.original.OUTPUT)]


@pytest.mark.parametrize("passed", [False, True])
def test_qualification_preserves_every_original_stage_and_gate(prefix, monkeypatch, passed):
    recovery.OUTPUT.mkdir()
    monkeypatch.setattr(recovery, "verify", lambda directory: None)
    monkeypatch.setattr(recovery, "require_completed_audit", lambda: None)
    monkeypatch.setattr(recovery, "require_cuda", lambda: {})
    monkeypatch.setattr(recovery, "sha256", lambda path: "digest")
    calls = []

    def child(output, label, script, arguments, cpu=False):
        calls.append((output, label, script, arguments, cpu))
        if label == "confirm":
            recovery.write_json(recovery.original.CONFIRMATION / "status.json", {"state": "awaiting_independent_audit"})
        if label == "audit-confirmation":
            recovery.write_json(recovery.original.CONFIRMATION / "gate.json", {"passed": passed})

    monkeypatch.setattr(recovery.original, "run_child", child)
    recovery.qualify()
    expected = ["replay", "audit-replay", "confirm", "audit-confirmation"]
    if passed:
        expected.extend(["benchmarks", "review"])
    assert [entry[1] for entry in calls] == expected
    assert [entry[4] for entry in calls] == ([False, True, False, True, False, True] if passed
                                             else [False, True, False, True])
    assert all(entry[0] == recovery.OUTPUT for entry in calls)
    assert all(entry[3] == ["--stage", entry[1]] for entry in calls[:4])
    if passed:
        assert calls[4][3] == ["--run", str(recovery.original.training.OUTPUT),
                              "--confirmation", str(recovery.original.CONFIRMATION),
                              "--source", str(recovery.original.PARENT_CACHE),
                              "--output", str(recovery.original.CAMPAIGN)]
    assert recovery.original.read(recovery.OUTPUT / "result.json")["goal_success_not_assessed"] is True


def test_failed_audit_launch_does_not_repeat_or_publish(prefix, monkeypatch):
    recovery.ENTRY.mkdir()
    monkeypatch.setattr(recovery, "verify", lambda directory: None)
    monkeypatch.setattr(recovery, "require_cuda", lambda: {})
    calls = []

    def child(command, **options):
        calls.append(command)
        return SimpleNamespace(pid=123, wait=lambda: 1)

    monkeypatch.setattr(recovery.subprocess, "Popen", child)
    with pytest.raises(RuntimeError, match="without repeating inference"):
        recovery.run_audit()
    assert len(calls) == 1
    assert not (recovery.ENTRY / "result.json").exists()


def test_live_original_owner_is_rejected():
    with pytest.raises(ValueError, match="must be stopped"):
        recovery.require_stopped({"pid": recovery.os.getpid()})
