"""Exercise the missing-import prefix and preserve the original scientific execution."""

from contextlib import contextmanager
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import recover_single_model_legacy_strata_replay as recovery


def test_clones_preserve_scientific_code_without_mutating_original_globals():
    before = dict(vars(recovery.original))
    namespace = recovery.execution_namespace()
    assert vars(recovery.original).keys() == before.keys()
    assert all(vars(recovery.original)[name] is value for name, value in before.items())
    assert "deepcopy" not in vars(recovery.original)
    assert namespace["deepcopy"] is deepcopy
    assert namespace["REPLAY"] == recovery.REPLAY != recovery.original.REPLAY
    assert namespace["CONFIRMATION"] == recovery.original.CONFIRMATION
    assert namespace["POLICY"] == recovery.original.POLICY
    for name, function in recovery.owned_functions().items():
        if name != "verify":
            assert inspect.unwrap(namespace[name]).__code__ is inspect.unwrap(function).__code__
            assert inspect.unwrap(namespace[name]).__globals__ is namespace
    assert namespace["training"] is recovery.original.training
    assert len(namespace["training"].confirmation_cases()) == 284


def test_global_dependency_check_includes_nested_code():
    def outer():
        def inner():
            return absent_nested_dependency
        return inner

    assert "absent_nested_dependency" in recovery.code_globals(outer.__code__)
    assert recovery.missing_globals(vars(recovery.original)) == {"materialize_endpoint_tasks": ["deepcopy"]}
    namespace = recovery.execution_namespace()
    assert recovery.missing_globals(namespace) == {}
    del namespace["compact_cache"]
    assert "compact_cache" in recovery.missing_globals(namespace)["materialize_endpoint_tasks"]


@pytest.mark.parametrize("name", ["confirm", "replay_actual"])
def test_cloned_neural_stages_preserve_no_grad(name):
    namespace = recovery.execution_namespace()

    def stop_before_science(output):
        assert not torch.is_grad_enabled()
        raise RuntimeError("test prerequisite sentinel")

    namespace["prerequisite"] = stop_before_science
    with torch.enable_grad(), pytest.raises(RuntimeError, match="test prerequisite sentinel"):
        namespace[name](Path("unused"))
    assert torch.is_grad_enabled()


@pytest.fixture
def materialization(tmp_path):
    namespace = recovery.execution_namespace()
    directory = tmp_path / "replay"
    directory.mkdir()
    namespace["REPLAY"] = directory
    namespace["actual_endpoint_step"] = lambda: 64
    tasks = [{"index": index, "name": f"task_{index}", "artifact": str(tmp_path / f"capsule_{index}.pt"),
              "sha256": f"capsule_hash_{index}"} for index in range(28)]
    specs = [{"ordinal": index} for index in range(28)]
    manifest = {"step": 64, "tasks": tasks, "entries": [{"cohort": "unaltered"}]}
    config = {"base": {"seed": 661100901}, "teacher_identities": {"general": "general", "ratings": "specialist"},
              "batches": [{"tasks": []} for step in range(63)] + [{"tasks": specs}]}
    capsules = {row["artifact"]: {"task": specs[row["index"]], "content_sha256": f"content_{row['index']}"} for row in tasks}
    calls = []

    @contextmanager
    def runtime(base, target, teachers):
        assert base == config["base"] and teachers == config["teacher_identities"]
        assert target == directory / "endpoint_tasks"
        calls.append("runtime")
        yield

    def setup(base):
        calls.append("setup")
        return object(), object(), "criterion", "teachers", "sampler"

    def regenerate(capsule, sampler, teachers, criterion, backend):
        assert (sampler, teachers, criterion) == ("sampler", "teachers", "criterion")
        calls.append(capsule["task"]["ordinal"])
        return {"value": torch.tensor([capsule["task"]["ordinal"]])}

    def save(payload, path):
        torch.save(payload, path)
        return recovery.sha256(path)

    backend = SimpleNamespace(setup=setup)
    namespace["training"] = SimpleNamespace(teacher_policy=SimpleNamespace(runtime=runtime),
        original=SimpleNamespace(save_tensor=save, alignment=SimpleNamespace(mixture=SimpleNamespace(backend=backend))))
    namespace["torch"] = SimpleNamespace(load=lambda path, **kwargs: capsules[str(path)])
    namespace["compact_cache"] = SimpleNamespace(regenerate=regenerate)
    namespace["sha256"] = lambda path: tasks[int(Path(path).stem.split("_")[-1])]["sha256"]
    return namespace, manifest, config, calls


def test_execute_all_28_materialized_tasks_without_mutating_inputs(materialization):
    namespace, manifest, config, calls = materialization
    before = deepcopy((manifest, config))
    source_path = Path(recovery.original.__file__)
    source_hash = recovery.sha256(source_path)
    result = namespace["materialize_endpoint_tasks"](manifest, config)
    assert (manifest, config) == before
    assert calls == ["runtime", "setup", *range(28)]
    assert result is not manifest and result["tasks"] is not manifest["tasks"]
    for expected, actual in zip(manifest["tasks"], result["tasks"]):
        assert actual["compact_artifact"] == expected["artifact"]
        assert actual["compact_sha256"] == expected["sha256"]
        assert actual["content_sha256"] == f"content_{expected['index']}"
        assert recovery.sha256(actual["artifact"]) == actual["sha256"]
        assert torch.load(actual["artifact"], weights_only=True)["value"].item() == expected["index"]
    assert recovery.original.read(namespace["REPLAY"] / "endpoint_tasks/manifest.json") == result
    assert recovery.sha256(source_path) == source_hash
    assert "deepcopy" not in vars(recovery.original)


def test_original_missing_dependency_fails_before_any_runtime_or_task(materialization):
    namespace, manifest, config, calls = materialization
    del namespace["deepcopy"]
    with pytest.raises(NameError, match="deepcopy"):
        namespace["materialize_endpoint_tasks"](manifest, config)
    assert calls == []
    assert list((namespace["REPLAY"] / "endpoint_tasks").iterdir()) == []


@pytest.mark.parametrize("mutation", ["step", "task_count", "capsule_hash", "task_spec"])
def test_materialization_retains_original_identity_guards(materialization, mutation):
    namespace, manifest, config, calls = materialization
    if mutation == "step":
        manifest["step"] = 63
    elif mutation == "task_count":
        manifest["tasks"].pop()
    elif mutation == "capsule_hash":
        namespace["sha256"] = lambda path: "changed"
    else:
        config["batches"][-1]["tasks"][0] = {"ordinal": 999}
    with pytest.raises(ValueError):
        namespace["materialize_endpoint_tasks"](manifest, config)
    assert not any(isinstance(call, int) for call in calls)


@pytest.fixture
def failed_prefix(tmp_path, monkeypatch):
    for name in ("OUTPUT", "REPLAY", "CONFIRMATION"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(recovery.original, name, directory)
        for filename in ("config.json", "registration.json", "status.json"):
            recovery.write_json(directory / filename, {})
    monkeypatch.setattr(recovery.original, "verify", lambda output: None)
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: None)
    monkeypatch.setattr(recovery.original, "CAMPAIGN", tmp_path / "campaign")
    monkeypatch.setattr(recovery.original, "REVIEW", tmp_path / "review")
    monkeypatch.setattr(recovery.original.training, "OUTPUT", tmp_path / "training")
    monkeypatch.setattr(recovery, "missing_globals", lambda namespace: {"materialize_endpoint_tasks": ["deepcopy"]})
    replay = recovery.original.REPLAY
    (replay / "endpoint_tasks").mkdir()
    (replay / "run.lock").touch()
    recovery.write_json(recovery.original.OUTPUT / "status.json", {"state": "failed", "pid": -1,
        "error": "RuntimeError('replay failed; preserve its artifacts without restarting')"})
    recovery.write_json(replay / "status.json", {"state": "failed", "pid": -1,
        "error": 'NameError("name \'deepcopy\' is not defined")'})
    recovery.write_json(recovery.original.CONFIRMATION / "status.json", {"state": "registered"})
    return replay, recovery.original.CONFIRMATION


def test_empty_failed_prefix_and_unused_confirmation_allowed(failed_prefix):
    recovery.require_failed_prefix()
    recovery.require_unused_confirmation()


@pytest.mark.parametrize("artifact", ["endpoint_tasks/task_00.pt", "before_task_00.npz", "candidate_task_00.npz",
                                     "selected_validation", "endpoint_tasks/manifest.json"])
def test_reject_any_scientific_artifact_in_failed_prefix(failed_prefix, artifact):
    replay, confirmation = failed_prefix
    (replay / artifact).touch()
    with pytest.raises(ValueError, match="task or prediction artifacts"):
        recovery.require_failed_prefix()


@pytest.mark.parametrize("artifact", ["bank", "manifest.json", "student.json", "decision.json", "run.lock"])
def test_reject_confirmation_consumption(failed_prefix, artifact):
    replay, confirmation = failed_prefix
    (confirmation / artifact).touch()
    with pytest.raises(ValueError, match="consumed confirmation"):
        recovery.require_unused_confirmation()


def test_reject_live_failed_prefix(failed_prefix, monkeypatch):
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: {"pid": pid})
    with pytest.raises(ValueError, match="stopped pre-inference"):
        recovery.require_failed_prefix()


def test_scratch_redirection_rejected(tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    scratch = scratch_root / "replay"
    scratch.mkdir()
    root = tmp_path / "checkpoint"
    root.mkdir()
    canonical = root / "replay"
    canonical.symlink_to(scratch)
    monkeypatch.setattr(recovery, "SCRATCH", scratch)
    monkeypatch.setattr(recovery, "REPLAY", canonical)
    monkeypatch.setattr(recovery.original, "ROOT", root)
    monkeypatch.setattr(recovery.original.diagnostics.storage, "SCRATCH", scratch_root)
    monkeypatch.setattr(recovery.original.diagnostics.storage, "require_mount", lambda: None)
    recovery.require_storage(True)
    canonical.unlink()
    canonical.mkdir()
    with pytest.raises(ValueError, match="canonical link"):
        recovery.require_storage(True)
