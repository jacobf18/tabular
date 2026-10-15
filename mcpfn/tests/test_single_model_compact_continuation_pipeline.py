from contextlib import nullcontext
from copy import deepcopy
import ast
import os
from types import SimpleNamespace

import pytest
import torch

import run_single_model_compact_continuation_pipeline as pipeline


def test_all_original_confirmation_and_real_comparisons_remain_mandatory():
    source = pipeline.ADAPTED_SOURCE
    for expression in ('audit["through_step"] != 96', 'audit["training_episodes"] != 2688',
        'audit["prediction_grids"] != 1368', 'require_compact_audit(audit)',
        'expected_count = 28 if payload["record"]["point_after"] is None else 56',
        'len(bank) != 284', 'count != 1136', 'len(decision["comparisons"]) != 22',
        'scripts/run_single_matrixformer_cached_benchmarks.py', 'scripts/review_single_matrixformer_campaign.py',
        'cached = check_materialized_task(row)', 'wait_for_training_audit(output)'):
        assert expression in source
    assert 'run_child(output, "training_audit"' not in source
    assert pipeline.PARENT_CACHE == pipeline.baseline.PARENT_CACHE
    assert pipeline.training is pipeline.training_audit.training
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("compact_continuation_v23")


def test_confirmation_and_promotion_function_bodies_are_unchanged():
    actual_tree = ast.parse(pipeline.ADAPTED_SOURCE)
    original_tree = ast.parse(pipeline.baseline.ADAPTED_SOURCE)
    for name in ("confirm", "audit_confirmation", "confirmation_setup", "no_confirmation_reason"):
        actual, original = getattr(pipeline, name).__code__, getattr(pipeline.baseline, name).__code__
        assert actual.co_code == original.co_code
        actual_node = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original_node = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual_node) == ast.dump(original_node)
    assert pipeline.no_confirmation_reason({"step": 0}, {}) is not None
    assert pipeline.no_confirmation_reason({"step": 16}, {"selected_validation_passed": False}) is not None


def test_live_process_identity_binds_start_ticks():
    current = pipeline.process_identity(os.getpid())
    assert current["pid"] == os.getpid() and current["start_ticks"] > 0 and current["command"]
    assert pipeline.process_identity(-1) is None


@pytest.mark.parametrize("label", list(pipeline.STAGE_WRITE_BUDGETS))
def test_capacity_gate_precedes_each_new_stage(monkeypatch, label):
    required = pipeline.MINIMUM_FREE + pipeline.STAGE_WRITE_BUDGETS[label]
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required))
    pipeline.require_stage_capacity(label)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1))
    with pytest.raises(OSError, match="Preserve18 GiB"):
        pipeline.require_stage_capacity(label)
    with pytest.raises(ValueError, match="declared capacity"):
        pipeline.require_stage_capacity("undeclared")


def test_waiter_uses_existing_chain_and_refuses_disappearance(tmp_path, monkeypatch):
    chain = {"pid": 123, "start_ticks": 10, "command": "original chain"}
    pipeline.write_json(tmp_path / "config.json", {"training_and_audit_chain": chain})
    monkeypatch.setattr(pipeline, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: {**chain, "start_ticks": 11})
    with pytest.raises(RuntimeError, match="stopped without"):
        pipeline.wait_for_training_audit(tmp_path)


def test_waiter_accepts_only_completed_audit(tmp_path, monkeypatch):
    pipeline.write_json(tmp_path / "config.json", {})
    monkeypatch.setattr(pipeline, "AUDIT", tmp_path / "audit")
    replay = tmp_path / "audit_task_replay"
    replay.mkdir()
    pipeline.write_json(replay / "status.json", {"state": "complete"})
    calls = []
    monkeypatch.setattr(pipeline, "prerequisite", lambda output: calls.append(output))
    pipeline.wait_for_training_audit(tmp_path)
    assert calls == [tmp_path]
    pipeline.write_json(replay / "status.json", {"state": "failed", "error": "bad digest"})
    with pytest.raises(RuntimeError, match="audit failed"):
        pipeline.wait_for_training_audit(tmp_path)


@pytest.fixture
def materialized(tmp_path, monkeypatch):
    run, replay = tmp_path / "run", tmp_path / "replay"
    replay.mkdir()
    directory = run / "training_batches" / "step_096"
    directory.mkdir(parents=True)
    monkeypatch.setattr(pipeline.training, "OUTPUT", run)
    monkeypatch.setattr(pipeline, "REPLAY", replay)
    tasks, records, episodes = [], [], {}
    for index in range(28):
        task = {"index": index, "name": f"task_{index}", "specification": {"seed": 100 + index}}
        episode = {"scaling": None, "source": torch.tensor([float(index), -0., float("nan")])}
        original = {"episode": episode, "teacher": torch.tensor([2., 3.]), "routing": {}, "task": task}
        path = directory / f"task_{index:02d}.pt"
        torch.save(pipeline.compact_cache.capsule_for(original), path)
        records.append({**task, "artifact": str(path), "sha256": pipeline.sha256(path)})
        tasks.append(task)
        episodes[task["specification"]["seed"]] = episode
    manifest = {"step": 96, "tasks": records, "entries": []}
    pipeline.write_json(directory / "manifest.json", manifest)
    config = {"base": {}, "teacher_identities": {}, "batches": [{"tasks": tasks}]}
    sampler = SimpleNamespace(sample=lambda step, seed: episodes[seed])
    backend = pipeline.training.original.alignment.mixture.backend
    monkeypatch.setattr(pipeline.training.teacher_policy, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(backend, "setup", lambda config: (None, None, None, None, sampler))
    monkeypatch.setattr(backend, "teacher_target", lambda *args: (torch.tensor([2., 3.]), {}))
    monkeypatch.setattr(backend, "cache_episode", lambda episode: episode)
    monkeypatch.setattr(pipeline.training.original, "save_tensor",
                        lambda payload, path: (torch.save(payload, path), pipeline.sha256(path))[1])
    result = pipeline.materialize_endpoint_tasks(manifest, config)
    return result, manifest


def test_materialized_final_tasks_preserve_original_capsules_and_all_typed_bytes(materialized):
    result, original = materialized
    assert result["entries"] == original["entries"] and len(result["tasks"]) == 28
    for task, old in zip(result["tasks"], original["tasks"]):
        assert task["compact_artifact"] == old["artifact"] and task["compact_sha256"] == old["sha256"]
        row = {"task": task["name"], "artifact": task["artifact"], "artifact_sha256": task["sha256"]}
        payload = pipeline.check_materialized_task(row)
        assert pipeline.compact_cache.content_digest(payload) == task["content_sha256"]


@pytest.mark.parametrize("mutation", ["row_digest", "path", "task_name", "teacher_bytes", "manifest_digest"])
def test_materialized_verifier_rejects_substitution(materialized, mutation):
    result, original = materialized
    task = result["tasks"][0]
    row = {"task": task["name"], "artifact": task["artifact"], "artifact_sha256": task["sha256"]}
    if mutation == "row_digest":
        row["artifact_sha256"] = "wrong"
    elif mutation == "path":
        row["artifact"] = original["tasks"][0]["artifact"]
    elif mutation == "task_name":
        row["task"] = "not-a-task"
    elif mutation == "manifest_digest":
        task["content_sha256"] = "0" * 64
        pipeline.write_json(pipeline.REPLAY / "endpoint_tasks" / "manifest.json", result)
    else:
        payload = torch.load(task["artifact"], weights_only=True)
        payload["teacher"][0] = 999.
        torch.save(payload, task["artifact"])
        task["sha256"] = row["artifact_sha256"] = pipeline.sha256(task["artifact"])
        pipeline.write_json(pipeline.REPLAY / "endpoint_tasks" / "manifest.json", result)
    with pytest.raises(ValueError):
        pipeline.check_materialized_task(row)


def test_reserved_confirmation_is_not_v22s_consumed_cases():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": pipeline.training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    altered = deepcopy(reservation)
    altered["confirmation_cases"] = pipeline.training.baseline.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(altered, "pipeline", "run", True)
