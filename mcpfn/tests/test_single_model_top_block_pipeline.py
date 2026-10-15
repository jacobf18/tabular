from contextlib import nullcontext
from copy import deepcopy
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import run_single_model_top_block_pipeline as pipeline


def test_original_qualification_scope_and_top_block_inventory():
    source = pipeline.ADAPTED_SOURCE
    for expression in ('audit["through_step"] != 64', 'audit["training_episodes"] != 1792',
        'audit["prediction_grids"] != 1216', 'require_compact_audit(audit)',
        'expected_count = 28 if payload["record"]["point_after"] is None else 56',
        'len(bank) != 284', 'count != 1136', 'len(decision["comparisons"]) != 22',
        'scripts/run_single_matrixformer_cached_benchmarks.py', 'scripts/review_single_matrixformer_campaign.py',
        'cached = check_materialized_task(row)', 'wait_for_training_audit(output)',
        'view = top_block.TopBlockParameterView(model)', 'result["frozen"]["frozen_tensors"] != 220',
        'unselected_parameters_and_all_buffers_bitwise_unchanged',
        'training.paired.metric_decision(measured["before"], measured["candidate"])',
        'training.paired.paired_displacement(', 'audit.get("top_block_training") is not True'):
        assert expression in source
    assert source.count('training.preflight.frozen_inventory(parent, model.state_dict(), mutable_names)') == 2
    for expression in ('run_child(output, "training_audit"', 'decoder.DecoderParameterView', 'frozen_tensors"] != 248'):
        assert expression not in source
    assert pipeline.PARENT_CACHE == pipeline.template.PARENT_CACHE
    assert pipeline.training is pipeline.training_audit.training
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("top_block_v24")
    for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
                 "materialize_endpoint_tasks", "check_materialized_task"):
        assert getattr(pipeline, name).__globals__ is pipeline.__dict__


def test_confirmation_and_promotion_bodies_and_capacity_budgets_remain_unchanged():
    current, previous = ast.parse(pipeline.ADAPTED_SOURCE), ast.parse(pipeline.template.ADAPTED_SOURCE)
    for name in ("confirm", "audit_confirmation", "confirmation_setup", "no_confirmation_reason"):
        actual = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original = next(node for node in previous.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(original)
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.no_confirmation_reason({"step": 0}, {}) is not None
    assert pipeline.no_confirmation_reason({"step": 25}, {"selected_validation_passed": False}) is not None


@pytest.mark.parametrize("label", list(pipeline.STAGE_WRITE_BUDGETS))
@pytest.mark.parametrize("path", [pipeline.ROOT, Path("/"), pipeline.training.SCRATCH.parent])
def test_capacity_gates_every_volume_before_each_stage(monkeypatch, label, path):
    capacity = {pipeline.ROOT: pipeline.MINIMUM_FREE + pipeline.STAGE_WRITE_BUDGETS[label],
                Path("/"): 6 * 1024**3, pipeline.training.SCRATCH.parent: 50 * 1024**3}
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda target: SimpleNamespace(free=capacity[target]))
    pipeline.require_stage_capacity(label)
    capacity[path] -= 1
    with pytest.raises(OSError, match="Preserve"):
        pipeline.require_stage_capacity(label)
    with pytest.raises(ValueError, match="declared capacity"):
        pipeline.require_stage_capacity("undeclared")


def test_live_identity_includes_start_ticks_and_exact_original_parent(tmp_path, monkeypatch):
    current = pipeline.process_identity(os.getpid())
    assert current["pid"] == os.getpid() and current["start_ticks"] > 0
    assert pipeline.process_identity(-1) is None
    monkeypatch.setattr(pipeline.training, "OUTPUT", tmp_path)
    pipeline.write_json(tmp_path / "status.json", {"pid": os.getpid()})
    identities = {os.getpid(): {"pid": os.getpid(), "command": "python train_single_matrixformer_top_block.py --stage train"},
                  os.getppid(): {"pid": os.getppid(), "start_ticks": 9,
                                "command": "bash python scripts/audit_single_model_top_block_training.py"}}
    monkeypatch.setattr(pipeline, "process_identity", identities.get)
    assert pipeline.register_chain_identity() == identities[os.getppid()]
    identities[os.getppid()]["command"] = "unrelated parent"
    with pytest.raises(ValueError, match="existing train-then-audit chain"):
        pipeline.register_chain_identity()
    identities[os.getpid()]["command"] = "previous trainer"
    with pytest.raises(ValueError, match="V24 one-shot"):
        pipeline.register_chain_identity()


def test_waiter_refuses_pid_reuse_and_accepts_only_completed_original_audit(tmp_path, monkeypatch):
    chain = {"pid": 123, "start_ticks": 10, "command": "original chain"}
    pipeline.write_json(tmp_path / "config.json", {"training_and_audit_chain": chain})
    monkeypatch.setattr(pipeline, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: {**chain, "start_ticks": 11})
    with pytest.raises(RuntimeError, match="stopped without"):
        pipeline.wait_for_training_audit(tmp_path)
    replay = tmp_path / "audit_task_replay"
    replay.mkdir()
    calls = []
    monkeypatch.setattr(pipeline, "prerequisite", lambda output: calls.append(output))
    pipeline.write_json(replay / "status.json", {"state": "complete"})
    pipeline.wait_for_training_audit(tmp_path)
    assert calls == [tmp_path]
    pipeline.write_json(replay / "status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="audit failed"):
        pipeline.wait_for_training_audit(tmp_path)


def test_complete_compact_audit_rejects_counts_flags_and_trajectory_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "AUDIT", tmp_path / "audit")
    pipeline.write_json(pipeline.AUDIT / "audit.json", {})
    directory = tmp_path / "audit_task_replay"
    pipeline.write_json(directory / "status.json", {"state": "complete"})
    pipeline.write_json(directory / "result.json", {"complete": True, "tasks": 1792,
                        "audit_sha256": pipeline.sha256(pipeline.AUDIT / "audit.json")})
    pipeline.write_json(directory / "config.json", {"source_sha256": {}})
    report = {"teacher_identity_checked_tasks": 1792, "training_teacher_policy": pipeline.training.teacher_policy.POLICY,
              "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
              "exact_complete_task_and_teacher_digest_checks": True, "decoder_only_training": False,
              "top_block_training": True, "autograd_replayed": False, "full_optimizer_trajectory_replayed": False}
    pipeline.require_compact_audit(report)
    for key, wrong in {"teacher_identity_checked_tasks": 1791, "independently_regenerated_training_tasks": 2688,
                       "independently_regenerated_rating_contexts": 192, "decoder_only_training": True,
                       "top_block_training": False, "exact_complete_task_and_teacher_digest_checks": False,
                       "autograd_replayed": True, "full_optimizer_trajectory_replayed": True}.items():
        with pytest.raises(ValueError, match="complete independent"):
            pipeline.require_compact_audit({**report, key: wrong})


@pytest.fixture
def materialized(tmp_path, monkeypatch):
    run, replay = tmp_path / "run", tmp_path / "replay"
    replay.mkdir()
    directory = run / "training_batches" / "step_064"
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
    manifest = {"step": 64, "tasks": records, "entries": []}
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
    return pipeline.materialize_endpoint_tasks(manifest, config), manifest


def test_materialized_tasks_preserve_all_original_typed_bytes(materialized):
    result, original = materialized
    assert result["entries"] == original["entries"] and len(result["tasks"]) == 28
    for task, old in zip(result["tasks"], original["tasks"]):
        assert task["compact_artifact"] == old["artifact"] and task["compact_sha256"] == old["sha256"]
        row = {"task": task["name"], "artifact": task["artifact"], "artifact_sha256": task["sha256"]}
        payload = pipeline.check_materialized_task(row)
        assert pipeline.compact_cache.content_digest(payload) == task["content_sha256"]


@pytest.mark.parametrize("mutation", ["row_digest", "path", "task_name", "teacher_bytes", "manifest_digest"])
def test_materialized_tasks_reject_substitution(materialized, mutation):
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


def test_confirmation_reservation_excludes_consumed_v23_cases():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": pipeline.training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    altered = deepcopy(reservation)
    altered["confirmation_cases"] = pipeline.template.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(altered, "pipeline", "run", True)
