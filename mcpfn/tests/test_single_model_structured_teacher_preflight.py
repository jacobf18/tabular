from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import preflight_single_model_structured_teacher as preflight
from recsys_evaluation_common import atomic_npz, sha256, write_json


def test_only_teacher_assignment_and_fresh_seeds_change():
    baseline = preflight.preceding.training.batches()[0]
    batch = preflight.batch()
    assert len(batch["tasks"]) == 28
    for actual, expected in zip(batch["tasks"], baseline["tasks"]):
        normalized = deepcopy(actual)
        normalized["gradient_seed"] = expected["gradient_seed"]
        recipe = normalized["specification"].get("structured", normalized["specification"])
        recipe["seed"] = expected["specification"].get("structured", expected["specification"])["seed"]
        assert normalized == expected
    seeds = preflight.training.alignment.fixed.seeds_in(batch)
    assert len(seeds) == 56
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(baseline))
    assert preflight.DATA_SEED == 371100901 and preflight.GRADIENT_SEED == 372100901
    assert preflight.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.RATE == 2e-5
    roles = [preflight.expected_role(task) for task in batch["tasks"]]
    assert roles.count("general") == 24 and roles.count("initialization") == roles.count("specialist") == 2


def test_training_scope_replay_and_original_optimizer_bindings():
    source = preflight.ADAPTED_SOURCE
    assert 'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):' in source
    assert "with training.alignment.mixture.runtime(" not in source
    assert "paired.protected_update(" in source and "paired.check_endpoint(" in source
    assert "teacher_evidence = audit_teacher_replay(output, config)" in source
    assert 'hashes.update(teacher_evidence["artifact_sha256"])' in source
    assert '"selection_qualified": False' in source
    assert preflight.previous_preflight.OUTPUT != preflight.OUTPUT
    assert "teacher_policy.runtime(" not in preflight.previous_preflight.ADAPTED_SOURCE


def test_storage_admission_retains_atomic_reserve(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 24 * 1024**3 - 1})())
    with pytest.raises(OSError, match="18 GiB"):
        preflight.require_space()


def replay_fixture(tmp_path, monkeypatch):
    batch = preflight.batch()
    identities = {}
    for role in ("initialization", "general", "specialist"):
        path = tmp_path / f"{role}.pth"
        path.write_bytes(role.encode())
        identities[role] = {"checkpoint": str(path), "sha256": sha256(path)}
    config = {"batch": batch, "teacher_identities": identities}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "result.json", {"training_pid": 123})
    (tmp_path / "training_batch").mkdir()
    (tmp_path / "teacher_replay").mkdir()
    rows, logged_tasks = [], []
    for task in batch["tasks"]:
        role = preflight.expected_role(task)
        target = torch.full((1, 2, 3), float(task["index"]))
        cached = {"teacher": target, "routing": {"target": role, "specialist_weight": float(role == "specialist"),
                  "policy": preflight.teacher_policy.POLICY, "teacher_identity": identities[role]}}
        path = tmp_path / "training_batch" / f'task_{task["index"]:02d}.pt'
        torch.save(cached, path)
        logged = {"artifact": str(path), "sha256": sha256(path)}
        logged_tasks.append(logged)
        prediction = tmp_path / "teacher_replay" / f'task_{task["index"]:02d}.npz'
        atomic_npz(prediction, teacher=target.numpy())
        rows.append({"task": task["name"], "index": task["index"], "role": role, "identity": identities[role],
                     "prediction": str(prediction), "prediction_sha256": sha256(prediction),
                     "task_sha256": logged["sha256"], "bitwise_equal": True})
    write_json(tmp_path / "training_batch/manifest.json", {"tasks": logged_tasks})
    report = {"complete": True, "rows": rows, "full_grid_bitwise_replay": True, "student_or_validation_evaluated": False,
              "training_pid": 123, "pid": 456, "config_sha256": sha256(tmp_path / "config.json"),
              "model_states": {role: {**identity, "bitwise_equal": True, "state_tensors": 1} for role, identity in identities.items()}}
    write_json(tmp_path / "teacher_replay/result.json", report)
    monkeypatch.setattr(preflight.task_audit, "check_task", lambda *args: {})
    return config, report


def test_cpu_audit_checks_all_original_targets_and_checkpoint_hashes(tmp_path, monkeypatch):
    config, report = replay_fixture(tmp_path, monkeypatch)
    result = preflight.audit_teacher_replay(tmp_path, config)
    assert result["complete"] and result["full_grids"] == 28 and result["bitwise_equal"]
    assert len(result["artifact_sha256"]) == 61


@pytest.mark.parametrize("corruption", ["grid", "routing", "role", "checkpoint", "omission", "same_process", "model_state"])
def test_teacher_labels_cannot_hide_wrong_targets(tmp_path, monkeypatch, corruption):
    config, report = replay_fixture(tmp_path, monkeypatch)
    if corruption == "grid":
        path = Path(report["rows"][1]["prediction"])
        atomic_npz(path, teacher=np.zeros((1, 2, 3), dtype=np.float32))
        report["rows"][1]["prediction_sha256"] = sha256(path)
    elif corruption == "routing":
        path = tmp_path / "training_batch/task_01.pt"
        cached = torch.load(path, weights_only=False)
        cached["routing"]["target"] = "initialization"
        torch.save(cached, path)
    elif corruption == "role":
        report["rows"][1]["role"] = "initialization"
    elif corruption == "checkpoint":
        Path(config["teacher_identities"]["general"]["checkpoint"]).write_bytes(b"replacement")
    elif corruption == "omission":
        report["rows"].pop()
    elif corruption == "same_process":
        report["pid"] = report["training_pid"]
    else:
        report["model_states"]["general"]["bitwise_equal"] = False
    write_json(tmp_path / "teacher_replay/result.json", report)
    with pytest.raises((ValueError, RuntimeError)):
        preflight.audit_teacher_replay(tmp_path, config)
