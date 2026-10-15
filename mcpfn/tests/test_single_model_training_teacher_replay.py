import pytest
import torch

import replay_single_model_training_teachers as replay
from recsys_evaluation_common import sha256, write_json


def test_tensor_digest_covers_shape_and_all_values():
    values = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    digest = replay.tensor_digest(values)
    assert replay.tensor_digest(values.clone()) == digest
    assert replay.tensor_digest(values.reshape(1, 3, 2)) != digest
    changed = values.clone()
    changed[0, 1, 2] += .000001
    assert replay.tensor_digest(changed) != digest
    with pytest.raises(ValueError):
        replay.tensor_digest(values.double())
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        replay.tensor_digest(values)


def row_fixture(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"step": 1})
    task = {"index": 0, "name": "legacy_draw_0"}
    logged = {"artifact": str(tmp_path / "task_00.pt"), "sha256": "task_hash"}
    cached = {"teacher": torch.ones(1, 2, 3)}
    config = {"teacher_identities": {"initialization": {"checkpoint": "v8", "sha256": "v8_hash"}}}
    row = {"step": 1, "index": 0, "task": task["name"], "role": "initialization",
           "identity": config["teacher_identities"]["initialization"], "manifest": str(manifest),
           "manifest_sha256": sha256(manifest), "artifact": logged["artifact"], "artifact_sha256": logged["sha256"],
           "shape": [1, 2, 3], "dtype": "float32", "prediction_sha256": replay.tensor_digest(cached["teacher"]),
           "bitwise_equal": True, "identical_replay_bytes_preserved_in": logged["artifact"], "tensor_key": "teacher"}
    arguments = (1, manifest, task, logged, cached, "initialization", config)
    return row, arguments


def test_independent_replay_bytes_are_bound_to_original_cached_tensor(tmp_path):
    row, arguments = row_fixture(tmp_path)
    replay.check_row(row, *arguments)


@pytest.mark.parametrize("field,value", [
    ("step", 2), ("role", "general"), ("prediction_sha256", "wrong"), ("shape", [1, 3, 2]),
    ("bitwise_equal", False), ("tensor_key", "source"), ("identical_replay_bytes_preserved_in", "other"),
    ("identity", {"checkpoint": "general", "sha256": "other"})])
def test_replay_metadata_cannot_hide_substitutions(tmp_path, field, value):
    row, arguments = row_fixture(tmp_path)
    row[field] = value
    with pytest.raises(ValueError, match="exact tensor bytes"):
        replay.check_row(row, *arguments)


def test_changed_cached_target_is_not_accepted_as_replayed(tmp_path):
    row, arguments = row_fixture(tmp_path)
    arguments[4]["teacher"][0, 0, 0] = 2
    with pytest.raises(ValueError):
        replay.check_row(row, *arguments)


@pytest.mark.parametrize("state,step,pid,count", [
    ("training", 64, 999999999, 1792), ("training_complete", 63, 999999999, 1792),
    ("training_complete", 64, 1, 1792), ("training_complete", 64, 999999999, 1791)])
def test_no_live_or_partial_teacher_replay(monkeypatch, state, step, pid, count):
    monkeypatch.setattr(replay, "read", lambda path: {"state": state, "step": step, "pid": pid})
    with pytest.raises(ValueError):
        replay.require_finished_training({"batches": [{"tasks": [None] * count}]})


def test_missing_batch_tasks_fail_before_any_neural_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.training, "OUTPUT", tmp_path)
    directory = tmp_path / "training_batches/step_001"
    directory.mkdir(parents=True)
    write_json(directory / "manifest.json", {"step": 1, "tasks": []})
    with pytest.raises(ValueError, match="full training batch"):
        list(replay.manifests({"batches": [{"step": 1, "tasks": []}]}))
