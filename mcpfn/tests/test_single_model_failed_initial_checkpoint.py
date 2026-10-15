from copy import deepcopy

import pytest
import torch

import relocate_single_model_failed_initial_checkpoint as relocation


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    parent = tmp_path / "selected_checkpoint.pth"
    model = {"weight": torch.tensor([0., 1.]), "buffer": torch.tensor([2.])}
    torch.save({"model": model}, parent)
    digest = relocation.sha256(parent)
    monkeypatch.setattr(relocation.memory.training, "PARENT", parent)
    monkeypatch.setattr(relocation.memory.training, "PARENT_SHA256", digest)
    payload = {"step": 0, "attempted_batches": 0, "gradient_updates_performed": 0, "best_step": 0,
        "parent_training_updates": 98, "config_sha256": "config", "parent_checkpoint_sha256": digest,
        "preflight_only": False, "independent_confirmation_passed": False, "full_network_training": True,
        "optimizer": {"state": {}, "param_groups": [{}]}, "model": deepcopy(model)}
    return payload, {"state": "failed", "error": "OutOfMemoryError"}, relocation.OWNERS["full_network_v34"], "config"


def test_initial_checkpoint_is_not_misrepresented_as_successful_training(tmp_path, monkeypatch):
    relocation.checkpoint_identity(*fixture(tmp_path, monkeypatch))
    assert '"failed_training_run_qualified": False' in relocation.MAIN_SOURCE
    assert 'len(records) != 1' in relocation.MAIN_SOURCE
    assert relocation.main.__globals__ is relocation.__dict__


@pytest.mark.parametrize("field,value", [("step", 8), ("attempted_batches", 8), ("gradient_updates_performed", 8),
    ("best_step", 10), ("parent_training_updates", 99), ("config_sha256", "changed"),
    ("parent_checkpoint_sha256", "changed"), ("preflight_only", True),
    ("independent_confirmation_passed", True), ("full_network_training", False), ("optimizer", {"state": {0: {}}})])
def test_no_trained_or_qualified_checkpoint_can_be_moved(tmp_path, monkeypatch, field, value):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch)
    payload[field] = value
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, expected, digest)


@pytest.mark.parametrize("mutation", ["weights", "signed_zero", "dtype", "missing_buffer", "link", "success"])
def test_every_tensor_and_original_failed_state_is_preserved(tmp_path, monkeypatch, mutation):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch)
    if mutation == "weights":
        payload["model"]["weight"][1] = 3.
    elif mutation == "signed_zero":
        payload["model"]["weight"][0] = -0.
    elif mutation == "dtype":
        payload["model"]["weight"] = payload["model"]["weight"].double()
    elif mutation == "missing_buffer":
        del payload["model"]["buffer"]
    elif mutation == "success":
        current["state"] = "training_complete"
    else:
        parent = relocation.memory.training.PARENT
        destination = tmp_path / "moved.pth"
        parent.rename(destination)
        parent.symlink_to(destination)
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, expected, digest)


def test_copy_then_publish_keeps_parent_regular_and_original_serialized_bytes(tmp_path, monkeypatch):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch)
    directory = tmp_path / "full_network_v34"
    directory.mkdir()
    relocation.write_json(directory / "config.json", {})
    payload["config_sha256"] = relocation.sha256(directory / "config.json")
    relocation.write_json(directory / "status.json", current)
    source = directory / "latest_checkpoint.pth"
    torch.save(payload, source)
    row = {"source": str(source), "destination": str(tmp_path / "scratch/latest_checkpoint.pth"),
           "bytes": source.stat().st_size, "sha256": relocation.sha256(source)}
    parent_hash = relocation.sha256(relocation.memory.training.PARENT)
    relocation.copy_and_verify(row)
    assert not source.is_symlink()
    relocation.copying.publish(row)
    assert source.is_symlink() and relocation.sha256(source) == row["sha256"]
    assert not relocation.memory.training.PARENT.is_symlink()
    assert relocation.sha256(relocation.memory.training.PARENT) == parent_hash
    loaded = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    relocation.checkpoint_identity(loaded, current, expected, payload["config_sha256"])
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)
