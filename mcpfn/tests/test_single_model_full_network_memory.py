from copy import deepcopy

import pytest
import torch

import probe_single_model_full_network_memory as probe


def fixture():
    task = deepcopy(probe.training.batches()[8]["tasks"][0])
    capsule = {"format": probe.training.compact_cache.FORMAT, "task": task,
               "content_sha256": "a" * 64, "scaling": None}
    records = [{"step": index, "accepted": True, "accepted_updates": index} for index in range(1, 9)]
    return ({"state": "failed", "error": "OutOfMemoryError:111.84 GiB allowed"},
            {"state": "failed"}, records, capsule, {"step": 0, "gradient_updates_performed": 0})


def test_only_the_actual_failed_task_and_eight_unqualified_training_records_are_admitted():
    probe.validate_failure(*fixture())


@pytest.mark.parametrize("mutation", ["live", "other_error", "different_cap", "missing_record", "rejected", "different_task", "later_checkpoint"])
def test_failure_binding_rejects_substitution(mutation):
    current, coordinator, records, capsule, latest = fixture()
    if mutation == "live":
        coordinator["state"] = "running"
    elif mutation == "other_error":
        current["error"] = "unrelated"
    elif mutation == "different_cap":
        current["error"] = "OutOfMemoryError:128 GiB allowed"
    elif mutation == "missing_record":
        records.pop()
    elif mutation == "rejected":
        records[2]["accepted"] = False
    elif mutation == "different_task":
        capsule["task"]["specification"]["seed"] += 1
    else:
        latest["step"] = 8
    with pytest.raises(ValueError):
        probe.validate_failure(current, coordinator, records, capsule, latest)


def configuration():
    return {"policy": probe.POLICY, "memory_fraction": .9, "gpu_headroom_vectors": 4,
            "parent_sha256": probe.training.PARENT_SHA256, "parent_checkpoint": str(probe.training.PARENT),
            "training_updates_performed": 0, "new_seed_consumption": False,
            "candidate_selection_performed": False, "failed_training_config": "old-config"}


def test_only_allocator_configuration_changes_without_mutating_registered_recipe(monkeypatch):
    original = {"base": {"memory_fraction": .8, "nested": {"learning_rate": 5e-6}}, "steps": 64}
    before = deepcopy(original)
    monkeypatch.setattr(probe.training, "read", lambda path: original)
    actual = probe.execution_config(configuration())
    assert original == before
    actual["base"]["memory_fraction"] = .8
    assert actual == original and actual is not original


@pytest.mark.parametrize("field,value", [("memory_fraction", .95), ("gpu_headroom_vectors", 0),
    ("parent_sha256", "different"), ("training_updates_performed", 1),
    ("new_seed_consumption", True), ("candidate_selection_performed", True)])
def test_probe_cannot_be_relabeled_as_training_or_selection(field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(ValueError, match="memory-only"):
        probe.execution_config(config)


def test_all_three_complete_gradients_are_finite_and_hashed(monkeypatch):
    monkeypatch.setattr(probe.training, "TRAINABLE_PARAMETERS", 4)
    gradients = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    entries = [{"kind": name, "index": index} for index, name in enumerate(("composite", "rmse", "common_nrmse"))]
    records = probe.gradient_records(gradients, entries)
    assert len(records) == 3 and all(record["parameters"] == 4 for record in records)
    assert len({record["sha256"] for record in records}) == 3
    gradients[1, 2] = float("nan")
    with pytest.raises(FloatingPointError):
        probe.gradient_records(gradients, entries)
    with pytest.raises(ValueError):
        probe.gradient_records(gradients[:2], entries[:2])


def test_capacity_preserves_each_volume_reserve(monkeypatch):
    for volume in probe.training.diagnostics.RESERVES:
        def usage(path):
            current = "checkpoint" if path == probe.training.ROOT else "root" if str(path) == "/root" else "scratch"
            free = probe.training.diagnostics.RESERVES[current] + 1024**2 - (current == volume)
            return type("Usage", (), {"free": free})()
        monkeypatch.setattr(probe.shutil, "disk_usage", usage)
        with pytest.raises(OSError, match=volume):
            probe.require_capacity()
