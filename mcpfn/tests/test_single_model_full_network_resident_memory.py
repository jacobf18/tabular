from copy import deepcopy

import pytest
import torch

import probe_single_model_full_network_resident_memory as probe
from test_single_model_full_network_memory import configuration, fixture


def test_probe_binds_the_same_failed_task_without_a_new_training_seed():
    probe.validate_failure(*fixture())
    assert probe.OUTPUT != probe.template.OUTPUT
    assert probe.capture.__globals__ is probe.__dict__
    assert probe.register.__globals__ is probe.__dict__
    assert 'torch.empty((84, training.TRAINABLE_PARAMETERS), dtype=torch.float32, device="cuda")' in probe.ADAPTED_SOURCE
    assert 'gradient_records(gradients[:3], entries)' in probe.ADAPTED_SOURCE
    assert 'paths.extend(template.OUTPUT / name' in probe.ADAPTED_SOURCE
    assert '"all_training_tasks_memory_qualified": False' in probe.ADAPTED_SOURCE


def test_allocator_is_still_the_only_recipe_change(monkeypatch):
    original = {"base": {"memory_fraction": .8, "learning_rate": 5e-6}, "steps": 64}
    before = deepcopy(original)
    monkeypatch.setattr(probe.training, "read", lambda path: original)
    config = {**configuration(), "policy": probe.POLICY, "gpu_gradient_rows": 84}
    updated = probe.execution_config(config)
    assert original == before
    updated["base"]["memory_fraction"] = .8
    assert updated == original
    config["gpu_gradient_rows"] = 3
    with pytest.raises(ValueError, match="memory-only"):
        probe.execution_config(config)


def test_report_does_not_read_the_unpopulated_gradient_rows(monkeypatch):
    monkeypatch.setattr(probe.training, "TRAINABLE_PARAMETERS", 4)
    gradients = torch.full((84, 4), float("nan"))
    gradients[:3] = torch.arange(12).reshape(3, 4)
    entries = [{"kind": name, "index": index} for index, name in enumerate(("composite", "rmse", "common_nrmse"))]
    records = probe.gradient_records(gradients[:3], entries)
    assert len(records) == 3
    assert all(record["parameters"] == 4 for record in records)
    with pytest.raises(ValueError):
        probe.gradient_records(gradients, entries)
