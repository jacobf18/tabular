from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_four_draw_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def tail():
    return {"step": 256, "epoch": 256, "attempted_batches": 256, "gradient_updates_performed": 111,
        "parent_training_updates": 98, "parent_checkpoint_sha256": capacity.SELECTED_SHA256,
        "config_sha256": capacity.TAIL_CONFIG_SHA256, "best_step": 0, "preflight_only": False,
        "independent_confirmation_passed": False, "full_network_training": True,
        "model": {"weight": torch.ones(3), "buffer": torch.zeros(1)},
        "optimizer": {"state": {0: {"step": torch.tensor(111.), "exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3)}},
            "param_groups": [{"params": [0]}]},
        "rng": {"cpu": torch.zeros(4, dtype=torch.uint8), "cuda": [torch.ones(4, dtype=torch.uint8)]}}


def test_only_the_closed_v56_tail_and_four_audited_vectors_are_declared():
    assert len(capacity.TARGETS) == 5
    assert set(capacity.TARGETS.values()) == {"tail", "vector"}
    assert list(capacity.TARGETS.values()).count("vector") == 4
    assert capacity.EXPECTED_BYTES == 936293291 + 4 * 311995561
    assert capacity.TRAINING_ADMISSION_BYTES == 22 * 1024**3
    assert not any("v28" in name or "selected" in name or "best_" in name for name in capacity.TARGETS)
    assert "four_draw_preflight_v1_mean_recovery_v1" in capacity.STATES
    for function in vars(capacity).values():
        if inspect.isfunction(function) and function.__globals__ is vars(capacity):
            assert not unresolved_globals(function), function.__name__


def test_complete_typed_payloads_are_accepted_without_relabeling_the_negative_tail():
    capacity.validate_payload(torch.ones(3), "vector", full_parameters=3)
    capacity.validate_payload(tail(), "tail", model_tensors=2, optimizer_tensors=1)


@pytest.mark.parametrize("value", [torch.ones(2), torch.ones(3, dtype=torch.float64), torch.tensor([1., float("nan"), 1.]), {"model": {}}])
def test_incomplete_nonfinite_or_retyped_diagnostic_vectors_fail(value):
    with pytest.raises(ValueError):
        capacity.validate_payload(value, "vector", full_parameters=3)


@pytest.mark.parametrize("field,value", [("step", 64), ("attempted_batches", 128), ("gradient_updates_performed", 110),
    ("parent_training_updates", 0), ("best_step", 128), ("independent_confirmation_passed", True), ("config_sha256", "changed")])
def test_tail_scientific_identity_cannot_change(field, value):
    payload = tail()
    payload[field] = value
    with pytest.raises(ValueError):
        capacity.validate_payload(payload, "tail", model_tensors=2, optimizer_tensors=1)


@pytest.mark.parametrize("mutation", ["model", "moments", "counter", "parameters", "rng"])
def test_complete_original_model_optimizer_and_rng_are_required(mutation):
    payload = deepcopy(tail())
    if mutation == "model":
        payload["model"]["weight"][0] = float("nan")
    elif mutation == "moments":
        del payload["optimizer"]["state"][0]["exp_avg_sq"]
    elif mutation == "counter":
        payload["optimizer"]["state"][0]["step"] -= 1
    elif mutation == "parameters":
        payload["optimizer"]["param_groups"][0]["params"] = [1]
    else:
        payload["rng"]["cuda"] = []
    with pytest.raises(ValueError):
        capacity.validate_payload(payload, "tail", model_tensors=2, optimizer_tensors=1)
