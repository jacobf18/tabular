from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_all_imputation_column_mean_capacity as capacity
import test_single_model_parent_structured_capacity as predecessor_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_exact_closed_scope_and_original_reserves():
    assert len(capacity.TARGETS) == 2
    assert set(capacity.TARGETS.values()) == {"tail", "vector"}
    assert capacity.SELECTED == capacity.probe.PARENT
    assert capacity.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    assert not any("v28" in name or "selected_checkpoint" in name for name in capacity.TARGETS)
    for function in vars(capacity).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(capacity):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize("mutation", [None, "step", "best_step", "gradient_updates_performed", "config_sha256",
    "parent_checkpoint_sha256", "independent_confirmation_passed", "full_network_training", "selected", "kind"])
def test_only_distinct_unqualified_v51_tail_is_eligible(mutation):
    payload = {"step": 64, "best_step": 0, "gradient_updates_performed": 58, "attempted_batches": 64,
        "config_sha256": capacity.TAIL_CONFIG_SHA256,
        "parent_checkpoint_sha256": capacity.SELECTED_SHA256, "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True, "optimizer": {"state": {}},
        "model": {"weight": torch.ones(3)}}
    selected = {"weight": torch.zeros(3)}
    kind = "tail"
    if mutation == "kind":
        kind = "selected"
    elif mutation == "selected":
        selected = deepcopy(payload["model"])
    elif mutation:
        payload[mutation] = {"step": 50, "best_step": 64, "gradient_updates_performed": 56,
            "config_sha256": "changed", "parent_checkpoint_sha256": "changed",
            "independent_confirmation_passed": True, "full_network_training": False}[mutation]
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, kind, model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(payload, kind, model_tensors=1, selected_model=selected)


def test_exact_copy_and_vector_contracts_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(predecessor_tests, "capacity", capacity)
    predecessor_tests.test_vectors_keep_full_float32_finite_coordinates()
    predecessor_tests.test_preservation_copy_is_exact_and_refuses_overwrite(tmp_path, monkeypatch)
