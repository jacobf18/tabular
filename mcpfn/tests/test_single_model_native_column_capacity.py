from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_native_column_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_scope_keeps_teachers_selected_and_unaudited_material():
    assert len(capacity.TARGETS) == 3
    assert capacity.TRAINING_ADMISSION_BYTES == 21877489664
    assert not any(token in name for name in capacity.TARGETS for token in ("selected_checkpoint", "best_validation", "anchor_rehearsal_v28"))
    assert capacity.SELECTED.parent.name == "legacy_strata_v29"
    for function in vars(capacity).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(capacity):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize("field,value", [(None, None), ("best_step", 10), ("step", 10), ("gradient_updates_performed", 55),
    ("attempted_batches", 63), ("config_sha256", "changed"), ("parent_checkpoint_sha256", "other"),
    ("independent_confirmation_passed", True), ("optimizer", {}), ("full_network_training", False)])
def test_only_original_v47_no_selection_tail_is_admitted(field, value):
    parent = {"matrix": torch.zeros(3)}
    payload = {"model": {"matrix": torch.ones(3)}, "step": 64, "best_step": 0,
        "gradient_updates_performed": 58, "attempted_batches": 64,
        "config_sha256": "1a5410dda6c87086a35801e646688f408751eec0d99a0c8d01865d57f6d9c0df",
        "parent_checkpoint_sha256": capacity.SELECTED_SHA256, "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True, "optimizer": {"state": {}}}
    if field:
        payload[field] = value
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=parent)
    else:
        capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=parent)
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=payload["model"])


@pytest.mark.parametrize("mutation", [None, "rate", "accepted", "reason", "rollback"])
def test_only_original_rejected_rate2_with_exact_rollback_is_admitted(monkeypatch, mutation):
    monkeypatch.setattr(capacity, "sha256", lambda path: "config")
    payload = {"config_sha256": "config", "record": {"rate": 1e-4, "accepted": False,
        "protected_tasks": 42, "reason": "finite_metrics_rejected"}, "optimizer": {"state": {}},
        "source_raw_sha256": "original", "snapshot": {name: torch.zeros(3) for name in
        ("before", "tentative", "after", "combined_gradient", "applied_gradient")}}
    if mutation == "rollback":
        payload["snapshot"]["after"][0] = 1.
    elif mutation:
        payload["record"][mutation] = {"rate": 2e-5, "accepted": True, "reason": "accepted"}[mutation]
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "rate_endpoint", rate_parameters=3)
    else:
        capacity.validate_payload(payload, "rate_endpoint", rate_parameters=3)


def test_vectors_keep_full_float32_finite_coordinates():
    capacity.validate_payload(torch.zeros(3), "vector", full_parameters=3)
    for value in (torch.zeros(2), torch.zeros(3, dtype=torch.float64), torch.full((3,), float("nan"))):
        with pytest.raises(ValueError):
            capacity.validate_payload(value, "vector", full_parameters=3)


def test_preservation_copy_is_exact_and_refuses_overwrite(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"immutable evidence")
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
        "sha256": capacity.sha256(source), "kind": "vector"}
    monkeypatch.setattr(capacity, "validate_file", lambda path, kind: None)
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)
