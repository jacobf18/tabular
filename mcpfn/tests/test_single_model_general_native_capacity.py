from copy import deepcopy

import pytest
import torch

import relocate_single_model_general_native_capacity as capacity


def test_scope_excludes_selected_teachers_and_unaudited_material():
    assert len(capacity.TARGETS) == 3
    assert capacity.TRAINING_ADMISSION_BYTES == 21877489664
    assert not any(token in name for name in capacity.TARGETS
                   for token in ("selected_checkpoint", "best_validation", "anchor_rehearsal_v28"))
    assert capacity.SELECTED.parent.name == "stronger_mean_distillation_v46"
    assert capacity.inventory.__globals__ is capacity.__dict__
    assert capacity.copy_and_verify.__globals__ is capacity.__dict__


@pytest.mark.parametrize("field,value", [(None, None), ("best_step", 50), ("step", 10),
    ("gradient_updates_performed", 56), ("attempted_batches", 63), ("config_sha256", "changed"),
    ("independent_confirmation_passed", True), ("optimizer", {}), ("full_network_training", False)])
def test_only_exact_nonselected_tail_is_admitted(field, value):
    selected = {"matrix": torch.zeros(3)}
    payload = {"model": {"matrix": torch.ones(3)}, "step": 64, "best_step": 10,
        "gradient_updates_performed": 55, "attempted_batches": 64,
        "config_sha256": "ac98d26801b73579928c510b6e76a7b8479cd4fa05c5690a5599e69127733c49",
        "independent_confirmation_passed": False, "preflight_only": False,
        "full_network_training": True, "optimizer": {"state": {}}}
    if field:
        payload[field] = value
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=payload["model"])


def test_vectors_preserve_shape_dtype_and_finiteness():
    capacity.validate_payload(torch.zeros(3), "vector", full_parameters=3)
    for value in (torch.zeros(2), torch.zeros(3, dtype=torch.float64), torch.full((3,), float("nan"))):
        with pytest.raises(ValueError):
            capacity.validate_payload(value, "vector", full_parameters=3)


def test_original_rate1_only(monkeypatch):
    monkeypatch.setattr(capacity, "sha256", lambda path: "config")
    payload = {"config_sha256": "config", "record": {"rate": 2e-5, "accepted": True, "protected_tasks": 42},
        "optimizer": {"state": {}}, "source_raw_sha256": "original",
        "snapshot": {name: torch.zeros(3) for name in
                     ("before", "tentative", "after", "combined_gradient", "applied_gradient")}}
    capacity.validate_payload(payload, "rate_endpoint", rate_parameters=3)
    for rate in (2e-6, 2e-4):
        invalid = deepcopy(payload)
        invalid["record"]["rate"] = rate
        with pytest.raises(ValueError):
            capacity.validate_payload(invalid, "rate_endpoint", rate_parameters=3)


def test_copy_is_exact_and_refuses_overwrite(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"immutable evidence")
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
           "sha256": capacity.sha256(source), "kind": "vector"}
    monkeypatch.setattr(capacity, "validate_file", lambda path, kind: None)
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)
