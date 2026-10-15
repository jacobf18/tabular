import inspect

import pytest
import torch

import relocate_single_model_parent_structured_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_scope_keeps_teachers_selected_and_unaudited_material():
    assert len(capacity.TARGETS) == 5
    assert capacity.TRAINING_ADMISSION_BYTES == 21877489664
    assert not any(token in name for name in capacity.TARGETS for token in ("selected_checkpoint", "best_validation", "anchor_rehearsal_v28"))
    assert capacity.SELECTED.parent.name == "native_column_mean_v48"
    for function in vars(capacity).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(capacity):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize("field,value", [(None, None), ("best_step", 64), ("step", 10), ("gradient_updates_performed", 55),
    ("attempted_batches", 63), ("config_sha256", "changed"), ("parent_checkpoint_sha256", "other"),
    ("independent_confirmation_passed", True), ("optimizer", {}), ("full_network_training", False)])
def test_only_original_v48_nonselected_tail_is_admitted(field, value):
    selected = {"matrix": torch.zeros(3)}
    payload = {"model": {"matrix": torch.ones(3)}, "step": 64, "best_step": 10,
        "gradient_updates_performed": 56, "attempted_batches": 64,
        "config_sha256": "9f34ec4f80daae3609135eb41ffe18c58492f6db62ebecbb53759bb0d4e8e1fa",
        "parent_checkpoint_sha256": capacity.probe.PARENT_SHA256, "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True, "optimizer": {"state": {}}}
    if field:
        payload[field] = value
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=payload["model"])


@pytest.mark.parametrize("kind", list(capacity.MATCHED_CASES))
@pytest.mark.parametrize("mutation", [None, "method", "rate", "accepted", "reason", "after", "coordinates"])
def test_complete_original_matched_control_outcomes_are_preserved(kind, mutation):
    method, rate, accepted = capacity.MATCHED_CASES[kind]
    payload = {"reference": {"path": str(capacity.ROOT / "point_inclusive_preflight_v1/raw_update.pt"),
        "sha256": "c17b357aa90ec33d66d9f11006c225e6f7354576537f1e3528ce13b63e501d0d"},
        "case": {"name": kind, "method": method, "rate": rate, "reused": False}, "method": method,
        "final_optimizer_bitwise_verified": True, "after_source": "candidate" if accepted else "initialization",
        "record": {"rate": rate, "accepted": accepted, "margin": .1, "composite_tasks": 28,
            "protected_tasks": 84, "step": 1, "accepted_updates": int(accepted),
            "reason": "accepted" if accepted else "finite_metrics_rejected"},
        "snapshot": {"optimizer_before": {"state": {}}, "optimizer_after": {"state": {}},
            **{name: torch.zeros(3) for name in ("tentative", "combined_gradient", "applied_gradient")}},
        "frozen": {"first": "a", "second": "b", "third": "c"}}
    if mutation == "method":
        payload["method"] = "changed"
    elif mutation == "after":
        payload["after_source"] = "changed"
    elif mutation == "coordinates":
        payload["snapshot"]["tentative"][0] = float("nan")
    elif mutation:
        payload["record"][mutation] = {"rate": 1., "accepted": not accepted, "reason": "changed"}[mutation]
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, kind, proposal_parameters=3)
    else:
        capacity.validate_payload(payload, kind, proposal_parameters=3)


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
