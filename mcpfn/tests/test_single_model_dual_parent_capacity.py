from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_dual_parent_capacity as capacity
import test_single_model_parent_structured_capacity as predecessor_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_exact_closed_scope_and_original_reserves():
    assert len(capacity.TARGETS) == 4
    assert set(capacity.TARGETS.values()) == {"tail", "vector", "decoder", "output_layer"}
    assert capacity.SELECTED.parent.name == "parent_structured_v49"
    assert capacity.SELECTED.name == "selected_checkpoint.pth"
    assert capacity.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    assert not any("v28" in name or "selected_checkpoint" in name for name in capacity.TARGETS)
    for function in vars(capacity).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(capacity):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize("mutation", [None, "step", "best_step", "gradient_updates_performed", "config_sha256",
    "parent_checkpoint_sha256", "independent_confirmation_passed", "full_network_training", "selected"])
def test_only_the_distinct_unqualified_v49_tail_is_eligible(mutation):
    payload = {"step": 64, "best_step": 25, "gradient_updates_performed": 62, "attempted_batches": 64,
        "config_sha256": "9eec9f2c3226153e68607fb42647fcc7c919940880c6530ef704cb2faae85b97",
        "parent_checkpoint_sha256": capacity.probe.PARENT_SHA256, "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True, "optimizer": {"state": {}},
        "model": {"weight": torch.ones(3)}}
    selected = {"weight": torch.zeros(3)}
    if mutation == "selected":
        selected = deepcopy(payload["model"])
    elif mutation:
        payload[mutation] = {"step": 50, "best_step": 64, "gradient_updates_performed": 61,
            "config_sha256": "changed", "parent_checkpoint_sha256": "changed",
            "independent_confirmation_passed": True, "full_network_training": False}[mutation]
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(payload, "tail", model_tensors=1, selected_model=selected)


@pytest.mark.parametrize("kind", ["decoder", "output_layer"])
@pytest.mark.parametrize("mutation", [None, "missing", "dtype", "shape", "nonfinite", "range", "name", "overlap"])
def test_both_audited_subspace_projections_preserve_exact_typed_coordinates(kind, mutation):
    payload = {name: torch.zeros(3) for name in ("before", "tentative", "candidate")}
    payload["ranges"] = [{"name": "decoder.layers.4.bias", "shape": [3], "start": 10, "stop": 13}]
    if mutation == "missing":
        payload.pop("candidate")
    elif mutation in {"dtype", "shape", "nonfinite"}:
        payload["candidate"] = {"dtype": torch.zeros(3).double(), "shape": torch.zeros(2),
            "nonfinite": torch.full((3,), float("nan"))}[mutation]
    elif mutation == "range":
        payload["ranges"][0]["shape"] = [2]
    elif mutation == "name":
        payload["ranges"][0]["name"] = "encoder.weight"
    elif mutation == "overlap":
        payload["ranges"].append(deepcopy(payload["ranges"][0]))
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, kind, decoder_parameters=3, output_parameters=3)
    else:
        capacity.validate_payload(payload, kind, decoder_parameters=3, output_parameters=3)


def test_full_vector_and_exact_copy_contracts_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(predecessor_tests, "capacity", capacity)
    predecessor_tests.test_vectors_keep_full_float32_finite_coordinates()
    predecessor_tests.test_preservation_copy_is_exact_and_refuses_overwrite(tmp_path, monkeypatch)
