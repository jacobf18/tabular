from copy import deepcopy

import pytest
import torch

import relocate_single_model_uniform_v8_capacity as relocation


@pytest.mark.parametrize("mutation", [None, "shape", "dtype", "nan", "kind"])
def test_only_complete_finite_probe_vectors_are_preserved(mutation):
    vector = torch.arange(5, dtype=torch.float32)
    kind = "vector"
    if mutation == "shape":
        vector = vector[:-1]
    elif mutation == "dtype":
        vector = vector.double()
    elif mutation == "nan":
        vector[0] = float("nan")
    elif mutation == "kind":
        kind = "selected"
    if mutation is None:
        relocation.validate_payload(vector, kind, full_parameters=5)
    else:
        with pytest.raises(ValueError):
            relocation.validate_payload(vector, kind, full_parameters=5)


@pytest.mark.parametrize("mutation", [None, "config", "rate", "accepted", "protected", "shape", "nan"])
def test_closed_rate_endpoint_keeps_original_scope_and_all_full_vectors(monkeypatch, mutation):
    monkeypatch.setattr(relocation, "sha256", lambda path: "fixed-config")
    payload = {"config_sha256": "fixed-config", "record": {"rate": 2e-6, "accepted": True, "protected_tasks": 42},
        "optimizer": {"state": {}}, "source_raw_sha256": "original-raw",
        "snapshot": {name: torch.arange(5, dtype=torch.float32) for name in
                     ("before", "tentative", "after", "combined_gradient", "applied_gradient")}}
    if mutation == "config":
        payload["config_sha256"] = "changed"
    elif mutation in ("rate", "accepted", "protected"):
        key = "protected_tasks" if mutation == "protected" else mutation
        payload["record"][key] = {"rate": 2e-5, "accepted": False, "protected": 14}[mutation]
    elif mutation == "shape":
        payload["snapshot"]["after"] = payload["snapshot"]["after"][:-1]
    elif mutation == "nan":
        payload["snapshot"]["applied_gradient"][0] = float("nan")
    if mutation is None:
        relocation.validate_payload(payload, "rate_endpoint", rate_parameters=5)
    else:
        with pytest.raises(ValueError):
            relocation.validate_payload(payload, "rate_endpoint", rate_parameters=5)


def test_copy_fsync_and_atomic_link_preserve_exact_original_serialization(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.pt", tmp_path / "scratch" / "copy.pt"
    torch.save(torch.arange(5, dtype=torch.float32), source)
    original = source.read_bytes()
    row = {"source": str(source), "destination": str(destination), "sha256": relocation.sha256(source),
           "bytes": source.stat().st_size, "kind": "vector"}
    validate = relocation.validate_payload
    monkeypatch.setattr(relocation, "validate_payload", lambda payload, kind: validate(payload, kind, full_parameters=5))
    relocation.copy_and_verify(row)
    assert not source.is_symlink()
    relocation.copying.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    assert source.read_bytes() == destination.read_bytes() == original


def test_existing_preservation_is_not_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(relocation, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError):
        relocation.main()


def test_scope_does_not_include_selected_models_or_teachers():
    assert len(relocation.TARGETS) == 3
    assert all("selected_checkpoint" not in name and "best_validation_checkpoint" not in name for name in relocation.TARGETS)
    assert relocation.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
