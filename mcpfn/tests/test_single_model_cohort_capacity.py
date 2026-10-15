from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import torch

import relocate_single_model_cohort_capacity as relocation
import test_single_model_legacy_blend_capacity as previous_tests


def probe_records():
    records = previous_tests.probe_records()
    records[2].update(reference={role: {**identity, "state_tensors": 254, "bitwise_equal": True}
        for role, identity in relocation.probe.reference_identities().items()},
        reference_role_counts={"v8": 4, "v29": 22}, endpoint_penalties=[{} for index in range(26)])
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("record", "displacement_fraction"), 1.), (1, ("cuda_initialized",), True),
    (1, ("teacher_replay", "full_grids"), 27), (1, ("teacher_replay", "component_grids"), 3),
    (1, ("teacher_replay", "observed_weights_refitted"), False), (2, ("reference_grids",), 25),
    (2, ("reference",), {}), (2, ("reference_role_counts", "v8"), 3), (2, ("reference_role_counts", "v29"), 23),
    (2, ("endpoint_penalties",), []), (3, ("passed",), False), (3, ("audit_sha256",), "wrong"),
    (4, ("state",), "running"), (4, ("retention_audited",), False),
])
def test_storage_requires_the_complete_new_cohort_probe(monkeypatch, entry, path, value):
    monkeypatch.setattr(relocation, "sha256", lambda path: "digest")
    records = probe_records()
    relocation.validate_probe(*records)
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        relocation.validate_probe(*records)


@pytest.mark.parametrize("mutation", [None, "step", "updates", "selected", "model", "optimizer", "config", "qualified", "scope"])
def test_preserves_unselected_step64_without_relabeling_selected_step50(mutation):
    payload = {"step": 64, "attempted_batches": 64, "gradient_updates_performed": 56, "best_step": 50,
        "best_score": 1., "config_sha256": "config", "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True,
        "model": {f"tensor_{index}": torch.ones(1) for index in range(254)}, "optimizer": {"state": {}}}
    current = {"state": "training_complete", "step": 64, "accepted_updates": 56, "best": {"step": 50, "score": 1.}}
    if mutation:
        field, value = {"step": ("step", 50), "updates": ("gradient_updates_performed", 55),
            "selected": ("best_step", 64), "model": ("model", {}), "optimizer": ("optimizer", {}),
            "config": ("config_sha256", "wrong"), "qualified": ("independent_confirmation_passed", True),
            "scope": ("full_network_training", False)}[mutation]
        payload[field] = value
    expected = relocation.OWNERS["legacy_blend_teacher_v41"]
    if mutation:
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, "config")
    else:
        relocation.checkpoint_identity(payload, current, expected, "config")


@pytest.mark.parametrize("mutation", [None, "norm", "nonfinite", "shape", "selected", "incomplete", "extra", "products", "missing_products"])
def test_reconstructed_vector_keeps_original_negative_result_and_full_shape(mutation):
    expected = {**relocation.OWNERS["finite_guarded_v13_endpoint_replay"], "parameters": 4}
    payload = {"parameters": torch.ones(4), "norm": expected["norm"], "products": [0.] * 42}
    current = {"state": "complete", "selected_validation_passed": False}
    if mutation == "norm":
        payload["norm"] += .1
    elif mutation == "nonfinite":
        payload["parameters"][0] = float("nan")
    elif mutation == "shape":
        payload["parameters"] = torch.ones(3)
    elif mutation == "selected":
        current["selected_validation_passed"] = True
    elif mutation == "incomplete":
        current["state"] = "running"
    elif mutation == "extra":
        payload["qualified"] = True
    elif mutation == "products":
        payload["products"][0] = float("nan")
    elif mutation == "missing_products":
        payload["products"].pop()
    if mutation:
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, "config")
    else:
        relocation.checkpoint_identity(payload, current, expected, "config")


def test_v17_reconstruction_preserves_its_original_positive_development_outcome():
    expected = {**relocation.OWNERS["v8_decoder_v17_endpoint_replay"], "parameters": 4}
    payload = {"parameters": torch.ones(4), "norm": expected["norm"], "products": [0.] * 42}
    current = {"state": "complete", "selected_validation_passed": True}
    relocation.checkpoint_identity(payload, current, expected, "config")
    current["selected_validation_passed"] = False
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, expected, "config")


@pytest.mark.parametrize("mutation", [None, "symlink", "bytes", "samefile"])
def test_selected_checkpoint_stays_distinct_and_physically_on_original_volume(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "legacy_blend_teacher_v41"
    directory.mkdir()
    selected = directory / "selected_checkpoint.pth"
    tail = directory / "latest_checkpoint.pth"
    selected.write_bytes(b"selected")
    tail.write_bytes(b"tail")
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "SELECTED_SHA256", relocation.sha256(selected))
    if mutation == "bytes":
        selected.write_bytes(b"changed")
    elif mutation == "symlink":
        selected.unlink()
        selected.symlink_to(tail)
    elif mutation == "samefile":
        tail.unlink()
        tail.hardlink_to(selected)
    if mutation:
        with pytest.raises(ValueError):
            relocation.require_selected_preserved()
    else:
        relocation.require_selected_preserved()


def test_copy_fsync_hash_mmap_precedes_publication_and_keeps_original_source(tmp_path, monkeypatch):
    assert len(relocation.OWNERS) == 7 and '"canonical_artifacts_checked": 7' in relocation.MAIN_SOURCE
    directory = tmp_path / "owner"
    directory.mkdir()
    path = directory / "reconstructed_candidate.pt"
    expected = {**relocation.OWNERS["finite_guarded_v13_endpoint_replay"], "parameters": 4}
    torch.save({"parameters": torch.arange(4, dtype=torch.float32), "norm": expected["norm"], "products": [0.] * 42}, path)
    relocation.write_json(directory / "config.json", {})
    relocation.write_json(directory / "status.json", {"state": "complete", "selected_validation_passed": False})
    monkeypatch.setattr(relocation, "OWNERS", {"owner": expected})
    destination = tmp_path / "scratch/owner/reconstructed_candidate.pt"
    row = {"source": str(path), "destination": str(destination), "sha256": relocation.sha256(path), "bytes": path.stat().st_size}
    relocation.copy_and_verify(row)
    assert not path.is_symlink() and relocation.sha256(path) == relocation.sha256(destination) == row["sha256"]
    assert torch.equal(torch.load(destination, mmap=True, weights_only=True)["parameters"], torch.arange(4, dtype=torch.float32))
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert 'tests/test_single_model_cohort_capacity.py' in relocation.MAIN_SOURCE
    assert relocation.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    for name in ("main", "copy_and_verify", "original_probe_evidence", "validate_original_probe"):
        assert getattr(relocation, name).__globals__ is vars(relocation)
    assert not torch.cuda.is_initialized()
