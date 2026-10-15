import builtins
import inspect

import pytest
import torch

import relocate_single_model_legacy_blend_capacity as relocation
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_unselected_tail_retains_exact_terminal_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_vectors_preserve_every_serialized_byte_and_canonical_path(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


def test_exact_scope_capacity_and_generated_function_bindings():
    assert set(relocation.OWNERS) == {"all_imputation_retention_v40", "all_imputation_retention_preflight_v1", "decoder_training_preflight_v1"}
    assert relocation.OWNERS["all_imputation_retention_v40"]["updates"] == 58
    assert relocation.OWNERS["all_imputation_retention_v40"]["selected"] == 0
    assert relocation.OWNERS["decoder_training_preflight_v1"] == {"kind": "vector", "parameters": 11227016,
        "frozen": 248, "tasks": 14, "grids": 28}
    assert relocation.probe.OUTPUT.name not in relocation.OWNERS
    assert relocation.TRAINING_ADMISSION_BYTES == 21877489664
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert 'tests/test_single_model_legacy_blend_capacity.py' in relocation.MAIN_SOURCE
    assert relocation.previous.OWNERS != relocation.OWNERS
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
    for name in ("inventory_original", "checkpoint_identity", "require_idle", "copy_and_verify", "main", "original_probe_evidence"):
        assert getattr(relocation, name).__globals__ is vars(relocation)


def probe_records():
    result = {"record": {"displacement_fraction": .5}}
    audit = {"cuda_initialized": False, "raw_endpoint": {"fixed_half_step": {"fraction": .5, "evaluated": True,
        "parameters": 77998496, "actual_saved_displacements_checked": True}},
        "teacher_replay": {"complete": True, "full_grids": 28, "bitwise_equal": True,
            "blend_tasks": 2, "component_grids": 4, "observed_weights_refitted": True}}
    return [result, audit, {"reference_grids": 26},
        {"passed": True, "preflight_only": True, "real_benchmark_success": False, "audit_sha256": "digest"},
        {"state": "complete", "retention_audited": True}]


@pytest.mark.parametrize("entry,path,value", [
    (0, ("record", "displacement_fraction"), 1.), (1, ("cuda_initialized",), True),
    (1, ("raw_endpoint", "fixed_half_step", "parameters"), 1),
    (1, ("teacher_replay", "complete"), False), (1, ("teacher_replay", "full_grids"), 27),
    (1, ("teacher_replay", "bitwise_equal"), False), (1, ("teacher_replay", "blend_tasks"), 1),
    (1, ("teacher_replay", "component_grids"), 2), (1, ("teacher_replay", "observed_weights_refitted"), False),
    (2, ("reference_grids",), 25), (3, ("passed",), False), (3, ("preflight_only",), False),
    (3, ("real_benchmark_success",), True), (3, ("audit_sha256",), "changed"),
    (4, ("state",), "awaiting_independent_audit"), (4, ("retention_audited",), False)])
def test_probe_requires_all_complete_independent_evidence(monkeypatch, entry, path, value):
    monkeypatch.setattr(relocation, "sha256", lambda path: "digest")
    records = probe_records()
    relocation.validate_probe(*records)
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        relocation.validate_probe(*records)


@pytest.mark.parametrize("mutation", [None, "tasks", "grids", "hash", "gate", "frozen", "parameters", "incomplete"])
def test_actual_inventory_requires_original14_task_decoder_evidence(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "owner"
    directory.mkdir()
    vector = directory / "candidate_parameters.pt"
    torch.save(torch.ones(4), vector)
    expected = {"kind": "vector", "parameters": 4, "frozen": 248, "tasks": 14, "grids": 28}
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "OWNERS", {"owner": expected})
    monkeypatch.setattr(relocation, "require_idle", lambda: None)
    monkeypatch.setattr(relocation.previous.copying, "require_mount", lambda: None)
    unit = tmp_path / "unit"
    unit.write_text("unit")
    monkeypatch.setattr(relocation.previous.copying, "UNIT", unit)
    audit = {"complete": True, "training_tasks": 14, "prediction_grids": 28,
        "frozen": {"frozen_tensors": 248}, "artifact_sha256": {str(vector): relocation.sha256(vector)}}
    result = {"preflight_only": True, "trainable_parameters": 4}
    if mutation == "tasks":
        audit["training_tasks"] = 28
    elif mutation == "grids":
        audit["prediction_grids"] = 56
    elif mutation == "hash":
        audit["artifact_sha256"][str(vector)] = "wrong"
    elif mutation == "frozen":
        audit["frozen"]["frozen_tensors"] = 16
    elif mutation == "parameters":
        result["trainable_parameters"] = 5
    elif mutation == "incomplete":
        audit["complete"] = False
    for name, payload in {"config.json": {}, "registration.json": {}, "status.json": {"state": "complete"},
            "result.json": result, "audit.json": audit}.items():
        relocation.write_json(directory / name, payload)
    relocation.write_json(directory / "gate.json", {"preflight_only": True,
        "audit_sha256": "wrong" if mutation == "gate" else relocation.sha256(directory / "audit.json")})
    if mutation:
        with pytest.raises(ValueError):
            relocation.inventory_original()
    else:
        rows, evidence = relocation.inventory_original()
        assert len(rows) == 1 and rows[0]["sha256"] == relocation.sha256(vector)
        assert str(directory / "audit.json") in evidence and not vector.is_symlink()
