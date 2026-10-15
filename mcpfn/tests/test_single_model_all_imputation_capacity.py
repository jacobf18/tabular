import builtins
from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_all_imputation_capacity as relocation
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_unselected_tail_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_vector_copy_preserves_bytes(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


@pytest.mark.parametrize("mutation", [None, "selected", "step", "optimizer", "config", "dtype", "nonfinite", "size", "live"])
def test_smoke_is_not_a_candidate_and_requires_exact_inventory(mutation):
    payload = {"model": {"weight": torch.ones(5)}, "step": 21, "not_a_candidate": True,
        "config_sha256": relocation.SMOKE_CONFIG}
    current = {"state": "complete", "updates": 21}
    expected = {"kind": "smoke", "tensors": 1, "elements": 5}
    if mutation == "selected":
        payload["not_a_candidate"] = False
    elif mutation == "step":
        payload["step"] = 20
    elif mutation == "optimizer":
        payload["optimizer"] = {}
    elif mutation == "config":
        payload["config_sha256"] = "other"
    elif mutation == "dtype":
        payload["model"]["weight"] = torch.ones(5, dtype=torch.float64)
    elif mutation == "nonfinite":
        payload["model"]["weight"][0] = float("nan")
    elif mutation == "size":
        payload["model"]["weight"] = torch.ones(4)
    elif mutation == "live":
        current["state"] = "running"
    if mutation is None:
        relocation.checkpoint_identity(payload, current, expected, relocation.SMOKE_CONFIG)
    else:
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, relocation.SMOKE_CONFIG)


def test_scope_capacity_and_closed_function_globals():
    assert len(relocation.OWNERS) == 3
    assert relocation.probe.OUTPUT.name not in relocation.OWNERS
    assert relocation.TRAINING_ADMISSION_BYTES == 21877489664
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name


@pytest.mark.parametrize("mutation", [None, "references", "status", "gate", "proof"])
def test_probe_requires_all_references_and_fixed_half_proof(monkeypatch, mutation):
    result = {"record": {"displacement_fraction": .5}}
    audit = {"raw_endpoint": {"fixed_half_step": {"fraction": .5, "evaluated": True,
        "parameters": 77998496, "actual_saved_displacements_checked": True}}}
    records = {"result.json": result, "audit.json": audit, "retention_audit.json": {"reference_grids": 26},
        "gate.json": {"passed": True, "preflight_only": True, "real_benchmark_success": False, "audit_sha256": "digest"},
        "status.json": {"retention_audited": True}}
    if mutation == "references":
        records["retention_audit.json"]["reference_grids"] = 25
    elif mutation == "status":
        records["status.json"] = {"accepted": True}
    elif mutation == "gate":
        records["gate.json"]["passed"] = False
    elif mutation == "proof":
        audit["raw_endpoint"]["fixed_half_step"]["fraction"] = 1.
    monkeypatch.setattr(relocation, "original_probe_evidence", lambda: {})
    monkeypatch.setattr(relocation.probe, "require_exact_predecessor", lambda: {})
    monkeypatch.setattr(relocation.probe, "read", lambda path: deepcopy(records[path.name]))
    monkeypatch.setattr(relocation, "sha256", lambda path: "digest")
    if mutation is None:
        assert len(relocation.require_probe()) == 1
    else:
        with pytest.raises(ValueError):
            relocation.require_probe()
