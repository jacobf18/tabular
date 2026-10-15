from copy import deepcopy
import builtins
import inspect

import pytest
import torch

import relocate_single_model_fixed_half_capacity as relocation
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_original_v38_tail_identity_is_unchanged(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_copy_and_publication_preserve_the_actual_serialized_vector(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


@pytest.mark.parametrize("field,value", [("state", "complete"), ("failed_stage", "run"), ("error", "other"), ("pid", 0), ("pid", True)])
def test_recovered_owner_keeps_the_original_failure_not_a_spoofed_complete_status(field, value):
    current = {"state": "failed", "failed_stage": "audit", "error": "ValueError('Audit must be CPU only')", "pid": 123}
    relocation.validate_recovered_failure(current)
    current[field] = value
    with pytest.raises(ValueError):
        relocation.validate_recovered_failure(current)


def test_recovered_vector_identity_binds_exact_configuration_and_complete_fp32_values(monkeypatch):
    expected = {"kind": "vector", "parameters": 5, "frozen": 16, "cpu_audit_recovered": True}
    monkeypatch.setitem(relocation.OWNERS, relocation.recovery.ORIGINAL.name, expected)
    current = {"state": "failed", "failed_stage": "audit", "error": "ValueError('Audit must be CPU only')", "pid": 123}
    relocation.checkpoint_identity(torch.ones(5), current, expected, relocation.recovery.CONFIG_SHA256)
    for payload, configuration in ((torch.ones(4), relocation.recovery.CONFIG_SHA256),
        (torch.ones(5, dtype=torch.float64), relocation.recovery.CONFIG_SHA256), (torch.ones(5), "changed")):
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, configuration)
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(torch.ones(5), {**current, "state": "complete"}, expected, relocation.recovery.CONFIG_SHA256)


def test_six_file_scope_keeps_the_fresh_probe_all_teachers_and_selections_untouched():
    assert set(relocation.OWNERS) == set(relocation.previous.OWNERS) | {relocation.recovery.ORIGINAL.name}
    assert relocation.probe.OUTPUT.name == "fixed_half_shrunk_retention_preflight_v1"
    assert relocation.probe.OUTPUT.name not in relocation.OWNERS
    assert relocation.TRAINING_ADMISSION_BYTES == 21743271936
    assert 'len(records) != 6' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 6' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert 'owner_terminal(ROOT / name, expected)' in relocation.IDLE_SOURCE
    assert 'vector_evidence(directory, expected)' in relocation.INVENTORY_SOURCE
    assert relocation.recovery.ORIGINAL.name not in relocation.previous.OWNERS


def test_helpers_use_only_new_globals_and_never_mutate_an_older_relocation():
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
    for name in ("inventory_original", "regular_checkpoint_identity", "checkpoint_identity", "require_idle", "copy_and_verify", "require_probe", "main"):
        assert getattr(relocation, name).__globals__ is vars(relocation)
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify


@pytest.mark.parametrize("mutation", [None, "fraction", "incomplete", "size", "unchecked"])
def test_requires_the_new_probe_independent_fixed_half_vector_audit(monkeypatch, mutation):
    contraction = {"evaluated": True, "fraction": .5, "parameters": 77998496, "actual_saved_displacements_checked": True}
    changes = {"fraction": ("fraction", .25), "incomplete": ("evaluated", False),
        "size": ("parameters", 5), "unchecked": ("actual_saved_displacements_checked", False)}
    if mutation:
        field, value = changes[mutation]
        contraction[field] = value
    reports = {"result.json": {"record": {"displacement_fraction": .5}}, "audit.json": {"raw_endpoint": {"fixed_half_step": contraction}}}
    monkeypatch.setattr(relocation, "require_accepted_retention_probe", lambda: {"accepted": "proof"})
    monkeypatch.setattr(relocation.probe, "read", lambda path: deepcopy(reports[path.name]))
    if mutation:
        with pytest.raises(ValueError):
            relocation.require_probe()
    else:
        assert relocation.require_probe() == {"accepted": "proof"}


def test_storage_entry_requires_explicitly_hidden_and_uninitialized_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        relocation.main()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CPU-only"):
        relocation.main()


@pytest.mark.parametrize("mutation", [None, "accepted", "changed_gate", "gpu_audit", "missing_rejection", "qualified"])
def test_recovered_vector_evidence_preserves_negative_outcome_and_additive_audit(tmp_path, monkeypatch, mutation):
    original_directory, recovered_directory = tmp_path / relocation.recovery.ORIGINAL.name, tmp_path / "recovery"
    original_directory.mkdir()
    recovered_directory.mkdir()
    monkeypatch.setattr(relocation.recovery, "ORIGINAL", original_directory)
    monkeypatch.setattr(relocation.recovery, "OUTPUT", recovered_directory)
    result = {"record": {"accepted": False, "accepted_updates": 0, "reason": "finite_metrics_rejected"}}
    recovered = {"complete": True, "accepted": False, "original_failure_preserved": True,
        "original_rejection_preserved": True, "gate_changed": False, "cuda_initialized": False}
    audit = {"cuda_initialized": False, "artifact_sha256": {"original": "digest"}}
    gate = {"passed": False, "reason": "finite_metrics_rejected"}
    retention = {"complete": True, "cuda_initialized": False, "original_audit_sha256": "digest", "artifact_sha256": {"reference": "digest"}}
    if mutation == "accepted":
        result["record"]["accepted"] = True
    elif mutation == "changed_gate":
        recovered["gate_changed"] = True
    elif mutation == "gpu_audit":
        audit["cuda_initialized"] = True
    elif mutation == "missing_rejection":
        recovered["original_rejection_preserved"] = False
    elif mutation == "qualified":
        gate["passed"] = True
    reports = {"result.json": result, "recovery_result.json": recovered, "audit.json": audit,
        "gate.json": gate, "retention_audit.json": retention,
        "recovery_config.json": {"input_sha256": {"original": "digest"}, "source_sha256": {"source": "digest"}}}
    monkeypatch.setattr(relocation, "owner_terminal", lambda *arguments: None)
    monkeypatch.setattr(relocation.previous.copying, "read", lambda path: deepcopy(reports[path.name]))
    monkeypatch.setattr(relocation, "sha256", lambda path: relocation.RECOVERY_SHA256 if path.name == "recovery_result.json" else "digest")
    verified = []
    monkeypatch.setattr(relocation, "verify_hashes", lambda values: verified.append(values))
    if mutation:
        with pytest.raises(ValueError):
            relocation.vector_evidence(relocation.recovery.ORIGINAL, relocation.OWNERS[relocation.recovery.ORIGINAL.name])
    else:
        path, measured, original, negative, paths = relocation.vector_evidence(relocation.recovery.ORIGINAL, relocation.OWNERS[relocation.recovery.ORIGINAL.name])
        assert path == relocation.recovery.OUTPUT / "audit.json"
        assert measured == audit and original == result and negative == gate
        assert len(verified) == 4
        assert relocation.recovery.ORIGINAL / "result.json" in paths
