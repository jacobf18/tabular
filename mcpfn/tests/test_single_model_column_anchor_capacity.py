import builtins
import inspect
from pathlib import Path

import pytest

import relocate_single_model_column_anchor_capacity as relocation
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_completed_v37_tail_must_retain_its_exact_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_old_vector_copy_and_publication_preserve_all_bytes(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


def test_scope_preserves_new_probe_all_selections_teachers_and_other_artifacts():
    assert set(relocation.OWNERS) == {"anchor_retention_v37", "anchor_retention_preflight_v1", "paired_decoder_preflight_v1"}
    assert relocation.OWNERS["anchor_retention_v37"] == {"kind": "unselected_tail", "run": "anchor_retention_v37",
        "step": 64, "updates": 60, "selected": 0, "grids": 1368, "view_flag": "full_network_training"}
    assert relocation.OWNERS["anchor_retention_preflight_v1"]["parameters"] == 77998496
    assert relocation.OWNERS["paired_decoder_preflight_v1"] == {"kind": "vector", "parameters": 11227016, "frozen": 248}
    assert relocation.probe.OUTPUT.name == "column_anchor_retention_preflight_v1"
    assert relocation.TRAINING_ADMISSION_BYTES == 21743271936
    assert 'len(records) != 3' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify


def test_generated_helpers_have_no_old_global_or_closure_bindings():
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
    for name in ("inventory_original", "checkpoint_identity", "require_idle", "copy_and_verify", "main"):
        assert getattr(relocation, name).__globals__ is vars(relocation)


@pytest.mark.parametrize("mutation", [None, "rejected", "coverage", "specification", "nonfinite", "incomplete", "gpu_audit"])
def test_new_probe_is_required_with_dynamic_complete_reference_coverage(monkeypatch, mutation):
    directory = relocation.probe.OUTPUT
    records = [{"eligible": index < 17} for index in range(28)]
    result = {"data_admission_only": False, "record": {"accepted": True}, "trainable_parameters": 77998496}
    original = {"complete": True, "training_tasks": 28, "prediction_grids": 56, "frozen": {"frozen_tensors": 16}, "artifact_sha256": {}}
    retention = {"complete": True, "eligibility_decisions": 28, "reference_grids": 17,
        "endpoint_penalties": [{"before": 0., "candidate": 1.} for index in range(17)],
        "no_inference_ensemble": True, "cuda_initialized": False, "original_audit_sha256": "digest", "artifact_sha256": {}}
    manifest = {"complete": True, "specification": relocation.probe.retention.SPECIFICATION, "records": records}
    if mutation == "rejected":
        result["record"]["accepted"] = False
    elif mutation == "coverage":
        retention["reference_grids"] = 16
    elif mutation == "specification":
        manifest["specification"] = {}
    elif mutation == "nonfinite":
        retention["endpoint_penalties"][0]["candidate"] = float("nan")
    elif mutation == "incomplete":
        original["complete"] = False
    elif mutation == "gpu_audit":
        retention["cuda_initialized"] = True
    inputs = {directory / name: payload for name, payload in zip(
        ("result.json", "audit.json", "retention_audit.json", "retention_targets/manifest.json"),
        (result, original, retention, manifest))}
    monkeypatch.setattr(relocation.probe, "verify", lambda output: None)
    monkeypatch.setattr(relocation.readers, "terminal", lambda path: None)
    monkeypatch.setattr(relocation.probe, "read", lambda path: inputs[Path(path)])
    monkeypatch.setattr(relocation, "sha256", lambda path: "digest")
    monkeypatch.setattr(relocation, "verify_hashes", lambda hashes: None)
    if mutation:
        with pytest.raises(ValueError):
            relocation.require_probe()
    else:
        assert len(relocation.require_probe()) == 7
